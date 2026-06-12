"""Segmentation interactive expérimentale par modèle de fondation (admin-only).

Principe — identique à la boucle « scribbles » du RandomForest, mais les
descripteurs faits-main sont remplacés par les embeddings denses d'un encodeur
de fondation gelé (DINOv2 par défaut). L'image est encodée UNE fois (puis mise
en cache) ; seul un petit classifieur se réentraîne à chaque trait → interactif.

Important :
- N'altère EN RIEN la segmentation RandomForest décrite dans le manuscrit.
- torch n'est pas une dépendance du cœur : import paresseux. Si torch/le modèle
  ne sont pas disponibles, available() renvoie False et l'onglet le signale.
- Les features sont calculées à la résolution de la grille de patchs (≈ image/14)
  pour rester légères en mémoire ; la prédiction est ré-échantillonnée ensuite.
"""
import hashlib
import os
import threading

import numpy as np

BACKEND = os.environ.get("FUNDUS_FM_BACKEND", "dinov2")
_PATCH = 14            # DINOv2 ViT-S/14
_LONGEST = 518         # côté le plus long après redimensionnement (multiple de 14)
_CACHE_MAX = 4

_model = None
_model_lock = threading.Lock()
_feat_cache = {}       # image_hash -> (gh, gw, C) float32
_load_error = None


def available():
    """torch est-il importable ? (le modèle est chargé paresseusement ensuite)."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def backend_info():
    if BACKEND == "dinov2":
        return "DINOv2 ViT-S/14 (frozen, CPU)"
    return BACKEND


def _torch_home():
    base = os.environ.get("FUNDUS_DATA_DIR", "userdata")
    path = os.path.join(base, "torch_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _load_model():
    global _model, _load_error
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        import torch
        torch.hub.set_dir(os.path.join(_torch_home(), "hub"))
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        if BACKEND == "dinov2":
            model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
            )
        else:
            raise RuntimeError(f"Unknown FUNDUS_FM_BACKEND: {BACKEND}")
        model.eval()
        _model = model
        return _model


def _image_hash(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def extract_grid_features(arr_rgb):
    """Renvoie les embeddings denses (gh, gw, C) de l'image, avec cache."""
    key = _image_hash(arr_rgb)
    cached = _feat_cache.get(key)
    if cached is not None:
        return cached

    import torch
    import torch.nn.functional as F
    from PIL import Image

    model = _load_model()
    h0, w0 = arr_rgb.shape[:2]
    scale = _LONGEST / float(max(h0, w0))
    hr = max(_PATCH, int(round(h0 * scale / _PATCH)) * _PATCH)
    wr = max(_PATCH, int(round(w0 * scale / _PATCH)) * _PATCH)

    img = Image.fromarray(arr_rgb).convert("RGB").resize((wr, hr), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0)

    with torch.no_grad():
        out = model.forward_features(x)
        tokens = out["x_norm_patchtokens"][0]   # (N, C)

    gh, gw = hr // _PATCH, wr // _PATCH
    feats = tokens.reshape(gh, gw, -1).contiguous().numpy().astype(np.float32)

    if len(_feat_cache) >= _CACHE_MAX:
        _feat_cache.pop(next(iter(_feat_cache)))
    _feat_cache[key] = feats
    return feats


def _labels_to_grid(label_mask, gh, gw):
    """Vote majoritaire des pixels marqués vers la grille de patchs."""
    h0, w0 = label_mask.shape
    ys, xs = np.nonzero(label_mask)
    grid = np.zeros((gh, gw), dtype=np.int32)
    if len(ys) == 0:
        return grid
    gy = np.clip((ys.astype(np.float64) / h0 * gh).astype(int), 0, gh - 1)
    gx = np.clip((xs.astype(np.float64) / w0 * gw).astype(int), 0, gw - 1)
    labels = label_mask[ys, xs].astype(int)
    k = int(labels.max())
    votes = np.zeros((gh, gw, k + 1), dtype=np.int32)
    np.add.at(votes, (gy, gx, labels), 1)
    has = votes[:, :, 1:].sum(axis=2) > 0
    grid[has] = votes[:, :, 1:].argmax(axis=2)[has] + 1
    return grid


def _refine_contours(prob_grid, classes, image_rgb, out_hw):
    """Affine les contours : probabilités ré-échantillonnées en bilinéaire puis
    moyennées par superpixel SLIC (les frontières suivent les bords de l'image).
    Supprime l'aspect « en blocs » dû à la résolution des patchs."""
    import cv2
    from skimage.segmentation import slic

    h0, w0 = out_hw
    prob_full = cv2.resize(prob_grid, (w0, h0), interpolation=cv2.INTER_LINEAR)
    if prob_full.ndim == 2:
        prob_full = prob_full[..., None]

    n_seg = int(np.clip((h0 * w0) // 1500, 400, 2500))
    seg = slic(image_rgb, n_segments=n_seg, compactness=15,
               start_label=0, channel_axis=-1)
    nseg = int(seg.max()) + 1
    k = prob_full.shape[2]
    flat_seg = seg.reshape(-1)
    sums = np.zeros((nseg, k), dtype=np.float64)
    np.add.at(sums, flat_seg, prob_full.reshape(-1, k))
    counts = np.bincount(flat_seg, minlength=nseg).reshape(-1, 1)
    means = sums / np.maximum(counts, 1)
    sp_label = classes[means.argmax(axis=1)]
    return sp_label[flat_seg].reshape(h0, w0).astype(np.uint8)


def segment(arr_rgb, label_mask, refine=True):
    """Segmente l'image à partir des traits.

    arr_rgb : HxWx3 uint8 ; label_mask : HxW (0 = non marqué, 1..K = classes).
    Renvoie un masque HxW uint8 de labels prédits (mêmes valeurs que les traits).
    """
    from sklearn.linear_model import LogisticRegression

    if arr_rgb.ndim == 2:
        arr_rgb = np.stack([arr_rgb] * 3, axis=-1)
    if arr_rgb.shape[2] == 4:
        arr_rgb = arr_rgb[:, :, :3]

    feats = extract_grid_features(arr_rgb)
    gh, gw, c = feats.shape
    flat = feats.reshape(-1, c)

    grid_labels = _labels_to_grid(label_mask, gh, gw)
    idx = grid_labels.reshape(-1) > 0
    if idx.sum() < 2 or len(np.unique(grid_labels[grid_labels > 0])) < 2:
        # Pas assez de classes après projection sur la grille.
        raise ValueError("not_enough_labels")

    # Normalisation simple (les embeddings ne sont pas centrés).
    mu = flat.mean(axis=0, keepdims=True)
    sd = flat.std(axis=0, keepdims=True) + 1e-6
    flat_n = (flat - mu) / sd

    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(flat_n[idx], grid_labels.reshape(-1)[idx])
    classes = clf.classes_.astype(np.uint8)
    proba = clf.predict_proba(flat_n)
    prob_grid = proba.reshape(gh, gw, proba.shape[1]).astype(np.float32)
    h0, w0 = label_mask.shape

    if refine:
        try:
            return _refine_contours(prob_grid, classes, arr_rgb, (h0, w0))
        except Exception:
            pass  # repli sur le plus proche voisin ci-dessous

    # Repli : prédiction par patch ré-échantillonnée au plus proche voisin.
    pred_grid = classes[prob_grid.argmax(axis=2)]
    yi = np.clip((np.arange(h0) / h0 * gh).astype(int), 0, gh - 1)
    xi = np.clip((np.arange(w0) / w0 * gw).astype(int), 0, gw - 1)
    return pred_grid[np.ix_(yi, xi)].astype(np.uint8)
