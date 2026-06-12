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


# ── Descripteurs rétiniens (résolution de travail) ─────────────────────────────
def _appearance_features(img_s):
    """Descripteurs locaux adaptés au fond d'œil : canal vert (où les lésions
    ressortent), R, B, moyenne/écart-type locaux, contraste local, et
    vascularisation (filtre de Frangi). Renvoie (H, W, A) float32 ~[0,1]."""
    import cv2
    from skimage.filters import frangi

    g = img_s[:, :, 1].astype(np.float32) / 255.0
    r = img_s[:, :, 0].astype(np.float32) / 255.0
    b = img_s[:, :, 2].astype(np.float32) / 255.0
    mean = cv2.blur(g, (15, 15))
    sq = cv2.blur(g * g, (15, 15))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    contrast = g - mean
    try:
        vessel = frangi(g, black_ridges=True).astype(np.float32)
        vessel = vessel / (float(vessel.max()) + 1e-6)
    except Exception:
        vessel = np.zeros_like(g)
    return np.stack([g, r, b, mean, std, contrast, vessel], axis=-1).astype(np.float32)


def _superpixel_dino(seg, nseg, gci, grid_feats):
    """Feature DINOv2 moyenne par superpixel via une matrice creuse
    superpixel→cellule-de-grille (évite d'upsampler densément les embeddings)."""
    gh, gw, c = grid_feats.shape
    ncell = gh * gw
    w = np.zeros((nseg, ncell), dtype=np.float32)
    np.add.at(w, (seg.reshape(-1), gci.reshape(-1)), 1.0)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1.0)
    return (w @ grid_feats.reshape(ncell, c)).astype(np.float32)


def _postprocess(pix_label, classes, min_size, round_disc, disc_label, lesion_label):
    """Nettoyage adapté aux lésions (trous bouchés, petites taches retirées,
    contour lissé) + a priori clinique « nerf optique rond » (ellipse)."""
    import cv2
    from scipy import ndimage as ndi
    from skimage.morphology import remove_small_objects, binary_closing, disk

    present = set(int(x) for x in classes)
    bg_label = 3 if 3 in present else next(
        (l for l in present if l not in (disc_label, lesion_label)), 3)
    out = pix_label.copy()

    if lesion_label in present:
        les = pix_label == lesion_label
        les = ndi.binary_fill_holes(les)
        if min_size > 0:
            les = remove_small_objects(les, int(min_size))
        les = binary_closing(les, disk(2))
        out[pix_label == lesion_label] = bg_label
        out[les] = lesion_label

    if disc_label in present:
        discm = pix_label == disc_label
        out[discm] = bg_label
        lbl, n = ndi.label(discm)
        if n >= 1:
            sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
            biggest = lbl == (1 + int(np.argmax(sizes)))
            reg = biggest
            if round_disc:
                ys, xs = np.nonzero(biggest)
                if len(xs) >= 5:
                    cnt = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
                    try:
                        canvas = np.zeros(pix_label.shape, np.uint8)
                        cv2.ellipse(canvas, cv2.fitEllipse(cnt), 1, -1)
                        reg = canvas.astype(bool)
                    except Exception:
                        reg = biggest
            out[reg] = disc_label
    return out


def _segment_from_features(grid_feats, arr_rgb, label_mask, *, sensitivity=0.5,
                           detail=0.5, min_lesion_frac=0.0003, round_disc=True,
                           disc_label=1, lesion_label=2, work_max=768):
    """Cœur de segmentation SANS torch (donc testable et utilisable une fois les
    embeddings DINOv2 calculés) : classification au niveau superpixel sur
    features DINOv2 + descripteurs rétiniens, avec réglages et a priori."""
    import cv2
    from skimage.segmentation import slic
    from sklearn.ensemble import RandomForestClassifier

    h0, w0 = label_mask.shape
    scale = min(1.0, work_max / float(max(h0, w0)))
    hw = max(1, int(round(h0 * scale)))
    ww = max(1, int(round(w0 * scale)))
    img_s = cv2.resize(arr_rgb, (ww, hw), interpolation=cv2.INTER_AREA)

    cell = float(np.interp(detail, [0.0, 1.0], [34.0, 12.0]))  # px / superpixel
    n_seg = int(np.clip((hw * ww) / (cell * cell), 300, 6000))
    seg = slic(img_s, n_segments=n_seg, compactness=12,
               start_label=0, channel_axis=-1)
    nseg = int(seg.max()) + 1

    gh, gw, _c = grid_feats.shape
    yy, xx = np.mgrid[0:hw, 0:ww]
    gci = (np.clip((yy / hw * gh).astype(int), 0, gh - 1) * gw
           + np.clip((xx / ww * gw).astype(int), 0, gw - 1))
    sp_dino = _superpixel_dino(seg, nseg, gci, grid_feats)

    app = _appearance_features(img_s)
    a = app.shape[2]
    sums = np.zeros((nseg, a), dtype=np.float64)
    np.add.at(sums, seg.reshape(-1), app.reshape(-1, a))
    counts = np.bincount(seg.reshape(-1), minlength=nseg).reshape(-1, 1)
    sp_app = (sums / np.maximum(counts, 1)).astype(np.float32)
    sp_feat = np.concatenate([sp_dino, sp_app], axis=1)

    # Traits → superpixels (poids = nombre de pixels marqués).
    lab_s = cv2.resize(label_mask, (ww, hw), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(lab_s)
    if len(ys) == 0:
        raise ValueError("not_enough_labels")
    sp_of = seg[ys, xs]
    lbls = lab_s[ys, xs].astype(int)
    votes = np.zeros((nseg, int(lbls.max()) + 1), dtype=np.int64)
    np.add.at(votes, (sp_of, lbls), 1)
    labeled = np.nonzero(votes[:, 1:].sum(axis=1) > 0)[0]
    sp_label = votes[labeled, 1:].argmax(axis=1) + 1
    sp_weight = votes[labeled, 1:].sum(axis=1)
    if len(np.unique(sp_label)) < 2:
        raise ValueError("not_enough_labels")

    clf = RandomForestClassifier(n_estimators=120, class_weight="balanced",
                                 random_state=0, n_jobs=1)
    clf.fit(sp_feat[labeled], sp_label, sample_weight=sp_weight)
    classes = clf.classes_.astype(np.uint8)
    proba = clf.predict_proba(sp_feat)

    # Sensibilité : biais monotone sur la classe « lésion » (0.5 = neutre).
    if lesion_label in classes:
        j = int(np.where(classes == lesion_label)[0][0])
        proba = proba.copy()
        proba[:, j] *= float(np.clip(sensitivity / 0.5, 0.0, 4.0))
        proba /= np.maximum(proba.sum(axis=1, keepdims=True), 1e-9)

    sp_pred = classes[proba.argmax(axis=1)]
    pix = sp_pred[seg]

    min_size = int(min_lesion_frac * hw * ww)
    pix = _postprocess(pix, classes, min_size, round_disc, disc_label, lesion_label)
    return cv2.resize(pix.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)


def segment(arr_rgb, label_mask, *, sensitivity=0.5, detail=0.5,
            min_lesion_frac=0.0003, round_disc=True, disc_label=1,
            lesion_label=2, refine=True):
    """Segmente l'image à partir des traits (modèle de fondation).

    arr_rgb : HxWx3 uint8 ; label_mask : HxW (0 = non marqué, 1..K = classes).
    Paramètres réglables : sensibilité (lésion), finesse (superpixels), taille
    minimale de lésion, a priori « nerf optique rond »."""
    if arr_rgb.ndim == 2:
        arr_rgb = np.stack([arr_rgb] * 3, axis=-1)
    if arr_rgb.shape[2] == 4:
        arr_rgb = arr_rgb[:, :, :3]
    grid_feats = extract_grid_features(arr_rgb)
    return _segment_from_features(
        grid_feats, arr_rgb, label_mask, sensitivity=sensitivity, detail=detail,
        min_lesion_frac=min_lesion_frac, round_disc=round_disc,
        disc_label=disc_label, lesion_label=lesion_label)
