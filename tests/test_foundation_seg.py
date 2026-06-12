"""Tests du backend de segmentation par modèle de fondation.

Le cœur `_segment_from_features` et le post-traitement sont testables SANS torch
(ils n'utilisent que numpy/opencv/skimage/sklearn), donc couverts en CI. Seuls
les tests qui appellent l'encodeur DINOv2 se sautent si torch est absent.
"""
import numpy as np
import pytest

import foundation_seg as fs


def test_available_returns_bool():
    assert isinstance(fs.available(), bool)


def test_backend_info_is_str():
    assert isinstance(fs.backend_info(), str) and fs.backend_info()


# ── Cœur du pipeline (sans torch) ──────────────────────────────────────────────
def _toy_inputs(n=80):
    # Image : moitié gauche sombre, moitié droite claire.
    img = np.zeros((n, n, 3), dtype=np.uint8)
    img[:, n // 2:] = 200
    # Features de grille : valeurs distinctes gauche/droite (sémantique simulée).
    g = 6
    gf = np.zeros((g, g, 4), dtype=np.float32)
    gf[:, : g // 2] = [1, 0, 0, 0]
    gf[:, g // 2:] = [0, 1, 0, 0]
    # Traits : classe 2 à gauche, classe 3 à droite.
    lab = np.zeros((n, n), dtype=np.uint8)
    lab[n // 2 - 3:n // 2 + 3, 5:11] = 2
    lab[n // 2 - 3:n // 2 + 3, n - 11:n - 5] = 3
    return gf, img, lab


def test_segment_from_features_separates_halves():
    gf, img, lab = _toy_inputs()
    out = fs._segment_from_features(gf, img, lab, round_disc=False)
    assert out.shape == img.shape[:2]
    assert set(np.unique(out)).issubset({2, 3})
    h, w = out.shape
    assert (out[:, : w // 2] == 2).mean() > 0.7
    assert (out[:, w // 2:] == 3).mean() > 0.7


def test_segment_from_features_raises_on_single_label():
    gf, img, lab = _toy_inputs()
    lab[lab == 3] = 0  # une seule classe restante
    with pytest.raises(ValueError):
        fs._segment_from_features(gf, img, lab)


def test_sensitivity_grows_lesion():
    gf, img, lab = _toy_inputs()
    low = fs._segment_from_features(gf, img, lab, sensitivity=0.2, round_disc=False)
    high = fs._segment_from_features(gf, img, lab, sensitivity=0.9, round_disc=False)
    assert (high == 2).sum() >= (low == 2).sum()


# ── Post-traitement ────────────────────────────────────────────────────────────
def test_postprocess_removes_small_lesions():
    pix = np.full((100, 100), 3, dtype=np.uint8)
    pix[10:40, 10:40] = 2          # grande lésion
    pix[80, 80] = 2                # speck isolé
    out = fs._postprocess(pix, np.array([2, 3], np.uint8), min_size=50,
                          round_disc=False, disc_label=1, lesion_label=2)
    assert out[25, 25] == 2        # grande lésion conservée
    assert out[80, 80] == 3        # speck retiré


def test_postprocess_rounds_optic_disc():
    # Disque « en croix » irrégulier → l'a priori doit le régulariser (ellipse,
    # composante unique, surface remplie).
    pix = np.full((120, 120), 3, dtype=np.uint8)
    pix[50:70, 20:100] = 1
    pix[20:100, 50:70] = 1
    out = fs._postprocess(pix, np.array([1, 3], np.uint8), min_size=0,
                          round_disc=True, disc_label=1, lesion_label=2)
    from scipy import ndimage as ndi
    _, ncc = ndi.label(out == 1)
    assert ncc == 1                       # une seule composante (pas une croix)
    assert (out == 1).sum() > (pix == 1).sum()  # ellipse pleine ⊇ la croix


def test_superpixel_dino_shape():
    seg = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
    grid = np.random.RandomState(0).rand(2, 2, 5).astype(np.float32)
    gci = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
    sp = fs._superpixel_dino(seg, 4, gci, grid)
    assert sp.shape == (4, 5)

# L'intégration de l'encodeur DINOv2 (torch) est vérifiée hors tests, dans le
# conteneur Linux (docker exec) — torch n'est pas requis pour la CI.
