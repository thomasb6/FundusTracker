"""Tests du backend de segmentation par modèle de fondation.

Les tests qui nécessitent torch se sautent automatiquement si torch n'est pas
installé (CI / image cœur), pour ne pas exiger la dépendance lourde.
"""
import numpy as np
import pytest

import foundation_seg as fs


def test_available_returns_bool():
    assert isinstance(fs.available(), bool)


def test_backend_info_is_str():
    assert isinstance(fs.backend_info(), str) and fs.backend_info()


def test_labels_to_grid_majority_vote():
    # Quadrants distincts → projection correcte sur une grille 2x2.
    lab = np.zeros((28, 28), dtype=np.uint8)
    lab[0:14, 0:14] = 2
    lab[0:14, 14:28] = 3
    grid = fs._labels_to_grid(lab, 2, 2)
    assert grid[0, 0] == 2
    assert grid[0, 1] == 3
    assert grid[1, 0] == 0  # aucun pixel marqué dans ce quadrant


@pytest.mark.skipif(not fs.available(), reason="torch not installed")
def test_segment_separates_regions():
    rng = np.random.default_rng(0)
    img = (np.ones((300, 300, 3)) * 40 + rng.normal(0, 5, (300, 300, 3))).clip(0, 255).astype("uint8")
    yy, xx = np.mgrid[0:300, 0:300]
    disc = (yy - 150) ** 2 + (xx - 150) ** 2 < 40 ** 2
    img[disc] = [220, 200, 180]
    spot = (yy - 80) ** 2 + (xx - 210) ** 2 < 25 ** 2
    img[spot] = [200, 60, 60]
    lab = np.zeros((300, 300), "uint8")
    lab[148:152, 148:152] = 1
    lab[78:82, 208:212] = 2
    lab[10:14, 10:14] = 3
    pred = fs.segment(img, lab)
    assert pred.shape == (300, 300)
    assert set(np.unique(pred)).issubset({1, 2, 3})
    # Les features de fondation doivent retrouver disque et tache.
    assert ((pred == 1) & disc).sum() / disc.sum() > 0.5
    assert ((pred == 2) & spot).sum() / spot.sum() > 0.5


@pytest.mark.skipif(not fs.available(), reason="torch not installed")
def test_segment_raises_on_single_label():
    img = np.zeros((100, 100, 3), "uint8")
    lab = np.zeros((100, 100), "uint8")
    lab[10:14, 10:14] = 1  # une seule classe
    with pytest.raises(ValueError):
        fs.segment(img, lab)
