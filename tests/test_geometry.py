"""Tests de régression du cœur géométrique (formules du manuscrit, section
« Measures of interest »). Valeurs de référence calculées à la main.

Ces tests verrouillent le comportement numérique : aucun refactor futur ne doit
modifier silencieusement une aire, un centroïde ou une distance — ce sont les
quantités sur lesquelles reposent les résultats publiés.
"""
import math

import pytest

import main


# ── Aire (Shoelace, valeur absolue) ────────────────────────────────────────────
class TestCalculateArea:
    def test_unit_square(self):
        assert main.calculate_area([(0, 0), (10, 0), (10, 10), (0, 10)]) == 100.0

    def test_right_triangle(self):
        # base 6, hauteur 8 → aire 24
        assert main.calculate_area([(0, 0), (6, 0), (0, 8)]) == 24.0

    def test_orientation_independent(self):
        # Le manuscrit : on prend la valeur absolue → sens horaire/anti-horaire
        # donne le même résultat.
        cw = [(0, 0), (0, 10), (10, 10), (10, 0)]
        ccw = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert main.calculate_area(cw) == main.calculate_area(ccw) == 100.0

    def test_degenerate_returns_zero(self):
        assert main.calculate_area([(0, 0), (1, 1)]) == 0
        assert main.calculate_area([]) == 0

    def test_translation_invariant(self):
        base = [(0, 0), (4, 0), (4, 4), (0, 4)]
        shifted = [(x + 100, y + 250) for x, y in base]
        assert main.calculate_area(shifted) == main.calculate_area(base) == 16.0


# ── Aire + centroïde (matching) ────────────────────────────────────────────────
class TestPolygonProperties:
    def test_square_area_and_centroid(self):
        area, centroid = main.calculate_polygon_properties_for_matching(
            [(0, 0), (10, 0), (10, 10), (0, 10)]
        )
        assert area == 100.0
        assert centroid == pytest.approx((5.0, 5.0))

    def test_triangle_centroid_is_vertex_mean(self):
        # Pour un triangle, le centroïde géométrique = moyenne des sommets.
        coords = [(0, 0), (9, 0), (0, 9)]
        area, centroid = main.calculate_polygon_properties_for_matching(coords)
        assert area == pytest.approx(40.5)
        assert centroid == pytest.approx((3.0, 3.0))

    def test_too_few_points(self):
        assert main.calculate_polygon_properties_for_matching([(0, 0), (1, 1)]) == (0, (0, 0))


# ── Point dans polygone (ray casting) ──────────────────────────────────────────
class TestPointInPolygon:
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]

    def test_inside(self):
        assert main.is_point_in_polygon((5, 5), self.square) is True

    def test_outside(self):
        assert main.is_point_in_polygon((15, 5), self.square) is False
        assert main.is_point_in_polygon((-1, -1), self.square) is False

    def test_concave_polygon(self):
        # Polygone en forme de L : un point dans le « creux » est dehors.
        l_shape = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        assert main.is_point_in_polygon((1, 1), l_shape) is True
        assert main.is_point_in_polygon((3, 3), l_shape) is False


# ── Parsing de path SVG ────────────────────────────────────────────────────────
class TestParsePath:
    def test_basic_path(self):
        assert main.parse_path_for_matching("M 1,2 L 3,4 L 5,6 Z") == [
            (1.0, 2.0), (3.0, 4.0), (5.0, 6.0)
        ]

    def test_decimal_coordinates(self):
        assert main.parse_path_for_matching("M 10.5,20.25 L 30.0,40.75 Z") == [
            (10.5, 20.25), (30.0, 40.75)
        ]

    def test_roundtrip_with_area(self):
        # Un path SVG rectangulaire se reparse en un polygone d'aire connue.
        path = "M 0,0 L 20,0 L 20,5 L 0,5 Z"
        assert main.calculate_area(main.parse_path_for_matching(path)) == 100.0


# ── Distance euclidienne ───────────────────────────────────────────────────────
def test_euclidean_distance_3_4_5():
    assert main.euclidean_distance((0, 0), (3, 4)) == 5.0


def test_euclidean_distance_symmetric():
    a, b = (1, 2), (4, 6)
    assert main.euclidean_distance(a, b) == main.euclidean_distance(b, a)


# ── Cercle → polygone ──────────────────────────────────────────────────────────
class TestCircleToCoords:
    def test_point_count(self):
        shape = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
        assert len(main.circle_to_coords(shape)) == 32

    def test_area_approximates_pi_r_squared(self):
        # Cercle de rayon 50 : aire ≈ π·r² = 7853,98. La discrétisation en
        # 64 points donne une erreur < 1 %, conforme au manuscrit (l.145-146).
        shape = {"x0": 0, "y0": 0, "x1": 100, "y1": 100}
        coords = main.circle_to_coords(shape, n_points=64)
        area = main.calculate_area(coords)
        assert area == pytest.approx(math.pi * 50 ** 2, rel=0.01)

    def test_ellipse_independent_axes(self):
        # rx et ry indépendants : ellipse 100×40 → aire ≈ π·50·20.
        shape = {"x0": 0, "y0": 0, "x1": 100, "y1": 40}
        coords = main.circle_to_coords(shape, n_points=128)
        assert main.calculate_area(coords) == pytest.approx(math.pi * 50 * 20, rel=0.01)


# ── Génération de path d'ellipse ───────────────────────────────────────────────
def test_generate_ellipse_path_format():
    path = main.generate_ellipse_path(0, 0, 10, 5, n_points=16)
    assert path.startswith("M ")
    assert path.endswith(" Z")
    assert path.count("L") == 15  # n_points - 1 segments avant la fermeture
