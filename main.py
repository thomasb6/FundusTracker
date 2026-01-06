from datetime import datetime
import base64
import io as io_buffer
import json
import os
import random
import re
import uuid
import zipfile
from PIL import Image
from dash import Dash, html, dcc, Input, Output, State, ctx
from dash import callback_context
from dash import clientside_callback, Output, Input
from dash.dependencies import ALL
from fpdf import FPDF
from fpdf.fonts import FontFace
from scipy.stats import norm
from scipy.stats import ttest_ind
from skimage import filters, color, feature
from skimage import measure
from sklearn.ensemble import RandomForestClassifier
import cv2
import dash
import dash_bootstrap_components as dbc
import dash_table
import gettext
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests


_ = gettext.gettext


REPO_OWNER = "thomasb6"
REPO_NAME = "FundusTracker"
FOLDER_PATH = "exemples"
GITHUB_API_URL = (
    f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FOLDER_PATH}"
)
GITHUB_TOKEN = "ghp_nwTO1ndY???????????rsxh9HxEJKi2QiZNDWGCSX?3?z?U?g?NP"
GITHUB_TOKEN = GITHUB_TOKEN.replace("?", "")

import numpy as np
import pandas as pd


def process_image_analysis_data(stored_shapes, file_val, uploaded_image, language, axial_length):
    _ = get_translator(language)

    processed_shapes_data = calculate_effective_areas(stored_shapes, language)

    optic_nerve_label = _("nerf optique")
    optic_nerve_labels = [optic_nerve_label.lower(), "optic nerve", "nerf optique"]

    area_no_px = 0
    optic_nerve_centroid = None

    for data in processed_shapes_data:
        classification = data["original_shape"].get("customdata", "").lower()
        if classification in optic_nerve_labels:
            area_no_px = data["raw_area"]
            optic_nerve_centroid = data["centroid"]
            break

    if optic_nerve_centroid is None:
        try:
            image_id = file_val or uploaded_image
            image = load_image_any(image_id)
            width, height = image.size
            optic_nerve_centroid = (width / 2, height / 2)
        except:
            optic_nerve_centroid = (350, 350)

    scale_mm2_per_px = 0
    if area_no_px > 0:
        scale_mm2_per_px = 2.54 / area_no_px

    bennett_factor = 1.0
    al_used = "Standard (24.2)"
    if axial_length and isinstance(axial_length, (int, float)) and axial_length > 0:
        try:
            bennett_factor = ((axial_length - 1.82) / (24.2 - 1.82)) ** 2
            al_used = f"{axial_length} mm"
        except:
            pass

    final_scale_area = scale_mm2_per_px * bennett_factor
    final_scale_linear = np.sqrt(final_scale_area) if final_scale_area > 0 else 0
    has_calibration = (area_no_px > 0)
    rows = []
    if has_calibration:
        rows.append({
            _("Zone"): "Ref",
            _("Classification"): _("Nerf Optique (Étalon)"),
            _("Aire (pixels²)"): area_no_px,
            _("Aire (mm²)"): 2.54 * bennett_factor,
            _("Grand Axe (px)"): None, _("Grand Axe (mm)"): None,
            _("Petit Axe (px)"): None, _("Petit Axe (mm)"): None,
            _("Distance au NO (px)"): 0, _("Distance au NO (mm)"): 0,
            _("Info Calibration"): _("Utilisé comme référence (2.54 mm²)")
        })

    def compute_ellipse_params(coords):
        arr = np.array(coords)
        if len(arr) < 3: return None, None, None, None
        centroid = np.mean(arr, axis=0)
        cov = np.cov(arr, rowvar=False)
        if np.linalg.matrix_rank(cov) < 2: return centroid, None, None, None
        eigenvals, eigenvecs = np.linalg.eig(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
        major = 4 * np.sqrt(eigenvals[0])
        minor = 4 * np.sqrt(eigenvals[1])
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        return centroid, major, minor, angle

    for data in processed_shapes_data:
        shape = data["original_shape"]
        classification = shape.get("customdata", "Tache")

        if classification in optic_nerve_labels:
            continue
        coords = data["coords"]
        if not coords: continue
        if classification == _("Exclusion"):
            area_px = data["raw_area"]
        else:
            area_px = data["effective_area"]
        area_mm2 = area_px * final_scale_area if has_calibration else None
        (lesion_centroid, major_axis_px, minor_axis_px, ellipse_angle) = compute_ellipse_params(coords)
        distance_px = None
        if lesion_centroid is not None and optic_nerve_centroid is not None:
            distance_px = np.linalg.norm(np.array(lesion_centroid) - np.array(optic_nerve_centroid))
        dist_mm = distance_px * final_scale_linear if has_calibration and distance_px else None
        major_mm = major_axis_px * final_scale_linear if has_calibration and major_axis_px else None
        minor_mm = minor_axis_px * final_scale_linear if has_calibration and minor_axis_px else None
        elongation_index = (major_axis_px / minor_axis_px) if (major_axis_px and minor_axis_px) else None
        perimeter = 0
        for j in range(len(coords)):
            p1 = coords[j]
            p2 = coords[(j + 1) % len(coords)]
            perimeter += np.linalg.norm(np.array(p1) - np.array(p2))
        circularity_index = (4 * np.pi * data["raw_area"]) / (perimeter ** 2) if perimeter > 0 else None
        alignement_radial = None
        if lesion_centroid is not None and ellipse_angle is not None and optic_nerve_centroid:
            angle_from_center = np.degrees(np.arctan2(
                lesion_centroid[1] - optic_nerve_centroid[1],
                lesion_centroid[0] - optic_nerve_centroid[0]
            ))
            delta_angle = ellipse_angle - angle_from_center
            alignement_radial = (delta_angle + 90) % 180 - 90
        parent_info = f"Zone {data['parent_index'] + 1}" if data["parent_index"] is not None else ""
        rows.append({
            _("Zone"): data["original_index"] + 1,
            _("Classification"): classification,
            _("Aire (pixels²)"): area_px,
            _("Distance au NO (px)"): distance_px,
            _("Grand Axe (px)"): major_axis_px,
            _("Petit Axe (px)"): minor_axis_px,
            _("Aire (mm²)"): area_mm2,
            _("Distance au NO (mm)"): dist_mm,
            _("Grand Axe (mm)"): major_mm,
            _("Petit Axe (mm)"): minor_mm,
            _("Info Calibration"): f"AL: {al_used}",
            _("Parent (si exclusion)"): parent_info,
            _("Indice d'Élongation"): elongation_index,
            _("Indice de Circularité"): circularity_index,
            _("Alignement Radial (degrés)"): alignement_radial,
        })

    df = pd.DataFrame(rows)
    desired_order = [
        _("Zone"), _("Classification"),
        _("Aire (pixels²)"), _("Distance au NO (px)"), _("Grand Axe (px)"), _("Petit Axe (px)"),
        _("Aire (mm²)"), _("Distance au NO (mm)"), _("Grand Axe (mm)"), _("Petit Axe (mm)"),
        _("Info Calibration"),
        _("Parent (si exclusion)"),
        _("Indice d'Élongation"), _("Indice de Circularité"), _("Alignement Radial (degrés)")
    ]

    final_cols = [c for c in desired_order if c in df.columns]
    df = df[final_cols]
    image_identifier = file_val or "local_image"
    filename = f"{image_identifier.split('/')[-1].rsplit('.', 1)[0]}.xlsx"

    return df, filename

def is_point_in_polygon(point, polygon_coords):

    x, y = point
    n = len(polygon_coords)
    inside = False
    p1x, p1y = polygon_coords[0]
    for i in range(n + 1):
        p2x, p2y = polygon_coords[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def calculate_effective_areas(shapes, language):

    _ = get_translator(language)
    exclusion_label = _("Exclusion")

    positive_shapes = [s for s in shapes if s.get("customdata") != exclusion_label]
    subtractive_shapes = [s for s in shapes if s.get("customdata") == exclusion_label]

    results = []

    all_shapes_data = {}
    for i, shape in enumerate(shapes):
        if shape.get("type") == "circle":
            coords = circle_to_coords(shape)
        else:
            path_str = shape.get("path", "")
            coords = parse_path_for_matching(path_str)

        area = calculate_area(coords)
        centroid = calculate_polygon_properties_for_matching(coords)[1]
        all_shapes_data[i] = {
            "original_shape": shape,
            "original_index": i,
            "coords": coords,
            "raw_area": area,
            "centroid": centroid,
            "parent_index": None,
            "subtracted_area": 0,
            "subtraction_from": [],
        }

    for sub_idx, sub_data in all_shapes_data.items():
        if sub_data["original_shape"].get("customdata") != exclusion_label:
            continue

        for pos_idx, pos_data in all_shapes_data.items():
            if pos_data["original_shape"].get("customdata") == exclusion_label:
                continue

            if is_point_in_polygon(sub_data["centroid"], pos_data["coords"]):

                pos_data["subtracted_area"] += sub_data["raw_area"]
                pos_data["subtraction_from"].append(sub_idx + 1)
                sub_data["parent_index"] = pos_idx
                break

    for i, data in all_shapes_data.items():
        effective_area = data["raw_area"] - data["subtracted_area"]
        data["effective_area"] = max(0, effective_area)
        results.append(data)

    return results


def image_to_base64(image_id):

    if isinstance(image_id, str) and image_id.startswith("data:image"):
        return image_id
    try:
        url = get_image_url(image_id)
        response = requests.get(url)
        response.raise_for_status()
        encoded = base64.b64encode(response.content).decode("utf-8")
        content_type = "image/png" if ".png" in image_id.lower() else "image/jpeg"
        return f"data:{content_type};base64,{encoded}"
    except Exception as e:
        print(f"Erreur lors de la conversion de l'image {image_id} en base64: {e}")
        return None


def parse_path_for_matching(path_string):

    points = re.findall(r"(\d+\.?\d*)", path_string)
    coords = [
        (float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)
    ]
    return coords


def calculate_polygon_properties_for_matching(coords):

    n = len(coords)
    if n < 3:
        return 0, (0, 0)
    area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        cross_product = (x1 * y2) - (x2 * y1)
        area += cross_product
        centroid_x += (x1 + x2) * cross_product
        centroid_y += (y1 + y2) * cross_product
    if area == 0:
        return 0, (
            (sum(p[0] for p in coords) / n, sum(p[1] for p in coords) / n)
            if n > 0
            else (0, 0)
        )
    signed_area = area / 2.0
    area = abs(signed_area)
    centroid_x /= 6.0 * signed_area
    centroid_y /= 6.0 * signed_area
    return area, (centroid_x, centroid_y)


def euclidean_distance(p1, p2):

    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def run_auto_matching(shapes_t0, shapes_t1, language):

    _ = get_translator(language)
    DISTANCE_THRESHOLD = 50.0
    AREA_CHANGE_THRESHOLD = 0.25
    STATUS_COLORS = {
        _("Initial"): "#f5f6fa",
        _("Apparition"): "lime",
        _("Stabilité"): "cyan",
        _("Progression"): "orange",
    }

    shapes_t0_processed = [s.copy() for s in shapes_t0]
    shapes_t1_processed = [s.copy() for s in shapes_t1]
    zones_t0, zones_t1 = [], []
    for i, s in enumerate(shapes_t0_processed):
        if s.get("customdata", "").lower() not in ["nerf optique", "optic nerve"]:
            coords = parse_path_for_matching(s.get("path", ""))
            area, centroid = calculate_polygon_properties_for_matching(coords)
            zones_t0.append(
                {"id": i, "area": area, "centroid": centroid, "shape_ref": s}
            )
    for i, s in enumerate(shapes_t1_processed):
        if s.get("customdata", "").lower() not in ["nerf optique", "optic nerve"]:
            coords = parse_path_for_matching(s.get("path", ""))
            area, centroid = calculate_polygon_properties_for_matching(coords)
            zones_t1.append(
                {"id": i, "area": area, "centroid": centroid, "shape_ref": s}
            )
    used_t0_indices = set()
    for zone_t1 in zones_t1:
        best_match_dist, best_match_idx_t0 = float("inf"), -1
        for zone_t0 in zones_t0:
            if zone_t0["id"] in used_t0_indices:
                continue
            dist = euclidean_distance(zone_t1["centroid"], zone_t0["centroid"])
            if dist < best_match_dist:
                best_match_dist, best_match_idx_t0 = dist, zone_t0["id"]
        t1_shape = zone_t1["shape_ref"]

        if best_match_dist < DISTANCE_THRESHOLD:
            zone_t0_match = next(
                (z for z in zones_t0 if z["id"] == best_match_idx_t0), None
            )
            if zone_t0_match:
                used_t0_indices.add(zone_t0_match["id"])
                area_change = (
                    (zone_t1["area"] - zone_t0_match["area"]) / zone_t0_match["area"]
                    if zone_t0_match["area"] > 0
                    else 0
                )
                status = (
                    _("Progression")
                    if area_change > AREA_CHANGE_THRESHOLD
                    else _("Stabilité")
                )

                if "line" not in t1_shape:
                    t1_shape["line"] = {}
                t1_shape["line"].update(
                    {"color": STATUS_COLORS[status], "width": 3, "dash": "solid"}
                )

                t0_shape = zone_t0_match["shape_ref"]
                t1_shape["unique_lesion_id"] = t0_shape.get(
                    "unique_lesion_id", str(uuid.uuid4())
                )

        else:
            status = _("Apparition")
            if "line" not in t1_shape:
                t1_shape["line"] = {}
            t1_shape["line"].update(
                {"color": STATUS_COLORS[status], "width": 3, "dash": "solid"}
            )

            if "unique_lesion_id" not in t1_shape or not t1_shape["unique_lesion_id"]:
                t1_shape["unique_lesion_id"] = str(uuid.uuid4())
    return {
        "augmented_shapes_t0": shapes_t0_processed,
        "augmented_shapes_t1": shapes_t1_processed,
    }


def calculate_permutation_test(areas_t0, areas_t1, num_permutations=10000):

    observed_diff = sum(areas_t1) - sum(areas_t0)

    combined_areas = np.concatenate([areas_t0, areas_t1])
    count_extreme_results = 0
    n0 = len(areas_t0)
    for _ in range(num_permutations):
        np.random.shuffle(combined_areas)
        perm_t0 = combined_areas[:n0]
        perm_t1 = combined_areas[n0:]
        current_diff = sum(perm_t1) - sum(perm_t0)
        if abs(current_diff) >= abs(observed_diff):
            count_extreme_results += 1
    p_value = count_extreme_results / num_permutations
    return p_value


def get_translator(language):

    lang_translation = gettext.translation(
        "main", localedir="locales", languages=[language]
    )
    return lang_translation.gettext


def load_image_any(img_id):

    if isinstance(img_id, str) and img_id.startswith("data:image"):
        content_type, content_string = img_id.split(",")
        return Image.open(io_buffer.BytesIO(base64.b64decode(content_string)))
    elif img_id:
        url = get_image_url(img_id)
        return Image.open(io_buffer.BytesIO(requests.get(url).content))
    return None


def mask_to_shapes(mask, label_value=1, min_area=200, contour_tolerance=3.0):
    mask_bin = (np.array(mask) == label_value).astype(np.uint8)
    labeled = measure.label(mask_bin)
    shapes = []
    idx = 1
    for region in measure.regionprops(labeled):
        if region.area < min_area:
            continue
        contours = measure.find_contours(labeled == region.label, level=0.5)
        if not contours:
            continue
        contour = max(contours, key=lambda c: len(c))
        contour = measure.approximate_polygon(contour, tolerance=contour_tolerance)
        path = "M " + " L ".join(f"{float(y)},{float(x)}" for x, y in contour) + " Z"
        shapes.append(
            {
                "type": "path",
                "path": path,
                "line": {"color": "yellow", "width": 2, "dash": "solid"},
                "customdata": "segmentation-auto",
                "customid": idx,
                "editable": True,
                "layer": "above",
            }
        )
        idx += 1
    return shapes


def transform_coords(coords, zoom, rotation_deg, center):
    from math import cos, sin, radians

    rot = radians(rotation_deg)
    cx, cy = center
    out = []
    for x, y in coords:
        tx, ty = x - cx, y - cy
        tx, ty = tx * zoom, ty * zoom
        rx = tx * cos(rot) - ty * sin(rot)
        ry = tx * sin(rot) + ty * cos(rot)
        out.append((rx + cx, ry + cy))
    return out


def transform_shape(shape, zoom, rotation_deg, center):
    s = shape.copy()
    if s.get("type") == "circle":
        coords = circle_to_coords(s)
        coords_t = transform_coords(coords, zoom, rotation_deg, center)
        xs = [pt[0] for pt in coords_t]
        ys = [pt[1] for pt in coords_t]
        s["x0"], s["x1"] = min(xs), max(xs)
        s["y0"], s["y1"] = min(ys), max(ys)
    elif "path" in s:
        import re

        path_str = s["path"]
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
        coords = [
            (float(matches[j]), float(matches[j + 1]))
            for j in range(0, len(matches), 2)
        ]
        coords_t = transform_coords(coords, zoom, rotation_deg, center)
        if coords_t:
            path = "M " + " L ".join(f"{x},{y}" for x, y in coords_t) + " Z"
            s["path"] = path
    return s


def get_filenames(path=FOLDER_PATH):

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
    items = response.json()
    filenames = []
    for item in items:
        if item["type"] == "file":
            filenames.append(item["path"])
        elif item["type"] == "dir":
            filenames.extend(get_filenames(item["path"]))
    return filenames


def get_image_url(filepath):

    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/{filepath}"


def calculate_area(coords):
    if len(coords) < 3:
        return 0
    x, y = zip(*coords)
    return 0.5 * abs(
        sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(coords) - 1))
    )


def circle_to_coords(shape, n_points=32):

    from math import cos, sin, pi

    x0, y0, x1, y1 = shape["x0"], shape["y0"], shape["x1"], shape["y1"]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    rx = abs(x1 - x0) / 2
    ry = abs(y1 - y0) / 2
    return [
        (cx + rx * cos(2 * pi * i / n_points), cy + ry * sin(2 * pi * i / n_points))
        for i in range(n_points)
    ]


def generate_figure(image, file_val=None, shapes=None, size="normal"):
    fig = px.imshow(image)
    fig.update_traces(hoverinfo="skip", hovertemplate=None)
    if size == "mini":
        width, height = 320, 320
    else:
        width, height = 700, 700
    fig.update_layout(
        dragmode="drawclosedpath",
        uirevision=file_val or str(random.random()),
        paper_bgcolor="black",
        plot_bgcolor="black",
        width=width,
        height=height,
        xaxis_visible=False,
        yaxis_visible=False,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=shapes if shapes is not None else [],
        newshape=dict(line=dict(color="white", width=2, dash="dot")),
        hovermode=False,
    )
    return fig


def shape_for_plotly(shape):

    return {
        k: v
        for k, v in shape.items()
        if k not in ["customdata", "customid", "unique_lesion_id"]
    }


def base64_to_cv2(base64_string):

    if isinstance(base64_string, str) and base64_string.startswith("data:image"):
        encoded_data = base64_string.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return None


def cv2_to_base64(img_array):

    _, buffer = cv2.imencode(".png", img_array)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def align_images_sift_simple(ref_img_b64, target_img_b64):

    try:
        img_ref = base64_to_cv2(ref_img_b64)
        img_target = base64_to_cv2(target_img_b64)

        if img_ref is None or img_target is None:
            return None, "Erreur de chargement des images."

        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_ref, None)
        kp2, des2 = sift.detectAndCompute(gray_target, None)

        if des1 is None or des2 is None:
            return None, "Pas assez de détails pour SIFT."

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 4:
            return None, "Pas assez de correspondances trouvées."

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            return None, "Transformation géométrique impossible."

        h, w, _ = img_ref.shape

        aligned_img = cv2.warpPerspective(img_target, M, (w, h))

        return cv2_to_base64(aligned_img), "Succès"

    except Exception as e:
        return None, str(e)


scatter_fig = go.Figure(
    go.Scattergl(
        x=np.random.randn(1000),
        y=np.random.randn(1000),
        mode="markers",
        marker=dict(
            color=random.sample(["#ecf0f1"] * 500 + ["#2d3436"] * 500, 1000),
            line_width=1,
        ),
    )
)
scatter_fig.update_layout(
    plot_bgcolor="#dfe6e9",
    width=700,
    height=700,
    xaxis_visible=False,
    yaxis_visible=False,
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),
    hovermode=False,
)
scatter_fig_mini = go.Figure(
    go.Scattergl(
        x=np.random.randn(400),
        y=np.random.randn(400),
        mode="markers",
        marker=dict(
            color=random.sample(["#ecf0f1"] * 200 + ["#2d3436"] * 200, 400),
            line_width=1,
        ),
    )
)
scatter_fig_mini.update_layout(
    plot_bgcolor="#dfe6e9",
    width=320,
    height=320,
    xaxis_visible=False,
    yaxis_visible=False,
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),
    hovermode=False,
)
config_graph = {
    "modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"],
    "displaylogo": False,
}


external_stylesheets = [
    dbc.themes.FLATLY,
    "https://use.fontawesome.com/releases/v5.15.3/css/all.css",
]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    title="FundusTracker",
    suppress_callback_exceptions=True,
)
server = app.server
filenames = get_filenames()


app.layout = html.Div(
    [dcc.Store(id="language-store", data="fr"), html.Div(id="page-content")]
)


def serve_layout(language):

    _ = get_translator(language)
    classification_options = [
        _("Tache"),
        _("Plaque"),
        _("Atrophie"),
        _("Pigment"),
        _("Nerf optique"),
        _("Plaque (incertain)"),
        _("Spot (incertain)"),
        _("Atrophie Péripapillaire"),
        _("CNV Actif"),
        _("CNV Cicatrice"),
        _("Exclusion"),
    ]

    def create_class_btn(opt):
        # Traducteur (assurez-vous que 'language' est accessible)
        _ = get_translator(language)

        # Styles de base
        btn_style = {
            "flex": "1",
            "fontSize": "0.75rem",
            "padding": "6px 2px",
            "whiteSpace": "normal",
            "lineHeight": "1.1",
            "margin": "2px",
            "border": "none"
        }
        if opt == _("Exclusion"):
            color = "dark"
        elif opt == _("Nerf optique"):
            color = "warning"
            btn_style["backgroundColor"] = "#e1b12c"
            btn_style["color"] = "white"
        else:
            color = "secondary"


        return dbc.Button(
            opt,
            id={"type": "classify-button", "index": opt},
            color=color,
            style=btn_style,
        )

    buttons_row_1 = [create_class_btn(opt) for opt in classification_options[:5]]
    buttons_row_2 = [create_class_btn(opt) for opt in classification_options[5:]]

    def layout_manual():
        return html.Div(
            [
                dbc.Container(
                    [
                        html.Div(
                            [
                                html.H2(_("Instructions d'utilisation")),
                                html.P(
                                    _(
                                        "1. Choisissez une image depuis le menu déroulant."
                                    )
                                ),
                                html.P(
                                    _(
                                        "2. Tracez le contour du nerf optique et d'une lésion sur l'image."
                                    )
                                ),
                                html.P(
                                    _(
                                        "3. Classez la zone en cliquant sur le type approprié."
                                    )
                                ),
                                html.H3(
                                    _(
                                        "Vous pouvez supprimer une zone en la sélectionnant."
                                    )
                                ),
                                html.H3(
                                    _(
                                        "Vous pouvez modifier une classification via le menu déroulant."
                                    )
                                ),
                                html.P(
                                    _(
                                        "4. Exportez les résultats vers Excel pour obtenir un résumé."
                                    )
                                ),
                                html.P(_("5. Téléchargez les zones annotées.")),
                                html.H3(
                                    _(
                                        "Vous pouvez importer un fichier avec les zones annotées."
                                    )
                                ),
                            ],
                            className="left-block",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="fig-image",
                                    config=config_graph,
                                    style={"width": "100%", "height": "auto"},
                                    className="graph-figure",
                                ),
                                html.Div(id="output-area", className="output-area"),
                            ],
                            className="middle-block",
                        ),
                        html.Div(
                            [
                                html.P(_("Choisir une image :")),
                                dcc.Dropdown(
                                    id="file-dropdown",
                                    options=[
                                        {"label": f.split("/")[-1], "value": f}
                                        for f in filenames
                                    ],
                                    placeholder=_("Choisissez une image"),
                                ),
                                html.Div(
                                    _("Ou chargez une image locale :"),
                                    style={"marginTop": "12px"},
                                ),
                                dcc.Upload(
                                    id="upload-image",
                                    children=html.Div(
                                        [
                                            _("Glissez-déposez une image ou "),
                                            html.A(_("cliquez ici")),
                                        ]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "40px",
                                        "lineHeight": "40px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px 0",
                                    },
                                    multiple=False,
                                ),
                                html.Hr(),
                                dbc.Button(
                                    [
                                        html.I(
                                            className="fas fa-crop-alt",
                                            style={"marginRight": "5px"},
                                        ),
                                        _("Aligner l'image sur une référence"),
                                    ],
                                    id="open-crop-modal-btn",
                                    color="warning",
                                    outline=True,
                                    className="mb-3",
                                    style={"width": "100%"},
                                ),
                                dbc.Modal(
                                    [
                                        dbc.ModalHeader(
                                            dbc.ModalTitle(
                                                _(
                                                    "Outil de recadrage automatique (SIFT algorithm)"
                                                )
                                            )
                                        ),
                                        dbc.ModalBody(
                                            [
                                                html.P(
                                                    _(
                                                        "L'image actuelle sera recadrée pour s'aligner sur une image de référence."
                                                    )
                                                ),
                                                html.Label(
                                                    _(
                                                        "Choisissez l'image de référence :"
                                                    ),
                                                    className="fw-bold",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Small(
                                                                    _(
                                                                        "Depuis la liste GitHub :"
                                                                    )
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="crop-ref-dropdown",
                                                                    options=[
                                                                        {
                                                                            "label": f.split(
                                                                                "/"
                                                                            )[
                                                                                -1
                                                                            ],
                                                                            "value": f,
                                                                        }
                                                                        for f in filenames
                                                                    ],
                                                                    placeholder=_(
                                                                        "Image référence..."
                                                                    ),
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Small(
                                                                    _(
                                                                        "Ou fichier local :"
                                                                    )
                                                                ),
                                                                dcc.Upload(
                                                                    id="crop-ref-upload",
                                                                    children=html.Div(
                                                                        [
                                                                            html.I(
                                                                                className="fas fa-upload"
                                                                            ),
                                                                            " "
                                                                            + _(
                                                                                "Choisir"
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    style={
                                                                        "borderWidth": "1px",
                                                                        "borderStyle": "dashed",
                                                                        "borderRadius": "5px",
                                                                        "textAlign": "center",
                                                                        "lineHeight": "36px",
                                                                        "marginTop": "2px",
                                                                    },
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Button(
                                                    [
                                                        _("Recadrer l'image"),
                                                        html.I(
                                                            className="fas fa-magic ms-2"
                                                        ),
                                                    ],
                                                    id="run-crop-sift-btn",
                                                    color="primary",
                                                    className="w-100 mb-3",
                                                ),
                                                dcc.Loading(
                                                    html.Div(
                                                        id="crop-result-display",
                                                        style={
                                                            "textAlign": "center",
                                                            "minHeight": "100px",
                                                        },
                                                    ),
                                                    type="circle",
                                                ),
                                                dcc.Store(id="crop-result-temp-store"),
                                                dcc.Download(
                                                    id="download-cropped-image"
                                                ),
                                            ]
                                        ),
                                        dbc.ModalFooter(
                                            [
                                                dbc.Button(
                                                    [
                                                        html.I(
                                                            className="fas fa-download me-2"
                                                        ),
                                                        _(
                                                            "Sauvegarder l'image localement"
                                                        ),
                                                    ],
                                                    id="download-cropped-img-btn",
                                                    color="secondary",
                                                    disabled=True,
                                                    style={"marginRight": "auto"},
                                                ),
                                                dbc.Button(
                                                    _("Fermer"),
                                                    id="close-crop-modal-btn",
                                                    color="light",
                                                ),
                                                dbc.Button(
                                                    [
                                                        _("Utiliser cette image"),
                                                        html.I(
                                                            className="fas fa-check ms-2"
                                                        ),
                                                    ],
                                                    id="apply-cropped-img-btn",
                                                    color="success",
                                                    disabled=True,
                                                ),
                                            ]
                                        ),
                                    ],
                                    id="crop-modal",
                                    size="lg",
                                    is_open=False,
                                    backdrop="static",
                                ),
                                html.P(_("Classification :"), className="mb-1"),

                                dbc.ButtonGroup(
                                    buttons_row_1,
                                    style={"width": "100%", "display": "flex"},
                                    className="mb-1"
                                ),

                                dbc.ButtonGroup(
                                    buttons_row_2,
                                    style={"width": "100%", "display": "flex"},
                                    className="mb-2"
                                ),

                                dbc.InputGroup(
                                    [
                                        dbc.Input(
                                            id="custom-label-input",
                                            placeholder=_("Autre"),
                                            type="text"
                                        ),
                                        dbc.Button(
                                            _("Appliquer"),
                                            id="apply-custom-label-btn",
                                            color="secondary"
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dcc.Dropdown(
                                    id="zone-selector",
                                    options=[],
                                    placeholder=_(
                                        "Sélectionnez une zone à reclassifier"
                                    ),
                                ),
                                html.P(_("Réinitialiser :")),
                                dbc.Button(
                                    [
                                        html.I(
                                            className="fas fa-undo",
                                            style={"margin-right": "5px"},
                                        ),
                                        _("Réinitialiser les zones annotées"),
                                    ],
                                    id="reset-button",
                                    color="danger",
                                    className="mb-2",
                                ),
                                html.P(_("Exporter :")),
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-download", style={"marginRight": "5px"}),
                                        html.Span(_("Tout exporter"))
                                    ],
                                    id="export-all-button",
                                    color="success",
                                    className="mb-3 w-100",
                                ),
                                dcc.Download(id="download-all-zip"),
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-table", style={"marginRight": "5px"}),
                                        _("Exporter les résultats (XLSX)"),
                                    ],
                                    id="export-button",
                                    color="dark",
                                    outline=True,
                                    className="mb-2 w-100",
                                ),
                                dcc.Download(id="download-dataframe-xlsx"),
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-file-code", style={"marginRight": "5px"}),
                                        _("Exporter les annotations (JSON)"),
                                    ],
                                    id="download-json-button",
                                    color="dark",
                                    outline=True,
                                    className="mb-2 w-100",
                                ),
                                dcc.Download(id="download-json"),
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-layer-group", style={"marginRight": "5px"}),
                                        _("Exporter la matrice de segmentation"),
                                    ],
                                    id="download-mask-button",
                                    color="dark",
                                    outline=True,
                                    className="mb-2 w-100",
                                ),
                                dcc.Download(id="download-mask-image"),
                                html.P(_("Paramètres d'affichage :")),
                                html.Div(
                                    [
                                        dbc.Checkbox(
                                            id="show-zone-numbers",
                                            value=True,
                                            className="form-check-input",
                                        ),
                                        dbc.Label(
                                            _(
                                                "Afficher le numéro des zones sur le dessin"
                                            ),
                                            html_for="show-zone-numbers",
                                            className="form-check-label",
                                        ),
                                    ],
                                    className="form-check mb-2",
                                ),
                                html.Div(
                                    [
                                        dbc.Checkbox(
                                            id="dashed-contour",
                                            value=False,
                                            className="form-check-input",
                                        ),
                                        dbc.Label(
                                            _("Contour pointillé des formes"),
                                            html_for="dashed-contour",
                                            className="form-check-label",
                                        ),
                                    ],
                                    className="form-check mb-2",
                                ),
                                html.P(_("Ajustements globaux :")),
                                dbc.Button(
                                    [
                                        html.I(
                                            className="fas fa-circle",
                                            style={"margin-right": "5px"},
                                        ),
                                        _("Ajouter le nerf optique"),
                                    ],
                                    id="add-nerf-optique-button",
                                    color="info",
                                    className="mb-2",
                                    style={"width": "100%"},
                                ),
                                html.Label(_("Zoom global :")),
                                dcc.Slider(
                                    id="zoom-slider",
                                    min=0.80,
                                    max=1.2,
                                    step=0.01,
                                    value=1.0,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                ),
                                html.Label(_("Rotation globale (°) :")),
                                dcc.Slider(
                                    id="rotation-slider",
                                    min=-30,
                                    max=30,
                                    step=0.5,
                                    value=0,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                ),
                                html.Hr(),
                                html.Label(_("Données Biométriques :")),
                                html.Div(
                                    [
                                        html.Label(_("Longueur Axiale (mm) :"), style={"fontSize": "0.9rem"}),
                                        dbc.Input(
                                            id="axial-length-input",
                                            type="number",
                                            placeholder="ex: 24.2",
                                            min=15,
                                            max=35,
                                            step=0.1,
                                            style={"marginBottom": "10px"},
                                        ),
                                        dbc.FormText(
                                            _("Permet la transformation px > mm"),
                                            color="secondary",
                                            style={"fontSize": "0.75rem"},
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Hr(),
                                html.P(
                                    _("Gestion Patient :"), style={"marginTop": "20px"}
                                ),
                                dcc.DatePickerSingle(
                                    id="annotation-date-picker-manuelle",
                                    placeholder=_("Date de l'examen..."),
                                    display_format="DD/MM/YYYY",
                                    style={"width": "100%"},
                                ),
                                dbc.Button(
                                    [_("Sauvegarder dans le dossier patient")],
                                    id="save-to-patient-btn-manuelle",
                                    color="success",
                                    className="mb-2 mt-2",
                                    style={"width": "100%"},
                                    disabled=True,
                                ),
                                html.Div(id="save-to-patient-feedback-manuelle"),
                                html.P(_("Importer :")),
                                html.Div(
                                    id="upload-div",
                                    children=[
                                        dcc.Upload(
                                            id="upload-annotations",
                                            children=html.Div(
                                                [
                                                    html.I(
                                                        className="fas fa-upload",
                                                        style={"margin-right": "5px"},
                                                    ),
                                                    _(
                                                        "Glissez-déposez ou sélectionnez un fichier annoté"
                                                    ),
                                                ]
                                            ),
                                            className="upload-area",
                                            style={"width": "100%"},
                                            multiple=False,
                                        )
                                    ],
                                ),
                                html.Div(id="output-text", className="output-text"),
                            ],
                            className="right-block",
                        ),
                    ],
                    fluid=True,
                    className="dashboard-container",
                    style={"display": "flex", "justify-content": "space-between"},
                ),
            ]
        )

    def layout_semiauto():
        return html.Div(
            [
                html.Div(
                    [
                        html.H2(_("Instructions d'utilisation")),
                        html.P(_("1. Choisissez une image.")),
                        html.P(
                            _(
                                "2. Dessinez à main levée les zones d'intéret sur l’image pour donner des exemples."
                            )
                        ),
                        html.P(
                            _(
                                "3. Cliquez sur Segmenter pour lancer la segmentation automatique."
                            )
                        ),
                        html.P(
                            _(
                                "4. Vous pouvez ajouter d’autres traits et relancer la segmentation pour affiner."
                            )
                        ),
                        html.P(
                            _(
                                "5. Acceptez la segmentation pour travailler manuellement les zones."
                            )
                        ),
                        html.Br(),
                    ],
                    className="left-block",
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="ml-image-graph",
                            config={
                                "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],
                                "displaylogo": False,
                            },
                            style={"width": "100%", "height": "auto"},
                            className="graph-figure",
                        ),
                        html.Div(id="ml-segment-result"),
                    ],
                    className="middle-block",
                ),
                html.Div(
                    [
                        html.P(_("Choix de l'image :")),
                        dcc.Dropdown(
                            id="ml-file-dropdown",
                            options=[
                                {"label": f.split("/")[-1], "value": f}
                                for f in filenames
                            ],
                            placeholder=_("Choisissez une image"),
                        ),
                        html.P(_("Pinceau/étiquette :")),
                        dcc.Dropdown(
                            id="ml-label-dropdown",
                            options=[
                                {"label": _("Papille (jaune)"), "value": 1},
                                {"label": _("Lésion (rouge)"), "value": 2},
                                {"label": _("Fond (vert)"), "value": 3},
                            ],
                            value=1,
                            style={"width": "100%"},
                        ),
                        html.P(_("Taille du pinceau :")),
                        dcc.Slider(
                            id="ml-line-width",
                            min=1,
                            max=20,
                            step=1,
                            value=7,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.P(_("Segmentation :")),
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-magic",
                                    style={"margin-right": "5px"},
                                ),
                                _("Segmenter les zones dessinées"),
                            ],
                            id="ml-segment-btn",
                            color="primary",
                            className="mb-2",
                        ),
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-check",
                                    style={"marginRight": "5px"},
                                ),
                                _("Accepter la segmentation"),
                            ],
                            id="ml-accept-zones-btn",
                            color="success",
                            className="mb-2",
                            style={"width": "100%"},
                        ),
                        html.P(_("Réinitialisation")),
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-undo",
                                    style={"marginRight": "5px"},
                                ),
                                _("Réinitialiser la segmentation"),
                            ],
                            id="ml-reset-btn",
                            color="danger",
                            className="mb-2",
                            style={"width": "100%"},
                        ),
                    ],
                    className="right-block",
                ),
            ],
            className="dashboard-container",
            style={"display": "flex", "justify-content": "space-between"},
        )

    def layout_patients():

        _ = get_translator(language)
        return html.Div(
            [
                html.Div(
                    [
                        html.H4(_("Gestion du Patient")),
                        html.Br(),
                        dbc.Label(_("Saisir un nouveau patient :")),
                        dcc.Input(
                            id="patient-nom-input",
                            placeholder=_("Nom"),
                            type="text",
                            className="mb-2",
                            style={"width": "100%"},
                        ),
                        dcc.Input(
                            id="patient-prenom-input",
                            placeholder=_("Prénom"),
                            type="text",
                            className="mb-2",
                            style={"width": "100%"},
                        ),
                        dbc.Label(_("Date de naissance")),
                        dcc.Input(
                            id="patient-ddn-input",
                            placeholder=_("Date de naissance"),
                            type="date",
                            className="mb-2",
                            style={"width": "100%"},
                        ),
                        dbc.Label(_("Sexe")),
                        dcc.Dropdown(
                            id="patient-sexe-input",
                            options=[
                                {"label": _("Non spécifié"), "value": "Non spécifié"},
                                {"label": _("Homme"), "value": "Homme"},
                                {"label": _("Femme"), "value": "Femme"},
                            ],
                            value="Non spécifié",
                            className="mb-2",
                            style={"width": "100%"},
                        ),
                        dbc.Button(
                            _("Créer un nouveau patient"),
                            id="create-patient-btn",
                            color="primary",
                            className="mb-2",
                            style={"width": "100%"},
                        ),
                        html.Hr(),
                        dbc.Label(_("Importer un dossier patient (.json) :")),
                        dcc.Upload(
                            id="import-patient-dossier",
                            children=html.Div(
                                [
                                    html.I(className="fas fa-upload me-2"),
                                    _("Sélectionner un fichier"),
                                ]
                            ),
                            className="upload-area mb-3",
                            style={
                                "height": "60px",
                                "lineHeight": "50px",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(id="import-feedback-message"),
                        html.Hr(),
                        dbc.Label(_("Ou sélectionner un patient existant :")),
                        dcc.Dropdown(
                            id="patient-dropdown",
                            options=[],
                            placeholder=_("Choisir un patient"),
                        ),
                        dbc.Button(
                            [_("Exporter le dossier du patient")],
                            id="export-patient-dossier-btn",
                            color="info",
                            outline=True,
                            className="mt-2",
                            style={"width": "100%"},
                            disabled=True,
                        ),
                        dcc.Download(id="download-patient-dossier-json"),
                        dbc.Button(
                            [
                                html.I(className="fas fa-file-pdf me-2"),
                                _("Exporter le rapport PDF"),
                            ],
                            id="export-patient-report-btn",
                            color="danger",
                            className="mt-2",
                            style={"width": "100%"},
                            disabled=True,
                        ),
                        dcc.Download(id="download-pdf-report"),
                        html.Div(id="patient-workflow-feedback", className="mt-2"),
                        html.Div(id="add-image-section", className="mt-4"),
                    ],
                    className="left-block",
                ),
                html.Div(
                    [
                        html.Div(
                            id="patient-active-display-header",
                            style={"padding": "0 15px"},
                        ),
                        dcc.Tabs(
                            id="patient-sub-tabs",
                            value="sub-tab-dossier",
                            children=[
                                dcc.Tab(
                                    label=_("Dossier patient"),
                                    value="sub-tab-dossier",
                                    children=html.Div(
                                        [
                                            html.Div(
                                                id="patient-identity-display",
                                                className="mb-3",
                                            ),
                                            html.Div(
                                                id="save-patient-changes-feedback",
                                                className="mt-2",
                                            ),
                                            dbc.Card(
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Strong(
                                                                    _("Trier par :"),
                                                                    style={
                                                                        "margin-right": "10px"
                                                                    },
                                                                ),
                                                                dbc.RadioItems(
                                                                    id="sort-by-radio",
                                                                    options=[
                                                                        {
                                                                            "label": _(
                                                                                "Date"
                                                                            ),
                                                                            "value": "exam_date",
                                                                        },
                                                                        {
                                                                            "label": _(
                                                                                "Nb lésions"
                                                                            ),
                                                                            "value": "nb_lesions",
                                                                        },
                                                                        {
                                                                            "label": _(
                                                                                "Aire totale"
                                                                            ),
                                                                            "value": "aire_totale",
                                                                        },
                                                                    ],
                                                                    value="exam_date",
                                                                    inline=True,
                                                                    labelClassName="me-3",
                                                                ),
                                                            ],
                                                            width="auto",
                                                        ),
                                                        dbc.Col(
                                                            dbc.ButtonGroup(
                                                                [
                                                                    dbc.Button(
                                                                        _(
                                                                            "Tout sélectionner"
                                                                        ),
                                                                        id="select-all-btn",
                                                                        size="sm",
                                                                        outline=True,
                                                                        color="primary",
                                                                    ),
                                                                    dbc.Button(
                                                                        _(
                                                                            "Tout désélectionner"
                                                                        ),
                                                                        id="deselect-all-btn",
                                                                        size="sm",
                                                                        outline=True,
                                                                        color="secondary",
                                                                    ),
                                                                ]
                                                            ),
                                                            width="auto",
                                                        ),
                                                    ],
                                                    justify="between",
                                                    align="center",
                                                    className="px-3 py-2",
                                                ),
                                                className="mb-3",
                                            ),
                                            html.Div(
                                                id="patient-matching-action-zone",
                                                className="mb-3",
                                            ),
                                            dbc.Modal(
                                                [
                                                    dbc.ModalHeader(
                                                        dbc.ModalTitle(
                                                            _(
                                                                "Atelier d'appariement en chaîne"
                                                            )
                                                        )
                                                    ),
                                                    dbc.ModalBody(
                                                        id="patient-matching-modal-body"
                                                    ),
                                                    dbc.ModalFooter(
                                                        id="patient-matching-modal-footer"
                                                    ),
                                                ],
                                                id="patient-matching-modal",
                                                size="xl",
                                                is_open=False,
                                            ),
                                            html.Div(id="patient-dossier-display"),
                                        ],
                                        style={"padding": "15px"},
                                    ),
                                ),
                                dcc.Tab(
                                    label=_("Analyse longitudinale"),
                                    value="sub-tab-analyse",
                                    children=html.Div(
                                        [
                                            html.P(
                                                _(
                                                    "Cette section présente l'analyse longitudinale des examens sélectionnés dans le dossier patient."
                                                ),
                                                className="mt-3",
                                            ),
                                            html.Hr(),
                                            html.H4(_("Évolution agrégée des lésions")),
                                            dcc.Graph(
                                                id="longitudinal-evolution-graph"
                                            ),
                                            html.Hr(),
                                            html.H4(_("Flux évolutif des lésions")),
                                            dcc.Graph(
                                                id="longitudinal-individual-graph",
                                                style={"height": "500px"},
                                            ),
                                            html.Hr(),
                                            html.H4(_("Analyses statistiques")),
                                            html.Div(id="longitudinal-stats-output"),
                                        ],
                                        style={"padding": "15px"},
                                    ),
                                ),
                            ],
                        ),
                    ],
                    className="main-content-block",
                ),
            ],
            className="dashboard-container two-columns",
        )

    def layout_compare():
        return html.Div(
            [
                html.Div(
                    [
                        html.H2(_("Informations du patient")),
                        html.Div(
                            [
                                dbc.Label(_("Nom du patient :")),
                                dbc.Input(
                                    id="patient-nom", type="text", placeholder=_("Nom")
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                dbc.Label(_("Prénom du patient :")),
                                dbc.Input(
                                    id="patient-prenom",
                                    type="text",
                                    placeholder=_("Prénom"),
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                dbc.Label(_("Date de naissance :")),
                                dbc.Input(id="patient-dob", type="date"),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                dbc.Label(_("Date image de référence :")),
                                dbc.Input(id="date-gauche", type="date"),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                dbc.Label(_("Date image de comparaison :")),
                                dbc.Input(id="date-droite", type="date"),
                            ],
                            className="mb-3",
                        ),
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-save",
                                    style={"marginRight": "8px"},
                                ),
                                _("Enregistrer infos patient"),
                            ],
                            id="save-patient-infos-btn",
                            color="primary",
                            className="mb-3",
                            style={
                                "marginTop": "10px",
                                "width": "100%",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                        ),
                        html.Div(id="save-patient-infos-feedback"),
                    ],
                    className="left-block",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="compare-fig-left",
                                            style={"width": "100%", "height": "320px"},
                                        ),
                                        html.P(
                                            id="compare-image-name-left",
                                            style={
                                                "textAlign": "center",
                                                "fontSize": "13px",
                                            },
                                        ),
                                    ],
                                    style={"flex": "1", "marginRight": "10px"},
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="compare-fig-right",
                                            style={"width": "100%", "height": "320px"},
                                        ),
                                        html.P(
                                            id="compare-image-name-right",
                                            style={
                                                "textAlign": "center",
                                                "fontSize": "13px",
                                            },
                                        ),
                                    ],
                                    style={"flex": "1", "marginLeft": "10px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "marginBottom": "20px",
                                "width": "100%",
                            },
                        ),
                        html.Hr(),
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-magic",
                                    style={"marginRight": "6px"},
                                ),
                                _("Lancer l'Appariement Automatique"),
                            ],
                            id="launch-matching-btn",
                            color="warning",
                            style={"marginTop": "12px", "width": "100%"},
                            n_clicks=0,
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalHeader(
                                    _("Proposition d'Appariement Automatique")
                                ),
                                dbc.ModalBody(id="matching-modal-body"),
                                dbc.ModalFooter(
                                    dbc.Button(
                                        _("Fermer"),
                                        id="close-matching-modal-btn",
                                        className="ml-auto",
                                        n_clicks=0,
                                    )
                                ),
                            ],
                            id="matching-modal",
                            size="xl",
                            is_open=False,
                        ),
                        html.Div(id="evolution-summary", style={"width": "100%"}),
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-file-pdf",
                                    style={"marginRight": "6px"},
                                ),
                                _("Exporter le rapport"),
                            ],
                            id="export-report-btn",
                            color="secondary",
                            style={"margin": "12px 0"},
                            n_clicks=0,
                        ),
                    ],
                    className="right-block",
                    style={"flex": "2"},
                ),
                html.Div(id="pdf-report-content", style={"display": "none"}),
            ],
            className="dashboard-container",
            style={"display": "flex", "justifyContent": "space-between"},
        )

    return html.Div(
        [
            html.Div(
                children=[
                    html.Img(
                        src=app.get_asset_url("logo.png"),
                        style={
                            "height": "100px",
                            "verticalAlign": "middle",
                            "marginRight": "10px",
                        },
                    ),
                    html.Span(
                        "FundusTracker",
                        style={
                            "fontSize": "40px",
                            "verticalAlign": "middle",
                            "fontWeight": "bold",
                        },
                    ),
                ],
                className="logo-container",
                style={"textAlign": "center"},
            ),
            dcc.Tabs(
                id="tabs",
                value="tab-ml",
                children=[
                    dcc.Tab(
                        label=_("Segmentation semi-automatique"),
                        value="tab-ml",
                        children=layout_semiauto(),
                    ),
                    dcc.Tab(
                        label=_("Segmentation manuelle"),
                        value="tab-manuelle",
                        children=layout_manual(),
                    ),
                    dcc.Tab(
                        label=_("Suivi de patients"),
                        value="tab-patients",
                        children=layout_patients(),
                    ),
                ],
            ),
            html.Footer(
                html.Div(
                    [
                        html.Div(
                            [
                                _("© 2026 – Réalisé par "),
                                html.A(
                                    "Thomas Foulonneau",
                                    href="https://www.linkedin.com/in/thomas-foulonneau?originalSubdomain=fr",
                                    target="_blank",
                                    style={
                                        "color": "#ffffff",
                                        "textDecoration": "underline",
                                    },
                                ),
                                _(
                                    " & Aliénor Vienne-Jumeau"
                                ),
                            ],
                            style={
                                "flex": "1",
                                "textAlign": "center",
                                "fontSize": "12px",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        html.Div(
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "🇫🇷",
                                        id="lang-fr",
                                        color="light",
                                        outline=(language != "fr"),
                                        size="sm",
                                        n_clicks=0,
                                        style={"marginRight": "2px"},
                                    ),
                                    dbc.Button(
                                        "🇬🇧",
                                        id="lang-en",
                                        color="light",
                                        outline=(language != "en"),
                                        size="sm",
                                        n_clicks=0,
                                    ),
                                ],
                                size="sm",
                            ),
                            style={"alignSelf": "center", "marginRight": "12px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "width": "100%",
                    },
                ),
                className="footer",
            ),
            dcc.Store(id="patient-infos-store", data={}),
            dcc.Store(id="compare-shapes-left", data=None),
            dcc.Store(id="compare-shapes-right", data=None),
            dcc.Store(id="compare-image-left", data=None),
            dcc.Store(id="compare-image-right", data=None),
            dcc.Store(id="stored-shapes", data=[]),
            dcc.Store(id="ml-image-store", data=None),
            dcc.Store(id="ml-squiggle-store", data=[]),
            dcc.Store(id="ml-segmentation-mask", data=None),
            dcc.Store(id="tab-value-store", data="tab-manuelle"),
            dcc.Store(id="trigger-print-store", data=False),
            dcc.Store(id="uploaded-image-store", data=None),
            dcc.Store(id="longitudinal-series-store", data={}),
            dcc.Store(id="longitudinal-table-store", data=[]),
            dcc.Store(id="longitudinal-active-patient-store", data=None),
            dcc.Store(id="matching-results-store"),
            dcc.Store(id="annotation-image-store", data=None),
            dcc.Store(id="selected-exams-store", data=[]),
            dcc.Store(id="local-filename-store", data=None),
        ]
    )


@app.callback(Output("page-content", "children"), Input("language-store", "data"))
def update_layout_on_language_change(language):

    return serve_layout(language)


@app.callback(
    Output("language-store", "data"),
    Input("lang-fr", "n_clicks"),
    Input("lang-en", "n_clicks"),
    prevent_initial_call=True,
)
def update_language(fr_clicks, en_clicks):

    triggered_id = ctx.triggered_id
    if triggered_id == "lang-fr":
        return "fr"
    elif triggered_id == "lang-en":
        return "en"
    return dash.no_update


@app.callback(
    Output("fig-image", "figure"),
    Input("file-dropdown", "value"),
    Input("uploaded-image-store", "data"),
    Input("annotation-image-store", "data"),
    Input("reset-button", "n_clicks"),
    Input("stored-shapes", "data"),
    Input("show-zone-numbers", "value"),
    Input("dashed-contour", "value"),
    Input("zoom-slider", "value"),
    Input("rotation-slider", "value"),
    State("fig-image", "figure"),
    State("language-store", "data"),
)
def update_figure(
    file_val,
    uploaded_image,
    annotation_image_id,
    reset_clicks,
    stored_shapes,
    show_zone_numbers,
    dashed_contour,
    zoom,
    rotation,
    current_fig,
    language,
):
    _ = get_translator(language)

    image_id = annotation_image_id or file_val or uploaded_image

    width, height = 700, 700
    fig = scatter_fig

    if image_id:
        try:
            img = load_image_any(image_id)
            if img:
                width, height = img.size
                fig = generate_figure(img, file_val=image_id)
        except Exception:
            fig = scatter_fig

    cx, cy = width / 2, height / 2

    if stored_shapes is not None:
        plotly_shapes = []
        for shape in stored_shapes:

            shape_t = transform_shape(shape, zoom, rotation, (cx, cy))

            shape_t["editable"] = True
            shape_t["layer"] = "above"
            shape_t["xref"] = "x"
            shape_t["yref"] = "y"

            if "line" not in shape_t:
                shape_t["line"] = {}
            shape_t["line"]["width"] = 3
            shape_t["line"]["dash"] = "dot" if dashed_contour else "solid"

            current_color = shape_t["line"].get("color", "white")

            if current_color == "yellow":

                shape_t["fillcolor"] = "rgba(255, 255, 0, 0.2)"
            elif shape_t.get("customdata") == _("Exclusion"):

                shape_t["fillcolor"] = "rgba(0, 0, 0, 0.4)"
            else:

                shape_t["fillcolor"] = "rgba(255, 255, 255, 0.2)"

            plotly_shapes.append(shape_for_plotly(shape_t))

        fig["layout"]["shapes"] = plotly_shapes

        annotations = []
        if show_zone_numbers:

            def centroid(coords):
                if not coords:
                    return 0, 0
                avg_x = sum(x for x, y in coords) / len(coords)
                avg_y = sum(y for x, y in coords) / len(coords)
                return avg_x, avg_y

            for i, shape in enumerate(stored_shapes):

                coords = []
                if shape.get("type") == "circle":
                    coords = circle_to_coords(shape)
                else:
                    path_str = shape.get("path", "")
                    matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
                    try:
                        coords = [
                            (float(matches[j]), float(matches[j + 1]))
                            for j in range(0, len(matches), 2)
                        ]
                    except Exception:
                        coords = []

                coords_t = transform_coords(coords, zoom, rotation, (cx, cy))
                cx_ann, cy_ann = centroid(coords_t)

                annotations.append(
                    dict(
                        x=cx_ann,
                        y=cy_ann,
                        text=str(i + 1),
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-20,
                        font=dict(color="white", size=12, shadow="1px 1px 2px black"),
                    )
                )
        fig["layout"]["annotations"] = annotations

    return fig


def find_optic_nerve(shapes, language):
    _ = get_translator(language)
    for i, shape in enumerate(shapes):
        if shape.get("customdata", "").lower() in ["nerf optique", "optic nerve"]:
            return i, shape
    return None, None


@app.callback(
    Output("stored-shapes", "data"),
    Output("output-area", "children"),
    Output("upload-div", "children"),
    Output("file-dropdown", "value"),
    Output("tab-value-store", "data"),
    Input("add-nerf-optique-button", "n_clicks"),
    Input("fig-image", "relayoutData"),
    Input("reset-button", "n_clicks"),
    Input({"type": "classify-button", "index": ALL}, "n_clicks"),
    Input("upload-annotations", "contents"),
    Input("file-dropdown", "value"),
    Input("ml-accept-zones-btn", "n_clicks"),
    Input("apply-custom-label-btn", "n_clicks"),
    Input("axial-length-input", "value"),
    State("stored-shapes", "data"),
    State("zone-selector", "value"),
    State("ml-segmentation-mask", "data"),
    State("ml-file-dropdown", "value"),
    State("language-store", "data"),
    State("custom-label-input", "value"),
    prevent_initial_call=True,
)
def update_shapes_combined(
    add_nerf_clicks,
    relayout_data,
    reset_clicks,
    classify_clicks,
    upload_contents,
    file_val,
    ml_accept_clicks,
    apply_custom_clicks,
    axial_length,
    stored_shapes,
    selected_zone_idx,
    mask_json,
    ml_file_val,
    language,
    custom_label_text,
):
    _ = get_translator(language)
    trigger_id_obj = ctx.triggered[0]["prop_id"] if ctx.triggered else "initial_load"
    trigger = ctx.triggered_id

    new_upload = dash.no_update
    shapes = stored_shapes.copy() if stored_shapes is not None else []
    optic_nerve_label = _("nerf optique")

    if trigger == "ml-accept-zones-btn":
        if not mask_json or not ml_file_val:
            return (
                dash.no_update,
                _("Aucune segmentation ML détectée."),
                new_upload,
                dash.no_update,
                dash.no_update,
            )
        mask = np.array(json.loads(mask_json))
        papille_shapes = mask_to_shapes(mask, label_value=1, min_area=20)
        lesion_shapes = mask_to_shapes(mask, label_value=2, min_area=200)
        new_shapes = []
        if papille_shapes:
            papille_shape = papille_shapes[0]
            papille_shape["customdata"] = optic_nerve_label
            papille_shape["line"] = {"color": "yellow", "width": 2, "dash": "dot"}
            new_shapes.append(papille_shape)
        for sh in lesion_shapes:
            sh["customdata"] = _("segmentation-auto")
            sh["line"] = {"color": "white", "width": 2, "dash": "dot"}
            new_shapes.append(sh)
        for i, sh in enumerate(new_shapes):
            sh["customid"] = i + 1
        summary = generate_summary(new_shapes, language, axial_length)
        return new_shapes, summary, new_upload, ml_file_val, "tab-manuelle"

    if trigger == "reset-button":
        new_upload = [
            dcc.Upload(
                id="upload-annotations",
                children=html.Div(
                    [
                        _("Glissez-déposez ou "),
                        html.A(_("sélectionnez un fichier annoté")),
                    ]
                ),
                className="upload-area",
                multiple=False,
            )
        ]
        return [], _("Annotations réinitialisées."), new_upload, None, dash.no_update

    if trigger == "upload-annotations" and upload_contents:
        content_type, content_string = upload_contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            new_annotations = json.loads(decoded.decode("utf-8"))
            optic_nerve_idx, _shape = find_optic_nerve(new_annotations, language)
            if optic_nerve_idx is not None and optic_nerve_idx > 0:
                optic_nerve_shape = new_annotations.pop(optic_nerve_idx)
                new_annotations.insert(0, optic_nerve_shape)
            shapes = new_annotations
        except Exception:
            shapes = []
        summary = generate_summary(shapes, language, axial_length)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    if trigger == "file-dropdown" and file_val:

        shapes = []

        summary = generate_summary(shapes, language, axial_length)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    if trigger == "add-nerf-optique-button":
        optic_nerve_idx, _shape = find_optic_nerve(shapes, language)
        if optic_nerve_idx is not None:
            summary = generate_summary(shapes, language, axial_length)
            return shapes, summary, new_upload, dash.no_update, dash.no_update
        image_id = file_val or None
        width, height = (700, 700)
        if image_id:
            try:
                img = load_image_any(image_id)
                width, height = img.size
            except Exception:
                pass
        cx, cy = width / 2, height / 2
        optic_nerve_circle = {
            "type": "circle",
            "xref": "x",
            "yref": "y",
            "x0": cx - 50,
            "y0": cy - 50,
            "x1": cx + 50,
            "y1": cy + 50,
            "line": {"color": "white", "width": 2, "dash": "dot"},
            "customdata": optic_nerve_label,
            "editable": True,
            "layer": "above",
        }
        shapes.insert(0, optic_nerve_circle)
        summary = generate_summary(shapes, language, axial_length)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    if isinstance(trigger, dict) and trigger.get("type") == "classify-button":
        label = trigger["index"]
        target_idx = selected_zone_idx if selected_zone_idx is not None else len(shapes) - 1

        if 0 <= target_idx < len(shapes):
            if label.lower() in ["nerf optique", "optic nerve"]:
                optic_nerve_idx, optic_nerve_shape_obj = find_optic_nerve(shapes, language)
                if optic_nerve_idx is not None and optic_nerve_idx != target_idx:
                    summary = dbc.Alert(
                        _("Un nerf optique existe déjà. Impossible d'en créer un second."),
                        color="danger",
                        duration=3000,
                    )
                    return shapes, summary, new_upload, dash.no_update, dash.no_update
                else:
                    shape_to_move = shapes.pop(target_idx)
                    shape_to_move["customdata"] = label
                    shape_to_move["line"]["dash"] = "dot"
                    shape_to_move["line"]["color"] = "yellow"
                    shapes.insert(0, shape_to_move)
            else:
                shapes[target_idx]["customdata"] = label
                shapes[target_idx]["line"]["dash"] = "dot"
                if label == _("Exclusion"):
                    shapes[target_idx]["line"]["color"] = "gray"
                else:
                    shapes[target_idx]["line"]["color"] = "white"

        summary = generate_summary(shapes, language, axial_length)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    if trigger == "apply-custom-label-btn":
        if not custom_label_text:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        label = custom_label_text
        target_idx = (
            selected_zone_idx if selected_zone_idx is not None else len(shapes) - 1
        )

        if 0 <= target_idx < len(shapes):
            shapes[target_idx]["customdata"] = label
            shapes[target_idx]["line"]["dash"] = "dot"
            shapes[target_idx]["line"]["color"] = "cyan"

        summary = generate_summary(shapes, language, axial_length)
        return shapes, summary, new_upload, dash.no_update, dash.no_update

    if trigger_id_obj == "fig-image.relayoutData":
        if relayout_data and "shapes" in relayout_data:
            new_plotly_shapes = relayout_data["shapes"]
            if len(new_plotly_shapes) > len(shapes):
                new_shape_plotly = new_plotly_shapes[-1]
                new_shape = {
                    k: v
                    for k, v in new_shape_plotly.items()
                    if k not in ["customdata", "customid"]
                }
                new_shape["customdata"] = _("Tache")
                shapes.append(new_shape)
            else:
                shapes = []
                for i, sh_plotly in enumerate(new_plotly_shapes):
                    sh = stored_shapes[i].copy()
                    sh.update(
                        {
                            k: v
                            for k, v in sh_plotly.items()
                            if k not in ["customdata", "customid"]
                        }
                    )
                    shapes.append(sh)
        elif relayout_data:
            for key, val in relayout_data.items():
                m = re.match(r"shapes\[(\d+)\]\.(\w+)", key)
                if m:
                    idx, prop = int(m.group(1)), m.group(2)
                    if idx < len(shapes):
                        shapes[idx][prop] = val

    for i, s in enumerate(shapes):
        s["customid"] = i + 1

    summary = generate_summary(shapes, language, axial_length)
    return shapes, summary, new_upload, dash.no_update, dash.no_update


def generate_summary(shapes, language, axial_length=None):
    _ = get_translator(language)

    processed_shapes = calculate_effective_areas(shapes, language)
    optic_nerve_label = _("nerf optique")

    optic_nerve_area_px = 0
    for data in processed_shapes:
        if data["original_shape"].get("customdata", "").lower() in ["nerf optique", "optic nerve"]:
            optic_nerve_area_px = data["raw_area"]
            break

    pixel_to_mm2_ratio = 0
    if optic_nerve_area_px > 0:
        pixel_to_mm2_ratio = 2.54 / optic_nerve_area_px

    bennett_factor = 1.0
    if axial_length and isinstance(axial_length, (int, float)) and axial_length > 0:
        try:
            ratio = (axial_length - 1.82) / (24.2 - 1.82)
            bennett_factor = ratio ** 2
        except:
            pass

    areas_text = []

    if pixel_to_mm2_ratio > 0:
        info_txt = _("Calibration sur Papille (std 1.8mm)")
        if bennett_factor != 1.0:
            info_txt += f" + Bennett (AL={axial_length})"
        areas_text.append(html.Li(html.B(info_txt), style={"color": "green"}))
        areas_text.append(html.Hr(style={"margin": "5px 0"}))
    else:
        areas_text.append(html.Li(html.I(_("Dessinez le nerf optique pour avoir des mm²")), style={"color": "orange"}))

    for data in processed_shapes:
        shape = data["original_shape"]
        idx = data["original_index"] + 1
        lab = shape.get("customdata", "Tache")

        raw_px = data['raw_area']
        eff_px = data['effective_area']

        val_str = f"{eff_px:.0f} px²"

        if pixel_to_mm2_ratio > 0:
            area_mm = eff_px * pixel_to_mm2_ratio * bennett_factor
            val_str = [f"{eff_px:.0f} px² ➔ ", html.B(f"{area_mm:.2f} mm²")]

        detail = ""
        if lab == _("Exclusion"):
            val_str = f"{raw_px:.0f} px²"
            parent = data["parent_index"]
            detail = f" (Zone {parent + 1})" if parent is not None else ""
        elif data["subtracted_area"] > 0:
            detail = f" (brut: {raw_px:.0f})"

        if isinstance(val_str, list):
            areas_text.append(html.Li([f"{_('Zone')} {idx} [{lab}] : "] + val_str + [detail]))
        else:
            areas_text.append(f"{_('Zone')} {idx} [{lab}] : {val_str}{detail}")

    return dbc.Card(
        [
            dbc.CardHeader(_("Résumé des zones annotées")),
            dbc.CardBody(html.Ul([html.Li(a) if not isinstance(a, html.Li) else a for a in areas_text])),
        ],
        style={"marginTop": "10px"}
    )


def generate_mask_from_shapes(image_shape, shapes, language="fr"):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    _ = get_translator(language)
    CLASS_IDS = {
        "nerf optique": 1,
        "optic nerve": 1,
        "atrophie": 2,
        "pigment": 3,
        "plaque (incertain)": 4,
        "spot (incertain)": 5,
        "atrophie péripapillaire": 6,
        "cnv actif": 7,
        "cnv cicatrice": 8,
        "plaque": 9,
    }
    EXCLUSION_LABEL = _("Exclusion").lower()
    for shape in shapes:
        label_raw = shape.get("customdata", "").lower()
        if label_raw == EXCLUSION_LABEL:
            draw_val = 0
        else:
            draw_val = CLASS_IDS.get(label_raw, 10)
        coords = []
        if shape.get("type") == "circle":
            coords = circle_to_coords(shape, n_points=64)
        else:
            path_str = shape.get("path", "")
            coords = parse_path_for_matching(path_str)

        if not coords:
            continue
        pts = np.array(coords, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], int(draw_val))

    return mask

@app.callback(
    Output("zone-selector", "options"),
    Input("stored-shapes", "data"),
    State("language-store", "data"),
)
def update_zone_selector_options(stored_shapes, language):
    _ = get_translator(language)
    if stored_shapes is None:
        return []
    return [
        {"label": f"{_('Zone')} {i + 1}", "value": i} for i in range(len(stored_shapes))
    ]

@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("export-button", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    State("uploaded-image-store", "data"),
    State("language-store", "data"),
    State("axial-length-input", "value"),
    prevent_initial_call=True,
)
def export_to_excel(n_clicks, stored_shapes, file_val, uploaded_image, language, axial_length):
    if not n_clicks or not stored_shapes:
        return dash.no_update
    df, filename = process_image_analysis_data(
        stored_shapes, file_val, uploaded_image, language, axial_length
    )
    _ = get_translator(language)
    def to_excel(bytes_io):
        with pd.ExcelWriter(bytes_io, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=_("Analyse Complète"))

    return dcc.send_bytes(to_excel, filename)


@app.callback(
    Output("download-json", "data"),
    Input("download-json-button", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    State("local-filename-store", "data"),
    prevent_initial_call=True,
)
def download_annotations(n_clicks, stored_shapes, file_val, local_filename):
    if not stored_shapes:
        return dash.no_update

    content = json.dumps(stored_shapes, indent=2)

    if file_val:

        base_name = file_val.split("/")[-1].rsplit(".", 1)[0]
        filename = f"{base_name}.json"
    elif local_filename:

        base_name = local_filename.rsplit(".", 1)[0]
        filename = f"{base_name}.json"
    else:

        filename = "annotations.json"

    return dcc.send_string(content, filename)


@app.callback(
    Output("ml-image-graph", "figure"),
    Output("ml-squiggle-store", "data", allow_duplicate=True),
    Output("ml-file-dropdown", "value", allow_duplicate=True),
    Input("ml-file-dropdown", "value"),
    Input("ml-image-store", "data"),
    Input("ml-image-graph", "relayoutData"),
    Input("ml-label-dropdown", "value"),
    Input("ml-line-width", "value"),
    Input("ml-reset-btn", "n_clicks"),
    State("ml-squiggle-store", "data"),
    prevent_initial_call=True,
)
def update_ml_figure(
    file_val_dropdown,
    file_val_store,
    relayout_data,
    label,
    width,
    ml_reset_clicks,
    squiggles,
):
    triggered_id = ctx.triggered_id if hasattr(ctx, "triggered_id") else None

    file_val = (
        file_val_store
        if triggered_id == "ml-image-store" and file_val_store
        else file_val_dropdown
    )

    if squiggles is None:
        squiggles = []

    if triggered_id in ["ml-file-dropdown", "ml-image-store", "ml-reset-btn"]:
        squiggles = []

    if triggered_id == "ml-reset-btn":
        if file_val:
            img = load_image_any(file_val)
            arr = np.array(img)
            fig = px.imshow(arr)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(
                dragmode="drawopenpath",
                width=700,
                height=700,
                margin=dict(l=0, r=0, t=0, b=0),
                shapes=[],
                uirevision=str(random.random()),
            )
        else:
            fig = scatter_fig
        return fig, [], file_val

    if file_val:
        try:
            img = load_image_any(file_val)
            arr = np.array(img)
            fig = px.imshow(arr)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(
                dragmode="drawopenpath",
                width=700,
                height=700,
                margin=dict(l=0, r=0, t=0, b=0),
                shapes=[],
                uirevision=file_val,
            )
        except Exception:
            fig = scatter_fig
    else:
        fig = scatter_fig

    shapes = []
    colors = {1: "yellow", 2: "red", 3: "lime"}
    if squiggles:
        for squig in squiggles:
            shapes.append(
                {
                    "type": "path",
                    "path": squig["path"],
                    "line": {
                        "color": colors.get(squig["label"], "yellow"),
                        "width": squig.get("width", 7),
                    },
                    "layer": "above",
                }
            )

    if relayout_data and "shapes" in relayout_data:
        for i, sh in enumerate(relayout_data["shapes"]):
            if i >= len(squiggles):
                squiggles.append({"path": sh["path"], "label": label, "width": width})
        shapes = []
        for squig in squiggles:
            shapes.append(
                {
                    "type": "path",
                    "path": squig["path"],
                    "line": {
                        "color": colors.get(squig["label"], "yellow"),
                        "width": squig.get("width", 7),
                    },
                    "layer": "above",
                }
            )

    fig.update_layout(shapes=shapes)

    return fig, squiggles, file_val


@app.callback(
    Output("ml-segment-result", "children"),
    Output("ml-image-graph", "figure", allow_duplicate=True),
    Output("ml-segmentation-mask", "data"),
    Input("ml-segment-btn", "n_clicks"),
    Input("ml-reset-btn", "n_clicks"),
    State("ml-file-dropdown", "value"),
    State("ml-squiggle-store", "data"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def ml_run_segmentation(n_seg, n_reset, file_val, squiggles, language):
    _ = get_translator(language)
    triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
    if triggered == "ml-reset-btn":
        return "", dash.no_update, None

    if not file_val or not squiggles or len(squiggles) < 2:
        return (
            _("Ajoutez au moins 2 squiggles (fond + lésion)."),
            dash.no_update,
            dash.no_update,
        )

    image = load_image_any(file_val)
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr_gray = color.rgb2gray(arr) if arr.ndim == 3 else arr
    h, w = arr_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for squig in squiggles:
        pts = re.findall(r"[-+]?\d*\.\d+|\d+", squig["path"])
        pts = np.array(
            [[float(pts[i]), float(pts[i + 1])] for i in range(0, len(pts), 2)]
        ).astype(int)
        for x, y in pts:
            for dx in range(-squig["width"] // 2, squig["width"] // 2 + 1):
                for dy in range(-squig["width"] // 2, squig["width"] // 2 + 1):
                    xi, yi = int(x + dx), int(y + dy)
                    if 0 <= xi < w and 0 <= yi < h:
                        mask[yi, xi] = squig["label"]
    idx = mask > 0
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    H_elems = hessian_matrix(arr_gray, sigma=2, order="rc")
    eigvals = hessian_matrix_eigvals(H_elems)
    eigval0 = eigvals[0].ravel()
    features = np.stack(
        [
            arr_gray.ravel(),
            filters.gaussian(arr_gray, sigma=1).ravel(),
            filters.sobel(arr_gray).ravel(),
            eigval0,
            Y.ravel() / h,
            X.ravel() / w,
        ],
        axis=1,
    )
    X_train = features[idx.ravel()]
    y_train = mask[idx]
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=25)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(features)
    mask_pred = y_pred.reshape(h, w)
    fig = px.imshow(arr)
    fig.add_trace(
        go.Heatmap(
            z=mask_pred,
            showscale=False,
            opacity=0.45,
            colorscale=[
                [0, "rgba(0,0,0,0)"],
                [0.5, "rgba(255,0,0,0.5)"],
                [1, "rgba(0,255,0,0.5)"],
            ],
        )
    )
    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)
    fig.update_layout(width=700, height=700, margin=dict(l=0, r=0, t=0, b=0))
    mask_json = json.dumps(mask_pred.tolist())
    return (
        html.Div(
            _(
                "Segmentation calculée ! Cliquez sur « Accepter comme zones » pour exporter les lésions."
            )
        ),
        fig,
        mask_json,
    )


@app.callback(
    Output("tabs", "value"), Input("tab-value-store", "data"), prevent_initial_call=True
)
def update_tabs(tab_value):
    return tab_value


@app.callback(
    Output("compare-shapes-right", "data"),
    Output("compare-image-right", "data"),
    Output("added-to-compare-feedback-right", "children"),
    Input("add-to-compare-right", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    State("uploaded-image-store", "data"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def add_right(n, shapes, file_val, uploaded_image, language):
    _ = get_translator(language)
    if not n or not shapes:
        return dash.no_update, dash.no_update, ""

    if file_val:
        image_id = file_val

    elif uploaded_image:
        image_id = uploaded_image
    else:
        image_id = None
    return (
        shapes,
        image_id,
        dbc.Alert(
            _("Ajouté comme image de comparaison !"), color="success", duration=1500
        ),
    )


@app.callback(
    Output("compare-fig-left", "figure"),
    Output("compare-fig-right", "figure"),
    Output("compare-image-name-left", "children"),
    Output("compare-image-name-right", "children"),
    Output("evolution-summary", "children"),
    Output("pdf-report-content", "children"),
    Input("compare-shapes-left", "data"),
    Input("compare-shapes-right", "data"),
    Input("compare-image-left", "data"),
    Input("compare-image-right", "data"),
    Input("patient-infos-store", "data"),
    State("language-store", "data"),
)
def update_comparison(
    shapes_left, shapes_right, img_left, img_right, patient_infos, language
):
    _ = get_translator(language)

    if img_left:
        img = load_image_any(img_left)
        fig_left = generate_figure(
            img,
            shapes=[shape_for_plotly(s) for s in shapes_left] if shapes_left else [],
            size="mini",
        )
    else:
        fig_left = scatter_fig_mini

    if img_right:
        img = load_image_any(img_right)
        fig_right = generate_figure(
            img,
            shapes=[shape_for_plotly(s) for s in shapes_right] if shapes_right else [],
            size="mini",
        )
    else:
        fig_right = scatter_fig_mini

    date_gauche = patient_infos.get("date_gauche")
    date_droite = patient_infos.get("date_droite")
    name_left = (
        f"{_('Image de référence')}  : {date_gauche}"
        if date_gauche
        else _("Image de référence")
    )
    name_right = (
        f"{_('Image de comparaison')} : {date_droite}"
        if date_droite
        else _("Image de comparaison")
    )
    summary = generate_comparison_summary(
        shapes_left, shapes_right, language, patient_infos
    )
    pdf_report_div = render_pdf_report(patient_infos, summary, language)
    return fig_left, fig_right, name_left, name_right, summary, pdf_report_div


def get_optic_nerve_area(shapes, language):
    _ = get_translator(language)

    for shape in shapes:
        if (
            shape.get("customdata") == _("nerf optique")
            or shape.get("customid", 0) == 1
        ):
            if shape.get("type") == "circle":
                coords = circle_to_coords(shape)
            else:
                path_str = shape.get("path", "")
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
                try:
                    coords = [
                        (float(matches[j]), float(matches[j + 1]))
                        for j in range(0, len(matches), 2)
                    ]
                except Exception:
                    coords = []
            return calculate_area(coords)
    return None


def get_lesion_areas(shapes, language):
    _ = get_translator(language)
    areas = []
    for shape in shapes:
        if shape.get("customdata", "").lower() not in [_("nerf optique")]:
            if shape.get("type") == "circle":
                coords = circle_to_coords(shape)
            else:
                path_str = shape.get("path", "")
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", path_str)
                try:
                    coords = [
                        (float(matches[j]), float(matches[j + 1]))
                        for j in range(0, len(matches), 2)
                    ]
                except Exception:
                    coords = []
            if coords:
                area = calculate_area(coords)
                areas.append(area)
    return areas


def generate_comparison_summary(
    shapes_left, shapes_right, language, patient_infos=None
):
    _ = get_translator(language)
    if not shapes_left or not shapes_right:
        return dbc.Alert(
            _("Les deux images doivent être ajoutées à la comparaison."),
            color="warning",
        )

    area_nerf_left = get_optic_nerve_area(shapes_left, language)
    area_nerf_right = get_optic_nerve_area(shapes_right, language)
    lesion_areas_left = get_lesion_areas(shapes_left, language)
    lesion_areas_right = get_lesion_areas(shapes_right, language)

    total_left = sum(lesion_areas_left)
    total_right = sum(lesion_areas_right)
    pct_left = (total_left / area_nerf_left * 100) if area_nerf_left else None
    pct_right = (total_right / area_nerf_right * 100) if area_nerf_right else None
    diff = total_right - total_left

    p_value_sum = None

    if lesion_areas_left and lesion_areas_right:
        p_value_sum = calculate_permutation_test(lesion_areas_left, lesion_areas_right)

    growth_txt = ""
    if patient_infos:
        date_gauche = patient_infos.get("date_gauche")
        date_droite = patient_infos.get("date_droite")
        if date_gauche and date_droite:
            try:
                d0 = datetime.strptime(date_gauche, "%Y-%m-%d")
                d1 = datetime.strptime(date_droite, "%Y-%m-%d")
                days = (d1 - d0).days
                if days > 0:
                    n_months = days / 30.44
                    n_years = days / 365.25
                    growth_month = diff / n_months if n_months != 0 else None
                    growth_year = diff / n_years if n_years != 0 else None
                    pct_month = (
                        (growth_month / total_left * 100)
                        if total_left and growth_month is not None
                        else None
                    )
                    pct_year = (
                        (growth_year / total_left * 100)
                        if total_left and growth_year is not None
                        else None
                    )

                    if n_years < 1:
                        growth_txt = (
                            (
                                _(
                                    "**Variation d’aire totale :** {val:.1f} pixels²/mois"
                                ).format(val=growth_month)
                                + "  \n"
                                + _("**Variation relative :** {val:.2f}%/mois").format(
                                    val=pct_month
                                )
                            )
                            if growth_month is not None
                            else ""
                        )
                    else:
                        growth_txt = (
                            (
                                _(
                                    "**Variation d’aire totale :** {val:.1f} pixels²/an"
                                ).format(val=growth_year)
                                + "  \n"
                                + _("**Variation relative :** {val:.2f}%/an").format(
                                    val=pct_year
                                )
                            )
                            if growth_year is not None
                            else ""
                        )
            except Exception:
                pass

    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th(""), html.Th("T0"), html.Th("Tx")])),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(_("Nombre de taches")),
                            html.Td(len(lesion_areas_left)),
                            html.Td(len(lesion_areas_right)),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(_("Aire totale (pixels²)")),
                            html.Td(f"{total_left:.0f}"),
                            html.Td(f"{total_right:.0f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(_("Aire totale (% du nerf optique)")),
                            html.Td(
                                f"{pct_left:.1f}%"
                                if pct_left is not None
                                else _("Non calculé")
                            ),
                            html.Td(
                                f"{pct_right:.1f}%"
                                if pct_right is not None
                                else _("Non calculé")
                            ),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(_("Différence brute (Tx-T0)")),
                            html.Td(colSpan=2, children=f"{diff:.0f} pixels²"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(_("p.value (diff. aire totale)")),
                            html.Td(
                                colSpan=2,
                                children=(
                                    f"{p_value_sum:.4f}"
                                    if p_value_sum is not None
                                    else _("Non calculable")
                                ),
                            ),
                        ]
                    ),
                ]
            ),
        ],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
    )

    mean_left = np.mean(lesion_areas_left) if len(lesion_areas_left) > 0 else 0
    mean_right = np.mean(lesion_areas_right) if len(lesion_areas_right) > 0 else 0
    date_gauche_label = (
        patient_infos.get("date_gauche", "T0") if patient_infos else "T0"
    )
    date_droite_label = (
        patient_infos.get("date_droite", "Tx") if patient_infos else "Tx"
    )

    fig = bar_dot_plot_evolution(
        total_left,
        total_right,
        lesion_areas_left,
        lesion_areas_right,
        mean_left,
        mean_right,
        language,
        date_left=date_gauche_label,
        date_right=date_droite_label,
    )

    growth_html = (
        html.Div([html.Hr(style={"margin": "8px 0"}), dcc.Markdown(growth_txt)])
        if growth_txt
        else None
    )

    return dbc.Card(
        [
            dbc.CardHeader(_("Synthèse de l'évolution")),
            dbc.CardBody(
                [
                    table,
                    dcc.Graph(
                        figure=fig,
                        config={"displayModeBar": False},
                        style={"height": "240px", "marginBottom": "6px"},
                    ),
                    growth_html if growth_html else None,
                ]
            ),
        ]
    )


def bar_dot_plot_evolution(
    total_left,
    total_right,
    lesion_areas_left,
    lesion_areas_right,
    mean_left,
    mean_right,
    language,
    date_left="T0",
    date_right="Tx",
):
    _ = get_translator(language)
    labels = [str(date_left or "T0"), str(date_right or "Tx")]

    lesion_areas_left = (
        lesion_areas_left if isinstance(lesion_areas_left, (list, np.ndarray)) else []
    )
    lesion_areas_right = (
        lesion_areas_right if isinstance(lesion_areas_right, (list, np.ndarray)) else []
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[total_left, total_right],
            name=_("Aire totale"),
            marker_color="#1976D2",
            width=0.5,
            opacity=0.25,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[labels[0]] * len(lesion_areas_left),
            y=lesion_areas_left,
            mode="markers",
            name=_("Aires lésionnelles T0"),
            marker=dict(
                size=9, color="#1abc9c", opacity=0.6, line=dict(width=1, color="white")
            ),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[labels[1]] * len(lesion_areas_right),
            y=lesion_areas_right,
            mode="markers",
            name=_("Aires lésionnelles Tx"),
            marker=dict(
                size=9, color="#e67e22", opacity=0.6, line=dict(width=1, color="white")
            ),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[mean_left, mean_right],
            mode="lines+markers",
            name=_("Moyenne"),
            marker=dict(size=16, symbol="diamond", color="#e74c3c"),
            line=dict(width=2, dash="dot", color="#e74c3c"),
            showlegend=True,
            hovertemplate=_("Moyenne : %{y:.0f} pixels²"),
        )
    )
    fig.update_layout(
        yaxis_title=_("Aire (pixels²)"),
        xaxis_title=_("Date"),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        height=260,
        margin=dict(l=12, r=12, t=38, b=10),
        xaxis=dict(type="category"),
    )
    return fig


@app.callback(
    Output("patient-infos-store", "data"),
    Output("save-patient-infos-feedback", "children"),
    Input("save-patient-infos-btn", "n_clicks"),
    State("patient-nom", "value"),
    State("patient-prenom", "value"),
    State("patient-dob", "value"),
    State("date-gauche", "value"),
    State("date-droite", "value"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def save_patient_infos(n, nom, prenom, dob, date_gauche, date_droite, language):
    _ = get_translator(language)
    if not n:
        return dash.no_update, ""
    infos = {
        "nom": nom,
        "prenom": prenom,
        "dob": dob,
        "date_gauche": date_gauche,
        "date_droite": date_droite,
        "last_save": datetime.now().isoformat(),
    }
    feedback = dbc.Alert(
        _("Informations patient enregistrées !"), color="success", duration=2000
    )
    return infos, feedback


@app.callback(
    Output("trigger-print-store", "data"),
    Input("export-report-btn", "n_clicks"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def trigger_pdf_export(n):
    if n:
        return True
    return dash.no_update


def render_pdf_report(patient_infos, summary_card, language):
    _ = get_translator(language)
    nom = patient_infos.get("nom", "") if patient_infos else ""
    prenom = patient_infos.get("prenom", "") if patient_infos else ""
    dob = patient_infos.get("dob", "") if patient_infos else ""

    return html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src="/assets/logo.png",
                        style={
                            "height": "60px",
                            "verticalAlign": "middle",
                            "marginRight": "10px",
                        },
                    ),
                    html.Span(
                        "FundusTracker",
                        style={
                            "fontSize": "28px",
                            "verticalAlign": "middle",
                            "fontWeight": "bold",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "12px",
                },
            ),
            html.Hr(),
            html.Div(
                [
                    html.P(_("Nom : {nom}").format(nom=nom)),
                    html.P(_("Prénom : {prenom}").format(prenom=prenom)),
                    html.P(_("Date de naissance : {dob}").format(dob=dob)),
                ],
                style={"marginBottom": "14px", "fontSize": "16px"},
            ),
            summary_card,
            html.Hr(),
            html.Footer(
                html.Div(
                    [
                        _("© 2025 – Réalisé par "),
                        html.A(
                            "Thomas Foulonneau",
                            href="https://www.linkedin.com/in/thomas-foulonneau?originalSubdomain=fr",
                            target="_blank",
                            style={"color": "#636e72", "textDecoration": "underline"},
                        ),
                        _(" & Aliénor Vienne-Jumeau"),
                    ]
                ),
                style={
                    "fontSize": "12px",
                    "textAlign": "center",
                    "color": "#636e72",
                    "background": "none",
                    "marginTop": "24px",
                    "padding": "0",
                },
            ),
        ],
        id="pdf-report-section",
        style={"background": "#fff", "color": "#222", "padding": "18px"},
    )


clientside_callback(
    """
    function(trigger) {
        if(trigger){
            setTimeout(function() {
                window.print();
            }, 500);
        }
        return false;
    }
    """,
    Output("trigger-print-store", "data"),
    Input("trigger-print-store", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("uploaded-image-store", "data"),
    Output("file-dropdown", "value", allow_duplicate=True),
    Output("local-filename-store", "data"),
    Output("upload-image", "contents", allow_duplicate=True),
    Output("upload-image", "filename", allow_duplicate=True),
    Output("upload-image", "last_modified", allow_duplicate=True),
    Input("upload-image", "contents"),
    Input("file-dropdown", "value"),
    State("upload-image", "filename"),
    State("upload-image", "last_modified"),
    prevent_initial_call=True,
)
def set_uploaded_image(contents, dropdown_value, filename, last_modified):
    triggered = (
        callback_context.triggered_id
        if hasattr(callback_context, "triggered_id")
        else None
    )

    if triggered == "upload-image" and contents is not None:

        return contents, None, filename, None, None, None

    elif triggered == "file-dropdown":

        if dropdown_value is None:

            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        return (
            None,
            dropdown_value,
            None,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    return None, None, None, None, None, None


@app.callback(
    Output("longitudinal-active-patient-store", "data"),
    Output("patient-dropdown", "options"),
    Output("patient-dropdown", "value"),
    Output("patient-workflow-feedback", "children"),
    Output("longitudinal-series-store", "data", allow_duplicate=True),
    Output("patient-active-display-header", "children"),
    Input("create-patient-btn", "n_clicks"),
    Input("patient-dropdown", "value"),
    Input("longitudinal-series-store", "data"),
    State("patient-nom-input", "value"),
    State("patient-prenom-input", "value"),
    State("patient-ddn-input", "value"),
    State("patient-sexe-input", "value"),
    Input("language-store", "data"),
    prevent_initial_call=True,
)
def manage_patient_selection(
    n_create, selected_patient_id, all_series, nom, prenom, ddn, sexe, language
):
    _ = get_translator(language)

    all_series = all_series or {}
    triggered_id = ctx.triggered_id

    options = [
        {"label": p_id.replace("_", " "), "value": p_id} for p_id in all_series.keys()
    ]
    header = dash.no_update

    if triggered_id == "create-patient-btn":
        if all([nom, prenom, ddn, sexe]):
            patient_id = f"{nom}_{prenom}_{ddn}_{sexe}".strip()
            if patient_id not in all_series:
                all_series[patient_id] = []
                options.append(
                    {"label": patient_id.replace("_", " "), "value": patient_id}
                )
                feedback = dbc.Alert(_("Patient créé."), color="success", duration=2000)
                return patient_id, options, patient_id, feedback, all_series, header
            else:
                feedback = dbc.Alert(_("Ce patient existe déjà."), color="warning")
                return patient_id, options, patient_id, feedback, dash.no_update, header
        else:
            return (
                dash.no_update,
                options,
                dash.no_update,
                dbc.Alert(_("Veuillez remplir tous les champs."), color="danger"),
                dash.no_update,
                dash.no_update,
            )

    elif triggered_id == "patient-dropdown" and selected_patient_id:
        feedback = dbc.Alert(_("Patient chargé."), color="info", duration=2000)

        return (
            selected_patient_id,
            options,
            selected_patient_id,
            feedback,
            dash.no_update,
            header,
        )

    if triggered_id == "longitudinal-series-store":
        return (
            dash.no_update,
            options,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    return dash.no_update, options, dash.no_update, "", dash.no_update, dash.no_update


@app.callback(
    Output("patient-dossier-display", "children"),
    Output("add-image-section", "children"),
    Input("longitudinal-active-patient-store", "data"),
    Input("longitudinal-series-store", "data"),
    Input("sort-by-radio", "value"),
    Input("selected-exams-store", "data"),
    Input("language-store", "data"),
)
def display_patient_dossier(
    active_patient_id, all_series, sort_by, selected_exam_ids, language
):
    _ = get_translator(language)
    if not active_patient_id:
        return (
            html.Div(
                _("Veuillez créer ou sélectionner un patient."),
                style={"textAlign": "center", "marginTop": "40px"},
            ),
            [],
        )

    add_new_image_section = dbc.Card(
        [
            dbc.CardHeader(html.H5(_("Ajouter un examen"))),
            dbc.CardBody(
                [
                    dbc.Label(_("Choisir une image :")),
                    dcc.Dropdown(
                        id="patient-image-dropdown",
                        options=[
                            {"label": f.split("/")[-1], "value": f} for f in filenames
                        ],
                    ),
                    html.P(_("ou"), style={"textAlign": "center", "margin": "10px 0"}),
                    dcc.Upload(
                        id="patient-image-upload",
                        children=html.Div([_("Importer un fichier local")]),
                        className="upload-area",
                        style={"height": "60px"},
                    ),
                    html.Hr(),
                    dbc.ButtonGroup(
                        [
                            dbc.Button(
                                [
                                    html.I(className="fas fa-pen-alt me-2"),
                                    _("Manuelle"),
                                ],
                                id="start-manual-annotation-btn",
                                color="primary",
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-magic me-2"), _("Semi-auto")],
                                id="start-ml-annotation-btn",
                                color="info",
                            ),
                        ],
                        className="mt-3",
                        style={"width": "100%"},
                    ),
                ]
            ),
        ]
    )

    patient_records = (all_series or {}).get(active_patient_id, [])
    dossier_content = []

    if not patient_records:
        dossier_content = [
            dbc.Alert(
                _("Ce dossier est vide. Ajoutez un premier examen."), color="info"
            )
        ]
    else:

        processed_records = []
        for record in patient_records:
            new_record = record.copy()
            shapes = new_record.get("shapes", [])
            lesions = [
                s
                for s in shapes
                if s.get("customdata", "").lower()
                not in ["nerf optique", "optic nerve"]
            ]
            new_record["nb_lesions"] = len(lesions)
            new_record["aire_totale"] = sum(
                calculate_area(parse_path_for_matching(s.get("path", "")))
                for s in lesions
            )
            processed_records.append(new_record)

        reverse_sort = sort_by != "exam_date"
        processed_records.sort(key=lambda r: r.get(sort_by, 0), reverse=reverse_sort)

        for record in processed_records:
            is_checked = record["record_id"] in selected_exam_ids
            try:
                img = load_image_any(record["image_id"])
                fig = generate_figure(
                    img,
                    shapes=[shape_for_plotly(s) for s in record.get("shapes", [])],
                    size="mini",
                )
                fig.update_layout(
                    width=200, height=200, margin=dict(l=5, r=5, t=5, b=5)
                )

                card_header = dbc.Row(
                    [
                        dbc.Col(
                            f"{_('Examen du')} {record.get('exam_date', 'N/A')}",
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Checkbox(
                                id={
                                    "type": "exam-checkbox",
                                    "index": record["record_id"],
                                },
                                value=is_checked,
                            ),
                            width="auto",
                        ),
                    ],
                    justify="between",
                    align="center",
                )

                aire_moyenne = (
                    record["aire_totale"] / record["nb_lesions"]
                    if record["nb_lesions"] > 0
                    else 0
                )

                aire_papille = 0
                for s in record.get("shapes", []):
                    if s.get("customdata", "").lower() in [
                        "nerf optique",
                        "optic nerve",
                    ]:
                        aire_papille = calculate_area(
                            parse_path_for_matching(s.get("path", ""))
                        )
                        break

                aire_normalisee = (
                    (record["aire_totale"] / aire_papille) if aire_papille > 0 else None
                )

                table_body = [
                    html.Tr(
                        [
                            html.Td(_("Nb. lésions"), className="fw-bold"),
                            html.Td(f"{record['nb_lesions']}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(_("Aire totale"), className="fw-bold"),
                            html.Td(f"{record['aire_totale']:.0f} px²"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(_("Aire moyenne"), className="fw-bold"),
                            html.Td(f"{aire_moyenne:.0f} px²"),
                        ]
                    ),
                ]

                if aire_normalisee is not None:
                    table_body.append(
                        html.Tr(
                            [
                                html.Td(
                                    _("Aire normalisée (ratio pupillaire)"),
                                    className="fw-bold",
                                ),
                                html.Td(f"{aire_normalisee:.3f}"),
                            ]
                        )
                    )

                info_table = dbc.Table(
                    html.Tbody(table_body),
                    bordered=False,
                    hover=True,
                    className="mb-0 record-info-table",
                )

                card_body = dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                                width=5,
                                className="record-thumbnail",
                            ),
                            dbc.Col(
                                info_table,
                                width=7,
                                className="record-info align-self-center",
                            ),
                        ],
                        align="center",
                    )
                )

                dossier_content.append(
                    dbc.Card(
                        [card_header, card_body], className="mb-3 patient-record-card"
                    )
                )
            except Exception as e:
                dossier_content.append(
                    dbc.Alert(
                        f"Erreur chargement examen du {record.get('exam_date')}: {e}",
                        color="danger",
                    )
                )

    return dossier_content, add_new_image_section


@app.callback(
    Output("patient-identity-display", "children"),
    Input("longitudinal-active-patient-store", "data"),
    State("language-store", "data"),
)
def display_patient_identity_card(active_patient_id, language):
    _ = get_translator(language)
    if not active_patient_id:
        return []

    try:
        nom, prenom, ddn, sexe = active_patient_id.split("_")
    except ValueError:

        return dbc.Alert("Format de l'ID patient incorrect.", color="danger")

    identity_card = dbc.Card(
        [
            dbc.CardHeader(_("Identité du patient actif")),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(_("Nom")),
                                    dbc.Input(
                                        id="edit-patient-nom", value=nom, type="text"
                                    ),
                                ],
                                width=6,
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(_("Prénom")),
                                    dbc.Input(
                                        id="edit-patient-prenom",
                                        value=prenom,
                                        type="text",
                                    ),
                                ],
                                width=6,
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(_("Date de naissance")),
                                    dbc.Input(
                                        id="edit-patient-ddn", value=ddn, type="date"
                                    ),
                                ],
                                width=6,
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(_("Sexe")),
                                    dcc.Dropdown(
                                        id="edit-patient-sexe",
                                        options=[
                                            {
                                                "label": _("Non spécifié"),
                                                "value": "Non spécifié",
                                            },
                                            {"label": _("Homme"), "value": "Homme"},
                                            {"label": _("Femme"), "value": "Femme"},
                                        ],
                                        value=sexe,
                                    ),
                                ],
                                width=6,
                                md=3,
                            ),
                        ]
                    ),
                    html.Hr(),
                    dbc.Button(
                        _("Enregistrer les modifications"),
                        id="save-patient-changes-btn",
                        color="success",
                        className="mt-2",
                    ),
                ]
            ),
        ]
    )
    return identity_card


@app.callback(
    Output("longitudinal-series-store", "data", allow_duplicate=True),
    Output("longitudinal-active-patient-store", "data", allow_duplicate=True),
    Output("save-patient-changes-feedback", "children"),
    Input("save-patient-changes-btn", "n_clicks"),
    State("longitudinal-active-patient-store", "data"),
    State("longitudinal-series-store", "data"),
    State("edit-patient-nom", "value"),
    State("edit-patient-prenom", "value"),
    State("edit-patient-ddn", "value"),
    State("edit-patient-sexe", "value"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def save_patient_identity_changes(
    n_clicks,
    old_patient_id,
    all_series,
    new_nom,
    new_prenom,
    new_ddn,
    new_sexe,
    language,
):
    _ = get_translator(language)
    if not n_clicks:
        return dash.no_update, dash.no_update, ""

    if not all([new_nom, new_prenom, new_ddn, new_sexe]):
        return (
            dash.no_update,
            dash.no_update,
            dbc.Alert(
                _("Tous les champs doivent être remplis."),
                color="danger",
                duration=4000,
            ),
        )

    new_patient_id = f"{new_nom}_{new_prenom}_{new_ddn}_{new_sexe}".strip()

    if new_patient_id == old_patient_id:
        return (
            dash.no_update,
            dash.no_update,
            dbc.Alert(_("Aucune modification détectée."), color="info", duration=3000),
        )

    if new_patient_id in all_series:
        return (
            dash.no_update,
            dash.no_update,
            dbc.Alert(
                _("Un patient avec cette identité existe déjà."),
                color="danger",
                duration=4000,
            ),
        )

    patient_data = all_series.pop(old_patient_id)
    all_series[new_patient_id] = patient_data

    feedback = dbc.Alert(
        _("Les informations du patient ont été mises à jour avec succès."),
        color="success",
        duration=3000,
    )

    return all_series, new_patient_id, feedback


@app.callback(
    Output("selected-exams-store", "data"),
    Input("select-all-btn", "n_clicks"),
    Input("deselect-all-btn", "n_clicks"),
    Input({"type": "exam-checkbox", "index": ALL}, "value"),
    State("longitudinal-active-patient-store", "data"),
    State("longitudinal-series-store", "data"),
    State({"type": "exam-checkbox", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def manage_exam_selection(
    n_all, n_none, checkbox_values, active_patient_id, all_series, checkbox_ids
):
    triggered_id = ctx.triggered_id
    if not active_patient_id or not all_series:
        return dash.no_update

    patient_records = all_series.get(active_patient_id, [])
    all_record_ids = [rec["record_id"] for rec in patient_records]

    if isinstance(triggered_id, str) and triggered_id == "select-all-btn":
        return all_record_ids

    if isinstance(triggered_id, str) and triggered_id == "deselect-all-btn":
        return []

    selected_ids = []
    for i, chk_id in enumerate(checkbox_ids):
        if i < len(checkbox_values) and checkbox_values[i]:
            selected_ids.append(chk_id["index"])
    return selected_ids


@app.callback(
    Output("tabs", "value", allow_duplicate=True),
    Output("annotation-image-store", "data", allow_duplicate=True),
    Output("stored-shapes", "data", allow_duplicate=True),
    Output("ml-image-store", "data", allow_duplicate=True),
    Output("ml-squiggle-store", "data", allow_duplicate=True),
    Input("start-manual-annotation-btn", "n_clicks"),
    Input("start-ml-annotation-btn", "n_clicks"),
    State("patient-image-dropdown", "value"),
    State("patient-image-upload", "contents"),
    prevent_initial_call=True,
)
def redirect_to_annotation_tab(manual_clicks, ml_clicks, image_file, image_upload):
    triggered_id = ctx.triggered_id
    if not triggered_id:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    image_id_to_annotate = image_file or image_upload
    if not image_id_to_annotate:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    if triggered_id == "start-manual-annotation-btn":
        return "tab-manuelle", image_id_to_annotate, [], dash.no_update, dash.no_update

    if triggered_id == "start-ml-annotation-btn":
        return "tab-ml", dash.no_update, dash.no_update, image_id_to_annotate, []

    return (
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
        dash.no_update,
    )


@app.callback(
    Output("longitudinal-series-store", "data", allow_duplicate=True),
    Output("save-to-patient-feedback-manuelle", "children"),
    Input("save-to-patient-btn-manuelle", "n_clicks"),
    State("longitudinal-active-patient-store", "data"),
    State("longitudinal-series-store", "data"),
    State("annotation-image-store", "data"),
    State("stored-shapes", "data"),
    State("annotation-date-picker-manuelle", "date"),
    Input("language-store", "data"),
    prevent_initial_call=True,
)
def save_annotation_to_patient(
    n_clicks, active_patient_id, all_series, image_id, shapes, exam_date, language
):
    _ = get_translator(language)
    if not n_clicks or not active_patient_id or not image_id or not shapes:
        return dash.no_update, dash.no_update
    if not exam_date:
        return dash.no_update, dbc.Alert(
            _("Veuillez spécifier la date de l'examen."), color="warning", duration=3000
        )

    new_record = {
        "record_id": str(uuid.uuid4()),
        "image_id": image_id,
        "shapes": shapes,
        "exam_date": exam_date,
        "saved_at": datetime.now().isoformat(),
    }
    all_series_updated = all_series or {}
    if active_patient_id not in all_series_updated:
        all_series_updated[active_patient_id] = []
    all_series_updated[active_patient_id].append(new_record)
    return all_series_updated, dbc.Alert(
        _("Annotation sauvegardée !"), color="success", duration=2000
    )


@app.callback(
    Output("longitudinal-evolution-graph", "figure"),
    Output("longitudinal-individual-graph", "figure"),
    Output("longitudinal-stats-output", "children"),
    Input("selected-exams-store", "data"),
    State("longitudinal-active-patient-store", "data"),
    State("longitudinal-series-store", "data"),
    Input("language-store", "data"),
)
def update_longitudinal_analysis_view(
    selected_exam_ids, active_patient_id, all_series, language
):

    _ = get_translator(language)
    import statsmodels.formula.api as smf
    import scipy.stats as stats
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    empty_fig = go.Figure().update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#F8F9FA",
        annotations=[dict(text=_("Données insuffisantes"), showarrow=False)],
    )
    if not active_patient_id or not all_series or not selected_exam_ids:
        return empty_fig, empty_fig, ""

    all_patient_records = all_series.get(active_patient_id, [])
    records_to_analyze = [
        rec for rec in all_patient_records if rec["record_id"] in selected_exam_ids
    ]
    if len(records_to_analyze) < 2:
        msg = dbc.Alert(
            _("Veuillez sélectionner au moins 2 examens pour une analyse complète."),
            color="warning",
        )
        return empty_fig, empty_fig, msg

    records_to_analyze.sort(key=lambda r: r.get("exam_date", "1900-01-01"))

    can_normalize = True
    optic_nerve_areas = {}
    for rec in records_to_analyze:
        on_area = 0
        for s in rec.get("shapes", []):
            if s.get("customdata", "").lower() in [
                "nerf optique".lower(),
                "optic nerve",
            ]:
                area = calculate_area(parse_path_for_matching(s.get("path", "")))
                if area > 0:
                    on_area = area
                break
        if on_area > 0:
            optic_nerve_areas[rec["record_id"]] = on_area
        else:
            can_normalize = False
            break

    all_lesion_data = []
    has_unique_ids = False
    for rec in records_to_analyze:
        norm_factor = (
            optic_nerve_areas.get(rec["record_id"], 1.0) if can_normalize else 1.0
        )
        exam_date = pd.to_datetime(rec.get("exam_date"))
        for s in rec.get("shapes", []):
            if s.get("customdata", "").lower() not in [
                "nerf optique".lower(),
                "optic nerve",
            ]:
                area = (
                    calculate_area(parse_path_for_matching(s.get("path", "")))
                    / norm_factor
                )
                lesion_info = {
                    "date": exam_date,
                    "area": area,
                    "unique_lesion_id": s.get("unique_lesion_id"),
                }
                if lesion_info["unique_lesion_id"]:
                    has_unique_ids = True
                all_lesion_data.append(lesion_info)

    if not all_lesion_data:
        return empty_fig, empty_fig, ""
    master_long_df = pd.DataFrame(all_lesion_data)
    num_time_points = len(master_long_df["date"].unique())

    fig_agg = go.Figure()
    unit_label_graph = _("Ratio papillaire") if can_normalize else _("pixels²")

    df_agg_total = master_long_df.groupby("date")["area"].sum().reset_index()
    stats_df = (
        master_long_df.groupby("date")["area"]
        .agg(
            [
                "median",
                "count",
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 75),
            ]
        )
        .reset_index()
    )
    stats_df.columns = ["date", "median", "count", "q1", "q3"]

    fig_agg.add_trace(
        go.Bar(
            x=df_agg_total["date"],
            y=df_agg_total["area"],
            name=_("Aire totale"),
            marker_color="#1976D2",
            opacity=0.3,
            yaxis="y1",
        )
    )
    fig_agg.add_trace(
        go.Scatter(
            x=stats_df["date"],
            y=stats_df["q3"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            yaxis="y2",
            hoverinfo="none",
        )
    )
    fig_agg.add_trace(
        go.Scatter(
            x=stats_df["date"],
            y=stats_df["q1"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 165, 0, 0.2)",
            name=_("Intervalle interquartile"),
            yaxis="y2",
            hoverinfo="none",
        )
    )
    fig_agg.add_trace(
        go.Scatter(
            x=stats_df["date"],
            y=stats_df["median"],
            mode="lines+markers",
            name=_("Médiane des aires"),
            line=dict(color="rgb(255, 165, 0)", width=3),
            marker=dict(size=7),
            yaxis="y2",
        )
    )

    fig_agg.update_layout(
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title=f"{_('Aire totale')} ({unit_label_graph})"),
        yaxis2=dict(
            title=f"{_('Aire individuelle')} ({unit_label_graph})",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        hovermode="x unified",
    )

    fig_sankey = empty_fig.update_layout(
        title=_("Suivi individuel non disponible (ID de lésion manquants)")
    )

    if has_unique_ids and not master_long_df.empty:
        STATUS_ORDER = [
            _("Initial"),
            _("Apparition"),
            _("Stabilité"),
            _("Progression"),
            _("Disparition"),
        ]
        STATUS_COLORS_HEX = {
            _("Initial"): "#636e72",
            _("Apparition"): "#55efc4",
            _("Stabilité"): "#0984e3",
            _("Progression"): "#fdcb6e",
            _("Disparition"): "#d63031",
        }

        y_margin = 0.02
        num_statuses = len(STATUS_ORDER)
        y_step = (1.0 - 2 * y_margin) / (num_statuses - 1) if num_statuses > 1 else 0
        Y_POSITIONS = {
            status: y_margin + i * y_step for i, status in enumerate(STATUS_ORDER)
        }

        AREA_CHANGE_THRESHOLD = 0.25
        time_points = sorted(master_long_df["date"].unique())
        num_time_points = len(time_points)

        lesion_history = {}
        for lesion_id in master_long_df["unique_lesion_id"].unique():
            lesion_history[lesion_id] = {}
            for t_idx, t_current in enumerate(time_points):
                record = master_long_df[
                    (master_long_df["unique_lesion_id"] == lesion_id)
                    & (master_long_df["date"] == t_current)
                ]
                if not record.empty:
                    area_current = record["area"].iloc[0]
                    is_first = not any(
                        t < t_idx for t in lesion_history[lesion_id].keys()
                    )
                    if is_first:
                        status = _("Initial") if t_idx == 0 else _("Apparition")
                    else:
                        prev_t_idx = max(
                            t for t in lesion_history[lesion_id].keys() if t < t_idx
                        )
                        area_previous = lesion_history[lesion_id][prev_t_idx]["area"]
                        change = (
                            (area_current - area_previous) / area_previous
                            if area_previous > 0
                            else float("inf")
                        )
                        status = (
                            _("Progression")
                            if change > AREA_CHANGE_THRESHOLD
                            else _("Stabilité")
                        )
                    lesion_history[lesion_id][t_idx] = {
                        "area": area_current,
                        "status": status,
                    }

        links_agg = {}
        if lesion_history:
            initial_areas = {}
            for lesion_id, history in lesion_history.items():
                if history:
                    first_appearance_idx = min(history.keys())
                    initial_areas[lesion_id] = history[first_appearance_idx]["area"]

            for t_idx in range(num_time_points - 1):
                for lesion_id, history in lesion_history.items():
                    if t_idx in history:
                        source_state = history[t_idx]
                        source_key = (t_idx, source_state["status"])
                        flow_value = initial_areas.get(lesion_id, 0)
                        target_key = (
                            (t_idx + 1, history[t_idx + 1]["status"])
                            if t_idx + 1 in history
                            else (t_idx + 1, _("Disparition"))
                        )
                        links_agg[(source_key, target_key)] = (
                            links_agg.get((source_key, target_key), 0) + flow_value
                        )

        if links_agg:
            all_node_keys = sorted(
                list(
                    set(k[0] for k in links_agg.keys())
                    | set(k[1] for k in links_agg.keys())
                )
            )
            node_map = {key: i for i, key in enumerate(all_node_keys)}

            inflows = {key: 0 for key in all_node_keys}
            outflows = {key: 0 for key in all_node_keys}
            for (source_key, target_key), area in links_agg.items():
                outflows[source_key] += area
                inflows[target_key] += area

            node_total_areas = {
                key: max(inflows.get(key, 0), outflows.get(key, 0))
                for key in all_node_keys
            }

            last_t_idx = num_time_points - 1
            final_column_keys = [key for key in all_node_keys if key[0] == last_t_idx]

            for key in final_column_keys:
                t_idx, status = key

                if status == _("Disparition"):
                    continue

                actual_area_at_tn = sum(
                    history[t_idx]["area"]
                    for lesion_id, history in lesion_history.items()
                    if t_idx in history and history[t_idx]["status"] == status
                )

                if actual_area_at_tn > 0:
                    node_total_areas[key] = actual_area_at_tn

            unit_sankey = unit_label_graph

            inflows = {key: 0 for key in all_node_keys}
            for (source_key, target_key), area in links_agg.items():
                inflows[target_key] += area

            node_total_areas = {}
            for key in all_node_keys:
                t_idx, status = key

                if status == _("Disparition"):
                    node_total_areas[key] = inflows.get(key, 0)
                else:

                    actual_area_at_t = sum(
                        history[t_idx]["area"]
                        for lesion_id, history in lesion_history.items()
                        if t_idx in history and history[t_idx]["status"] == status
                    )
                    node_total_areas[key] = actual_area_at_t

            unit_sankey = unit_label_graph

            inflows = {key: 0 for key in all_node_keys}
            for (source_key, target_key), area in links_agg.items():
                inflows[target_key] += area

            node_total_areas = {}
            for key in all_node_keys:
                t_idx, status = key

                if status == _("Disparition"):
                    node_total_areas[key] = inflows.get(key, 0)
                else:

                    actual_area_at_t = sum(
                        history[t_idx]["area"]
                        for lesion_id, history in lesion_history.items()
                        if t_idx in history and history[t_idx]["status"] == status
                    )
                    node_total_areas[key] = actual_area_at_t

            unit_sankey = unit_label_graph
            node_labels = [
                (
                    f"{node_total_areas.get(key, 0):.3f} {unit_sankey}"
                    if can_normalize
                    else f"{node_total_areas.get(key, 0):.0f} {unit_sankey}"
                )
                for key in all_node_keys
            ]

            fig_sankey = go.Figure(
                data=[
                    go.Sankey(
                        arrangement="perpendicular",
                        node=dict(
                            pad=100,
                            thickness=10,
                            line=dict(color="black", width=0.5),
                            label=node_labels,
                            x=[
                                key[0] / max(1, num_time_points - 1)
                                for key in all_node_keys
                            ],
                            y=[Y_POSITIONS[key[1]] for key in all_node_keys],
                            color=[
                                STATUS_COLORS_HEX.get(key[1]) for key in all_node_keys
                            ],
                            hovertemplate="%{label}<extra></extra>",
                        ),
                        link=dict(
                            source=[node_map[link[0]] for link in links_agg.keys()],
                            target=[node_map[link[1]] for link in links_agg.keys()],
                            value=[area for area in links_agg.values()],
                        ),
                    )
                ]
            )

            for status, color in STATUS_COLORS_HEX.items():
                fig_sankey.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(size=10, color=color, symbol="square"),
                        name=status,
                        showlegend=True,
                    )
                )

            sankey_annotations = []
            unique_dates = sorted(master_long_df["date"].unique())
            for i, date_val in enumerate(unique_dates):
                date_str = pd.to_datetime(date_val).strftime("%d/%m/%Y")
                x_pos = i / max(1, len(unique_dates) - 1)
                sankey_annotations.append(
                    dict(
                        x=x_pos,
                        y=-0.05,
                        text=f"<b>{date_str}</b>",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        xanchor="center",
                        yanchor="top",
                    )
                )

            fig_sankey.update_layout(
                font=dict(size=12),
                height=500,
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis_visible=False,
                yaxis_visible=False,
                margin=dict(l=60, r=60, t=80, b=50),
                annotations=sankey_annotations,
                legend=dict(
                    title=_("Statut"),
                    orientation="h",
                    yanchor="bottom",
                    y=1.0,
                    xanchor="center",
                    x=0.5,
                    itemclick=False,
                    itemdoubleclick=False,
                ),
            )

    stats_content = []
    unit_label = _("pixels²")
    normalization_footer = None

    if num_time_points >= 2:

        can_normalize = True
        optic_nerve_areas = {}
        for rec in records_to_analyze:
            rec_id = rec["record_id"]
            on_area = 0
            for s in rec.get("shapes", []):
                if s.get("customdata", "").lower() in [
                    "nerf optique".lower(),
                    "optic nerve",
                ]:
                    area = calculate_area(parse_path_for_matching(s.get("path", "")))
                    if area > 0:
                        on_area = area
                    break
            if on_area > 0:
                optic_nerve_areas[rec_id] = on_area
            else:
                can_normalize = False
                break

        unit_label = _("d'aire papillaire") if can_normalize else _("pixels²")
        normalization_footer_text = (
            _(
                "Analyse réalisée sur des aires normalisées par la surface de la papille."
            )
            if can_normalize
            else _(
                "Analyse réalisée sur des aires brutes (pixels²), car la papille n'a pas été détectée sur au moins un examen."
            )
        )
        normalization_footer = html.P(
            normalization_footer_text, className="text-muted small mt-2 fst-italic"
        )

        all_lesion_data = []
        has_unique_ids = False
        for rec in records_to_analyze:
            norm_factor = (
                optic_nerve_areas.get(rec["record_id"], 1.0) if can_normalize else 1.0
            )
            exam_date = pd.to_datetime(rec.get("exam_date"))

            for s in rec.get("shapes", []):
                if s.get("customdata", "").lower() not in [
                    "nerf optique".lower(),
                    "optic nerve",
                ]:
                    area = (
                        calculate_area(parse_path_for_matching(s.get("path", "")))
                        / norm_factor
                    )
                    lesion_info = {
                        "date": exam_date,
                        "area": area,
                        "unique_lesion_id": s.get("unique_lesion_id"),
                    }
                    if lesion_info["unique_lesion_id"]:
                        has_unique_ids = True
                    all_lesion_data.append(lesion_info)

        master_long_df = pd.DataFrame(all_lesion_data)

        if not master_long_df.empty:
            dates = sorted(master_long_df["date"].unique())
            areas_t0 = master_long_df[master_long_df["date"] == dates[0]][
                "area"
            ].tolist()
            areas_t_end = master_long_df[master_long_df["date"] == dates[-1]][
                "area"
            ].tolist()

            if areas_t0 and areas_t_end and (len(areas_t0) + len(areas_t_end)) > 2:
                p_value_perm = calculate_permutation_test(areas_t0, areas_t_end)
                card_permutation = dbc.Card(
                    [
                        dbc.CardHeader(
                            _("Analyse des aires globales (Test de permutation)")
                        ),
                        dbc.CardBody(
                            [
                                html.P(
                                    _(
                                        "Ce test non-paramétrique compare la distribution de l'ensemble des aires lésionnelles entre le premier et le dernier examen."
                                    )
                                ),
                                html.P(
                                    [
                                        html.Strong(
                                            _(
                                                "Significativité de la variation globale (p-value) : "
                                            )
                                        ),
                                        html.Span(
                                            f"{p_value_perm:.4f}",
                                            style={
                                                "fontWeight": "bold",
                                                "color": "#007BFF",
                                            },
                                        ),
                                    ]
                                ),
                                normalization_footer,
                            ]
                        ),
                    ],
                    color="info",
                    outline=True,
                    class_name="mb-3",
                )
                stats_content.append(card_permutation)

        if has_unique_ids and not master_long_df.empty:
            if num_time_points == 2:
                paired_data = master_long_df.pivot(
                    index="unique_lesion_id", columns="date", values="area"
                ).dropna()

                if not paired_data.empty and len(paired_data) >= 3:
                    t_stat, p_value_ttest = stats.ttest_rel(
                        paired_data.iloc[:, 0], paired_data.iloc[:, 1]
                    )
                    mean_diff = (paired_data.iloc[:, 1] - paired_data.iloc[:, 0]).mean()

                    card_ttest = dbc.Card(
                        [
                            dbc.CardHeader(
                                _("Analyse des lésions appariées (Test t de Student)")
                            ),
                            dbc.CardBody(
                                [
                                    html.P(
                                        _(
                                            "Ce test compare les aires des {n_lesions} lésions appariées pour déterminer si la variation moyenne est significative."
                                        ).format(n_lesions=len(paired_data))
                                    ),
                                    html.P(
                                        [
                                            html.Strong(
                                                _("Variation moyenne par lésion : ")
                                            ),
                                            html.Span(
                                                f"{mean_diff:+.3f} {unit_label}",
                                                style={
                                                    "fontWeight": "bold",
                                                    "color": (
                                                        "#28A745"
                                                        if mean_diff >= 0
                                                        else "#DC3545"
                                                    ),
                                                },
                                            ),
                                        ]
                                    ),
                                    html.P(
                                        [
                                            html.Strong(
                                                _(
                                                    "Significativité de la variation (p-value) : "
                                                )
                                            ),
                                            html.Span(
                                                f"{p_value_ttest:.4f}",
                                                style={
                                                    "fontWeight": "bold",
                                                    "color": "#007BFF",
                                                },
                                            ),
                                        ]
                                    ),
                                    normalization_footer,
                                ]
                            ),
                        ],
                        color="success",
                        outline=True,
                        class_name="mb-3",
                    )
                    stats_content.append(card_ttest)

            elif num_time_points >= 3:
                df_for_model = master_long_df.copy()
                df_for_model["time"] = (
                    df_for_model["date"] - df_for_model["date"].min()
                ).dt.days
                lesion_counts = df_for_model.dropna(subset=["unique_lesion_id"])[
                    "unique_lesion_id"
                ].value_counts()
                tracked_lesions_ids = lesion_counts[lesion_counts >= 2].index

                if len(tracked_lesions_ids) >= 5:
                    try:
                        model_df = df_for_model[
                            df_for_model["unique_lesion_id"].isin(tracked_lesions_ids)
                        ]
                        mixed_model = smf.mixedlm(
                            "area ~ time", model_df, groups=model_df["unique_lesion_id"]
                        )
                        result = mixed_model.fit(method=["lbfgs"])

                        p_value_mixed = result.pvalues["time"]
                        slope = result.params["time"]
                        progression_par_an = slope * 365.25

                        card_mixed_model = dbc.Card(
                            [
                                dbc.CardHeader(
                                    _("Modèle de progression linéaire à effets mixtes")
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            _(
                                                "Ce modèle estime une trajectoire de croissance moyenne pour {n_lesions} lésions suivies sur {n_dates} examens."
                                            ).format(
                                                n_lesions=len(tracked_lesions_ids),
                                                n_dates=num_time_points,
                                            )
                                        ),
                                        html.P(
                                            [
                                                html.Strong(
                                                    _(
                                                        "Progression annuelle moyenne estimée : "
                                                    )
                                                ),
                                                html.Span(
                                                    f"{progression_par_an:+.3f} {unit_label}/an",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "color": (
                                                            "#28A745"
                                                            if progression_par_an >= 0
                                                            else "#DC3545"
                                                        ),
                                                    },
                                                ),
                                            ]
                                        ),
                                        html.P(
                                            [
                                                html.Strong(
                                                    _(
                                                        "Significativité de la progression (p-value) : "
                                                    )
                                                ),
                                                html.Span(
                                                    f"{p_value_mixed:.4f}",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "color": "#007BFF",
                                                    },
                                                ),
                                            ]
                                        ),
                                        normalization_footer,
                                    ]
                                ),
                            ],
                            color="primary",
                            outline=True,
                            class_name="mb-3",
                        )
                        stats_content.append(card_mixed_model)
                    except Exception as e:
                        stats_content.append(
                            dbc.Alert(
                                f"Le modèle à effets mixtes n'a pas pu converger. Erreur : {e}",
                                color="warning",
                            )
                        )

    if not stats_content:
        stats_content = dbc.Alert(
            _("Données insuffisantes pour une analyse statistique détaillée."),
            color="secondary",
        )
    return fig_agg, fig_sankey, stats_content


@app.callback(
    Output("matching-modal", "is_open"),
    Output("matching-modal", "children"),
    Output("matching-results-store", "data"),
    Input("launch-matching-btn", "n_clicks"),
    Input("close-matching-modal-btn", "n_clicks"),
    State("matching-modal", "is_open"),
    State("compare-shapes-left", "data"),
    State("compare-shapes-right", "data"),
    State("compare-image-left", "data"),
    State("compare-image-right", "data"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def toggle_matching_modal(
    n_launch,
    n_close,
    is_open,
    shapes_left,
    shapes_right,
    img_left_id,
    img_right_id,
    language,
):
    _ = get_translator(language)
    triggered_id = ctx.triggered_id

    if not triggered_id:
        return False, None, dash.no_update

    if triggered_id == "launch-matching-btn":
        error_messages = []
        if not shapes_left:
            error_messages.append(
                _("Les annotations de référence (gauche) sont manquantes.")
            )
        if not shapes_right:
            error_messages.append(
                _("Les annotations de comparaison (droite) sont manquantes.")
            )
        if not img_left_id:
            error_messages.append(_("L'image de référence (gauche) est manquante."))
        if not img_right_id:
            error_messages.append(_("L'image de comparaison (droite) est manquante."))
        if error_messages:
            modal_body = [html.P(_("Erreur de workflow :"))] + [
                html.P(msg, className="text-danger") for msg in error_messages
            ]
            return (
                True,
                [
                    dbc.ModalHeader(_("Données Manquantes")),
                    dbc.ModalBody(modal_body),
                    dbc.ModalFooter(
                        dbc.Button(_("Fermer"), id="close-matching-modal-btn")
                    ),
                ],
                dash.no_update,
            )

        try:
            results = run_auto_matching(shapes_left, shapes_right, language)
            fig_left = generate_figure(
                load_image_any(img_left_id),
                shapes=[shape_for_plotly(s) for s in results["shapes_t0_colored"]],
                size="mini",
            )
            fig_left.update_layout(title=_("Image de référence (T0)"))
            fig_right = generate_figure(
                load_image_any(img_right_id),
                shapes=[shape_for_plotly(s) for s in results["shapes_t1_colored"]],
                size="mini",
            )
            fig_right.update_layout(title=_("Image de comparaison (T1)"))

            modal_children = [
                dbc.ModalHeader(_("Appariement Automatique")),
                dbc.ModalBody(
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(figure=fig_left), width=6),
                            dbc.Col(dcc.Graph(figure=fig_right), width=6),
                        ]
                    )
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            _("Exporter T0 Augmenté"),
                            id="export-augmented-t0-btn",
                            color="primary",
                        ),
                        dbc.Button(
                            _("Exporter T1 Augmenté"),
                            id="export-augmented-t1-btn",
                            color="info",
                        ),
                        dbc.Button(
                            _("Fermer"),
                            id="close-matching-modal-btn",
                            className="ml-auto",
                        ),
                        dcc.Download(id="download-augmented-t0"),
                        dcc.Download(id="download-augmented-t1"),
                    ]
                ),
            ]
            return True, modal_children, results
        except Exception as e:
            return (
                True,
                [
                    dbc.ModalHeader(_("Erreur d'Exécution")),
                    dbc.ModalBody(dbc.Alert(f"Erreur d'analyse: {e}", color="danger")),
                    dbc.ModalFooter(
                        dbc.Button(_("Fermer"), id="close-matching-modal-btn")
                    ),
                ],
                dash.no_update,
            )

    if triggered_id == "close-matching-modal-btn":
        return False, None, dash.no_update

    return False, None, dash.no_update


@app.callback(
    Output("download-augmented-t0", "data"),
    Input("export-augmented-t0-btn", "n_clicks"),
    State("matching-results-store", "data"),
    State("compare-image-left", "data"),
    prevent_initial_call=True,
)
def export_augmented_t0(n_clicks, matching_results, img_id_t0):

    if not n_clicks or not matching_results:
        return dash.no_update

    shapes_t0 = matching_results.get("augmented_shapes_t0")
    if not shapes_t0:
        return dash.no_update

    def create_filename(original_id, suffix="_augmented.json"):
        if not original_id or not isinstance(original_id, str):
            return f"annotations_T0{suffix}"
        if original_id.startswith("data:image"):
            return f"local_image_T0{suffix}"
        base_name = original_id.split("/")[-1].rsplit(".", 1)[0]
        return f"{base_name}{suffix}"

    return dcc.send_string(json.dumps(shapes_t0, indent=2), create_filename(img_id_t0))


@app.callback(
    Output("download-augmented-t1", "data"),
    Input("export-augmented-t1-btn", "n_clicks"),
    State("matching-results-store", "data"),
    State("compare-image-right", "data"),
    prevent_initial_call=True,
)
def export_augmented_t1(n_clicks, matching_results, img_id_t1):

    if not n_clicks or not matching_results:
        return dash.no_update

    shapes_t1 = matching_results.get("augmented_shapes_t1")
    if not shapes_t1:
        return dash.no_update

    def create_filename(original_id, suffix="_augmented.json"):
        if not original_id or not isinstance(original_id, str):
            return f"annotations_T1{suffix}"
        if original_id.startswith("data:image"):
            return f"local_image_T1{suffix}"
        base_name = original_id.split("/")[-1].rsplit(".", 1)[0]
        return f"{base_name}{suffix}"

    return dcc.send_string(json.dumps(shapes_t1, indent=2), create_filename(img_id_t1))


@app.callback(
    Output("patient-matching-action-zone", "children"),
    Input("selected-exams-store", "data"),
    Input("language-store", "data"),
)
def update_matching_action_zone(selected_ids, language):
    _ = get_translator(language)
    if selected_ids and len(selected_ids) >= 2:
        return dbc.Button(
            [
                html.I(className="fas fa-link me-2"),
                _("Lancer l'Atelier d'Appariement en Chaîne (%s examens)")
                % len(selected_ids),
            ],
            id="launch-patient-matching-btn",
            color="warning",
            className="w-100",
        )
    return dbc.Alert(
        _(
            "Veuillez sélectionner au moins 2 examens pour lancer un appariement en chaîne."
        ),
        color="info",
        className="text-center",
    )


@app.callback(
    Output("patient-matching-modal", "is_open"),
    Output("patient-matching-modal-body", "children"),
    Output("patient-matching-modal-footer", "children"),
    Input("launch-patient-matching-btn", "n_clicks"),
    State("selected-exams-store", "data"),
    State("longitudinal-active-patient-store", "data"),
    State("longitudinal-series-store", "data"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def open_matching_confirmation_modal(
    n_clicks, selected_ids, active_patient_id, all_series, language
):
    _ = get_translator(language)
    if not n_clicks:
        return False, None, None

    patient_records = all_series.get(active_patient_id, [])
    records_to_process = [
        rec for rec in patient_records if rec["record_id"] in selected_ids
    ]

    records_to_process.sort(key=lambda r: r.get("exam_date", "1900-01-01"))

    if len(records_to_process) < 2:
        return False, None, None

    list_items = []
    for i, rec in enumerate(records_to_process):
        arrow = (
            html.I(className="fas fa-arrow-down my-2")
            if i < len(records_to_process) - 1
            else None
        )
        list_items.append(
            html.Div(
                [
                    dbc.ListGroupItem(
                        f"Étape {i + 1}: Examen du {rec.get('exam_date')}"
                    ),
                    arrow,
                ],
                className="text-center",
            )
        )

    body = [
        dbc.Alert(
            _(
                "Vous êtes sur le point de lancer un appariement en chaîne automatisé. Les identifiants des lésions seront propagés séquentiellement à travers les examens dans l'ordre chronologique suivant :"
            ),
            color="info",
        ),
        dbc.ListGroup(
            list_items, flush=True, style={"maxHeight": "300px", "overflowY": "auto"}
        ),
        html.P(
            _(
                "Cette opération écrasera les appariements précédents pour les examens sélectionnés. Êtes-vous sûr ?"
            ),
            className="mt-4 fw-bold",
        ),
    ]

    footer = [
        dbc.Button(
            _("Annuler"), id="close-patient-matching-modal-btn", color="secondary"
        ),
        dbc.Button(
            [_("Confirmer et Lancer la Chaîne"), html.I(className="fas fa-play ms-2")],
            id="confirm-and-run-chain-btn",
            color="primary",
            className="ms-2",
        ),
    ]

    return True, body, footer


@app.callback(
    Output("longitudinal-series-store", "data", allow_duplicate=True),
    Output("patient-matching-modal", "is_open", allow_duplicate=True),
    Output("patient-workflow-feedback", "children", allow_duplicate=True),
    Input("confirm-and-run-chain-btn", "n_clicks"),
    State("selected-exams-store", "data"),
    State("longitudinal-active-patient-store", "data"),
    State("longitudinal-series-store", "data"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def run_chain_and_save_all(
    n_clicks, selected_ids, active_patient_id, all_series, language
):
    _ = get_translator(language)
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update

    all_series_updated = json.loads(json.dumps(all_series))
    patient_records = all_series_updated.get(active_patient_id, [])
    records_map = {rec["record_id"]: rec for rec in patient_records}
    records_to_process = [
        records_map[rec_id] for rec_id in selected_ids if rec_id in records_map
    ]
    records_to_process.sort(key=lambda r: r.get("exam_date", "1900-01-01"))

    if records_to_process:
        first_record = records_to_process[0]
        initial_color = "#f5f6fa"
        initial_status = _("Initial")
        for shape in first_record.get("shapes", []):
            if shape.get("customdata") not in ["nerf optique", "optic nerve"]:
                if "line" not in shape:
                    shape["line"] = {}
                shape["line"].update(
                    {"color": initial_color, "width": 3, "dash": "solid"}
                )

                if "unique_lesion_id" not in shape or not shape["unique_lesion_id"]:
                    shape["unique_lesion_id"] = str(uuid.uuid4())
        first_record["last_matched"] = datetime.now().isoformat()

    for i in range(len(records_to_process) - 1):
        record_t0 = records_to_process[i]
        record_t1 = records_to_process[i + 1]

        shapes_t0 = record_t0.get("shapes", [])
        shapes_t1 = record_t1.get("shapes", [])

        results = run_auto_matching(shapes_t0, shapes_t1, language)

        record_t1["shapes"] = results["augmented_shapes_t1"]
        record_t1["last_matched"] = datetime.now().isoformat()

    feedback = dbc.Alert(
        _("Appariement en chaîne simplifié terminé !"), color="success", duration=5000
    )

    return all_series_updated, False, feedback


@app.callback(
    Output("patient-matching-modal", "is_open", allow_duplicate=True),
    Input("close-patient-matching-modal-btn", "n_clicks"),
    prevent_initial_call=True,
)
def close_patient_matching_modal(n_clicks):
    if n_clicks:
        return False
    return dash.no_update


@app.callback(
    Output("export-patient-dossier-btn", "disabled"),
    Input("longitudinal-active-patient-store", "data"),
)
def toggle_export_button(active_patient_id):

    return not active_patient_id


@app.callback(
    Output("download-patient-dossier-json", "data"),
    Input("export-patient-dossier-btn", "n_clicks"),
    State("longitudinal-active-patient-store", "data"),
    State("longitudinal-series-store", "data"),
    prevent_initial_call=True,
)
def export_patient_dossier(n_clicks, active_patient_id, all_series):

    if not n_clicks or not active_patient_id:
        return dash.no_update

    patient_records = (all_series or {}).get(active_patient_id, [])
    if not patient_records:
        return dash.no_update

    export_data = []
    for record in patient_records:
        new_record = record.copy()

        new_record["image_id"] = image_to_base64(record["image_id"])
        export_data.append(new_record)

    dossier_complet = {"patient_id": active_patient_id, "records": export_data}

    filename = f"dossier_{active_patient_id}.json"
    return dcc.send_string(json.dumps(dossier_complet, indent=2), filename)


@app.callback(
    Output("longitudinal-series-store", "data", allow_duplicate=True),
    Output("import-feedback-message", "children"),
    Output("longitudinal-active-patient-store", "data", allow_duplicate=True),
    Output("patient-dropdown", "value", allow_duplicate=True),
    Output("import-patient-dossier", "contents", allow_duplicate=True),
    Output("import-patient-dossier", "filename", allow_duplicate=True),
    Output("import-patient-dossier", "last_modified", allow_duplicate=True),
    Input("import-patient-dossier", "contents"),
    State("longitudinal-series-store", "data"),
    State("language-store", "data"),
    State("import-patient-dossier", "filename"),
    State("import-patient-dossier", "last_modified"),
    prevent_initial_call=True,
)
def import_patient_dossier(contents, all_series, language, filename, last_modified):

    _ = get_translator(language)
    if not contents:

        return dash.no_update, "", dash.no_update, dash.no_update, None, None, None

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        imported_dossier = json.loads(decoded.decode("utf-8"))
        patient_id = imported_dossier.get("patient_id")
        records = imported_dossier.get("records")

        if not patient_id or not isinstance(records, list):
            raise ValueError("Format de fichier invalide.")

        all_series_updated = all_series or {}
        all_series_updated[patient_id] = records

        feedback = dbc.Alert(
            _("Dossier pour '%s' importé et chargé avec succès.")
            % patient_id.replace("_", " "),
            color="success",
            duration=5000,
        )

        return all_series_updated, feedback, patient_id, patient_id, None, None, None

    except Exception as e:
        feedback = dbc.Alert(
            _("Erreur lors de l'importation du fichier : %s") % str(e),
            color="danger",
            duration=8000,
        )

        return (
            dash.no_update,
            feedback,
            dash.no_update,
            dash.no_update,
            None,
            None,
            None,
        )


@app.callback(
    Output("pdf-patient-report-content", "children"),
    Output("trigger-patient-report-print-store", "data"),
    Input("export-patient-report-btn", "n_clicks"),
    State("longitudinal-active-patient-store", "data"),
    State("longitudinal-series-store", "data"),
    State("selected-exams-store", "data"),
    Input("language-store", "data"),
    prevent_initial_call=True,
)
def generate_and_trigger_patient_report(
    n_clicks, active_patient_id, all_series, selected_ids, language
):
    _ = get_translator(language)
    if not n_clicks or not active_patient_id:
        return dash.no_update, dash.no_update

    patient_info_parts = active_patient_id.split("_")
    patient_name = (
        f"{patient_info_parts[0]} {patient_info_parts[1]}"
        if len(patient_info_parts) > 1
        else patient_info_parts[0]
    )
    patient_dob = patient_info_parts[2] if len(patient_info_parts) > 2 else "N/A"

    header = html.Div(
        [
            html.H1(_("Rapport de Suivi FundusTracker")),
            html.Hr(),
            html.P(f"{_('Patient')}: {patient_name}"),
            html.P(f"{_('Date de naissance')}: {patient_dob}"),
            html.P(f"{_('Date du rapport')}: {datetime.now().strftime('%d/%m/%Y')}"),
        ],
        className="report-header",
    )

    all_patient_records = (all_series or {}).get(active_patient_id, [])
    records_to_display = [
        rec for rec in all_patient_records if rec["record_id"] in selected_ids
    ]
    if not records_to_display:
        return dash.no_update, dash.no_update

    records_to_display.sort(key=lambda r: r.get("exam_date", "1900-01-01"))

    exam_section_content = []
    for rec in records_to_display:
        try:
            img_b64 = image_to_base64(rec["image_id"])
            pil_img = Image.open(
                io_buffer.BytesIO(base64.b64decode(img_b64.split(",")[1]))
            )
            fig = generate_figure(
                pil_img, shapes=[shape_for_plotly(s) for s in rec.get("shapes", [])]
            )
            fig.update_layout(width=400, height=400, margin=dict(l=5, r=5, t=5, b=5))
            lesions = [
                s
                for s in rec.get("shapes", [])
                if s.get("customdata", "").lower() not in ["nerf optique", "optic nerve"]
            ]
            aire_totale = sum(
                calculate_area(parse_path_for_matching(s.get("path", "")))
                for s in lesions
            )
            exam_card = html.Div(
                [
                    html.H4(f"{_('Examen du')} {rec.get('exam_date', 'N/A')}"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.P(
                                        [
                                            html.Strong(_("Nombre de lésions: ")),
                                            len(lesions),
                                        ]
                                    ),
                                    html.P(
                                        [
                                            html.Strong(_("Aire totale des lésions: ")),
                                            f"{aire_totale:.0f} px²",
                                        ]
                                    ),
                                ],
                                width=6,
                                style={"paddingTop": "20px"},
                            ),
                        ]
                    ),
                ],
                className="report-exam-card",
            )
            exam_section_content.append(exam_card)
        except Exception as e:
            print(
                f"Erreur lors de la génération de la carte d'examen pour le rapport : {e}"
            )
            continue

    exams_section = html.Div(
        [html.H2(_("Détail des Examens Sélectionnés")), *exam_section_content],
        className="report-section",
    )

    fig_agg, fig_sankey, stats_content = update_longitudinal_analysis_view(
        selected_ids, active_patient_id, all_series, language
    )
    analysis_section = html.Div(
        [
            html.H2(_("Analyse Longitudinale")),
            dcc.Graph(figure=fig_agg),
            html.Hr(style={"margin": "20px 0"}),
            dcc.Graph(figure=fig_sankey),
            html.Div(stats_content, style={"marginTop": "20px"}),
        ],
        className="report-section",
    )

    final_report = html.Div([header, exams_section, analysis_section])

    return final_report, n_clicks


if pio.kaleido.scope is not None:
    pio.kaleido.scope.mathjax = None

    current_args = pio.kaleido.scope.chromium_args
    new_args = current_args + ("--disable-gpu", "--no-sandbox")
    pio.kaleido.scope.chromium_args = new_args


class PDF(FPDF):
    def __init__(
        self, *args, creator_line="FundusTracker © 2025", show_timestamp=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.creator_line = creator_line
        self.show_timestamp = show_timestamp

    def header(self):
        title = "Rapport FundusTracker"
        logo_path = "assets/logo.png"
        page_w = self.w
        usable_w = page_w - self.l_margin - self.r_margin

        logo_w = 24.0
        logo_h = 0.0
        if os.path.exists(logo_path):
            try:
                from PIL import Image as PILImage

                iw, ih = PILImage.open(logo_path).size
                logo_h = logo_w * (ih / iw) if iw else logo_w
            except Exception:
                logo_h = logo_w
        else:
            logo_w = 0.0
            logo_h = 0.0

        self.set_font("helvetica", "B", 16)
        text_h = self.font_size
        tw = self.get_string_width(title)
        gap = 2.0 if logo_w > 0 else 0.0

        top_y = self.t_margin
        bar_h = max(logo_h, text_h)
        group_w = (logo_w if logo_w > 0 else 0.0) + gap + tw
        start_x = self.l_margin + (usable_w - group_w) / 2.0

        if logo_w > 0:
            logo_y = top_y + (bar_h - logo_h) / 2.0
            self.image(logo_path, x=start_x, y=logo_y, w=logo_w)

        title_x = start_x + (logo_w + gap if logo_w > 0 else 0.0)
        title_y = top_y + (bar_h - text_h) / 2.0
        self.set_xy(title_x, title_y)
        self.cell(tw, text_h, title)

        y_line = top_y + bar_h + 2
        self.set_draw_color(210, 210, 210)
        self.set_line_width(0.6)
        self.line(self.l_margin, y_line, page_w - self.r_margin, y_line)
        self.set_y(y_line + 8)

    def footer(self):

        self.set_y(-18)
        y_line = self.get_y()
        self.set_draw_color(210, 210, 210)
        self.set_line_width(0.4)
        self.line(self.l_margin, y_line, self.w - self.r_margin, y_line)

        self.set_y(-14)
        self.set_font("helvetica", "", 8)

        self.set_x(self.l_margin)
        self.cell(self.epw / 2.0, 5, f"Page {self.page_no()}/{{nb}}", align="L")

        self.set_x(self.l_margin + self.epw / 2.0)
        self.cell(self.epw / 2.0, 5, self.creator_line, align="R")


def generate_pdf_report(active_patient_id, all_series, selected_ids, language):
    _ = get_translator(language)

    def s(text):
        return str(text).encode("latin-1", "replace").decode("latin-1")

    def hsep(pdf, pad=6, lw=0.5):
        pdf.ln(pad)
        x = pdf.l_margin
        y = pdf.get_y()
        pdf.set_draw_color(210, 210, 210)
        pdf.set_line_width(lw)
        pdf.line(x, y, x + pdf.epw, y)
        pdf.set_draw_color(0, 0, 0)
        pdf.ln(pad)

    def section_title(pdf, title):
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 9, s(title), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def kv(pdf, label, value, label_w=58, h=6, x_anchor=None):
        if x_anchor is not None:
            pdf.set_x(x_anchor)
        pdf.set_font("helvetica", "", 11)
        pdf.cell(label_w, h, s(f"{label} :"), align="R")
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, h, s(value))
        pdf.ln(h)

    def mcell(pdf, text, h=5, indent=0, border=0, align="L"):
        if getattr(pdf, "epw", 0) <= 10:
            pdf.set_left_margin(10)
            pdf.set_right_margin(10)
        pdf.set_x(pdf.l_margin + indent)
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(pdf.epw - indent, h, s(text), border=border, align=align)

    def dash_children_to_text(node):
        if isinstance(node, (list, tuple)):
            return " ".join(dash_children_to_text(n) for n in node)
        if hasattr(node, "children"):
            return dash_children_to_text(node.children)
        return "" if node is None else str(node)

    def ensure_space(pdf, needed_mm):

        remaining = (pdf.h - pdf.b_margin) - pdf.get_y()
        if remaining < needed_mm:
            pdf.add_page()

    parts = (active_patient_id or "").split("_")
    patient_name = f"{parts[0]} {parts[1]}" if len(parts) > 1 else parts[0]
    patient_dob = parts[2] if len(parts) > 2 else "N/A"

    all_patient_records = (all_series or {}).get(active_patient_id, [])
    records = [
        rec
        for rec in all_patient_records
        if rec.get("record_id") in (selected_ids or [])
    ]
    records.sort(key=lambda r: r.get("exam_date", "1900-01-01"))

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    section_title(pdf, _("Informations patient"))

    TOP_GAP = 4
    FRAME_PAD_TOP = 6
    FRAME_PAD_BOTTOM = 8

    pdf.ln(TOP_GAP)
    frame_top_y = pdf.get_y()
    pdf.ln(FRAME_PAD_TOP)

    content_start_y = pdf.get_y()
    kv(pdf, _("Patient"), patient_name)
    kv(pdf, _("Date de naissance"), patient_dob)
    kv(pdf, _("Date du rapport"), datetime.now().strftime("%d/%m/%Y"))
    content_end_y = pdf.get_y()

    box_h = (content_end_y - frame_top_y) + FRAME_PAD_BOTTOM
    pdf.set_draw_color(220, 220, 220)
    pdf.rect(pdf.l_margin, frame_top_y, pdf.epw, box_h)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_y(content_end_y + FRAME_PAD_BOTTOM)

    hsep(pdf)

    section_title(pdf, _("Examens réalisés"))
    IMG_W, GAP = 60, 8

    for rec in records:
        ensure_space(pdf, IMG_W + 20)

        pdf.set_font("helvetica", "B", 12)
        pdf.cell(
            0,
            7,
            s(f"{_('Examen du')} {rec.get('exam_date', 'N/A')}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        content_y = pdf.get_y()

        img_bottom_y = content_y
        try:
            img_b64 = image_to_base64(rec["image_id"])
            pil_img = Image.open(
                io_buffer.BytesIO(base64.b64decode(img_b64.split(",")[1]))
            )
            shapes_for_fig = rec.get("shapes", [])
            fig = generate_figure(
                pil_img, shapes=[shape_for_plotly(shape) for shape in shapes_for_fig]
            )
            fig.update_layout(
                width=500,
                height=500,
                margin=dict(l=0, r=0, t=0, b=0),
                title=None,
                annotations=[],
            )
            img_bytes = pio.to_image(fig, format="png", scale=2)
            pdf.image(
                io_buffer.BytesIO(img_bytes),
                x=pdf.l_margin,
                y=content_y,
                w=IMG_W,
                h=IMG_W,
            )
            img_bottom_y = content_y + IMG_W
        except Exception as e:
            pdf.set_xy(pdf.l_margin, content_y)
            pdf.set_font("helvetica", "I", 8)
            pdf.set_text_color(255, 0, 0)
            pdf.multi_cell(IMG_W, 5, s(f"Erreur Image:\n{e}"), border=1, align="C")
            pdf.set_text_color(0, 0, 0)
            img_bottom_y = pdf.get_y()

        right_x = pdf.l_margin + IMG_W + GAP

        shapes = rec.get("shapes", []) or []
        lesions = [
            shape
            for shape in shapes
            if shape.get("customdata", "").lower()
            not in [_("nerf optique").lower(), "optic nerve"]
        ]
        nb_lesions = len(lesions)
        aire_totale = sum(
            calculate_area(parse_path_for_matching(shape.get("path", "")))
            for shape in lesions
        )
        aire_moyenne = aire_totale / nb_lesions if nb_lesions > 0 else 0

        aire_papille = 0
        for shape in shapes:
            if shape.get("customdata", "").lower() in [
                _("nerf optique").lower(),
                "optic nerve",
            ]:
                aire_papille = calculate_area(
                    parse_path_for_matching(shape.get("path", ""))
                )
                break
        aire_normalisee = (aire_totale / aire_papille) if aire_papille > 0 else None

        table_data = [
            (_("Nb. lésions"), f"{nb_lesions}"),
            (_("Aire totale"), f"{aire_totale:.0f} px²"),
            (_("Aire moyenne"), f"{aire_moyenne:.0f} px²"),
        ]
        if aire_normalisee is not None:
            table_data.append((_("Ratio papillaire"), f"{aire_normalisee:.3f}"))

        LINE_H_TABLE = 6
        table_height = len(table_data) * LINE_H_TABLE
        metrics_y = content_y + max(0.0, (IMG_W - table_height) / 2.0)
        pdf.set_y(metrics_y)

        col1_w = 40
        for label, value in table_data:
            pdf.set_x(right_x)
            pdf.set_font("helvetica", "B", 9)
            pdf.cell(col1_w, LINE_H_TABLE, s(label), border=0)
            pdf.set_font("helvetica", "", 9)
            pdf.cell(0, LINE_H_TABLE, s(value), border=0, new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(max(img_bottom_y, pdf.get_y()) + 3)
        hsep(pdf, pad=3, lw=0.3)

    if records:
        pdf.add_page()

    section_title(pdf, _("Analyse longitudinale"))

    try:
        fig_agg, fig_sankey, stats_content = update_longitudinal_analysis_view(
            selected_ids, active_patient_id, all_series, language
        )

        pdf.set_font("helvetica", "B", 12)
        pdf.cell(
            0, 7, s(_("Évolution agrégée des lésions")), new_x="LMARGIN", new_y="NEXT"
        )
        img_bytes_agg = pio.to_image(
            fig_agg, format="png", width=900, height=420, scale=2
        )

        agg_drawn_h = pdf.epw * (420 / 900)
        ensure_space(pdf, agg_drawn_h + 8)
        pdf.image(io_buffer.BytesIO(img_bytes_agg), w=pdf.epw)
        pdf.ln(4)

        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 7, s(_("Flux évolutif des lésions")), new_x="LMARGIN", new_y="NEXT")
        img_bytes_sankey = pio.to_image(
            fig_sankey, format="png", width=900, height=520, scale=2
        )
        sankey_drawn_h = pdf.epw * (520 / 900)
        ensure_space(pdf, sankey_drawn_h + 8)
        pdf.image(io_buffer.BytesIO(img_bytes_sankey), w=pdf.epw)

        hsep(pdf, pad=8, lw=0.6)

        section_title(pdf, _("Analyses statistiques"))

        HEADER_COLORS = {
            "info": (23, 162, 184),
            "success": (40, 167, 69),
            "primary": (0, 123, 255),
        }

        if isinstance(stats_content, list) and stats_content:
            for card in stats_content:

                ensure_space(pdf, 40)

                try:

                    card_color_key = card.color
                    header_text = dash_children_to_text(
                        getattr(card, "children", [{}])[0].children
                    )
                    body_children = getattr(card, "children", [{}, {}])[1].children

                    start_y = pdf.get_y()
                    pdf.set_font("helvetica", "B", 11)
                    header_fill_color = HEADER_COLORS.get(
                        card_color_key, (220, 220, 220)
                    )
                    pdf.set_fill_color(*header_fill_color)
                    pdf.set_text_color(255, 255, 255)
                    pdf.cell(
                        0,
                        8,
                        s(header_text),
                        new_x="LMARGIN",
                        new_y="NEXT",
                        fill=True,
                        border=1,
                    )

                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("helvetica", "", 9)

                    content_to_write = []
                    for item in body_children:

                        if "fst-italic" not in getattr(item, "className", ""):
                            text = dash_children_to_text(item)
                            is_strong = any(
                                "bold" in str(getattr(c, "style", {}))
                                for c in getattr(item, "children", [])
                            )
                            content_to_write.append({"text": text, "strong": is_strong})

                    footer_item = next(
                        (
                            item
                            for item in body_children
                            if "fst-italic" in getattr(item, "className", "")
                        ),
                        None,
                    )
                    footer_text = (
                        dash_children_to_text(footer_item) if footer_item else ""
                    )

                    for content in content_to_write:
                        if content["strong"]:
                            pdf.set_font("helvetica", "B", 9)
                        else:
                            pdf.set_font("helvetica", "", 9)
                        pdf.multi_cell(
                            0, 5, s(content["text"]), new_x="LMARGIN", new_y="NEXT"
                        )

                    if footer_text:
                        pdf.set_font("helvetica", "I", 8)
                        pdf.set_text_color(108, 117, 125)
                        pdf.multi_cell(
                            0, 5, s(footer_text), new_x="LMARGIN", new_y="NEXT"
                        )
                        pdf.set_text_color(0, 0, 0)

                    end_y = pdf.get_y()
                    pdf.set_draw_color(220, 220, 220)
                    pdf.rect(pdf.l_margin, start_y, pdf.epw, end_y - start_y)
                    pdf.set_draw_color(0, 0, 0)
                    pdf.ln(5)

                except Exception as e:
                    pdf.set_font("helvetica", "I", 8)
                    pdf.set_text_color(255, 0, 0)
                    pdf.multi_cell(0, 4, s(f"Erreur rendu carte stat: {e}"), border=1)
                    pdf.set_text_color(0, 0, 0)

        else:
            pdf.set_font("helvetica", "I", 10)
            pdf.multi_cell(0, 5, s(dash_children_to_text(stats_content)))

    except Exception as e:
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(255, 0, 0)
        mcell(
            pdf,
            f"{_('Erreur lors de la génération de l analyse :')}\n{e}",
            h=5,
            indent=0,
            border=1,
        )
        pdf.set_text_color(0, 0, 0)

    return pdf.output()


@app.callback(
    Output("download-pdf-report", "data"),
    Input("export-patient-report-btn", "n_clicks"),
    State("longitudinal-active-patient-store", "data"),
    State("longitudinal-series-store", "data"),
    State("selected-exams-store", "data"),
    Input("language-store", "data"),
    prevent_initial_call=True,
)
def download_pdf_report_callback(
    n_clicks, active_patient_id, all_series, selected_ids, language
):
    if not n_clicks or not active_patient_id or not selected_ids:
        return dash.no_update

    try:
        pdf_bytes = generate_pdf_report(
            active_patient_id, all_series, selected_ids, language
        )

        pdf_bytes = bytes(pdf_bytes)

        patient_name = active_patient_id.split("_")[0]
        filename = f"Rapport_FundusTracker_{patient_name}_{datetime.now().strftime('%Y-%m-%d')}.pdf"

        return dcc.send_bytes(lambda buf: buf.write(pdf_bytes), filename)

    except Exception as e:
        print(f"Erreur majeure lors de la génération du PDF : {e}")
        return dash.no_update


@app.callback(
    Output("export-patient-report-btn", "disabled"),
    Input("longitudinal-active-patient-store", "data"),
    Input("selected-exams-store", "data"),
)
def toggle_export_pdf_button(active_patient_id, selected_exam_ids):
    if not active_patient_id or not selected_exam_ids:
        return True
    return False


@app.callback(
    Output("save-to-patient-btn-manuelle", "disabled"),
    Input("longitudinal-active-patient-store", "data"),
    Input("annotation-image-store", "data"),
    Input("annotation-date-picker-manuelle", "date"),
    Input("stored-shapes", "data"),
)
def toggle_save_to_patient_button(active_patient_id, image_id, exam_date, shapes):

    if active_patient_id and image_id and exam_date and shapes:
        return False
    return True


@app.callback(
    Output("crop-modal", "is_open"),
    Output("crop-result-display", "children"),
    Output("crop-result-temp-store", "data"),
    Output("apply-cropped-img-btn", "disabled"),
    Output("download-cropped-img-btn", "disabled"),
    Input("open-crop-modal-btn", "n_clicks"),
    Input("close-crop-modal-btn", "n_clicks"),
    Input("run-crop-sift-btn", "n_clicks"),
    State("crop-ref-dropdown", "value"),
    State("crop-ref-upload", "contents"),
    State("file-dropdown", "value"),
    State("uploaded-image-store", "data"),
    State("crop-modal", "is_open"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def manage_crop_workflow(
    n_open, n_close, n_run, ref_drop, ref_up, main_drop, main_up, is_open, language
):
    ctx_id = ctx.triggered_id
    _ = get_translator(language)

    if ctx_id == "open-crop-modal-btn":
        return True, None, None, True, True

    if ctx_id == "close-crop-modal-btn":
        return False, dash.no_update, None, True, True

    if ctx_id == "run-crop-sift-btn":

        target_b64 = image_to_base64(main_up if main_up else main_drop)

        ref_b64 = None
        if ref_up:
            ref_b64 = ref_up
        elif ref_drop:
            ref_b64 = image_to_base64(ref_drop)

        if not target_b64:
            return (
                True,
                dbc.Alert(
                    _("Erreur : Aucune image principale chargée."), color="danger"
                ),
                None,
                True,
                True,
            )
        if not ref_b64:
            return (
                True,
                dbc.Alert(
                    _("Erreur : Veuillez choisir une image de référence."),
                    color="warning",
                ),
                None,
                True,
                True,
            )

        aligned_b64, msg = align_images_sift_simple(ref_b64, target_b64)

        if aligned_b64:
            preview = html.Div(
                [
                    html.H6(_("Résultat visuel :"), className="mt-2"),
                    html.Img(
                        src=aligned_b64,
                        style={
                            "maxWidth": "100%",
                            "maxHeight": "350px",
                            "border": "2px solid #28a745",
                        },
                    ),
                    dbc.Alert(
                        _("Alignement réussi !"), color="success", className="mt-2 py-2"
                    ),
                ]
            )
            return True, preview, aligned_b64, False, False
        else:
            return (
                True,
                dbc.Alert(f"Échec SIFT : {msg}", color="danger"),
                None,
                True,
                True,
            )

    return is_open, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("uploaded-image-store", "data", allow_duplicate=True),
    Output("file-dropdown", "value", allow_duplicate=True),
    Output("crop-modal", "is_open", allow_duplicate=True),
    Input("apply-cropped-img-btn", "n_clicks"),
    State("crop-result-temp-store", "data"),
    prevent_initial_call=True,
)
def apply_cropped_image(n_clicks, aligned_b64):
    if not n_clicks or not aligned_b64:
        return dash.no_update, dash.no_update, dash.no_update

    return aligned_b64, None, False


@app.callback(
    Output("download-cropped-image", "data"),
    Input("download-cropped-img-btn", "n_clicks"),
    State("crop-result-temp-store", "data"),
    prevent_initial_call=True,
)
def download_cropped_image_file(n_clicks, aligned_b64):
    if not n_clicks or not aligned_b64:
        return dash.no_update

    import datetime

    content_string = aligned_b64.split(",")[1]

    image_bytes = base64.b64decode(content_string)

    filename = f"recadrage_sift_{datetime.datetime.now().strftime('%H%M%S')}.png"

    return dcc.send_bytes(lambda f: f.write(image_bytes), filename)


@app.callback(
    Output("download-mask-image", "data"),
    Input("download-mask-button", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    State("uploaded-image-store", "data"),
    State("language-store", "data"),
    prevent_initial_call=True,
)
def export_segmentation_mask(n_clicks, stored_shapes, file_val, uploaded_image, language):
    if not n_clicks:
        return dash.no_update
    image_id = file_val or uploaded_image
    if not image_id:
        return dash.no_update
    try:
        pil_img = load_image_any(image_id)
        width, height = pil_img.size
        image_shape = (height, width)
    except Exception as e:
        print(f"Erreur chargement image pour masque: {e}")
        return dash.no_update
    shapes = stored_shapes if stored_shapes else []
    mask_array = generate_mask_from_shapes(image_shape, shapes, language)
    is_success, buffer = cv2.imencode(".png", mask_array)
    if not is_success:
        return dash.no_update

    if isinstance(image_id, str) and not image_id.startswith("data:"):
        base_name = image_id.split("/")[-1].rsplit(".", 1)[0]
    else:
        base_name = "local_image"

    filename = f"{base_name}_mask.png"

    return dcc.send_bytes(buffer.tobytes(), filename)


@app.callback(
    Output("download-all-zip", "data"),
    Input("export-all-button", "n_clicks"),
    State("stored-shapes", "data"),
    State("file-dropdown", "value"),
    State("uploaded-image-store", "data"),
    State("language-store", "data"),
    State("axial-length-input", "value"),
    prevent_initial_call=True,
)
def export_all_zip(n_clicks, stored_shapes, file_val, uploaded_image, language, axial_length):
    if not n_clicks or not stored_shapes:
        return dash.no_update
    df, filename_xlsx = process_image_analysis_data(
        stored_shapes, file_val, uploaded_image, language, axial_length
    )
    base_name = filename_xlsx.replace(".xlsx", "")
    json_str = json.dumps(stored_shapes, indent=2)
    mask_bytes = None
    try:
        image_id = file_val or uploaded_image
        if image_id:
            pil_img = load_image_any(image_id)
            width, height = pil_img.size
            mask_arr = generate_mask_from_shapes((height, width), stored_shapes, language)
            is_success, buffer = cv2.imencode(".png", mask_arr)
            if is_success:
                mask_bytes = buffer.tobytes()
    except Exception as e:
        print(f"Erreur lors de la génération du masque pour le ZIP : {e}")
    zip_buffer = io_buffer.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        with io_buffer.BytesIO() as excel_buffer:
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Analyse")
            zf.writestr(f"{base_name}_analyse.xlsx", excel_buffer.getvalue())
        zf.writestr(f"{base_name}_annotations.json", json_str)
        if mask_bytes:
            zf.writestr(f"{base_name}_mask.png", mask_bytes)
    return dcc.send_bytes(zip_buffer.getvalue(), f"{base_name}_complet.zip")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
