"""Structured3D: floorplans drawn from the annotations, cameras and ground-truth geometry.

    python -m scenealigner.structured3d <scene directory> <image>   # draws the floorplan of a scene
"""
from __future__ import annotations

import argparse
import io
import json
import os
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import cv2
import matplotlib
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union

from .preprocess import resize_plan
from .semantics import align_mask_to_grid, normalize_label

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from .preprocess import ReconMode
    from .semantics import ClipTextEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PlanParams: TypeAlias = tuple[tuple[float, float], float, tuple[int, int]]

MM_PER_PIXEL = 10.0  # of the drawn floorplan, before it is resized
CANVAS = 80000  # mm: the drawing covers 40 m around the origin of a scene
TRAIN_SCENES, TEST_SCENES = range(0, 3000), range(3250, 3500)
OFF_CANVAS_SCENES = (2119, 2208, 2401)  # train scenes whose floorplans lie (partly) outside the drawing


def convert_lines_to_vertices(lines: Sequence[Sequence[int]] | NDArray[np.integer[Any]]) -> list[list[int]]:
    """Chains line segments (pairs of junctions) into polygons, as the Structured3D toolbox does. Polygons of
    fewer than three junctions are dropped."""
    polygons, polygon = [], None

    def close(polygon: list[int]) -> None:
        polygon = [v for i, v in enumerate(polygon) if i == 0 or v != polygon[i - 1]]
        if len(set(polygon)) >= 3:
            polygons.append(polygon)

    lines = np.array(lines, dtype=int)
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, axis=0)
            continue
        line_ids, columns = np.where(lines == polygon[-1])
        if len(line_ids) == 0:
            close(polygon)
            polygon = None
            continue
        vertex = lines[line_ids[0], 1 - columns[0]]
        lines = np.delete(lines, line_ids[0], axis=0)
        if vertex in polygon:
            close(polygon)
            polygon = None
        else:
            polygon.append(vertex)
    if polygon is not None:
        close(polygon)
    return polygons


def load_polygons(annos: dict[str, Any]) -> list[tuple[list[int], str]]:
    """Outlines of the rooms, doors and windows (their floor planes) and of the outer wall, as junction indices."""
    def junction_pairs(line_ids: Sequence[int] | NDArray[np.integer[Any]]) -> list[list[int]]:
        pairs = [np.where(np.array(annos["lineJunctionMatrix"][line_id]))[0].tolist() for line_id in line_ids]
        return [pair for pair in pairs if len(pair) == 2]

    def lines_of(plane_id: int) -> list[int]:
        return np.where(np.array(annos["planeLineMatrix"][plane_id]))[0].tolist()

    polygons = []
    for semantic in annos["semantics"]:
        for plane_id in semantic["planeID"]:
            if annos["planes"][plane_id]["type"] == "floor":
                polygon = convert_lines_to_vertices(junction_pairs(lines_of(plane_id)))
                if polygon:
                    polygons.append((polygon[0], semantic["type"]))

    # Outer wall: its lines on the floor, without those of doors and windows
    holes = [line_id for semantic in annos["semantics"] if semantic["type"] in ("window", "door")
             for plane_id in semantic["planeID"] for line_id in lines_of(plane_id)]
    on_floor = np.where(np.isclose(np.array([junction["coordinate"][2] for junction in annos["junctions"]]), 0))[0]
    outer_lines = []
    for semantic in annos["semantics"]:
        if semantic["type"] == "outwall":
            for plane_id in semantic["planeID"]:
                outer_lines += [pair for pair in junction_pairs(np.setdiff1d(lines_of(plane_id), np.unique(holes)))
                                if pair[0] in on_floor and pair[1] in on_floor]
    outer_polygon = convert_lines_to_vertices(outer_lines)
    if outer_polygon:
        polygons.append((outer_polygon[0], "outwall"))
    return polygons


def render_floorplan(annos: dict[str, Any], dpi: int = 100) -> tuple[Image.Image, tuple[float, float]]:
    """Draws the floorplan of a scene at 1 cm per pixel, cropped to its content: walls black, rooms white. Doors
    and windows are not drawn. Returns the image and the position of its top-left corner, in pixels from the
    origin of the scene."""
    junctions = np.array([junction["coordinate"][:2] for junction in annos["junctions"]])
    rooms: Any
    rooms, outer_wall = [], None
    for indices, polygon_type in load_polygons(annos):
        polygon = Polygon(junctions[np.array(indices, dtype=int)])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        if polygon_type == "outwall":
            outer_wall = polygon
        elif polygon_type not in ("door", "window"):
            rooms += list(polygon.geoms) if isinstance(polygon, MultiPolygon) else [polygon]
    rooms = unary_union(rooms)  # no seams between adjacent rooms
    if isinstance(rooms, GeometryCollection):
        rooms = unary_union([g for g in rooms.geoms if isinstance(g, (Polygon, MultiPolygon))])

    def parts(shape: Any) -> list[Polygon]:
        if shape is None or shape.is_empty:
            return []
        return [g for g in getattr(shape, "geoms", [shape]) if isinstance(g, Polygon) and not g.is_empty]

    size = CANVAS / MM_PER_PIXEL / dpi
    _, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    ax.set_xlim(-CANVAS / 2, CANVAS / 2)
    ax.set_ylim(-CANVAS / 2, CANVAS / 2)
    ax.set_aspect("equal")
    line_style = {"color": "black", "linewidth": 1, "solid_capstyle": "round", "solid_joinstyle": "round", "antialiased": True}
    for polygon in parts(outer_wall):
        ax.fill(*polygon.exterior.xy, color="black", zorder=1)
    for polygon in parts(rooms):
        ax.fill(*polygon.exterior.xy, color="white", edgecolor="none", zorder=2)
        ax.plot(*polygon.exterior.xy, **line_style, zorder=3)
    for polygon in parts(outer_wall):
        ax.plot(*polygon.exterior.xy, **line_style, zorder=4)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buffer = io.BytesIO()
    plt.savefig(buffer, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close()
    img = Image.open(buffer).convert("RGB")
    box: Any = ImageOps.invert(img.convert("L")).getbbox()
    return img.crop(box), (box[0] - img.width / 2, box[1] - img.height / 2)


def load_floorplan(scene_dir: str, plan_size: int = 512) -> tuple[Image.Image, PlanParams]:
    """Floorplan of a scene, resized as the floorplans of C3, and the parameters of world_to_plan."""
    with open(os.path.join(scene_dir, "annotation_3d.json")) as f:
        annos = json.load(f)
    plan_img, origin = render_floorplan(annos)
    plan_img, (scale, off_x, off_y) = resize_plan(plan_img, plan_size)
    return plan_img, (origin, scale, (off_x, off_y))


def _poly_parts(geom: BaseGeometry | None) -> list[Polygon]:
    """Flatten a Shapely geometry into a list of non-empty Polygon objects."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [part for sub in geom.geoms for part in _poly_parts(sub)]
    return []


def _line_parts(geom: BaseGeometry | None) -> list[LineString]:
    """Flatten a Shapely geometry into a list of standalone LineString segments."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, (MultiLineString, GeometryCollection)):
        return [part for sub in geom.geoms for part in _line_parts(sub)]
    return []


def _to_pixel_ring(coords: Sequence[Sequence[float]], plan_params: PlanParams) -> NDArray[np.int32]:
    pts = world_to_plan(np.asarray(coords, dtype=np.float64)[..., :2], plan_params)
    return np.round(pts).astype(np.int32)


def _fill_polygon(
    canvas: NDArray[np.uint8], geom: BaseGeometry | None, label_id: int, plan_params: PlanParams
) -> None:
    """Rasterize polygon areas onto an integer index canvas without anti-aliasing."""
    for poly in _poly_parts(geom):
        ext = _to_pixel_ring(list(poly.exterior.coords), plan_params)
        if len(ext) < 3:
            continue
        rings = [ext]
        for interior in poly.interiors:
            hole = _to_pixel_ring(list(interior.coords), plan_params)
            if len(hole) >= 3:
                rings.append(hole)
        cv2.fillPoly(canvas, rings, color=int(label_id), lineType=cv2.LINE_8)


def _stroke_polygon(
    canvas: NDArray[np.uint8],
    geom: BaseGeometry | None,
    label_id: int,
    plan_params: PlanParams,
    thickness: int = 3,
) -> None:
    """Stroke thin polygon outlines or collinear LineStrings onto an integer index canvas."""
    for poly in _poly_parts(geom):
        rings: list[NDArray[np.int32]] = []
        for ring_coords in [poly.exterior.coords, *(interior.coords for interior in poly.interiors)]:
            pts = _to_pixel_ring(list(ring_coords), plan_params)
            if len(pts) >= 2:
                rings.append(pts)
        if rings:
            cv2.polylines(canvas, rings, isClosed=True, color=int(label_id), thickness=int(thickness), lineType=cv2.LINE_8)

    for line in _line_parts(geom):
        pts = _to_pixel_ring(list(line.coords), plan_params)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], isClosed=False, color=int(label_id), thickness=int(thickness), lineType=cv2.LINE_8)


def _extract_elevated_openings(annos: dict[str, Any]) -> list[tuple[BaseGeometry, str]]:
    """Project XY footprints for door/window annotations lacking a valid 'floor' polygon in Structured3D."""
    junctions = annos["junctions"]
    plane_lines = np.array(annos["planeLineMatrix"])
    line_junctions = np.array(annos["lineJunctionMatrix"])
    openings: list[tuple[BaseGeometry, str]] = []
    for semantic in annos["semantics"]:
        sem_type = str(semantic["type"])
        if sem_type not in ("door", "window"):
            continue
        def _plane_has_floor_poly(pid: int) -> bool:
            if annos["planes"][pid]["type"] != "floor":
                return False
            pairs = [np.where(line_junctions[lid])[0].tolist() for lid in np.where(plane_lines[pid])[0]]
            valid_pairs = [pair for pair in pairs if len(pair) == 2]
            return bool(valid_pairs and convert_lines_to_vertices(valid_pairs))

        if any(_plane_has_floor_poly(pid) for pid in semantic["planeID"]):
            continue

        plane_geoms: list[BaseGeometry] = []
        for pid in semantic["planeID"]:
            plane_jids: set[int] = set()
            for lid in np.where(plane_lines[pid])[0]:
                plane_jids.update(np.where(line_junctions[lid])[0].tolist())
            if len(plane_jids) < 2:
                continue
            pts_xy = [junctions[jid]["coordinate"][:2] for jid in plane_jids]
            hull = MultiPoint(pts_xy).convex_hull
            if not hull.is_empty:
                plane_geoms.append(hull)
        if plane_geoms:
            combined = unary_union(plane_geoms)
            if not combined.is_empty:
                openings.append((combined, sem_type))
    return openings


def render_semantic_floorplan(
    annos: dict[str, Any], plan_params: PlanParams, plan_size: int = 512, include_rooms: bool = False
) -> tuple[NDArray[np.uint8], list[str]]:
    """Rasterize floorplan semantics aligned 1-to-1 with load_floorplan.

    Layering order:
      Z=1: outer wall footprint filled with 'wall'
      Z=2: room floor polygons carve interiors (labeled if include_rooms=True, else 0),
           followed by 1-px wall boundary stroke around unioned rooms & outer wall
      Z=3: doors and windows filled and stroked on top of walls

    Returns:
      sem_idx: uint8 array (plan_size, plan_size), 0 = unlabeled, 1..N = index into labels
      labels: list[str] of normalized concept labels for indices 1..N
    """
    canvas = np.zeros((plan_size, plan_size), dtype=np.uint8)
    junctions = annos["junctions"]
    polygons = load_polygons(annos)

    labels: list[str] = []
    label_to_id: dict[str, int] = {}

    def get_id(clean_label: str) -> int:
        if not clean_label:
            return 0
        if clean_label not in label_to_id:
            label_to_id[clean_label] = len(labels) + 1
            labels.append(clean_label)
        return label_to_id[clean_label]

    def to_geom(poly_indices: Sequence[int]) -> BaseGeometry | None:
        pts = [junctions[idx]["coordinate"][:2] for idx in poly_indices]
        if len(pts) >= 3:
            geom = Polygon(pts)
            return geom if geom.is_valid else geom.buffer(0)
        if len(pts) == 2:
            return LineString(pts)
        return None

    # Z=1: outer wall fill establishes the structural envelope
    outwall_geoms: list[BaseGeometry] = []
    for poly_indices, raw_type in polygons:
        if raw_type == "outwall":
            geom = to_geom(poly_indices)
            if geom is not None:
                outwall_geoms.append(geom)
                _fill_polygon(canvas, geom, get_id(normalize_label("outwall")), plan_params)

    # Z=2: room polygons carve interiors out of the solid outer wall fill
    room_geoms: list[BaseGeometry] = []
    for poly_indices, raw_type in polygons:
        if raw_type in ("outwall", "door", "window"):
            continue
        geom = to_geom(poly_indices)
        if geom is None:
            continue
        if isinstance(geom, (Polygon, MultiPolygon)):
            room_geoms.append(geom)
        clean = normalize_label(raw_type) if include_rooms else ""
        _fill_polygon(canvas, geom, get_id(clean), plan_params)

    # Restore 1-px wall strokes along unioned room boundaries & outer wall (without open-plan seams)
    if outwall_geoms or room_geoms:
        wall_id = get_id(normalize_label("outwall"))
        rooms_union = unary_union(room_geoms) if room_geoms else None
        if isinstance(rooms_union, GeometryCollection):
            rooms_union = unary_union([g for g in rooms_union.geoms if isinstance(g, (Polygon, MultiPolygon))])
        if rooms_union is not None and not rooms_union.is_empty:
            _stroke_polygon(canvas, rooms_union, wall_id, plan_params, thickness=1)
        for geom in outwall_geoms:
            _stroke_polygon(canvas, geom, wall_id, plan_params, thickness=1)

    # Z=3: doors and windows (both floor-plane polygons and projected elevated openings)
    _, scale, _ = plan_params
    stroke_px = max(2, min(8, int(round(120.0 / MM_PER_PIXEL * scale))))
    opening_geoms: list[tuple[BaseGeometry, str]] = []
    for poly_indices, raw_type in polygons:
        if raw_type in ("door", "window"):
            geom = to_geom(poly_indices)
            if geom is not None:
                opening_geoms.append((geom, raw_type))
    opening_geoms.extend(_extract_elevated_openings(annos))

    for geom, raw_type in opening_geoms:
        clean = normalize_label(raw_type)
        if not clean:
            continue
        open_id = get_id(clean)
        _fill_polygon(canvas, geom, open_id, plan_params)
        _stroke_polygon(canvas, geom, open_id, plan_params, thickness=stroke_px)

    return canvas, labels


def load_semantic_floorplan(
    scene_dir: str,
    text_encoder: ClipTextEncoder | None = None,
    plan_size: int = 512,
    include_rooms: bool = False,
    plan_params: PlanParams | None = None,
    annos: dict[str, Any] | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float16], list[str]]:
    """Load a scene's aligned semantic index map, CLIP palette, and label list."""
    from .semantics import ClipTextEncoder

    if annos is None:
        with open(os.path.join(scene_dir, "annotation_3d.json")) as f:
            scene_annos: dict[str, Any] = json.load(f)
    else:
        scene_annos = annos
    if plan_params is None:
        plan_img, origin = render_floorplan(scene_annos)
        _, (scale, off_x, off_y) = resize_plan(plan_img, plan_size)
        plan_params = (origin, scale, (off_x, off_y))

    sem_idx, labels = render_semantic_floorplan(
        scene_annos, plan_params, plan_size=plan_size, include_rooms=include_rooms
    )
    if text_encoder is None:
        text_encoder = ClipTextEncoder()
    palette = text_encoder.encode_palette(labels)
    return sem_idx, palette, labels


_FIXED_COLORS: dict[str, tuple[int, int, int]] = {
    "wall": (40, 40, 40),
    "door": (214, 96, 77),
    "window": (67, 147, 195),
    "living room": (253, 219, 199),
    "bedroom": (209, 229, 240),
    "kitchen": (254, 224, 139),
    "bathroom": (171, 217, 233),
    "corridor": (215, 215, 215),
    "balcony": (166, 217, 106),
}


def colorize_semantic_floorplan(sem_idx: NDArray[np.integer[Any]], labels: Sequence[str]) -> Image.Image:
    """Render an RGB visualization PIL.Image of a semantic floorplan index map."""
    cmap = plt.get_cmap("tab20")
    lut = np.full((len(labels) + 1, 3), 255, dtype=np.uint8)
    for idx, label in enumerate(labels, start=1):
        if label in _FIXED_COLORS:
            lut[idx] = _FIXED_COLORS[label]
        else:
            rgb = cmap((sum(ord(ch) for ch in label) % 20) / 19.0)[:3]
            lut[idx] = np.round(np.array(rgb) * 255).astype(np.uint8)
    return Image.fromarray(lut[np.asarray(sem_idx, dtype=np.int64)])


def world_to_plan(points: NDArray[np.floating[Any]], plan_params: PlanParams) -> NDArray[np.floating[Any]]:
    """World coordinates (..., 2 or 3) in mm -> floorplan pixels (..., 2). The rows of the floorplan run along -y."""
    (delta_x, delta_y), scale, (off_x, off_y) = plan_params
    return np.stack([(points[..., 0] / MM_PER_PIXEL - delta_x) * scale + off_x,
                     (-points[..., 1] / MM_PER_PIXEL - delta_y) * scale + off_y], axis=-1)


def list_views(scene_dir: str) -> list[str]:
    """Directories of the furnished perspective views of a scene, in order."""
    rendering_dir = os.path.join(scene_dir, "2D_rendering")
    views = []
    for room_id in sorted(os.listdir(rendering_dir)):
        room_dir = os.path.join(rendering_dir, room_id, "perspective", "full")
        if os.path.isdir(room_dir):
            views += [os.path.join(room_dir, position_id) for position_id in sorted(os.listdir(room_dir))]
    return [view for view in views if all(os.path.exists(os.path.join(view, name)) for name in ("camera_pose.txt", "depth.png", "rgb_rawlight.png"))]


def load_camera(
    view_dir: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], np.float64, np.float64]:
    """Camera of a view: position (mm), viewing direction, up direction and half fields of view (rad)."""
    with open(os.path.join(view_dir, "camera_pose.txt")) as f:
        values = np.array(f.read().split(), dtype=np.float64)
    return values[0:3], values[3:6], values[6:9], values[9], values[10]


def load_world_points(view_dir: str, size: int = 518) -> NDArray[np.float64]:
    """3D point (mm, world frame) of every pixel of a view as loaded by pi3 (resized and padded to 518 x 518),
    from its depth map and camera. NaN where there is no depth."""
    eye, view, up, xfov, yfov = load_camera(view_dir)
    depth = cv2.imread(os.path.join(view_dir, "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float64)  # type: ignore[union-attr]
    height, width = depth.shape
    new_height = int(round(height * (size / width) / 14) * 14)
    pad_top = (size - new_height) // 2
    depth = cv2.resize(depth, (size, new_height), interpolation=cv2.INTER_NEAREST)
    depth = cv2.copyMakeBorder(depth, pad_top, size - new_height - pad_top, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Camera axes (right, down, forward) and the intrinsics of the resized, padded view
    view, up = view / (np.linalg.norm(view) + 1e-10), up / (np.linalg.norm(up) + 1e-10)
    right = np.cross(view, up)
    R_world2cam = np.stack([right / (np.linalg.norm(right) + 1e-10), -up, view], axis=0)
    fx, fy = (width / 2.0) / np.tan(xfov) * (size / width), (height / 2.0) / np.tan(yfov) * (new_height / height)
    cx, cy = width / 2.0 * (size / width), height / 2.0 * (new_height / height) + pad_top
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    points = np.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], axis=-1) @ R_world2cam + eye
    points[depth <= 0] = np.nan
    return points


NYU40_LABELS: dict[int, str] = {
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refrigerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "shower curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "nightstand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
}
STRUCTURAL_PHOTO_LABELS: frozenset[str] = frozenset({"wall", "door", "window"})
DEFAULT_EXCLUDED_PHOTO_LABELS: frozenset[str] = frozenset({"floor", "ceiling"})


def _read_raw_label_png(sem_path: str) -> NDArray[np.integer[Any]]:
    """Read raw NYUv2 class indices from a PNG without palette-to-RGB color expansion."""
    with Image.open(sem_path) as img:
        raw = np.asarray(img)
    if raw.ndim == 3:
        if raw.shape[2] >= 3 and np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 0], raw[..., 2]):
            raw = raw[..., 0]
        else:
            raise ValueError(f"Expected single-channel or equal-RGB label PNG at {sem_path}, got shape={raw.shape}")
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"Expected 2D integer label PNG at {sem_path}, got shape={raw.shape} dtype={raw.dtype}")
    return raw


def load_view_gt_semantics(
    view_dir: str,
    text_encoder: ClipTextEncoder | None = None,
    *,
    allowed_labels: Collection[str] | None = None,
    excluded_labels: Collection[str] = DEFAULT_EXCLUDED_PHOTO_LABELS,
    target_size: int | None = 518,
    mode: ReconMode = "pad",
) -> tuple[NDArray[np.uint8], NDArray[np.float16], list[str]]:
    """Load Structured3D view semantic.png (NYUv2 40 classes) into a compact CLIP palette and aligned index map."""
    sem_path = os.path.join(view_dir, "semantic.png")
    raw_ids = _read_raw_label_png(sem_path)
    aligned_raw = align_mask_to_grid(raw_ids, target_size=target_size, mode=mode) if target_size is not None else raw_ids

    allowed_set = (
        None
        if allowed_labels is None
        else frozenset(clean for lbl in allowed_labels if (clean := normalize_label(lbl)))
    )
    excluded_set = frozenset(clean for lbl in excluded_labels if (clean := normalize_label(lbl)))

    present_ids = sorted(int(cid) for cid in np.unique(aligned_raw) if int(cid) in NYU40_LABELS)
    labels: list[str] = []
    lut_size = max(NYU40_LABELS) + 1
    lut = np.zeros(lut_size, dtype=np.uint8)
    label_to_compact: dict[str, int] = {}
    for cid in present_ids:
        clean = normalize_label(NYU40_LABELS[cid])
        if not clean or clean in excluded_set:
            continue
        if allowed_set is not None and clean not in allowed_set:
            continue
        if clean not in label_to_compact:
            label_to_compact[clean] = len(labels) + 1
            labels.append(clean)
        lut[cid] = np.uint8(label_to_compact[clean])

    in_range = (aligned_raw >= 0) & (aligned_raw < lut_size)
    sem_idx = np.where(in_range, lut[np.clip(aligned_raw, 0, lut_size - 1)], np.uint8(0)).astype(np.uint8)

    if text_encoder is None:
        from .semantics import ClipTextEncoder

        text_encoder = ClipTextEncoder()
    palette = text_encoder.encode_palette(labels)
    return sem_idx, palette, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a Structured3D geometric or semantic floorplan.")
    parser.add_argument("scene_dir", help="path to a Structured3D scene_* directory")
    parser.add_argument("output", help="output image path")
    parser.add_argument("--semantic", action="store_true", help="render colorized semantic floorplan")
    parser.add_argument("--include_rooms", action="store_true", help="include room-type labels in semantic mode")
    parser.add_argument("--plan_size", type=int, default=512, help="output square size in semantic mode")
    args = parser.parse_args()

    if args.semantic:
        with open(os.path.join(args.scene_dir, "annotation_3d.json")) as f:
            annos = json.load(f)
        _, plan_params = load_floorplan(args.scene_dir, plan_size=args.plan_size)
        sem_idx, labels = render_semantic_floorplan(
            annos, plan_params, plan_size=args.plan_size, include_rooms=args.include_rooms
        )
        colorize_semantic_floorplan(sem_idx, labels).save(args.output)
    else:
        with open(os.path.join(args.scene_dir, "annotation_3d.json")) as f:
            render_floorplan(json.load(f))[0].save(args.output)
