"""Tests for open-vocabulary Structured3D floorplan rasterization and plan_params reuse."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scenealigner import structured3d as s3d


def _build_scene(
    outer_box: tuple[float, float, float, float],
    room_specs: list[tuple[str, tuple[float, float, float, float]]],
    extra_openings: list[tuple[str, list[tuple[str, list[tuple[float, float, float]]]]]] | None = None,
) -> dict[str, Any]:
    """Build a minimal valid Structured3D annotation_3d dict with rectangular rooms and optional openings."""
    junctions: list[dict[str, list[float]]] = []
    planes: list[dict[str, str]] = []
    semantics: list[dict[str, Any]] = []
    plane_line_edges: list[tuple[int, int]] = []
    line_junction_edges: list[tuple[int, int]] = []
    num_lines = 0

    def add_loop(pts: list[tuple[float, float, float]], plane_type: str) -> int:
        nonlocal num_lines
        base_j = len(junctions)
        for x, y, z in pts:
            junctions.append({"coordinate": [float(x), float(y), float(z)]})
        pid = len(planes)
        planes.append({"type": plane_type})
        n = len(pts)
        for i in range(n):
            lid = num_lines
            num_lines += 1
            plane_line_edges.append((pid, lid))
            line_junction_edges.append((lid, base_j + i))
            line_junction_edges.append((lid, base_j + (i + 1) % n))
        return pid

    def box_pts(b: tuple[float, float, float, float], z: float = 0.0) -> list[tuple[float, float, float]]:
        x0, y0, x1, y1 = b
        return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]

    out_pid = add_loop(box_pts(outer_box, z=0.0), "floor")
    semantics.append({"type": "outwall", "planeID": [out_pid]})

    for room_type, rbox in room_specs:
        rpid = add_loop(box_pts(rbox, z=0.0), "floor")
        semantics.append({"type": room_type, "planeID": [rpid]})

    for sem_type, plane_loops in extra_openings or []:
        pids = [add_loop(pts, ptype) for ptype, pts in plane_loops]
        semantics.append({"type": sem_type, "planeID": pids})

    plane_line = np.zeros((len(planes), num_lines), dtype=int)
    for pid, lid in plane_line_edges:
        plane_line[pid, lid] = 1

    line_junction = np.zeros((num_lines, len(junctions)), dtype=int)
    for lid, jid in line_junction_edges:
        line_junction[lid, jid] = 1

    return {
        "junctions": junctions,
        "planes": planes,
        "semantics": semantics,
        "planeLineMatrix": plane_line.tolist(),
        "lineJunctionMatrix": line_junction.tolist(),
    }


def test_corner_window_does_not_fill_room_interior(tmp_path: Path) -> None:
    corner_window = (
        "window",
        [
            ("wall", [(200.0, 200.0, 900.0), (2200.0, 200.0, 900.0), (2200.0, 200.0, 2100.0), (200.0, 200.0, 2100.0)]),
            ("wall", [(200.0, 200.0, 900.0), (200.0, 2200.0, 900.0), (200.0, 2200.0, 2100.0), (200.0, 200.0, 2100.0)]),
        ],
    )
    # Also verify a window with a degenerate (2-junction) "floor" plane still gets extracted.
    degenerate_floor_door = (
        "door",
        [
            ("floor", [(2600.0, 3800.0, 0.0), (3400.0, 3800.0, 0.0)]),
            ("wall", [(2600.0, 3800.0, 0.0), (3400.0, 3800.0, 0.0), (3400.0, 3800.0, 2100.0), (2600.0, 3800.0, 2100.0)]),
        ],
    )
    annos = _build_scene(
        outer_box=(0.0, 0.0, 4000.0, 4000.0),
        room_specs=[("living room", (200.0, 200.0, 3800.0, 3800.0))],
        extra_openings=[corner_window, degenerate_floor_door],
    )
    (tmp_path / "annotation_3d.json").write_text(json.dumps(annos))
    _, plan_params = s3d.load_floorplan(str(tmp_path), plan_size=512)

    sem_idx, labels = s3d.render_semantic_floorplan(annos, plan_params, plan_size=512, include_rooms=False)
    win_id = labels.index("window") + 1
    door_id = labels.index("door") + 1

    # Window edges along bottom & left walls should be marked, while interior point (800, 800) mm stays unlabeled.
    edge_px = np.round(s3d.world_to_plan(np.array([[1200.0, 200.0], [200.0, 1200.0]]), plan_params)).astype(int)
    assert sem_idx[edge_px[0, 1], edge_px[0, 0]] == win_id
    assert sem_idx[edge_px[1, 1], edge_px[1, 0]] == win_id

    interior_px = np.round(s3d.world_to_plan(np.array([800.0, 800.0]), plan_params)).astype(int)
    assert sem_idx[interior_px[1], interior_px[0]] != win_id

    door_px = np.round(s3d.world_to_plan(np.array([3000.0, 3800.0]), plan_params)).astype(int)
    assert sem_idx[door_px[1], door_px[0]] == door_id


@pytest.mark.parametrize("include_rooms", [False, True])
def test_thin_interior_partition_wall_survives_adjacent_room_fills(tmp_path: Path, include_rooms: bool) -> None:
    # 24 m house with a 40 mm interior partition wall (sub-pixel at 512x512).
    annos = _build_scene(
        outer_box=(0.0, 0.0, 24000.0, 24000.0),
        room_specs=[
            ("bedroom", (200.0, 200.0, 11980.0, 23800.0)),
            ("kitchen", (12020.0, 200.0, 23800.0, 23800.0)),
        ],
    )
    (tmp_path / "annotation_3d.json").write_text(json.dumps(annos))
    _, plan_params = s3d.load_floorplan(str(tmp_path), plan_size=512)

    sem_idx, labels = s3d.render_semantic_floorplan(annos, plan_params, plan_size=512, include_rooms=include_rooms)
    assert "wall" in labels
    wall_id = labels.index("wall") + 1

    scan_world = np.stack([np.linspace(11000.0, 13000.0, 100), np.full(100, 12000.0)], axis=-1)
    scan_px = np.round(s3d.world_to_plan(scan_world, plan_params)).astype(int)
    sampled = sem_idx[scan_px[:, 1], scan_px[:, 0]]
    assert np.any(sampled == wall_id), "Thin interior partition wall was erased by adjacent room fills"


def test_load_semantic_floorplan_reuses_plan_params_without_rerendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annos = _build_scene(
        outer_box=(0.0, 0.0, 4000.0, 4000.0),
        room_specs=[("bedroom", (200.0, 200.0, 3800.0, 3800.0))],
    )
    (tmp_path / "annotation_3d.json").write_text(json.dumps(annos))
    _, plan_params = s3d.load_floorplan(str(tmp_path), plan_size=512)

    class _StubEncoder:
        def encode_palette(self, labels: list[str]) -> np.ndarray:
            out = np.zeros((len(labels) + 1, 16), dtype=np.float16)
            for idx in range(1, len(labels) + 1):
                out[idx, 0] = 1.0
            return out

    def _fail_render(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("render_floorplan should not be called when plan_params is supplied")

    monkeypatch.setattr(s3d, "render_floorplan", _fail_render)
    sem_idx, palette, labels = s3d.load_semantic_floorplan(
        str(tmp_path),
        text_encoder=_StubEncoder(),  # type: ignore[arg-type]
        plan_size=512,
        include_rooms=True,
        plan_params=plan_params,
    )
    assert sem_idx.shape == (512, 512)
    assert palette.shape == (len(labels) + 1, 16)


def test_valid_concave_floor_door_skips_elevated_hull(tmp_path: Path) -> None:
    l_door = (
        "door",
        [
            (
                "floor",
                [
                    (1000.0, 1000.0, 0.0),
                    (2600.0, 1000.0, 0.0),
                    (2600.0, 1400.0, 0.0),
                    (1400.0, 1400.0, 0.0),
                    (1400.0, 2600.0, 0.0),
                    (1000.0, 2600.0, 0.0),
                ],
            ),
        ],
    )
    annos = _build_scene(
        outer_box=(0.0, 0.0, 4000.0, 4000.0),
        room_specs=[("living room", (200.0, 200.0, 3800.0, 3800.0))],
        extra_openings=[l_door],
    )
    (tmp_path / "annotation_3d.json").write_text(json.dumps(annos))
    _, plan_params = s3d.load_floorplan(str(tmp_path), plan_size=512)

    assert s3d._extract_elevated_openings(annos) == []

    sem_idx, labels = s3d.render_semantic_floorplan(annos, plan_params, plan_size=512, include_rooms=False)
    door_id = labels.index("door") + 1
    arm_px = np.round(s3d.world_to_plan(np.array([1200.0, 2000.0]), plan_params)).astype(int)
    notch_px = np.round(s3d.world_to_plan(np.array([1900.0, 1900.0]), plan_params)).astype(int)
    assert sem_idx[arm_px[1], arm_px[0]] == door_id
    assert sem_idx[notch_px[1], notch_px[0]] != door_id
