"""Tests for resumable training/eval state recovery, schedule determinism, and dataset filtering."""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

import evaluate_s3d
import prepare_s3d
import train
from scenealigner.training import DensityFloorplanDataset


class _ExploitPickle:
    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (eval, ("1 + 1",))


class _TinyDataset(Dataset[dict[str, Any]]):
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"idx": index}


def test_epoch_sample_order_invariants() -> None:
    n = 26
    o_a = train.epoch_sample_order(n, epoch=0, seed=7)
    o_b = train.epoch_sample_order(n, epoch=0, seed=7)
    o_next = train.epoch_sample_order(n, epoch=1, seed=7)
    assert o_a == o_b
    assert sorted(o_a) == list(range(n))
    assert o_next != o_a


def test_empty_dataset_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        train.compute_epoch_schedule(dataset_len=0, epochs=20)


def test_should_save_mid_epoch_skips_epoch_boundary() -> None:
    assert train.should_save_mid_epoch(step=5, save_every=5, steps_per_epoch=7) is True
    assert train.should_save_mid_epoch(step=7, save_every=7, steps_per_epoch=7) is False
    assert train.should_save_mid_epoch(step=10, save_every=0, steps_per_epoch=7) is False


def test_train_resume_rejects_unsafe_pickle_and_resumes_mid_epoch_and_heals_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "train_out"
    out_dir.mkdir()
    data_dir = tmp_path / "s3d_data"
    data_dir.mkdir()

    class _DummyLoRA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0]))

        def save_pretrained(self, save_dir: str) -> None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_dir) / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
            (Path(save_dir) / "adapter_model.safetensors").write_bytes(b"safe")

    class _DummySemanticAdapter(torch.nn.Module):
        def __init__(self, *_a: Any, **_k: Any) -> None:
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor([2.0]))

    seen_batches: list[list[int]] = []
    rsync_calls: list[list[str]] = []

    def _fake_compute_loss(
        model: _DummyLoRA,
        batch: dict[str, Any],
        *_a: Any,
        semantic_adapter: _DummySemanticAdapter | None = None,
        **_k: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        seen_batches.append(list(batch["idx"]))
        if crash_on_second_step[0] and len(seen_batches) == 2:
            raise RuntimeError("simulated interruption")
        loss = (model.w ** 2).sum()
        if semantic_adapter is not None:
            loss = loss + (semantic_adapter.p ** 2).sum()
        return loss, {"feat": loss.detach()}

    def _fake_save_semantic_adapter(save_dir: str, _adapter: Any, lambda_sem: float | None = 0.0) -> None:
        torch.save({"lambda_sem": lambda_sem}, Path(save_dir) / "semantic_adapter.pt")

    def _inprocess_dataloader(dataset: Any, **kwargs: Any) -> TorchDataLoader[Any]:
        kwargs["num_workers"] = 0
        return TorchDataLoader(dataset, **kwargs)

    def _fake_subproc_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        rsync_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(train, "DensityFloorplanDataset", lambda *_a, **_k: _TinyDataset(8))
    monkeypatch.setattr(train, "collate", lambda batch, **_k: {"idx": [b["idx"] for b in batch]})
    monkeypatch.setattr(train, "DataLoader", _inprocess_dataloader)
    monkeypatch.setattr(train, "LoraConfig", lambda **_k: object())
    monkeypatch.setattr(train, "load_model", lambda *_a, **_k: _DummyLoRA())
    monkeypatch.setattr(train, "get_peft_model", lambda m, _cfg: m)
    monkeypatch.setattr(train, "get_peft_model_state_dict", lambda m: {"w": m.w.detach().clone()})
    monkeypatch.setattr(train, "set_peft_model_state_dict", lambda m, sd: m.w.data.copy_(sd["w"]))
    monkeypatch.setattr(train, "SemanticAdapter", _DummySemanticAdapter)
    monkeypatch.setattr(train, "save_semantic_adapter", _fake_save_semantic_adapter)
    monkeypatch.setattr(train, "compute_loss", _fake_compute_loss)
    monkeypatch.setattr(train.subprocess, "run", _fake_subproc_run)

    # 1. Unsafe pickled object in checkpoint.pt is rejected by weights_only=True
    torch.save({"step": 1, "bad": _ExploitPickle()}, out_dir / "checkpoint.pt")
    base_argv = [
        "train.py",
        "--data",
        str(data_dir),
        "--output",
        str(out_dir),
        "--epochs",
        "1",
        "--resolutions",
        "512",
        "--save_every",
        "1",
        "--sem_mode",
        "consensus",
        "--resume",
        "--sync_gcs_uri",
        "gs://bucket/runs/r1",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        train.main()
    (out_dir / "checkpoint.pt").unlink()

    # 2. First run saves step=1 mid-epoch then aborts before step=2
    crash_on_second_step = [True]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train.main()
    expected_perm = train.epoch_sample_order(8, epoch=0, seed=0)
    assert seen_batches == [expected_perm[:4], expected_perm[4:8]]

    # 3. Resuming recovers at step=1 and executes only the second batch
    seen_batches.clear()
    crash_on_second_step[0] = False
    train.main()
    assert seen_batches == [expected_perm[4:8]]
    assert torch.load(out_dir / "checkpoint.pt", weights_only=True)["step"] == 2

    # 4. If inference exports are missing at completed step (start_epoch >= epochs), resume heals & syncs them
    for export_name in ("adapter_config.json", "adapter_model.safetensors", "semantic_adapter.pt"):
        (out_dir / export_name).unlink()
    rsync_calls.clear()
    seen_batches.clear()
    train.main()
    assert seen_batches == []
    assert (out_dir / "adapter_config.json").exists()
    assert (out_dir / "adapter_model.safetensors").exists()
    assert (out_dir / "semantic_adapter.pt").exists()
    assert len(rsync_calls) == 1
    assert rsync_calls[0][:4] == ["gcloud", "storage", "rsync", "--recursive"]


def test_save_checkpoint_sync_gcs_uri_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class _DummyModel:
        def save_pretrained(self, out_dir: str) -> None:
            (Path(out_dir) / "adapter_model.safetensors").write_bytes(b"ok")

    def _fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        returncode = 1 if len(calls) > 1 else 0
        stderr = "transient error" if returncode else ""
        return subprocess.CompletedProcess(cmd, returncode=returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(train, "get_peft_model_state_dict", lambda _m: {"w": torch.tensor([1.0])})
    monkeypatch.setattr(train.subprocess, "run", _fake_run)

    layer = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()

    train.save_checkpoint(str(out_dir), _DummyModel(), opt, step=5, semantic_adapter=None, sync_gcs_uri=None)
    assert calls == []

    train.save_checkpoint(
        str(out_dir), _DummyModel(), opt, step=10, semantic_adapter=None, sync_gcs_uri="gs://my-bucket/runs/r1/"
    )
    assert (out_dir / "checkpoint.pt").exists()
    assert calls == [["gcloud", "storage", "rsync", "--recursive", "--quiet", str(out_dir), "gs://my-bucket/runs/r1"]]

    # Second call simulates rsync failure and must remain non-fatal
    train.save_checkpoint(
        str(out_dir), _DummyModel(), opt, step=15, semantic_adapter=None, sync_gcs_uri="gs://my-bucket/runs/r1"
    )
    assert len(calls) == 2
    assert torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=True)["step"] == 15


def test_prepare_s3d_per_scene_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s3d_root = tmp_path / "Structured3D"
    out_dir = tmp_path / "s3d_data"
    out_dir.mkdir()
    for s in (0, 1, 2):
        (s3d_root / f"scene_{s:05d}" / "2D_rendering" / "room0" / "perspective" / "full" / "0").mkdir(parents=True)

    load_model_calls = 0
    processed_scenes: list[str] = []

    def _fake_load_models(**_kwargs: Any) -> tuple[object, object]:
        nonlocal load_model_calls
        load_model_calls += 1
        return (object(), object())

    def _fake_load_floorplan(scene_dir: str) -> tuple[Image.Image, Any]:
        processed_scenes.append(Path(scene_dir).name)
        return Image.new("RGB", (16, 16)), ((0.0, 0.0), 1.0)

    def _fake_list_views(scene_dir: str) -> list[str]:
        return [str(Path(scene_dir) / "2D_rendering" / "room0" / "perspective" / "full" / "0")]

    def _fake_sample(*_args: Any, **_kwargs: Any) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        pts = np.zeros((200, 2), dtype=np.float32)
        return Image.new("L", (16, 16)), pts, pts

    part_file = out_dir / "scenes_0_3.csv.jsonl.part"
    cached_row = {
        "scene_id": "scene_00000",
        "density": "pi3/scene_00000/room0_0.png",
        "floorplan": "floorplan/scene_00000.png",
        "correspondences": "pi3/scene_00000/room0_0.npy",
    }
    part_file.write_text(
        json.dumps({"scene_id": "scene_00000", "rows": [cached_row]}) + "\n"
        + '{"scene_id": "scene_00001", "rows": [',
        encoding="utf-8",
    )

    monkeypatch.setattr(prepare_s3d.s3d, "TRAIN_SCENES", range(0, 10))
    monkeypatch.setattr(prepare_s3d.s3d, "OFF_CANVAS_SCENES", ())
    monkeypatch.setattr(prepare_s3d, "load_models", _fake_load_models)
    monkeypatch.setattr(prepare_s3d.s3d, "load_floorplan", _fake_load_floorplan)
    monkeypatch.setattr(prepare_s3d.s3d, "list_views", _fake_list_views)
    monkeypatch.setattr(prepare_s3d.s3d, "load_world_points", lambda _v: np.zeros((10, 3), dtype=np.float32))
    monkeypatch.setattr(prepare_s3d.s3d, "world_to_plan", lambda _w, _p: np.zeros((10, 2), dtype=np.float32))
    monkeypatch.setattr(prepare_s3d, "reconstruct", lambda *_a, **_k: {"points_cam0": np.zeros((1, 4, 4, 3))})
    monkeypatch.setattr(prepare_s3d, "depth_sample", _fake_sample)
    monkeypatch.setattr(prepare_s3d, "pi3_sample", _fake_sample)

    argv = ["prepare_s3d.py", "--s3d_root", str(s3d_root), "--output", str(out_dir), "--scenes", "0", "3", "--no_semantics", "--resume"]
    monkeypatch.setattr(sys, "argv", argv)
    prepare_s3d.main()

    assert load_model_calls == 1
    assert processed_scenes == ["scene_00001", "scene_00002"]
    assert not part_file.exists()
    df = pd.read_csv(out_dir / "scenes_0_3.csv")
    assert set(df["scene_id"]) == {"scene_00000", "scene_00001", "scene_00002"}

    # Re-running --resume when final CSV exists skips before loading models
    prepare_s3d.main()
    assert load_model_calls == 1


def test_density_dataset_deduplicates_overlapping_csvs(tmp_path: Path) -> None:
    row_a = {
        "scene_id": "scene_00000",
        "density": "pi3/scene_00000/r0_0.png",
        "floorplan": "floorplan/scene_00000.png",
        "correspondences": "pi3/scene_00000/r0_0.npy",
    }
    row_b = {
        "scene_id": "scene_00001",
        "density": "pi3/scene_00001/r0_0.png",
        "floorplan": "floorplan/scene_00001.png",
        "correspondences": "pi3/scene_00001/r0_0.npy",
    }
    pd.DataFrame([row_a]).to_csv(tmp_path / "scenes_0_1.csv", index=False)
    pd.DataFrame([row_a, row_b]).to_csv(tmp_path / "scenes_0_2.csv", index=False)
    ds = DensityFloorplanDataset(str(tmp_path), load_semantics=False)
    assert len(ds) == 2
    assert list(ds.data["scene_id"]) == ["scene_00000", "scene_00001"]


def test_evaluate_s3d_single_pass_multimode_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s3d_root = tmp_path / "Structured3D"
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    for scene in (3250, 3251):
        (s3d_root / f"scene_{scene:05d}").mkdir(parents=True)

    load_model_calls = 0
    recon_calls = 0
    photo_sem_calls = 0
    seen_sem_modes: list[str] = []
    rsync_calls: list[list[str]] = []

    def _fake_load_models(**_kwargs: Any) -> tuple[object, object]:
        nonlocal load_model_calls
        load_model_calls += 1
        return (object(), object())

    def _fake_reconstruct(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal recon_calls
        recon_calls += 1
        return {
            "conf": np.ones((1, 4, 4), dtype=np.float32),
            "extrinsics": np.eye(4, dtype=np.float32)[None],
            "points_cam0": np.zeros((1, 4, 4, 3), dtype=np.float32),
            "est_gravity_cam": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

    def _fake_load_view_gt_semantics(*_args: Any, **_kwargs: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
        nonlocal photo_sem_calls
        photo_sem_calls += 1
        return np.zeros((4, 4), dtype=np.uint8), np.zeros((2, 512), dtype=np.float16), ["wall"]

    def _fake_align(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        seen_sem_modes.append(str(kwargs.get("sem_mode", "none")))
        return {"proj_params": object(), "M": np.eye(2, 3)}

    def _fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        rsync_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    progress_file = out_dir / "eval_progress_none_uniform_consensus.jsonl"
    prior_scene = {
        "scene": 3250,
        "modes": {
            "none": {"t_errs": [0.2], "r_errs": [5.0], "failed": 0},
            "uniform": {"t_errs": [0.2], "r_errs": [5.0], "failed": 0},
            "consensus": {"t_errs": [0.2], "r_errs": [5.0], "failed": 0},
        },
    }
    progress_file.write_text(
        json.dumps(prior_scene) + "\n" + '{"scene": 3251, "modes": ',
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_s3d.s3d, "TEST_SCENES", [3250, 3251])
    monkeypatch.setattr(evaluate_s3d, "load_models", _fake_load_models)
    monkeypatch.setattr(evaluate_s3d, "load_model", lambda *_a, **_k: torch.nn.Linear(2, 2))
    monkeypatch.setattr(evaluate_s3d, "load_semantic_adapter", lambda *_a, **_k: (torch.nn.Identity(), 0.5))
    monkeypatch.setattr(evaluate_s3d, "ClipTextEncoder", lambda *_a, **_k: object())
    monkeypatch.setattr(evaluate_s3d.s3d, "load_floorplan", lambda _d: (Image.new("RGB", (16, 16)), ((0.0, 0.0), 1.0)))
    monkeypatch.setattr(
        evaluate_s3d.s3d,
        "load_semantic_floorplan",
        lambda *_a, **_k: (np.zeros((8, 8), dtype=np.uint8), np.zeros((2, 512), dtype=np.float16), ["wall"]),
    )
    monkeypatch.setattr(evaluate_s3d.s3d, "list_views", lambda d: [str(Path(d) / "v0")])
    monkeypatch.setattr(evaluate_s3d, "reconstruct", _fake_reconstruct)
    monkeypatch.setattr(evaluate_s3d.s3d, "load_view_gt_semantics", _fake_load_view_gt_semantics)
    monkeypatch.setattr(evaluate_s3d, "align_to_floorplan", _fake_align)
    monkeypatch.setattr(evaluate_s3d, "pose_errors", lambda *_a, **_k: (0.2, 5.0))
    monkeypatch.setattr(evaluate_s3d.subprocess, "run", _fake_run)

    argv = [
        "evaluate_s3d.py",
        "--s3d_root",
        str(s3d_root),
        "--checkpoint_dir",
        str(out_dir),
        "--sem_modes",
        "none",
        "uniform",
        "consensus",
        "--output_dir",
        str(out_dir),
        "--resume",
        "--sync_gcs_uri",
        "gs://my-bucket/runs/r1",
        "--sync_every_scenes",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate_s3d.main()

    assert load_model_calls == 1
    assert recon_calls == 1
    assert photo_sem_calls == 1
    assert seen_sem_modes == ["none", "uniform", "consensus"]
    assert not progress_file.exists()
    assert len(rsync_calls) >= 2
    for mode in ("none", "uniform", "consensus"):
        out_json = out_dir / f"eval_s3d_{mode}.json"
        assert out_json.exists()
        assert json.loads(out_json.read_text())["Pos@0.5m"] == pytest.approx(100.0)

    # Completed modes are skipped on a second --resume invocation before model load
    evaluate_s3d.main()
    assert load_model_calls == 1

    # Corrupt eval output JSON is detected on --resume, removed, and re-evaluated
    seen_sem_modes.clear()
    (out_dir / "eval_s3d_none.json").write_text("{corrupt", encoding="utf-8")
    evaluate_s3d.main()
    assert load_model_calls == 2
    assert seen_sem_modes == ["none", "none"]
    assert json.loads((out_dir / "eval_s3d_none.json").read_text())["Pos@0.5m"] == pytest.approx(100.0)


def test_density_dataset_filters_by_scene_ranges(tmp_path: Path) -> None:
    def _row(sid: str) -> dict[str, str]:
        return {
            "scene_id": sid,
            "density": f"pi3/{sid}/r0_0.png",
            "floorplan": f"floorplan/{sid}.png",
            "correspondences": f"pi3/{sid}/r0_0.npy",
        }

    pd.DataFrame([_row("scene_00050"), _row("scene_00199"), _row("scene_00200"), _row("scene_00250")]).to_csv(
        tmp_path / "scenes_0_300.csv", index=False
    )
    pd.DataFrame([_row("scene_00250"), _row("scene_00350")]).to_csv(
        tmp_path / "scenes_200_400.csv", index=False
    )

    ds = DensityFloorplanDataset(str(tmp_path), load_semantics=False, scene_ranges=[(0, 200)])
    assert len(ds) == 2
    assert list(ds.data["scene_id"]) == ["scene_00050", "scene_00199"]

    with pytest.raises(ValueError, match="overlap scene_ranges"):
        DensityFloorplanDataset(str(tmp_path), load_semantics=False, scene_ranges=[(400, 600)])


def test_train_parse_scene_ranges() -> None:
    assert train.parse_scene_ranges(["0:200", "400:600"]) == [(0, 200), (400, 600)]
    assert train.parse_scene_ranges(None) is None
    for bad in ("200:100", "bad", "200:200", "-1:10", ":100", "0:"):
        with pytest.raises(ValueError):
            train.parse_scene_ranges([bad])


def test_evaluate_s3d_recon_cache_miss_hit_and_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s3d_root = tmp_path / "Structured3D"
    scene_dir = s3d_root / "scene_03250"
    view_dir = scene_dir / "2D_rendering" / "0" / "perspective" / "full" / "0"
    view_dir.mkdir(parents=True)
    recon_cache_dir = tmp_path / "recon_cache"

    load_model_calls = 0
    recon_calls = 0
    base_recon = {
        "conf": np.full((1, 4, 4), 2.5, dtype=np.float32),
        "extrinsics": np.eye(4, dtype=np.float32)[None],
        "points_cam0": np.arange(48, dtype=np.float32).reshape(1, 4, 4, 3),
        "est_gravity_cam": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    def _fake_load_models(**_kwargs: Any) -> tuple[object, object]:
        nonlocal load_model_calls
        load_model_calls += 1
        return (object(), object())

    def _fake_reconstruct(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal recon_calls
        recon_calls += 1
        return {k: v.copy() for k, v in base_recon.items()}

    monkeypatch.setattr(evaluate_s3d.s3d, "TEST_SCENES", [3250])
    monkeypatch.setattr(evaluate_s3d, "load_models", _fake_load_models)
    monkeypatch.setattr(evaluate_s3d, "load_model", lambda *_a, **_k: torch.nn.Linear(2, 2))
    monkeypatch.setattr(evaluate_s3d.s3d, "load_floorplan", lambda _d: (Image.new("RGB", (16, 16)), ((0.0, 0.0), 1.0)))
    monkeypatch.setattr(evaluate_s3d.s3d, "list_views", lambda d: [str(Path(d) / "2D_rendering/0/perspective/full/0")])
    monkeypatch.setattr(evaluate_s3d, "reconstruct", _fake_reconstruct)
    monkeypatch.setattr(evaluate_s3d, "align_to_floorplan", lambda *_a, **_k: {"proj_params": object(), "M": np.eye(2, 3)})
    monkeypatch.setattr(
        evaluate_s3d,
        "pose_errors",
        lambda recon, *_a, **_k: (0.05 if recon["conf"].dtype == np.float32 else 5.0, 5.0),
    )

    def _run_eval(out_dir: Path) -> dict[str, float]:
        out_dir.mkdir(exist_ok=True)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_s3d.py",
                "--s3d_root",
                str(s3d_root),
                "--checkpoint_dir",
                str(out_dir),
                "--sem_modes",
                "none",
                "--output_dir",
                str(out_dir),
                "--recon_cache",
                str(recon_cache_dir),
            ],
        )
        evaluate_s3d.main()
        return json.loads((out_dir / "eval_s3d_none.json").read_text(encoding="utf-8"))

    # 1. Cache miss lazily loads Pi3 models once and persists lossless float32 .npz
    m1 = _run_eval(tmp_path / "out_miss")
    assert load_model_calls == 1
    assert recon_calls == 1
    cache_files = list((recon_cache_dir / "scene_03250").glob("*.npz"))
    assert len(cache_files) == 1
    cache_file = cache_files[0]
    with np.load(cache_file, allow_pickle=False) as cached:
        for k in ("points_cam0", "conf", "extrinsics", "est_gravity_cam"):
            assert cached[k].dtype == np.float32
            assert np.array_equal(cached[k], base_recon[k])

    # 2. Cache hit skips load_models and reconstruct and matches bit-for-bit metrics
    load_model_calls = 0
    recon_calls = 0
    m2 = _run_eval(tmp_path / "out_hit")
    assert load_model_calls == 0
    assert recon_calls == 0
    assert m2 == m1

    # 3a. Corrupt .npz invalidates and recomputes
    cache_file.write_bytes(b"corrupt")
    m3 = _run_eval(tmp_path / "out_corrupt")
    assert load_model_calls == 1
    assert recon_calls == 1
    assert m3 == m1

    # 3b. float16-downgraded .npz invalidates and rewrites float32
    np.savez(
        cache_file,
        points_cam0=base_recon["points_cam0"],
        conf=base_recon["conf"].astype(np.float16),
        extrinsics=base_recon["extrinsics"],
        est_gravity_cam=base_recon["est_gravity_cam"],
    )
    load_model_calls = 0
    recon_calls = 0
    m4 = _run_eval(tmp_path / "out_f16")
    assert load_model_calls == 1
    assert recon_calls == 1
    assert m4 == m1
    with np.load(cache_file, allow_pickle=False) as repaired:
        assert repaired["conf"].dtype == np.float32


def test_stop_after_epoch_preserves_curriculum_and_writes_train_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "train_stop_out"
    out_dir.mkdir()
    data_dir = tmp_path / "s3d_data"
    data_dir.mkdir()

    class _DummyLoRA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0]))

        def save_pretrained(self, save_dir: str) -> None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_dir) / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
            (Path(save_dir) / "adapter_model.safetensors").write_bytes(b"safe")

    class _DummySemanticAdapter(torch.nn.Module):
        def __init__(self, *_a: Any, **_k: Any) -> None:
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor([2.0]))

    seen_calls: list[tuple[list[int], int, int]] = []
    rsync_calls: list[list[str]] = []

    def _fake_compute_loss(
        model: _DummyLoRA,
        batch: dict[str, Any],
        step: int = 0,
        total_steps: int = 0,
        *,
        semantic_adapter: _DummySemanticAdapter | None = None,
        **_k: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        seen_calls.append((list(batch["idx"]), step, total_steps))
        loss = (model.w ** 2).sum()
        if semantic_adapter is not None:
            loss = loss + (semantic_adapter.p ** 2).sum()
        return loss, {"feat": loss.detach()}

    def _inprocess_dataloader(dataset: Any, **kwargs: Any) -> TorchDataLoader[Any]:
        kwargs["num_workers"] = 0
        return TorchDataLoader(dataset, **kwargs)

    monkeypatch.setattr(train, "DensityFloorplanDataset", lambda *_a, **_k: _TinyDataset(8))
    monkeypatch.setattr(train, "collate", lambda batch, **_k: {"idx": [b["idx"] for b in batch]})
    monkeypatch.setattr(train, "DataLoader", _inprocess_dataloader)
    monkeypatch.setattr(train, "LoraConfig", lambda **_k: object())
    monkeypatch.setattr(train, "load_model", lambda *_a, **_k: _DummyLoRA())
    monkeypatch.setattr(train, "get_peft_model", lambda m, _cfg: m)
    monkeypatch.setattr(train, "get_peft_model_state_dict", lambda m: {"w": m.w.detach().clone()})
    monkeypatch.setattr(train, "set_peft_model_state_dict", lambda m, sd: m.w.data.copy_(sd["w"]))
    monkeypatch.setattr(train, "SemanticAdapter", _DummySemanticAdapter)
    monkeypatch.setattr(
        train,
        "save_semantic_adapter",
        lambda save_dir, _ad, lambda_sem=0.0: torch.save({"lambda_sem": lambda_sem}, Path(save_dir) / "semantic_adapter.pt"),
    )
    monkeypatch.setattr(train, "compute_loss", _fake_compute_loss)
    monkeypatch.setattr(
        train.subprocess,
        "run",
        lambda cmd, **_k: (rsync_calls.append(cmd), subprocess.CompletedProcess(cmd, 0, "", ""))[1],
    )

    def _run(stop_ep: int) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--data",
                str(data_dir),
                "--output",
                str(out_dir),
                "--epochs",
                "2",
                "--stop_after_epoch",
                str(stop_ep),
                "--resolutions",
                "512",
                "--save_every",
                "10",
                "--sem_mode",
                "consensus",
                "--resume",
                "--sync_gcs_uri",
                "gs://bucket/runs/r2",
            ],
        )
        train.main()

    _run(1)
    assert [c[1:] for c in seen_calls] == [(0, 4), (1, 4)]
    assert json.loads((out_dir / "train_state.json").read_text(encoding="utf-8")) == {
        "step": 2,
        "steps_per_epoch": 2,
        "completed_epochs": 1,
        "total_epochs": 2,
    }
    assert len(rsync_calls) == 1

    # Removing train_state.json simulates resuming a legacy Epoch 1 checkpoint
    (out_dir / "train_state.json").unlink()
    seen_calls.clear()
    rsync_calls.clear()
    _run(1)
    assert seen_calls == []
    assert json.loads((out_dir / "train_state.json").read_text(encoding="utf-8"))["completed_epochs"] == 1
    assert len(rsync_calls) == 1

    _run(2)
    assert [c[1:] for c in seen_calls] == [(2, 4), (3, 4)]
    assert json.loads((out_dir / "train_state.json").read_text(encoding="utf-8")) == {
        "step": 4,
        "steps_per_epoch": 2,
        "completed_epochs": 2,
        "total_epochs": 2,
    }

    with pytest.raises(ValueError, match="--stop_after_epoch"):
        _run(3)
