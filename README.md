# Semantic-Aware 3D-Grounded Floorplan Localization

**CS 5787 Deep Learning (Fall 2026) final project.** Manaswita Kodali (mk2774), Liheng Cao (lc2332), Jiatong Xu (jx429)

A fork of [SceneAligner](https://github.com/Cornell-VAILab/SceneAligner) ([paper](https://arxiv.org/abs/2605.22581), [project page](https://Cornell-VAILab.github.io/SceneAligner)) by Cho, Cai and Averbuch-Elor (NeurIPS'26). `main` holds the original code; the `liheng` branch adds **multi-view, open-vocabulary semantic consensus** to its floorplan localization.

![teaser](./assets/teaser.png)
> SceneAligner reconstructs a gravity-aligned 3D point cloud from photos and aligns it to a 2D floorplan, localizing the photos within it.

## What the project does

SceneAligner matches DINOv3 features of a top-down density map (from the 3D reconstruction) against DINOv3 features of the floorplan. Purely geometric features are ambiguous when different regions have similar layouts. We add a semantic channel:

1. **Semantic maps on both sides.** Floorplan cells and photo pixels get labels (wall, door, window, optionally rooms), encoded with a CLIP text encoder (`openai/clip-vit-base-patch16`, 3 prompt templates), so labels live in an open-vocabulary embedding space.
2. **Multi-view consensus.** Photo semantics are lifted onto the reconstructed points and projected top-down. Per cell, CLIP embeddings are fused by confidence-weighted averaging, and the agreement between views gives a reliability score `rel = ‖μ‖ · (1 − exp(−n_obs/τ)) · cos(μ̂, label)` in [0, 1].
3. **Adaptive fusion.** A small trainable `SemanticAdapter` (MLP 512→256→128) projects CLIP patch features, and patch similarity becomes `S_ij = ⟨g_i, g_j⟩ + λ_sem · rel_i · rel_j · ⟨s_i, s_j⟩` (`g` = DINOv3, `s` = semantic). Reliable semantics disambiguate similar regions; unreliable ones fall back to geometry. Training adds a semantic contrastive loss (`--alpha_sem`).

`--sem_mode` selects the variant, which doubles as our ablation: `none` (original SceneAligner), `uniform` (semantics without reliability weighting), `consensus` (full method).

## Changes We Made

**New files**

| Path | Purpose |
|---|---|
| `scenealigner/semantics.py` | CLIP text encoder, compact semantic-map I/O, patch pooling, and multi-view top-down consensus with reliability |
| `runners/` (`pipeline.py`, `run_vm.py`, `setup_vm.sh`, `sync.sh`, `colab.ipynb`) | Resumable prep → train → eval runner for a GPU VM or Colab, with optional GCS backup |
| `tests/` | 67 unit tests (semantics, fusion, training resume, runner), run with `pytest` |
| `pyproject.toml` | pytest and pyright config |

**Modified files**

| Path | Change |
|---|---|
| `scenealigner/inference.py` | `SemanticAdapter`, `compute_fused_sim_matrix`, adapter save/load; `estimate_correspondences` and `align_to_floorplan` take `sem_mode`, `lambda_sem` and semantic maps |
| `scenealigner/training.py` | Dataset loads and rotates/crops semantic maps with the images; collate pools them to patches; adds `semantic_contrastive_loss`; fused logits in the feature-matching loss |
| `scenealigner/structured3d.py` | `render_semantic_floorplan`, `load_semantic_floorplan`, `load_view_gt_semantics` (NYU40 labels → CLIP palette), colorized semantic floorplans |
| `train.py` | New flags `--sem_mode --lambda_sem --alpha_sem --sem_dim --init_lora --freeze_lora` (train only the adapter), `--resume --save_every --scene_ranges --stop_after_epoch --sync_gcs_uri` |
| `prepare_s3d.py` | Also writes floorplan, photo and consensus-density semantic maps; `--no_semantics --include_rooms --photo_sem_source --structural_photo_semantics --clip_model --resume` |
| `evaluate_s3d.py` | `--sem_mode/--sem_modes` (several modes in one pass, shared reconstructions), `--semantic_adapter --lambda_sem --recon_cache --output_dir --resume --max_scenes`, GCS sync |
| `requirements.txt`, `.gitignore` | Adds `transformers` (CLIP); ignores `notes/` and `.pytest_cache/` |
| `demo.py`, `evaluate_c3.py`, `prepare_c3.py`, `scenealigner/{preprocess,evaluation,viewer,visualization}.py` | Type annotations only (no behavior change) |

With `--sem_mode none` (the default) the code reproduces the original method; C3 evaluation and training are unchanged.

## How to run our additions

Install as below, then verify with `python -m pytest -q` (no GPU or data needed). Our semantic experiments use [Structured3D](https://structured3d-dataset.org/) (train scenes 0–2999, test 3250–3499), extracted to `<S3D>/scene_*`.

```bash
# 1. Training data incl. semantic maps and reliabilities
python prepare_s3d.py --s3d_root <S3D> --output <s3d_data>

# 2. Train the semantic adapter on top of the frozen Structured3D LoRA (use --sem_mode uniform for the ablation) the command trains the semantic adapter on top of the authors’ released, frozen Structured3D LoRA. The adapter is what we add.
python train.py --data <s3d_data> --output <run> --init_lora jhcho99/SceneAligner/structured3d --freeze_lora \
  --sem_mode consensus --lambda_sem 0.5 --alpha_sem 0.2 --epochs 10 --resolutions 512 --min_points 1000 --resume

# 3. Evaluate all three modes in one pass (recalls as in Table 3: Pos@0.1/0.5/1.0m, Ang&Pos@(30,1m))
python evaluate_s3d.py --s3d_root <S3D> --checkpoint_dir <run> --sem_modes none uniform consensus \
  --recon_cache <recon_cache> --output_dir <eval_out>          # results: <eval_out>/eval_s3d_<mode>.json
```

Or run everything with the runner (needs `S3D_BASE_URL` in `.env`; `runners/colab.ipynb` does the same on Colab):

```bash
python runners/run_vm.py --stages prep train eval --chunks 0 1 2 --epochs 10 --sem-mode consensus --no-gcs
python runners/run_vm.py --stages smoke      # pytest + demo
```


## Roadmap

- [x] Implement Structured3D semantic floorplans, CLIP category embeddings, reliability-aware semantic projection, adaptive fusion, semantic adapter training, and `none/uniform/consensus` evaluation.
- [x] Implement and unit-test multi-view semantic aggregation and consensus reliability estimation.
- [ ] Integrate and evaluate multi-view semantic consensus in the end-to-end localization pipeline, comparing single-view and multi-view reliability.
- [ ] Replace ground-truth photo semantics with SAM3-based semantic predictions and CLIP embeddings (`--photo_sem_source` currently supports `gt`/`none`).
- [ ] Extend semantic annotations and localization evaluation to C3, including annotation propagation and manual validation.
- [ ] Compare against SemRayLoc and evaluate correspondence accuracy using PCK/RMSE.

---

# Original SceneAligner usage

### Installation

```bash
conda create -n scenealigner python=3.11
conda activate scenealigner
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Optional, to render images of the 3D viewer without a browser (`--render`):

```bash
pip install playwright && playwright install chromium
```

### Demo

The example is the Friedrichswerder Church in Berlin from the [C3 dataset](https://c3po-correspondence.github.io/):
20 interior and 40 exterior photos, which share one floorplan.

```bash
python demo.py --images examples/S221/interior examples/S221/exterior --floorplan examples/S221/floorplan.jpg --output outputs/S221
```

Each directory is a photo collection, reconstructed and aligned to the floorplan independently; collections without visual overlap thus end up in a common frame. Collections of more than 150 photos are split into several reconstructions. For each reconstruction, the demo saves the floorplan-aligned 3D scene (`.npz`), the density map and its alignment with the reliable correspondences, and it draws the cameras on the floorplan (`cameras.png`). It then starts the 3D viewer at http://localhost:8080. With `--share`, the viewer is also served at a public viser share URL, valid for 24 hours.

```bash
python -m scenealigner.viewer outputs/S221                                # interactive viewer
python -m scenealigner.viewer outputs/S221 --render first_frame.png       # image of the first-frame view
python -m scenealigner.viewer outputs/S221 --render demo.png --angle 270  # image after a 270° turn of the turntable
```

![demo](./assets/demo.png)
> View of the example after a 270° turn of the viewer's turntable.

### Dataset

We use a clean subset of [C3](https://c3po-correspondence.github.io/). Its test split is listed in `data/c3/test.csv` and its train split in `train.csv` of our [dataset repository](https://huggingface.co/datasets/jhcho99/SceneAligner):

| Column | Description |
|---|---|
| `uid` | Identifier of the photo-floorplan pair in C3 |
| `scene_id` | `S<scene>-<I or E>-F<floorplan>`: interior (I) or exterior (E) photos of a scene, paired with a floorplan |
| `chunk_idx` | Test only: reconstruction the photo belongs to (scenes of more than 150 photos are split) |
| `scene_name`, `plan_path`, `photo_path` | As in C3; paths are relative to `<C3>/visual/<scene_name>/` |
| `type` | `interior` or `exterior` |

Download [C3](https://huggingface.co/datasets/kwhuang/C3) (427 GB) and extract its archives (the evaluation only needs `geometric/geometric_test.tar.gz` and the `visual/` archives of the test scenes, 58 GB):

```bash
hf download kwhuang/C3 --repo-type dataset --local-dir <C3>
(cd <C3>/geometric && for f in *.tar.gz; do tar -xzf "$f"; done)
(cd <C3>/visual && for f in *.tar.gz; do tar -xzf "$f"; done)
```

The dataset repository (52 GB) also holds our cached test data: the π³ reconstructions of the test photos (with the scene gravity from GeoCalib), their density maps, and the GeoCalib gravity of each photo:

```bash
hf download jhcho99/SceneAligner --repo-type dataset --local-dir <cache>
```

### Evaluation

On our cached reconstructions, which reproduces Tables 1 and 2 of the paper up to small numerical differences between GPUs:

```bash
python evaluate_c3.py --c3_root <C3> --recon_cache <cache>/test/reconstructions                               # Ours
python evaluate_c3.py --c3_root <C3> --recon_cache <cache>/test/reconstructions --checkpoint_dir pre-trained  # pretrained DINOv3
```

End to end from the photos, with the public π³ weights of MegaDepth-X; the numbers may differ slightly:

```bash
python evaluate_c3.py --c3_root <C3>
```

### Training

The training data are density maps of π³ and VGGT reconstructions of the C3 train split, of photo collections and of single photos, with correspondences derived from the C3 annotations. VGGT is only needed for the second command:

```bash
pip install einops && pip install --no-deps git+https://github.com/facebookresearch/vggt.git@a288dd0f14786c93483e45524328726ab7b1b4ce
```

```bash
python prepare_c3.py --c3_root <C3> --train_csv <cache>/train.csv --output <data>                 # π³, photo collections
python prepare_c3.py --c3_root <C3> --train_csv <cache>/train.csv --output <data> --model vggt    # VGGT, photo collections
python prepare_c3.py --c3_root <C3> --train_csv <cache>/train.csv --output <data> --single_view   # π³, single photos
python train.py --data <data> --output <run>
python evaluate_c3.py --c3_root <C3> --recon_cache <cache>/test/reconstructions --checkpoint_dir <run>
```

Each `prepare_c3.py` command goes through five confidence thresholds, which `--conf` can split over several runs.

### Structured3D

Single-view localization on [Structured3D](https://structured3d-dataset.org/) (Table 3) uses its perspective images (`full`) and annotations, extracted to `<S3D>/scene_*`. Structured3D is a synthetic indoor dataset, so its views are reconstructed with the original π³ weights rather than those fine-tuned on Internet photos by MegaDepth-X. The floorplans are drawn from `annotation_3d.json` at 1 cm per pixel: walls in black, rooms in white, doors and windows closed.

```bash
python evaluate_s3d.py --s3d_root <S3D>
```

The released weights were trained for 20 epochs instead of 10, as we observed that training longer leads to a higher accuracy than reported in Table 3:

| Recall (%) | Pos@0.1m | Pos@0.5m | Pos@1.0m | Ang&Pos@(30,1m) |
|---|---|---|---|---|
| Table 3 | 3.5 | 37.5 | 53.8 | 51.6 |
| Released weights | 4.7 | 44.3 | 59.7 | 58.0 |


Training: every view of the train scenes gives two samples, a density map of its π³ reconstruction and one of its ground-truth depth map.

```bash
python prepare_s3d.py --s3d_root <S3D> --output <s3d_data>
python train.py --data <s3d_data> --output <s3d_run> --epochs 20 --resolutions 512 --min_points 1000
python evaluate_s3d.py --s3d_root <S3D> --checkpoint_dir <s3d_run>
```

## License
Our additions are released under the same [MIT License](LICENSE) as the original code. The models and data used keep their own licenses, listed in [THIRD_PARTY.md](THIRD_PARTY.md); the π³ weights are for non-commercial use only.

## Citation
If you build on SceneAligner, please cite the original paper:

````BibTeX
@inproceedings{cho2026scenealigner,
  title={SceneAligner: 3D-Grounded Floorplan Localization in the Wild},
  author={Junhyeong Cho and Ruojin Cai and Hadar Averbuch-Elor},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
````
