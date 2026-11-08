# DermaPath

Automated skin lesion segmentation using machine learning. This repository accompanies a
BSc Computer Science thesis at Middle East College and compares three segmentation
architectures under identical experimental conditions on the ISIC 2018 Task 1 dataset
(binary lesion segmentation of dermoscopic images):

1. **UNet** – pure CNN baseline, implemented from scratch.
2. **TransUNet** – hybrid CNN + Vision Transformer encoder.
3. **Swin-UNet** – pure transformer encoder/decoder (Swin-Tiny).

"Identical conditions" means the same data splits, preprocessing, augmentation, loss,
optimiser, epoch/early-stopping rule, random seeds, and hardware for all three models.

## Metrics

- Dice similarity coefficient (DSC) – primary metric
- Intersection over Union (IoU / Jaccard)
- 95th-percentile Hausdorff Distance (HD95)
- Pixel accuracy

Interpretability is provided through Grad-CAM heatmaps for each model. The final
deliverable is a small web dashboard: upload an image, choose a model, view the mask
overlay, and download the predicted mask.

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/` | `base.yaml` with shared hyperparameters plus one YAML per model. |
| `data/` | `raw/` and `processed/` datasets. Contents are gitignored; only the README is committed. |
| `scripts/` | `download_isic2018.py` (dataset) and `download_weights.py` (pretrained weights). |
| `src/datasets/` | ISIC dataset class, split logic, and shared transforms. |
| `src/models/` | `unet.py`, `transunet/`, `swin_unet/` (adapted code with original licence files kept). |
| `src/training/` | `train.py` – the single YAML-driven training entry point. |
| `src/evaluation/` | `metrics.py`, `evaluate.py`, `gradcam.py`. |
| `src/utils/` | Seeding, logging, checkpointing, and device selection. |
| `dashboard/` | Web app (built last). |
| `notebooks/` | Exploratory data analysis only. |
| `results/` | Metrics CSVs and figures. Gitignored except the final summary table. |
| `tests/` | pytest tests and `tests/fixtures/` with tiny mock images/masks. |

## Environment

Target machine: NVIDIA RTX 4070 Ti (12 GB VRAM), AMD Ryzen 7 9800X3D, Windows 11,
Python 3.11, PyTorch (CUDA build). See Task 2 setup notes for exact package versions.

## Training

```
python -m src.training.train --config configs/unet.yaml
```

Swap the config path for `configs/transunet.yaml` or `configs/swin_unet.yaml` to train the
other architectures under the same harness.

## Data

The ISIC 2018 Task 1 data is not distributed with this repository. Use
`scripts/download_isic2018.py` to fetch it from the official ISIC challenge archive. See
`data/README.md` for details.

## Notebooks

The exploratory data analysis lives in `notebooks/01_eda.ipynb`. It runs against the
project virtual environment through a dedicated Jupyter kernel. After creating the venv and
installing `requirements-dev.txt`, register the kernel once:

```
python -m ipykernel install --user --name dermapath-venv --display-name "DermaPath (.venv)"
```

Then run it from the command line (figures are written to `results/figures/eda/`):

```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```
