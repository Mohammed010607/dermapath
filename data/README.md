# Data

This directory holds the ISIC 2018 Task 1 dataset. **Nothing here except this README is
committed to git** — the raw and processed data are large and are excluded via
`.gitignore`.

## Layout

- `raw/` – the unmodified files downloaded from the official ISIC 2018 challenge archive
  (dermoscopic images plus binary ground-truth masks), organised by split.
- `processed/` – any intermediate artefacts produced by the data pipeline.

## How to obtain the data

Run the download script from the repository root:

```
python scripts/download_isic2018.py
```

It downloads the official ISIC 2018 Task 1 files, unzips them into `raw/`, and prints a
verification report (file counts per split, image/mask pairing, resolution range, mask
value range, and any corrupt or mismatched files).

The dataset is provided by the ISIC 2018 challenge and is subject to the licence and terms
published on the official challenge site. Refer to that site for citation requirements.
