# SMAP2FarmNet — Farm‑scale Soil Moisture Retrieval (IGARSS 2026)

Code repository for:

> Rafique, Hamza and Muhammad, Abubakr, **“A deep learning framework for farm-scale soil moisture retrievals: A case study for a data-scarce region,”** IGARSS 2026 — 2026 IEEE International Geoscience and Remote Sensing Symposium.

This repo provides a Python (PyTorch) workflow to train and evaluate an LSTM-based model (with Monte‑Carlo Dropout uncertainty) for predicting soil moisture using SMAP coarse-resolution (9 km) soil moisture features.

---

## Table of Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Data](#data)
  - [Expected CSV format](#expected-csv-format)
  - [Folder layout](#folder-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [1) Configure experiment](#1-configure-experiment)
  - [2) Train](#2-train)
  - [3) Evaluate](#3-evaluate)
  - [4) Farm-level aggregation (optional)](#4-farm-level-aggregation-optional)
- [Configuration reference (`config.py`)](#configuration-reference-configpy)
- [Model notes](#model-notes)
- [Outputs](#outputs)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

At a high level, this project:

1. Loads multiple site CSV files (each CSV corresponds to a sensor/site).
2. Builds sliding-window sequences of length `SEQ_LEN` per site.
3. Trains an LSTM to predict **soil moisture** (scaled to **m³/m³** internally).
4. Runs **Monte‑Carlo Dropout** at inference time to quantify predictive uncertainty.
5. Saves trained model weights, scalers, and evaluation metrics/plots.

Key scripts:

- `train.py`: training (standard and MC-dropout variants) + per-site metrics on the training set.
- `test.py`: evaluation on a folder of site CSVs (standard and MC-dropout variants).
- `witsms_farm_stats.py`: aggregates sensor-level metrics to farm-level metrics and makes boxplots.
- `config.py`: experiment configuration (data folders, feature selection, model mode, etc.).
- `model.py`: LSTM model definitions (standard + MC-dropout).
- `utils.py`: data loading, sequence generation, scaling, metrics, plotting, cleaning utilities.

---

## Repository structure

```text
.
├── README.md
├── config.py
├── model.py
├── train.py
├── test.py
├── utils.py
├── witsms_farm_stats.py
├── training/
│   ├── train/        # (you provide) holdout training set CSVs
│   ├── test/         # (you provide) holdout test set CSVs
│   └── complete/     # (optional) full dataset CSVs
└── checkpoints/
    ├── holdout/      # saved models/scalers for holdout runs
    └── complete/     # saved models/scalers for complete runs
```

Notes:
- `training/` is expected to contain your CSV files (see [Data](#data)).
- `checkpoints/` will be created/used automatically for model + scaler.
- Results are written into `results/holdout` or `results/complete` depending on configuration.

---

## Data

### Expected CSV format

Each site CSV should contain:

- A timestamp column (default): `TimeStamp`
- One target column: `VolumetricWaterContent1`
- One or more feature columns depending on `MODE` (see `config.py`):
  - `SM_AM_9km` for AM mode
  - `SM_PM_9km` for PM mode
  - `SM_9km` for combined mode

### Folder layout

Controlled by `SET` in `config.py`:

- If `SET = "holdout"`:
  - `training/train/` contains training CSVs
  - `training/test/` contains test CSVs

- If `SET = "complete"`:
  - `training/complete/` contains CSVs used for both training and evaluation

---

## Installation

### Requirements

This is a Python-only project using common scientific libraries and PyTorch.

Recommended:
- Python 3.9+ (or newer)
- PyTorch (CPU or CUDA)
- NumPy, Pandas, scikit-learn, SciPy, Matplotlib

### Create environment (example)

Using `venv`:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1
```

Install dependencies (typical):

```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib torch
```

If you use CUDA, install the correct PyTorch build per the official PyTorch instructions.

---

## Quickstart

### 1) Configure experiment

Edit `config.py`:

- `MODE`: `"AM"`, `"PM"`, or `"comb"`
- `SET`: `"holdout"` or `"complete"`
- `SEQ_LEN`: sequence length (default 7)
- `EPOCHS`: training epochs (default 5000)
- `MODEL_UNCERTAINTY`: `True` for MC-dropout model, else standard model
- `TRAIN_ON_TRUE_VALUES`:
  - `True`: train to predict soil moisture directly
  - `False`: train to predict residuals w.r.t. persistence baseline (see notes below)
- `USE_SCALING`: standardize inputs/targets/residuals using `StandardScaler`

Also verify these are correct for your data:
- `TARGET_COL = "VolumetricWaterContent1"`
- `DATE_COL = "TimeStamp"`
- `FEATURE_COLS` are set automatically based on `MODE`

### 2) Train

Run:

```bash
python train.py
```

What it does (default path in `train.py` main):
- Trains an MC-dropout model using `train_global_model_mc()`
- Evaluates per-site on the dataset it trained on via `evaluate_per_site_mc(...)`
- Writes metrics CSV into `OUT_DIR`
- Saves model and scalers into `MODEL_DIR`

### 3) Evaluate

Run:

```bash
python test.py
```

What it does:
- Loads the saved model + scalers from `MODEL_DIR`
- Evaluates:
  - Either deterministic evaluation, or MC-dropout evaluation (depending on `MODEL_UNCERTAINTY`)
- For MC-dropout evaluation on all sites, it uses `TEST_DATA_FOLDER` (from `config.py`)
- Saves a metrics CSV into `OUT_DIR` with a name like:
  - `results/holdout/AM_metrics_per_site_mc_test.csv` (example)

### 4) Farm-level aggregation (optional)

If you produced a sensor-level metrics CSV from `test.py`, you can aggregate to farm-level metrics:

```bash
python witsms_farm_stats.py
```

This script expects:
- Input metrics CSV path based on `OUT_DIR` and `MODE`, e.g.:
  - `results/holdout/AM_metrics_per_site_mc_test.csv`

It produces:
- Farm-level aggregated metrics CSV in:
  - `results/<set>/farmlevel/`
- Boxplots saved alongside the CSV.

---

## Configuration reference (`config.py`)

Key settings:

- **Experiment selection**
  - `MODE = "AM" | "PM" | "comb"`
  - `SET = "holdout" | "complete"`

- **Data locations**
  - `DATA_FOLDER`:
    - holdout: `training/train`
    - complete: `training/complete`
  - `TEST_DATA_FOLDER`:
    - holdout: `training/test`
    - complete: same as `DATA_FOLDER`

- **Model/training**
  - `HIDDEN_DIM`, `NUM_LAYERS`, `LR`, `EPOCHS`
  - `TRAIN_ON_TRUE_VALUES`: train on true values vs residuals
  - `MODEL_UNCERTAINTY`: use MC-dropout model or standard model

- **Scaling**
  - `USE_SCALING`: uses `StandardScaler` on inputs and targets/residuals

- **Device**
  - `DEVICE = cuda if available else cpu`

- **Output paths**
  - `OUT_DIR`: results folder
  - `MODEL_DIR`: checkpoints folder
  - `MODEL_NAME`, `SCALARS_NAME`: depend on `MODE`

---

## Model notes

### Standard LSTM vs MC-dropout LSTM

- Standard model is `LSTM` via `make_model(...)` in `model.py`
- MC-dropout model is `LSTM_MC` via `make_model_mc(...)` in `model.py`
  - During MC evaluation, the model is set to `.train()` mode to activate dropout.
  - Multiple forward passes are taken (`n_mc`), then:
    - mean prediction: `preds_mean`
    - uncertainty proxy: `preds_std` (std across MC samples)

### Predicting true values vs residuals

The repo supports two target formulations:

- `TRAIN_ON_TRUE_VALUES = True`:
  - Model predicts soil moisture directly.

- `TRAIN_ON_TRUE_VALUES = False`:
  - A persistence-like baseline is computed:
    - baseline = last value of the last timestep feature (`X_seq[:, -1, 0]`)
  - The model predicts the residual:
    - residual = y_true - baseline
  - Final prediction is:
    - y_pred = clip(baseline + residual_pred, 0, 1)

Make sure the first (or only) feature column is the intended baseline series if using residual mode.

---

## Outputs

Depending on run mode, you should expect:

### Saved artifacts

- Model weights:
  - `checkpoints/<set>/<MODE>_model_mc.pt` (or similar)
- Scalers:
  - `checkpoints/<set>/<MODE>_scalars.pkl`

### Metrics (CSV)

Examples:
- Training per-site metrics:
  - `results/<set>/AM_metrics_per_site_train.csv` (name depends on `MODE`)
- Test per-site metrics (MC):
  - `results/<set>/AM_metrics_per_site_mc_test.csv`

### Plots

Generated by evaluation scripts depending on flags and calls, for example:
- Scatter plots (observed vs predicted)
- Uncertainty distribution
- Uncertainty time series
- Time series with uncertainty band

## Citation

If you use this code, please cite the IGARSS 2026 paper:

```bibtex
@inproceedings{rafique2026smap2farmnet,
  title     = {A deep learning framework for farm-scale soil moisture retrievals: A case study for a data-scarce region},
  author    = {Rafique, Hamza and Muhammad, Abubakr},
  booktitle = {Proceedings of the 2026 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year      = {2026},
  note      = {Code: https://github.com/LUMS-WIT/SMAP2FarmNet}
}
```

---

## Data Access

The soil moisture sensor data used for this research is under the ownership of Centre for Water Informatics & Technology [(WIT)](https://wit.lums.edu.pk/SoilMoistureSensor) LUMS, Lahore

---

## Contact

For questions or collaborations, please contact the the authors via [LinkdIn] (www.linkedin.com/in/hamza-rafique-ac952) or [email](22060019@lums.edu.pk).
