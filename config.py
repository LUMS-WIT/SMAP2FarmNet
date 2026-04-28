# config.py

import torch

MODE = "AM" # "PM", "comb", "AM"
SET = "holdout" # "holdout", "complete"

# ---------------- DATA ----------------
if SET == "holdout":
    DATA_FOLDER = "training/train"          # folder containing CSV(s)
    TEST_DATA_FOLDER = "training/test"
elif SET == "complete":
    DATA_FOLDER = "training/complete"       # folder containing CSV(s)
    TEST_DATA_FOLDER = DATA_FOLDER  # can be same as training for complete set

TARGET_COL = "VolumetricWaterContent1"
DATE_COL = "TimeStamp"
SEQ_LEN = 7
if MODE == "AM":
    FEATURE_COLS = ["SM_AM_9km"]
elif MODE == "PM":
    FEATURE_COLS = ["SM_PM_9km"]
else:
    FEATURE_COLS = ["SM_9km"]

# ---------------- MODEL ----------------
HIDDEN_DIM = 32
NUM_LAYERS = 1
LR = 0.01
EPOCHS = 5000
TRAIN_ON_TRUE_VALUES = True    # True: train on true values, False: train on residuals

MODEL_UNCERTAINTY = True  # True: use MC Dropout model, False: use standard model

# ---------------- SCALING & DEVICE ----------------
USE_SCALING = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- OUTPUT ----------------
if SET == "holdout":
    OUT_DIR = "results\holdout"
    MODEL_DIR = "checkpoints\holdout"
elif SET == "complete":
    OUT_DIR = "results\complete"
    MODEL_DIR = "checkpoints\complete"

if MODE == "AM":
    MODEL_NAME = "AM_model_mc.pt"
    SCALARS_NAME = "AM_scalars.pkl"
elif MODE == "PM":
    MODEL_NAME = "PM_model_mc.pt"
    SCALARS_NAME = "PM_scalars.pkl"
else:    
    MODEL_NAME = "comb_model_mc.pt"
    SCALARS_NAME = "comb_scalars.pkl"

PLOT_TIMESERIES = False

# --------------- Extras ---------------
METRIC_UNITS = {
    "RMSE": "m³/m³",
    "ubRMSE": "m³/m³",
    "MSE": "m³/m³",
    "Bias": "m³/m³",
    "NSE": "-",
    "KGE": "-",
    "Pearson r": "-"
}




