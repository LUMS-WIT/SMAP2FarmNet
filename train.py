# train_updated.py
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import *
from utils import (
    load_all_sites, make_sequences_per_site, create_residuals,
    fit_scalers, apply_scalers_to_seq, save_model_torch, save_scalers,
    compute_classic_metrics, plot_uncertainty_timeseries, plot_uncertainty_distribution,
    plot_time_series_with_uncertainty
)
from model import LSTM, make_model_mc

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def train_global_model():
    # --- Load all site CSVs ---
    df_all = load_all_sites(DATA_FOLDER, FEATURE_COLS, TARGET_COL, DATE_COL)
    
    # --- Make sequences ---
    X_seq_raw, y_seq_raw, _, _ = make_sequences_per_site(
        df_all, SEQ_LEN, FEATURE_COLS, TARGET_COL, DATE_COL
    )

    # --- Decide target ---
    if TRAIN_ON_TRUE_VALUES:
        target_raw = y_seq_raw.reshape(-1,1)
        baseline_last_raw = None
        residuals_raw = None
        # --- Fit scalers ---
        if USE_SCALING:
            scaler_x = StandardScaler().fit(X_seq_raw.reshape(-1, X_seq_raw.shape[2]))
            scaler_y = StandardScaler().fit(target_raw)
            scaler_r = None
    else:
        residuals_raw, baseline_last_raw = create_residuals(X_seq_raw, y_seq_raw)
        target_raw = residuals_raw.reshape(-1,1)
        # --- Fit scalers ---
        if USE_SCALING:
            scaler_x, scaler_r, scaler_y = fit_scalers(X_seq_raw, residuals_raw, y_seq_raw)

    # --- Apply scaling ---
    X_seq_scaled = apply_scalers_to_seq(X_seq_raw, scaler_x) if USE_SCALING else X_seq_raw
    y_scaled = None
    if TRAIN_ON_TRUE_VALUES:
        y_scaled = scaler_y.transform(target_raw) if USE_SCALING else target_raw
    else:
        y_scaled = scaler_r.transform(target_raw) if USE_SCALING else target_raw

    X_tensor = torch.from_numpy(X_seq_scaled).float().to(DEVICE)
    y_tensor = torch.from_numpy(y_scaled).float().to(DEVICE)

    # --- Initialize model ---
    model = LSTM(input_dim=X_tensor.shape[2], hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    # --- Training loop ---
    print(f"Training global model for {EPOCHS} epochs on all sites...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.6f}")
    
    # --- Save final model & scalers ---
    save_model_torch(model, os.path.join(MODEL_DIR, MODEL_NAME))
    save_scalers((scaler_x, scaler_r, scaler_y), os.path.join(MODEL_DIR, SCALARS_NAME))
    
    print("Training complete. Model and scalers saved.")

    return model, (scaler_x, scaler_r, scaler_y), df_all, baseline_last_raw

# --- Per-site evaluation ---
def evaluate_per_site(model, scalers, df_all):
    scaler_x, scaler_r, scaler_y = scalers
    X_seq, y_seq, dates_seq, site_seq = make_sequences_per_site(
        df_all, SEQ_LEN, FEATURE_COLS, TARGET_COL, DATE_COL
    )
    baseline_last = X_seq[:, -1, 0]  # last feature

    if USE_SCALING:
        X_seq_scaled = apply_scalers_to_seq(X_seq, scaler_x)
    else:
        X_seq_scaled = X_seq

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(X_seq_scaled).float().to(DEVICE)).cpu().numpy()
    
    # --- Decide prediction path ---
    if TRAIN_ON_TRUE_VALUES:
        preds_all = preds_scaled.ravel()
        if USE_SCALING:
            preds_all = scaler_y.inverse_transform(preds_all.reshape(-1,1)).ravel()
    else:
        preds_residual = preds_scaled.ravel()
        if USE_SCALING:
            preds_residual = scaler_r.inverse_transform(preds_residual.reshape(-1,1)).ravel()
        preds_all = np.clip(baseline_last + preds_residual, 0, 1)

    # --- Compute per-site metrics ---
    metrics_per_site = {}
    for site in np.unique(site_seq):
        mask = site_seq == site
        y_true = y_seq[mask]
        y_pred = preds_all[mask]
        metrics_per_site[site] = compute_classic_metrics(y_true, y_pred)
    
    # --- Save metrics ---
    metrics_df = pd.DataFrame.from_dict(metrics_per_site, orient='index')
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_per_site_train.csv"))
    print("Per-site metrics saved to metrics_per_site_train.csv")

    # --- Global metrics ---
    global_metrics = compute_classic_metrics(y_seq, preds_all)
    print("Global metrics:", global_metrics)
    return metrics_df, global_metrics, preds_all, y_seq


def train_global_model_mc():
    df_all = load_all_sites(DATA_FOLDER, FEATURE_COLS, TARGET_COL, DATE_COL)

    # --- Build sequences ---
    X_seq_raw, y_seq_raw, _, _ = make_sequences_per_site(
        df_all, SEQ_LEN, FEATURE_COLS, TARGET_COL, DATE_COL
    )
    residuals_raw, baseline_last_raw = create_residuals(X_seq_raw, y_seq_raw)

    # ============================
    # ► CHOOSE TARGET BASED ON TRAIN_ON_TRUE_VALUES
    # ============================
    if TRAIN_ON_TRUE_VALUES:
        train_target_raw = y_seq_raw.reshape(-1, 1)
    else:
        train_target_raw = residuals_raw.reshape(-1, 1)

    # ============================
    # ► SCALING
    # ============================
    if USE_SCALING:
        scaler_x, scaler_r, scaler_y = fit_scalers(
            X_seq_raw, residuals_raw, y_seq_raw
        )

        X_scaled = apply_scalers_to_seq(X_seq_raw, scaler_x)

        if TRAIN_ON_TRUE_VALUES:
            y_scaled = scaler_y.transform(train_target_raw)
        else:
            y_scaled = scaler_r.transform(train_target_raw)
    else:
        X_scaled = X_seq_raw
        y_scaled = train_target_raw
        scaler_x = scaler_r = scaler_y = None

    # --- Convert to tensors ---
    X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
    y_tensor = torch.from_numpy(y_scaled).float().to(DEVICE)

    # --- Model Config ---
    cfg = {
        'input_dim': X_tensor.shape[2],
        'hidden_dim': 32,
        'num_layers': 1,
        'dropout': 0.2,
        'output_residual': not TRAIN_ON_TRUE_VALUES
    }
    model = make_model_mc(**cfg).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    print(f"Training global MC model ({'TRUE' if TRAIN_ON_TRUE_VALUES else 'RESIDUAL'}) for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss.item():.6f}")

    # --- Save Model & Scalers ---
    save_model_torch(model, os.path.join(MODEL_DIR, MODEL_NAME))
    save_scalers((scaler_x, scaler_r, scaler_y), os.path.join(MODEL_DIR, SCALARS_NAME))

    print("MC model training complete.")

    return model, (scaler_x, scaler_r, scaler_y), df_all, baseline_last_raw

def evaluate_per_site_mc(model, scalers, df_all, n_mc=50, save_csv=None):
    """
    MC-dropout evaluation:
    - Multiple stochastic forward passes (n_mc)
    - Supports TRAIN_ON_TRUE_VALUES = True/False
    - Returns mean prediction, uncertainty, and metrics
    """
    scaler_x, scaler_r, scaler_y = scalers

    # --- Build sequences ---
    X_seq, y_seq, dates_seq, site_seq = make_sequences_per_site(
        df_all, SEQ_LEN, FEATURE_COLS, TARGET_COL, DATE_COL
    )

    baseline_last = X_seq[:, -1, 0]

    # --- Apply scaling ---
    if USE_SCALING:
        X_seq_scaled = apply_scalers_to_seq(X_seq, scaler_x)
    else:
        X_seq_scaled = X_seq

    X_tensor = torch.from_numpy(X_seq_scaled).float().to(DEVICE)

    # =====================================================
    #              MC DROP OUT PASSES
    # =====================================================
    model.train()   # enable dropout

    preds_mc = []
    for _ in range(n_mc):
        with torch.no_grad():
            preds_scaled = model(X_tensor).cpu().numpy().ravel()

        # =====================================
        #  TRUE-VALUE PREDICTION PATH
        # =====================================
        if TRAIN_ON_TRUE_VALUES:
            preds = preds_scaled

            if USE_SCALING:
                preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()

        # =====================================
        #  RESIDUAL PREDICTION PATH
        # =====================================
        else:
            preds_res = preds_scaled

            if USE_SCALING:
                preds_res = scaler_r.inverse_transform(preds_res.reshape(-1, 1)).ravel()

            preds = np.clip(baseline_last + preds_res, 0, 1)

        preds_mc.append(preds)

    # stack MC predictions
    preds_mc = np.stack(preds_mc, axis=0)   # (n_mc, N)
    preds_mean = preds_mc.mean(axis=0)
    preds_std  = preds_mc.std(axis=0)

    # =====================================================
    #              PER-SITE METRICS + UNCERTAINTY
    # =====================================================
    sites = np.unique(site_seq)
    metrics_per_site = {}
    unc_mean_per_site = {}        # NEW**
    unc_ts_per_site = {}   

    for site in np.unique(site_seq):
        mask = site_seq == site
        metrics_per_site[site] = compute_classic_metrics(
            y_seq[mask], preds_mean[mask]
        )
        # --- per-site mean uncertainty ---
        unc_mean_per_site[site] = preds_std[mask].mean()
        unc_ts_per_site[site] = {
            "dates": dates_seq[mask],
            "uncertainty": preds_std[mask]
        }

    metrics_df = pd.DataFrame.from_dict(metrics_per_site, orient="index")
    metrics_df["unc_mean"] = metrics_df.index.map(unc_mean_per_site)


    global_metrics = compute_classic_metrics(y_seq, preds_mean)
    global_metrics["unc_mean"] = preds_std.mean()

    # =====================================================
    #              SAVE CSV 
    # ====================================================

    combined_row = pd.DataFrame(
        {"site": ["Combined"], **{k:[v] for k,v in global_metrics.items()}}
    ).set_index("site")

    full_df = pd.concat([metrics_df, combined_row], axis=0)
    if save_csv is not None:
        full_df.to_csv(save_csv)
        print(f"Saved per-site + combined MC metrics to: {save_csv}")

    # ===========================
    # Print Output
    # ===========================
    print("\n===== Monte-Carlo Dropout Evaluation =====")
    print(f"MC Samples: {n_mc}")
    print("Global Metrics:")
    for k, v in global_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nMean predictive uncertainty (std of MC samples): {preds_std.mean():.4f}")

    return (
        metrics_df, global_metrics,
        preds_mean, preds_std, preds_mc,
        y_seq, site_seq, 
        unc_mean_per_site, unc_ts_per_site
    )


# =============================
if __name__ == "__main__":
    # model, scalers, df_all, baseline_last = train_global_model()
    # metrics_df, global_metrics, preds, obs = evaluate_per_site(model, scalers, df_all)

    model_mc, scalers_mc, df_all, baseline_last = train_global_model_mc()

    (metrics_df, global_metrics,
    preds_mean, preds_std, preds_mc,
    y_true, site_seq,
    unc_mean_per_site, unc_ts_per_site) = evaluate_per_site_mc(model_mc, scalers_mc, df_all, n_mc=50, 
                                                               save_csv=os.path.join(OUT_DIR, f"{MODE}_metrics_per_site_train.csv"))

    # Plot a site:
    site = list(unc_ts_per_site.keys())[0]
    plot_uncertainty_timeseries(
        unc_ts_per_site[site]["dates"],
        unc_ts_per_site[site]["uncertainty"],
        title=f"Uncertainty TS – {site}"
    )

    site = list(unc_ts_per_site.keys())[0]
    mask = site_seq == site
    dates = unc_ts_per_site[site]["dates"]

    plot_time_series_with_uncertainty(
        dates=dates,
        y_true=y_true[mask],
        y_pred_mean=preds_mean[mask],
        y_pred_std=preds_std[mask],
        site_name=site,
        save_path=f"{OUT_DIR}/uncertainty_site_{site}.png"
    )


    plot_uncertainty_distribution(preds_std, title="Global Uncertainty Distribution")



#####
# =============================
# HOLDOUT SET
# =============================


# test set results for AM_9km feature, 5000 epochs
# ===== Monte-Carlo Dropout Evaluation =====
# Global Metrics:
#   Bias: -0.0010
#   MSE: 0.0015
#   RMSE: 0.0383
#   ubRMSE: 0.0383
#   Pearson r: 0.8032
#   NSE: 0.6425
#   KGE: 0.6854
#   N: 2972.0000
#   unc_mean: 0.0089

# Mean predictive uncertainty (std of MC samples): 0.0088

# test set results for PM_9km feature, 5000 epochs
# ===== Monte-Carlo Dropout Evaluation =====
# Global Metrics:
#   Bias: 0.0003
#   MSE: 0.0015
#   RMSE: 0.0384
#   ubRMSE: 0.0384
#   Pearson r: 0.8034
#   NSE: 0.6448
#   KGE: 0.7020
#   N: 2693.0000
#   unc_mean: 0.0079

# Mean predictive uncertainty (std of MC samples): 0.0083

# test set results for SM_9km feature, 5000 epochs
# ===== Monte-Carlo Dropout Evaluation =====
# Global Metrics:
#   Bias: 0.0001
#   MSE: 0.0015
#   RMSE: 0.0393
#   ubRMSE: 0.0393
#   Pearson r: 0.7853
#   NSE: 0.6139
#   KGE: 0.6568
#   N: 3858.0000
#   unc_mean: 0.0094

# Mean predictive uncertainty (std of MC samples): 0.0094

# =============================
# COMPLETE SET


# # AM_9km feature, 5000 epochs
# MC Samples: 50
# Global Metrics:
#   Bias: -0.0007
#   MSE: 0.0015
#   RMSE: 0.0384
#   ubRMSE: 0.0384
#   Pearson r: 0.8067
#   NSE: 0.6502
#   KGE: 0.7119
#   N: 3695.0000
#   unc_mean: 0.0089


# PM_9km feature, 5000 epochs
# ===== Monte-Carlo Dropout Evaluation =====
# MC Samples: 50
# Global Metrics:
#   Bias: 0.0012
#   MSE: 0.0015
#   RMSE: 0.0388
#   ubRMSE: 0.0388
#   Pearson r: 0.8015
#   NSE: 0.6413
#   KGE: 0.6984
#   N: 3364.0000
#   unc_mean: 0.0081

# Mean predictive uncertainty (std of MC samples): 0.0081

# SM_9km feature, 5000 epochs
# ===== Monte-Carlo Dropout Evaluation =====
# MC Samples: 50
# Global Metrics:
#   Bias: 0.0001
#   MSE: 0.0015
#   RMSE: 0.0393
#   ubRMSE: 0.0393
#   Pearson r: 0.7971
#   NSE: 0.6342
#   KGE: 0.6878
#   N: 5359.0000
#   unc_mean: 0.0094