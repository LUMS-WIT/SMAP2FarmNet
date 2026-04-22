# evaluate_global_model.py
import os
import numpy as np
import torch
import pandas as pd
from config import *
from utils import (
    load_single_csv, make_sequences_per_site,
    load_scalers, load_model_torch, apply_scalers_to_seq,
    compute_classic_metrics, plot_scatter, plot_timeseries, plot_scatter_density, plot_uncertainty_distribution,
    plot_uncertainty_timeseries, plot_time_series_with_uncertainty
)
from model import make_model_mc, make_model
os.makedirs(OUT_DIR, exist_ok=True)


if MODEL_UNCERTAINTY:
    scaler_path = os.path.join(MODEL_DIR, SCALARS_NAME)
    model_path  = os.path.join(MODEL_DIR, MODEL_NAME)

    scaler_x, scaler_r, scaler_y = load_scalers(scaler_path)

    cfg = {
        'input_dim': len(FEATURE_COLS),
        'hidden_dim': 32,
        'num_layers': 1,
        'dropout': 0.2,
        'output_residual': not TRAIN_ON_TRUE_VALUES
    }

    cfg = {'input_dim': len(FEATURE_COLS), 'hidden_dim': 32, 'num_layers': 1, 'output_residual': not TRAIN_ON_TRUE_VALUES}
    model = load_model_torch(make_model_mc, model_path, map_location=DEVICE, weights_only=True, **cfg)
    model.train()
else:
    # --- Load global model & scalers ---
    scaler_path = os.path.join(MODEL_DIR, SCALARS_NAME)
    model_path  = os.path.join(MODEL_DIR, MODEL_NAME)
    scaler_x, scaler_r, scaler_y = load_scalers(scaler_path)

    cfg = {'input_dim': len(FEATURE_COLS), 'hidden_dim': 32, 'num_layers': 1, 'output_residual': not TRAIN_ON_TRUE_VALUES}
    model = load_model_torch(make_model, model_path, map_location=DEVICE, weights_only=True, **cfg)
    model.eval()


def evaluate_site(site_csv_path):
    site_name = os.path.basename(site_csv_path)
    print(f"\n=== Evaluating site: {site_name} ===")

    # load data
    df_site = load_single_csv(site_csv_path, FEATURE_COLS, TARGET_COL, DATE_COL)
    if df_site.empty:
        print(f"No valid data found in {site_csv_path}")
        return None

    # make sequences
    X_seq, y_seq, dates_seq, _ = make_sequences_per_site(
        df_site.assign(site_id=site_name),
        SEQ_LEN, FEATURE_COLS, TARGET_COL, DATE_COL
    )

    if len(X_seq) == 0:
        print(f"Not enough rows to form sequences for {site_name}")
        return None

    # apply scaling
    X_scaled = apply_scalers_to_seq(X_seq, scaler_x) if scaler_x is not None else X_seq

    # predict
    X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
    with torch.no_grad():
        preds_scaled = model(X_tensor).cpu().numpy().ravel()

    # --- Decide prediction path ---
    if TRAIN_ON_TRUE_VALUES:
        preds_all = preds_scaled
        if scaler_y is not None:
            preds_all = scaler_y.inverse_transform(preds_all.reshape(-1,1)).ravel()
    else:
        preds_res = preds_scaled
        if scaler_r is not None:
            preds_res = scaler_r.inverse_transform(preds_res.reshape(-1,1)).ravel()
        baseline_last = X_seq[:, -1, 0]
        preds_all = np.clip(baseline_last + preds_res, 0, 1)

    # compute metrics
    metrics = compute_classic_metrics(y_seq, preds_all)
    print(f"Metrics for {site_name}: {metrics}")
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    # plots
    if PLOT_TIMESERIES:
        baseline_last = X_seq[:, -1, 0] if not TRAIN_ON_TRUE_VALUES else None
        plot_scatter(y_seq, preds_all, title=f"Observed vs Predicted: {site_name}", metrics_text=metrics_text)
        if baseline_last is not None:
            plot_timeseries(dates_seq, y_seq, preds_all, baseline_last, title=f"Time series: {site_name}")

    return metrics


def evaluate_all_sites_in_folder(folder_path, out_metrics_csv=None):
    site_metrics_list = []
    combined_preds = []
    combined_obs = []

    baseline_preds_combined = []  # for baseline persistence
    baseline_obs_combined = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith('.csv'):
            continue
        site_csv = os.path.join(folder_path, fname)
        metrics = evaluate_site(site_csv)
        if metrics is None:
            continue
        site_metrics_list.append({'site': fname, **metrics})

        # load sequences for combined metric
        df_site = load_single_csv(site_csv, FEATURE_COLS, TARGET_COL, DATE_COL)
        X_seq, y_seq, _, _ = make_sequences_per_site(
            df_site.assign(site_id=fname),
            SEQ_LEN, FEATURE_COLS, TARGET_COL, DATE_COL
        )

        # --- MODEL PREDICTIONS ---
        X_scaled = apply_scalers_to_seq(X_seq, scaler_x) if scaler_x is not None else X_seq
        X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
        with torch.no_grad():
            preds_scaled = model(X_tensor).cpu().numpy().ravel()

        if TRAIN_ON_TRUE_VALUES:
            preds_all = preds_scaled
            if scaler_y is not None:
                preds_all = scaler_y.inverse_transform(preds_all.reshape(-1,1)).ravel()
        else:
            preds_res = preds_scaled
            if scaler_r is not None:
                preds_res = scaler_r.inverse_transform(preds_res.reshape(-1,1)).ravel()
            baseline_last = X_seq[:, -1, 0]
            preds_all = np.clip(baseline_last + preds_res, 0, 1)

        combined_preds.extend(preds_all)
        combined_obs.extend(y_seq)

        # --- BASELINE PREDICTIONS ---
        baseline_preds = X_seq[:, -1, 0]  # last-step persistence
        baseline_preds_combined.extend(baseline_preds)
        baseline_obs_combined.extend(y_seq)

    df_metrics = pd.DataFrame(site_metrics_list)

    # overall metrics
    combined_metrics = compute_classic_metrics(np.array(combined_obs), np.array(combined_preds))
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in combined_metrics.items()])
    plot_scatter(np.array(combined_obs), np.array(combined_preds), title="Observed vs Predicted: Combined", metrics_text=metrics_text)
    print("\n=== Combined Metrics Across All Sites ===")
    print("MODEL:", combined_metrics)

    # --- Baseline metrics ---
    baseline_metrics = compute_classic_metrics(np.array(baseline_obs_combined), np.array(baseline_preds_combined))
    baseline_text = "\n".join([f"{k}: {v:.4f}" for k, v in baseline_metrics.items()])
    plot_scatter(np.array(baseline_obs_combined), np.array(baseline_preds_combined), title="Baseline: Last-step Prediction", metrics_text=baseline_text)
    print("BASELINE:", baseline_metrics)
    
    df_metrics.loc['Combined'] = {'site': 'Combined', **combined_metrics}

    if out_metrics_csv:
        df_metrics.to_csv(out_metrics_csv, index=False)
        print(f"Saved metrics to {out_metrics_csv}")

    return df_metrics

def evaluate_site_mc(site_csv_path, n_mc=50):
    site_name = os.path.basename(site_csv_path)
    df_site = load_single_csv(site_csv_path, FEATURE_COLS, TARGET_COL, DATE_COL)
    if df_site.empty: 
        print(f"No data in {site_csv_path}")
        return None

    X_seq, y_seq, dates_seq, _ = make_sequences_per_site(
        df_site.assign(site_id=site_name),
        SEQ_LEN, FEATURE_COLS, TARGET_COL, DATE_COL
    )
    if len(X_seq) == 0: 
        print(f"Not enough samples for sequences at {site_name}")
        return None

    X_scaled = apply_scalers_to_seq(X_seq, scaler_x) if scaler_x is not None else X_seq
    X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
    baseline_last = X_seq[:, -1, 0]

    # ---- MC Dropout ----
    model.train()  # enable dropout
    preds_mc = []

    for _ in range(n_mc):
        with torch.no_grad():
            preds_scaled = model(X_tensor).cpu().numpy().ravel()

        if TRAIN_ON_TRUE_VALUES:
            preds = preds_scaled
            if USE_SCALING and scaler_y is not None:
                preds = scaler_y.inverse_transform(preds.reshape(-1,1)).ravel()
        else:
            preds_res = preds_scaled
            if USE_SCALING and scaler_r is not None:
                preds_res = scaler_r.inverse_transform(preds_res.reshape(-1,1)).ravel()
            preds = np.clip(baseline_last + preds_res, 0, 1)

        preds_mc.append(preds)

    preds_mc = np.stack(preds_mc, axis=0)
    preds_mean = preds_mc.mean(axis=0)
    preds_std  = preds_mc.std(axis=0)

    unc_mean = preds_std.mean()
    unc_ts = {"dates": dates_seq, "uncertainty": preds_std}

    metrics = compute_classic_metrics(y_seq, preds_mean)
    
    return metrics, preds_mean, preds_std, y_seq, unc_mean, unc_ts


def evaluate_all_sites_in_folder_mc(folder_path, n_mc=50, save_csv=None):
    site_metrics_list = []
    combined_preds, combined_obs, combined_unc = [], [], []
    unc_mean_per_site, unc_ts_per_site = {}, {}

    # Baseline persistence predictions
    baseline_preds_combined = []   
    baseline_obs_combined = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".csv"):
            continue
        site_csv = os.path.join(folder_path, fname)
        out = evaluate_site_mc(site_csv, n_mc=n_mc)
        if out is None: 
            continue

        metrics, preds_mean, preds_std, obs, unc_mean, unc_ts = out
        site_metrics_list.append({'site': fname, **metrics, 'unc_mean': unc_mean})
        unc_mean_per_site[fname] = unc_mean
        unc_ts_per_site[fname] = unc_ts

        combined_preds.extend(preds_mean)
        combined_obs.extend(obs)
        combined_unc.extend(preds_std)

        # ===== BASELINE SECTION =====
        # Load raw CSV to reconstruct sequences for baseline
        df_site = load_single_csv(site_csv, FEATURE_COLS, TARGET_COL, DATE_COL)
        X_seq, y_seq, _, _ = make_sequences_per_site(
            df_site.assign(site_id=fname),
            SEQ_LEN, FEATURE_COLS, TARGET_COL, DATE_COL
        )

        # last-step value as baseline prediction
        baseline_preds = X_seq[:, -1, 0]
        baseline_preds_combined.extend(baseline_preds)
        baseline_obs_combined.extend(y_seq)
        # ============================================================

    df_metrics = pd.DataFrame(site_metrics_list)
    combined_metrics = compute_classic_metrics(np.array(combined_obs), np.array(combined_preds))

    # # metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in combined_metrics.items()])
    # metrics_text = "\n".join([
    #     f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
    #     for k, v in combined_metrics.items()
    # ])

    plot_scatter_density(np.array(combined_obs), np.array(combined_preds), title="Observed vs Predicted", 
                         metrics_dict=combined_metrics,include_rmse=False)

                        #  metrics_text=metrics_text)

    combined_metrics["unc_mean"] = np.mean(combined_unc)
    print("\n=== Combined Metrics Across All Sites ===")
    print("MODEL:", combined_metrics)


    baseline_preds_combined = np.array(baseline_preds_combined)
    baseline_obs_combined = np.array(baseline_obs_combined)

    # ---- Baseline Metrics ----
    baseline_metrics = compute_classic_metrics(baseline_obs_combined, baseline_preds_combined)

    # baseline_text = "\n".join([f"{k}: {v:.4f}" for k, v in baseline_metrics.items()])
    # plot_scatter_density(baseline_obs_combined, baseline_preds_combined,
    #                      title="Baseline: Last-step (Persistence)",
    #                      metrics_text=baseline_text)

    print("\n=== Baseline Metrics (Persistence) ===")
    print("BASELINE:", baseline_metrics)


    # ===== ADD TWO SUMMARY ROWS TO CSV =====
    combined_row = pd.DataFrame(
        {"site": ["Combined"], **{k: [v] for k, v in combined_metrics.items()}}
    ).set_index("site")

    baseline_row = pd.DataFrame(
        {"site": ["Baseline"], **{k: [v] for k, v in baseline_metrics.items()}}
    ).set_index("site")

    full_df = pd.concat(
        [df_metrics.set_index("site"), combined_row, baseline_row],
        axis=0
    )

    if save_csv:
        full_df.to_csv(save_csv)
        print(f"Saved per-site + combined + baseline MC metrics to: {save_csv}")
    return df_metrics, combined_metrics, unc_mean_per_site, unc_ts_per_site, combined_unc



if __name__ == '__main__':
    
    if MODEL_UNCERTAINTY:
        site_csv = os.path.join(DATA_FOLDER, '2206_NARC_A07_Plot01_Pear.csv')

        metrics, preds_mean, preds_std, y_true, unc_mean, unc_ts = evaluate_site_mc(
            site_csv,
            n_mc=50  # number of MC dropout passes
        )

        # Print results
        print("Metrics:", metrics)
        print("Mean predictive uncertainty:", unc_mean)

        site = os.path.basename(site_csv)
        dates = unc_ts["dates"]         # array of dates
        mask = np.arange(len(dates))    # all indices

        plot_time_series_with_uncertainty(
        dates=dates,
        y_true=y_true[mask],
        y_pred_mean=preds_mean[mask],
        y_pred_std=preds_std[mask],
        site_name=site,
        save_path=f"{OUT_DIR}/{site}.png"
        )

        df_metrics, combined_metrics, unc_mean_per_site, unc_ts_per_site, combined_unc = evaluate_all_sites_in_folder_mc(
            TEST_DATA_FOLDER,
            n_mc=50,
            save_csv=os.path.join(OUT_DIR, f"{MODE}_metrics_per_site_mc_test.csv")
        )
        
        plot_uncertainty_distribution(combined_unc, title="Model Uncertainty Distribution")

    else:
    
        evaluate_site(os.path.join(DATA_FOLDER, '2310_Sultan_A13_Plot06_Citrus.csv'))  # example for single site
        
        PLOT_TIMESERIES = False  # disable plots for all-sites eval
        metrics_df = evaluate_all_sites_in_folder(
            DATA_FOLDER, 
            out_metrics_csv=os.path.join(OUT_DIR, f"{MODE}_metrics_per_site_test.csv")
        )

      # enable MC Dropout evaluation
