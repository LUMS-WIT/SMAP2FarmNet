import pandas as pd
import os
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
from config import METRIC_UNITS


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#------------------- DATA CLEANING UTILITIES : witsms_clean.py ------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def remove_nan_and_bounds(df, val_col, lower=0.0, upper=100.0):
    """
    Remove rows with NaN in val_col or values outside [lower, upper].
    Returns a new dataframe (reset index).
    """
    df_clean = df.copy().reset_index(drop=True)
    mask = df_clean[val_col].notnull() & (df_clean[val_col] >= lower) & (df_clean[val_col] <= upper)
    return df_clean.loc[mask].reset_index(drop=True)


def remove_single_jump(df, val_col, pct_threshold=0.10, neighbor_similarity_tol=0.03):
    """
    Remove single isolated values that differ from BOTH immediate neighbors by > pct_threshold (relative).
    Also require the two neighbors themselves to be similar (so we don't remove gradual trends).
    Args:
        df: sorted by time
        val_col: column name
        pct_threshold: e.g. 0.10 for 10%
        neighbor_similarity_tol: tolerance for neighbors to be considered 'similar' (fraction)
    Returns:
        df with single-point anomalies removed
    """
    df = df.copy().reset_index(drop=True)
    vals = df[val_col].values
    n = len(vals)
    remove = set()

    for i in range(1, n-1):  # skip first and last (no two-sided neighborhood)
        left = float(vals[i-1])
        mid  = float(vals[i])
        right= float(vals[i+1])

        # neighbors must be reasonably similar (so we are isolating a single spike)
        denom_neighbors = max(abs(np.median([left, right])), 1e-6)
        neighbors_rel_diff = abs(left - right) / denom_neighbors

        if neighbors_rel_diff > neighbor_similarity_tol:
            # neighbors not similar enough -> don't mark a single-point removal
            continue

        # compare mid to median of neighbors
        neighbor_median = np.median([left, right])
        denom = neighbor_median if abs(neighbor_median) > 1e-6 else 1.0  # avoid /0
        pct_diff = abs(mid - neighbor_median) / abs(denom)

        if pct_diff > pct_threshold:
            remove.add(i)

    if not remove:
        return df

    return df.drop(index=list(remove)).reset_index(drop=True)



def remove_consecutive_jump(df, val_col, pct_threshold=0.10, 
                                     neighbor_similarity_tol=0.03,
                                     max_run_length=3):
    """
    Remove runs of 2 to max_run_length consecutive points that deviate strongly
    from their surrounding neighbors.

    Args:
        df: DataFrame sorted by time
        val_col: column name of values
        pct_threshold: percentage deviation threshold (0.10 = 10%)
        neighbor_similarity_tol: tolerance for neighbors to be considered similar
        max_run_length: maximum run length to consider (2=pair, 3=triplet)

    Returns:
        df with those runs removed
    """
    df = df.copy().reset_index(drop=True)
    vals = df[val_col].values
    n = len(vals)
    remove = set()

    i = 1
    while i < n - 2:   # need space for left and right neighbors
        left = float(vals[i-1])

        # For each run length L = 2..max_run_length
        removed_any = False
        for L in range(2, max_run_length+1):

            # Ensure enough points remain to test run
            if i + L >= n:
                continue

            right = float(vals[i+L])

            # Are neighbors similar?
            denom_neighbors = max(abs(np.median([left, right])), 1e-6)
            neighbors_rel_diff = abs(left - right) / denom_neighbors

            if neighbors_rel_diff > neighbor_similarity_tol:
                continue  # trend, not an anomaly run

            # Check all points in the run
            neighbor_median = np.median([left, right])
            denom = neighbor_median if abs(neighbor_median) > 1e-6 else 1.0

            run_vals = vals[i:i+L]  # L consecutive candidate values
            pct_diffs = [abs(v - neighbor_median)/abs(denom) for v in run_vals]

            # If ALL L points exceed 10% deviation → remove entire run
            if all(p > pct_threshold for p in pct_diffs):
                for j in range(i, i+L):
                    remove.add(j)
                i += L             # skip past the removed block
                removed_any = True
                break  # break out of the L-loop

        if not removed_any:
            i += 1  # move one step forward

    if not remove:
        return df

    return df.drop(index=sorted(remove)).reset_index(drop=True)


def remove_spike(df, val_col, slope_threshold=0.05, window=1):
    """
    Remove values based on sharp rise or fall (spike detection).
    Args:
        df: pandas dataframe sorted by time.
        val_col: column name containing soil moisture values.
        slope_threshold: maximum allowed change per step.
        window: points on each side to examine (1 = immediate neighbors).
    Returns:
        df with spike-like points removed.
    """
    
    df = df.copy().reset_index(drop=True)
    values = df[val_col].values

    remove_idx = []

    for i in range(window, len(df) - window):
        prev_val = values[i - window]
        curr_val = values[i]
        next_val = values[i + window]

        # First derivative slopes
        slope_prev = curr_val - prev_val
        slope_next = next_val - curr_val

        # Condition 1: Very sharp rise/fall (slope threshold exceeded)
        if abs(slope_prev) > slope_threshold and abs(slope_next) > slope_threshold:
            # Spike shape: low → high → low OR high → low → high
            if (curr_val > prev_val and curr_val > next_val) or \
               (curr_val < prev_val and curr_val < next_val):
                remove_idx.append(i)

        # Condition 2: Single-side sharp spike (low → high → gradual? No → mark out)
        elif abs(slope_prev) > slope_threshold and abs(slope_next) < slope_threshold/4:
            remove_idx.append(i)

        elif abs(slope_next) > slope_threshold and abs(slope_prev) < slope_threshold/4:
            remove_idx.append(i)

    df_cleaned = df.drop(remove_idx).reset_index(drop=True)
    return df_cleaned


def smooth(df, val_col, window=7, polyorder=2):
    """
    Apply Savitzky-Golay smoothing to the soil moisture series.
    Args:
        df : dataframe sorted by time
        val_col : column to smooth
        window : window size (must be odd)
        polyorder : polynomial order (2 is typical for soil moisture)
    Returns:
        df with new smoothed column '<val_col>_smooth'
    """
    df = df.copy().reset_index(drop=True)
    # Ensure window is not too large
    window = min(window, len(df)-1 if (len(df)-1)%2==1 else len(df)-2)
    if window % 2 == 0: window += 1

    df[val_col] = savgol_filter(df[val_col], window, polyorder)
    return df


def clean_sms_pipeline(
        df, 
        val_col='VolumetricWaterContent1',
        lower=0.0,
        upper=100.0,
        pct_threshold=0.10,
        neighbor_similarity_tol=0.03,
        max_run_length=3,
        slope_threshold=0.05,
        slope_window=1,
        smooth_window=7,
        smooth_polyorder=2,
        apply_slope_filter=True
    ):
    """
    Full soil-moisture cleaning pipeline:
    1. remove NaN and out-of-range
    2. remove single-point jumps
    3. remove consecutive 2–max_run_length jumps
    4. optional: remove slope-based spike anomalies
    5. smoothing filter (Savitzky–Golay)

    Returns:
        Cleaned + smoothed dataframe
    """

    # -----------------------------
    # Step 1 — Clean NaN and bounds
    # -----------------------------
    df1 = remove_nan_and_bounds(df, val_col, lower=lower, upper=upper)

    # -----------------------------------
    # Step 2 — Remove isolated 1-point jumps
    # -----------------------------------
    df2 = remove_single_jump(df1, val_col,
                             pct_threshold=pct_threshold,
                             neighbor_similarity_tol=neighbor_similarity_tol)

    # ---------------------------------------------------------
    # Step 3 — Remove consecutive 2–max_run_length value jumps
    # ---------------------------------------------------------
    df3 = remove_consecutive_jump(df2, val_col,
                                  pct_threshold=pct_threshold,
                                  neighbor_similarity_tol=neighbor_similarity_tol,
                                  max_run_length=max_run_length)

    # ---------------------------------------------------------
    # Step 4 — Slope-based spike removal (optional)
    # ---------------------------------------------------------
    if apply_slope_filter:
        df4 = remove_spike(df3, val_col,
                           slope_threshold=slope_threshold,
                           window=slope_window)
    else:
        df4 = df3

    # ---------------------------------------------------------
    # Step 5 — Savitzky-Golay smoothing
    # ---------------------------------------------------------
    df5 = smooth(df4, val_col,
                 window=smooth_window,
                 polyorder=smooth_polyorder)

    return df5.reset_index(drop=True)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# ------------- TIME-BASED AVERAGING UTILITIES: witsms_average.py ---------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def accumulate(df, value_cols, timestamp_col, average='daily'):
    """
    Computes time-based averages (e.g., daily, 3hourly, hourly and 30minute) for specified value columns 
    in a dataframe, setting the timestamp for each average to a fixed time during the interval.

    Args:
        df: The pandas dataframe with value and timestamp columns.
        value_cols: List of column names containing values to average.
        timestamp_col: Name of the column containing timestamps.
        average: The type of averaging interval ('daily', 'hourly', etc.).

    Returns:
        A new pandas dataframe with averages over specified intervals and fixed timestamps for each interval.
    """
    # Ensure the timestamp column is in datetime format
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    if average == 'daily':
        # Set the time part to 12:00 PM for daily averages
        # df['TimeStamp'] = df[timestamp_col].dt.floor('D') + pd.Timedelta(hours=12)
        df['TimeStamp'] = df[timestamp_col].dt.floor('D')

        # Group by the new 'FixedTime' column and compute the mean for each group
        avg_df = df.groupby('TimeStamp')[value_cols].mean().reset_index()

    elif average == '3hourly':
        df['TimeStamp'] = df[timestamp_col].dt.floor('3h')
        avg_df = df.groupby('TimeStamp')[value_cols].mean().reset_index()
        # Ensure every 3-hour interval is represented
        if not avg_df.empty:
            all_times = pd.date_range(start=avg_df['TimeStamp'].min(), end=avg_df['TimeStamp'].max(), freq='3h')
            avg_df = avg_df.set_index('TimeStamp').reindex(all_times).reset_index().rename(columns={'index': 'TimeStamp'})
  
    elif average == 'hourly':
        # Set time to every hour
        df['TimeStamp'] = df[timestamp_col].dt.floor('h')
        avg_df = df.groupby('TimeStamp')[value_cols].mean().reset_index()

    elif average == '30minute':
        # Set time to every thirty minutes
        df['TimeStamp'] = df[timestamp_col].dt.floor('30min')
        avg_df = df.groupby('TimeStamp')[value_cols].mean().reset_index()
    
    # Round the averages to 3 decimal places
    try:
        for col in value_cols:
            avg_df[col] = avg_df[col].round(3)
    except:
        avg_df = (
            df.groupby('TimeStamp')[value_cols]
            .mean()                       # still float64 here
            .round(3)                     # round
            .astype('float64')            # ← FORCES real float, no hidden objects
            .reset_index()
        )
    
    return avg_df



#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# -------------------- PLOTTING FUNCTION: combine_data.py -----------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def plot_merged(csv_file):
    """
    Plot merged WITSMS + SMAP soil moisture.
    
    Parameters
    ----------
    csv_file : str
        Path to merged CSV file.
    """

    df = pd.read_csv(csv_file)

    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    plt.figure(figsize=(12, 6))

    # ---- WITSMS (solid blue) ----
    plt.plot(
        df['TimeStamp'], df['VolumetricWaterContent1']/100.0, 
        label="WITSMS",
        marker='x',
        linestyle='None',
        color='blue'
    )

    # ---- SMAP AM (dotted) ----
    if 'SM_AM_9km' in df.columns:
        plt.plot(
            df['TimeStamp'], df['SM_AM_9km'],
            # linestyle='--',
            marker='o',
            linestyle='None',
            alpha=0.7,
            label="SMAP AM (9km)"
        )

    # ---- SMAP PM (dotted) ----
    if 'SM_PM_9km' in df.columns:
        plt.plot(
            df['TimeStamp'], df['SM_PM_9km'],
            # linestyle='--',
            marker='o',
            linestyle='None',
            alpha=0.7,
            label="SMAP PM (9km)"
        )

    # ---- SMAP 9km (solid) ----
    if 'SM_9km' in df.columns:
        plt.plot(
            df['TimeStamp'], df['SM_9km'],
            # linestyle='-',
            marker='o',
            linestyle='None',
            label="SMAP Combined (9km)"
        )

    plt.xlabel("Date")
    plt.ylabel("Soil Moisture")
    plt.title("WITSMS vs SMAP Soil Moisture")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# ---------------------------- Training utilities  ------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# utils.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch
import joblib
import matplotlib.pyplot as plt

DATE_COL_DEFAULT = "TimeStamp"

# ---------- IO / loading ----------

def load_single_csv(path, feature_cols, target_col, date_col=DATE_COL_DEFAULT):
    df = pd.read_csv(path)
    # keep only expected columns that exist
    cols = [date_col] + list(feature_cols) + [target_col]
    df = df.loc[:, df.columns.intersection(cols)].dropna()
    # ensure ordering of columns
    df = df[[c for c in cols if c in df.columns]]
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def load_all_sites(folder, feature_cols, target_col, date_col=DATE_COL_DEFAULT):
    dfs = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith('.csv'):
            full = os.path.join(folder, fname)
            df = load_single_csv(full, feature_cols, target_col, date_col)
            df = df.copy()
            df['site_id'] = fname
            dfs.append(df)
    if len(dfs) == 0:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return pd.concat(dfs, ignore_index=True)


# ---------- sequencing / features ----------

def make_sequences_per_site(df_all, seq_len, feature_cols, target_col, date_col=DATE_COL_DEFAULT):
    """
    Build sliding sequences per site. Returns arrays (X, y, dates, site_ids)
    X shape: (N_sequences, seq_len, n_features)
    y shape: (N_sequences,)
    """
    X_list, y_list, date_list, site_list = [], [], [], []

    for site, df_site in df_all.groupby('site_id'):
        df_site = df_site.sort_values(date_col).reset_index(drop=True)

        # skip tiny sites
        if len(df_site) <= seq_len:
            continue

        X_raw = df_site[feature_cols].values.astype(np.float32)
        y_raw = (df_site[target_col].values.astype(np.float32)) / 100.0  # keep same scaling as original
        dates = df_site[date_col].values

        for i in range(len(df_site) - seq_len):
            X_list.append(X_raw[i:i+seq_len])
            y_list.append(y_raw[i+seq_len])
            date_list.append(dates[i+seq_len])
            site_list.append(site)

    if len(X_list) == 0:
        return (np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype='datetime64[ns]'),
                np.empty((0,), dtype=object))

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(date_list),
        np.array(site_list)
    )


def create_residuals(X_seq, y_seq):
    """
    baseline_last: the last time-step feature (assumes the first feature column is the coarse baseline like SM_PM_9km)
    residual = y - baseline_last
    """
    baseline_last = X_seq[:, -1, 0]
    residuals = y_seq - baseline_last
    return residuals, baseline_last


# ---------- scaling helpers ----------

def fit_scalers(X_train, residuals_train, y_train):
    scaler_x = StandardScaler()
    scaler_r = StandardScaler()
    scaler_y = StandardScaler()

    # X_train shape: (N, seq_len, n_features) -> flatten along time for fitting
    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    scaler_x.fit(X_train_flat)
    scaler_r.fit(residuals_train.reshape(-1, 1))
    scaler_y.fit(y_train.reshape(-1, 1))

    return scaler_x, scaler_r, scaler_y


def apply_scalers_to_seq(X_seq, scaler_x):
    """
    Apply a fitted StandardScaler (scaler_x) to each timestep of the sequence array.
    
    Args:
        X_seq: np.array of shape (num_sequences, seq_len, num_features)
        scaler_x: fitted sklearn StandardScaler
    
    Returns:
        X_scaled: np.array with same shape as X_seq, scaled
    """
    X_scaled = X_seq.copy()
    seq_len = X_seq.shape[1]
    for ts in range(seq_len):
        X_scaled[:, ts, :] = scaler_x.transform(X_scaled[:, ts, :])
    return X_scaled


# ---------- metrics ----------

def compute_ubrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_d = y_true - y_true.mean()
    y_pred_d = y_pred - y_pred.mean()
    return np.sqrt(np.mean((y_true_d - y_pred_d) ** 2))


def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_bias(y_true, y_pred):
    """Mean bias (difference of means): pred - true"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_pred) - np.mean(y_true)


def compute_pearson(y_true, y_pred):
    try:
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = np.nan
    return r

def compute_nse(y_true, y_pred):
    """Nash–Sutcliffe Efficiency"""
    denom = np.sum((y_true - np.mean(y_true))**2)
    num = np.sum((y_true - y_pred)**2)
    return 1 - num/denom if denom > 0 else np.nan

def compute_kge(y_true, y_pred):
    """Kling-Gupta Efficiency"""
    r = compute_pearson(y_true, y_pred)
    alpha = np.std(y_pred)/np.std(y_true) if np.std(y_true) > 0 else np.nan
    beta = np.mean(y_pred)/np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)


def compute_classic_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {'RMSE': np.nan, 'UBRMSE': np.nan, 'Pearson': np.nan, 'MSE': np.nan, 'N': 0}
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    mse = mean_squared_error(y_true_m, y_pred_m)
    
    return {
        'Bias': float(compute_bias(y_true_m, y_pred_m)),
        'MSE': float(mse),
        'RMSE': float(np.sqrt(mse)),
        'ubRMSE': float(compute_ubrmse(y_true_m, y_pred_m)),
        'Pearson r': float(compute_pearson(y_true_m, y_pred_m)),
        'NSE': float(compute_nse(y_true_m, y_pred_m)),
        'KGE': float(compute_kge(y_true_m, y_pred_m)),
        'N': int(len(y_true_m))
    }

    # return {
    #     r'Bias (m³/m³)': float(compute_bias(y_true_m, y_pred_m)),
    #     r'MSE (m³/m³)': float(mse),
    #     # r'RMSE (m³/m³)': float(np.sqrt(mse)),
    #     r'ubRMSE (m³/m³)': float(compute_ubrmse(y_true_m, y_pred_m)),
    #     r'Pearson r': float(compute_pearson(y_true_m, y_pred_m)),
    #     r'NSE': float(compute_nse(y_true_m, y_pred_m)),
    #     r'KGE': float(compute_kge(y_true_m, y_pred_m)),
    #     r'N': str(len(y_true_m))
    # }


# ---------- model save/load ----------

def save_model_torch(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_torch(model_class, state_path, map_location=None, **model_kwargs):
    device = map_location if map_location is not None else torch.device('cpu')
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_scalers(scalers_tuple, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'scaler_x': scalers_tuple[0], 'scaler_r': scalers_tuple[1], 'scaler_y': scalers_tuple[2]}, path)


def load_scalers(path):
    d = joblib.load(path)
    return d['scaler_x'], d['scaler_r'], d['scaler_y']


# ---------- plotting ----------

def plot_scatter(obs, preds, title='Observed vs Predicted', metrics_text=None, out_path=None):
    plt.figure(figsize=(7,7))
    plt.scatter(obs, preds, edgecolor='k', alpha=0.7)
    mn = min(np.nanmin(obs), np.nanmin(preds))
    mx = max(np.nanmax(obs), np.nanmax(preds))
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Observed'); plt.ylabel('Predicted')
    plt.title(title)
    if metrics_text:
        plt.gcf().text(0.65, 0.15, metrics_text, fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
    plt.grid(True)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde

# def plot_scatter_density(obs, preds, title='Observed vs Predicted', metrics_text=None, out_path=None):
#     xy = np.vstack([obs, preds])
#     z = gaussian_kde(xy)(xy)  # density values for each point

#     plt.figure(figsize=(7,7))
#     plt.scatter(obs, preds, c=z, s=30, edgecolor='none', cmap='turbo')  # color by density
#     mn = min(np.nanmin(obs), np.nanmin(preds))
#     mx = max(np.nanmax(obs), np.nanmax(preds))
#     plt.plot([mn, mx], [mn, mx], 'r--')
#     plt.xlabel('Observed'); plt.ylabel('Predicted')
#     plt.title(title)
#     if metrics_text:
#         plt.gcf().text(0.65, 0.15, metrics_text, fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
#     plt.colorbar(label='Point Density')
#     plt.grid(True)
#     plt.tight_layout()
#     if out_path:
#         plt.savefig(out_path)
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def format_metrics_text(metrics_dict, include_rmse=True):
    """
    Format metrics text with units for plotting.
    """
    lines = []

    for k, v in metrics_dict.items():
        if k == "RMSE" and not include_rmse:
            continue

        if k=='N':
            lines.append(f"{k}: {v}")
            continue

        unit = METRIC_UNITS.get(k, "")
        if unit and unit != "-":
            lines.append(f"{k}: {v:.4f} ({unit})")
        else:
            lines.append(f"{k}: {v:.4f}")

    return "\n".join(lines)


# def plot_scatter_density(obs, preds, title='Observed vs Predicted', metrics_text=None, out_path=None):
# def plot_scatter_density(
#     obs,
#     preds,
#     title='Observed vs Predicted',
#     metrics_dict=None,
#     include_rmse=True,
#     out_path=None
# ):

#     # Compute density
#     xy = np.vstack([obs, preds])
#     z = gaussian_kde(xy)(xy)

#     # Normalize to 0–100 for nicer colorbar ticks
#     z_norm = 100 * (z - z.min()) / (z.max() - z.min())

#     fig = plt.figure(figsize=(7,7))
#     ax = plt.gca()

#     # Scatter with density coloring
#     sc = ax.scatter(obs, preds, c=z_norm, s=30, edgecolors='none', cmap='turbo')

#     # 1:1 line
#     mn = min(np.nanmin(obs), np.nanmin(preds))
#     mx = max(np.nanmax(obs), np.nanmax(preds))
#     ax.plot([mn, mx], [mn, mx], 'r--')

#     ax.set_xlabel('Observed soil moisture (m³/m³)')
#     ax.set_ylabel('Predicted soil moisture (m³/m³)')
#     ax.set_title(title)

#     # Metrics box
#     # if metrics_text:
#     #     fig.text(0.67, 0.12, metrics_text, fontsize=12,
#     #              bbox=dict(facecolor='white', edgecolor='black'))

#     # Metrics box (with units)
#     if metrics_dict:
#         metrics_text = format_metrics_text(
#             metrics_dict,
#             include_rmse=include_rmse
#         )

#         fig.text(
#             0.67, 0.12,
#             metrics_text,
#             fontsize=12,
#             bbox=dict(facecolor='white', edgecolor='black')
#         )


#     # ---------------------------------------------------------
#     # Small inset colorbar inside the plot
#     # ---------------------------------------------------------
#     cax = inset_axes(ax,
#                      width="28%",   # width relative to plot
#                      height="3%",   # height relative to plot
#                      loc='upper right',
#                      borderpad=1.0)

#     cb = plt.colorbar(sc, cax=cax, orientation='horizontal')

#     # Show only 0 and 100 ticks
#     cb.set_ticks([0, 100])
#     cb.set_ticklabels(['0', '100'])
#     cb.ax.tick_params(labelsize=8)
#     cb.set_label("Point Density (scaled)", fontsize=8)

#     ax.grid(True)
#     plt.tight_layout()

#     # Save if needed
#     if out_path:
#         plt.savefig(out_path, dpi=300, bbox_inches='tight')

#     plt.show()


def plot_scatter_density(
    obs, 
    preds, 
    title='Observed vs Predicted', 
    metrics_dict=None, 
    include_rmse=True, 
    out_path=None,
    min_value=0.01   # New parameter: minimum value to plot
):
    import numpy as np
    from scipy.stats import gaussian_kde
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.pyplot as plt
    
    # Convert to numpy arrays if not already
    obs = np.asarray(obs)
    preds = np.asarray(preds)
    
    # === Filter: Keep only values > min_value ===
    mask = (obs > min_value) & (preds > min_value)
    
    if mask.sum() == 0:
        raise ValueError(f"No data points remain after filtering values > {min_value}")
    
    obs_filtered = obs[mask]
    preds_filtered = preds[mask]
    
    print(f"Plotting {mask.sum():,} out of {len(obs):,} points (values > {min_value})")
    
    # Compute density on filtered data
    xy = np.vstack([obs_filtered, preds_filtered])
    z = gaussian_kde(xy)(xy)
    
    # Normalize density to 0–100 for nicer colorbar
    z_norm = 100 * (z - z.min()) / (z.max() - z.min())
    
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    
    # Scatter with density coloring
    sc = ax.scatter(obs_filtered, preds_filtered, 
                    c=z_norm, 
                    s=30, 
                    edgecolors='none', 
                    cmap='turbo')
    
    # 1:1 line - use filtered min/max
    mn = min(np.nanmin(obs_filtered), np.nanmin(preds_filtered))
    mx = max(np.nanmax(obs_filtered), np.nanmax(preds_filtered))
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='1:1 line')
    
    ax.set_xlabel('Observed soil moisture (m³/m³)')
    ax.set_ylabel('Predicted soil moisture (m³/m³)')
    ax.set_title(title)
    
    # Metrics box (only on filtered data!)
    if metrics_dict:
        # Important: You should recompute metrics on filtered data if needed
        metrics_text = format_metrics_text(metrics_dict, include_rmse=include_rmse)
        fig.text(0.67, 0.12, metrics_text, fontsize=12,
                 bbox=dict(facecolor='white', edgecolor='black'))
    
    # Small inset colorbar
    cax = inset_axes(ax, width="28%", height="3%", 
                     loc='upper right', borderpad=1.0)
    cb = plt.colorbar(sc, cax=cax, orientation='horizontal')
    cb.set_ticks([0, 100])
    cb.set_ticklabels(['0', '100'])
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Point Density (scaled)", fontsize=8)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
    plt.show()



def plot_timeseries(dates, obs, preds, baseline=None, title='Time Series', out_path=None):
    plt.figure(figsize=(14,6))
    plt.plot(dates, obs, label='Observed', marker='o')
    plt.plot(dates, preds, label='Predicted', marker='*')
    if baseline is not None:
        plt.plot(dates, baseline, label='Feature', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Volumetric Water Content')
    plt.title(title)
    plt.legend(); plt.grid(True); plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()


# ---------- uncertainty plotting ----------

def plot_uncertainty_timeseries(dates, uncertainty, title="Uncertainty Time Series"):
    plt.figure(figsize=(10,4))
    plt.plot(dates, uncertainty)
    plt.ylabel("Predictive Std")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_distribution(unc_values, title="Uncertainty Distribution"):
    plt.figure(figsize=(6,4))
    plt.hist(unc_values, bins=30)
    plt.xlabel("Predictive Std (m³/m³)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_vs_error(unc, error, title="Uncertainty vs Absolute Error"):
    plt.figure(figsize=(6,4))
    plt.scatter(unc, np.abs(error), s=8)
    plt.xlabel("Predictive Std (Uncertainty)")
    plt.ylabel("|Error|")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_time_series_with_uncertainty(
        dates,
        y_true,
        y_pred_mean,
        y_pred_std,
        site_name=None,
        title=None,
        save_path=None, 
        fontsize=12,
        fontsize_ticks=14,
        fontsize_legend=14


    ):
    """
    Plots:
        - True values
        - Predicted mean
        - Uncertainty band = mean ± std
    """

    plt.figure(figsize=(14,5))

    # Prediction mean line
    plt.plot(dates, y_pred_mean, label="Prediction (Mean)", linewidth=2)

    # True values
    plt.plot(dates, y_true, label="Measured", color="blue", linestyle='--', linewidth=1.5)

    # Shaded uncertainty region
    lower = y_pred_mean - y_pred_std
    upper = y_pred_mean + y_pred_std

    plt.fill_between(
        dates,
        lower,
        upper,
        alpha=0.25,
        label="Model uncertainty (± Std)"
    )

    plt.xlabel("Date")
    plt.ylabel("Soil Moisture (m³/m³)", fontsize=fontsize_legend)
    t = title if title else f"Prediction With Uncertainty {'' if site_name is None else f' - Site {site_name}'}"
    plt.title(t)
    plt.yticks(fontsize=fontsize_ticks)

    # Legend
    plt.legend(fontsize=fontsize_legend)
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
