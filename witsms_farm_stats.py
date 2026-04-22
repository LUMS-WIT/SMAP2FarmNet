import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from config import OUT_DIR, MODE

EXTRA_COLS = ['Combined', 'Baseline']

# ===============================
# Helper functions
# ===============================

def fisher_z(r):
    """Fisher Z-transform for correlation coefficients"""
    r = np.clip(r, -0.999999, 0.999999)  # numerical safety
    return np.arctanh(r)

def inverse_fisher_z(z):
    """Inverse Fisher Z-transform"""
    return np.tanh(z)


# ===============================
# Main aggregation function
# ===============================

def compute_farm_level_metrics(
    metrics_csv,
    out_csv="farm_level_metrics.csv"
):
    """
    Compute farm-level mean and std from sensor-level metrics.

    Farm ID extraction:
    2001_NARC_A07_Plot01_Pear.csv → 2001

    Metrics handled correctly:
    - RMSE / UBRMSE : quadratic → mean(square) → sqrt
    - Pearson      : Fisher Z mean
    - Bias          : arithmetic mean
    - MSE          : arithmetic mean
    - NSE          : arithmetic mean
    - KGE          : arithmetic mean
    - unc_mean     : arithmetic mean
    """

    # ===============================
    # Read metrics
    # ===============================
    df = pd.read_csv(metrics_csv)

    # -------------------------------
    # Clean site name
    # -------------------------------
    df["site_clean"] = df["site"].str.replace(".csv", "", regex=False)
    
    # Remove non-site rows (e.g. Combined, Baseline)
    df = df[~df["site"].isin(EXTRA_COLS)].copy()
    # -------------------------------
    # Extract FARM ID
    # Example:
    # 2001_NARC_A07_Plot01_Pear.csv → 2001_NARC
    # -------------------------------
    df["farm_id"] = df["site_clean"].str.split("_").str[:2].str.join("_")
    # df["farm_id"] = df["site_clean"].str.extract(r"^(\d+)").astype(int)

    # ===============================
    # Transformed columns
    # ===============================
    df["RMSE_sq"]   = df["RMSE"] ** 2
    df["ubRMSE_sq"] = df["ubRMSE"] ** 2
    df["Pearson r_z"] = fisher_z(df["Pearson r"])

    # ===============================
    # Group by farm
    # ===============================
    grouped = df.groupby("farm_id")

    # ===============================
    # Aggregate metrics
    # ===============================
    farm_df = pd.DataFrame({
        "farm_id": grouped.size().index,
        "sensors": grouped.size().values,
        "overlaps": grouped["N"].sum().values,

        # ---- Bias ----
        "Bias_mean": grouped["Bias"].mean(),
        "Bias_std": grouped["Bias"].std(),

        # ---- RMSE ----
        "RMSE_mean": np.sqrt(grouped["RMSE_sq"].mean()),
        "RMSE_std": np.sqrt(grouped["RMSE_sq"].std()),

        # ---- UBRMSE ----
        "ubRMSE_mean": np.sqrt(grouped["ubRMSE_sq"].mean()),
        "ubRMSE_std": np.sqrt(grouped["ubRMSE_sq"].std()),

        # ---- Pearson r ----
        "Pearson r_mean": inverse_fisher_z(grouped["Pearson r_z"].mean()),
        "Pearson r_std": grouped["Pearson r"].std(),

        # ---- MSE ----
        "MSE_mean": grouped["MSE"].mean(),
        "MSE_std": grouped["MSE"].std(),

        # ---- NSE ----
        "NSE_mean": grouped["NSE"].mean(),
        "NSE_std": grouped["NSE"].std(),

        # ---- KGE ----
        "KGE_mean": grouped["KGE"].mean(),
        "KGE_std": grouped["KGE"].std(),

        # ---- Uncertainty ----
        "unc_mean_mean": grouped["unc_mean"].mean(),
        "unc_mean_std": grouped["unc_mean"].std(),
    })

    # ===============================
    # Save output
    # ===============================
    farm_df.to_csv(out_csv, index=False)

    print("Farm-level aggregation completed successfully.")
    print(f"Total farms: {len(farm_df)}")
    print(f"Output file: {out_csv}")

    return farm_df

def prettify_metric_label(col):
    """
    Convert metric column names to clean plot labels.
    """
    if col == "unc_mean_mean":
        return "model_unc"
    return col.split("_")[0]


# def print_boxplot_medians(data, labels, title):
#     print(f"\nMedian values for {title}:")
#     for label, values in zip(labels, data):
#         median_val = np.median(values)
#         print(f"  {label}: {median_val:.4f}")


def print_and_annotate_medians(data, labels, bp):
    """
    Print medians to console and annotate them on boxplots.

    Parameters
    ----------
    data : list of np.ndarray
        Data arrays used for boxplots.
    labels : list of str
        Labels corresponding to each box.
    bp : dict
        Dictionary of matplotlib boxplot artists.
    """
    for i, (label, values) in enumerate(zip(labels, data)):
        if len(values) == 0:
            continue

        median_val = np.median(values)
        print(f"  {label}: {median_val:.4f}")

        # Annotate median on plot
        x, y = bp["medians"][i].get_xydata()[1]
        plt.text(
            x, y,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black"
        )


def plot_farm_level_boxplots(
    farm_df,
    out_dir="checkpoints/plots",
    figsize_skill=(7, 4),
    figsize_error=(7, 4)
):
    os.makedirs(out_dir, exist_ok=True)

    # ==========================
    # Skill metrics
    # ==========================
    skill_metrics = [
        c for c in ["Pearson r_mean", "NSE_mean", "KGE_mean"]
        if c in farm_df.columns
    ]

    skill_data = [farm_df[c].dropna().values for c in skill_metrics]
    skill_labels = [prettify_metric_label(c) for c in skill_metrics]

    plt.figure(figsize=figsize_skill)
    # plt.boxplot(skill_data, labels=skill_labels, showfliers=True)
    bp_skill = plt.boxplot(skill_data, labels=skill_labels, showfliers=True)
    plt.ylabel("Skill metric value")
    plt.title("Farm-level distribution of skill metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    print("\nMedian values for Skill metrics:")
    print_and_annotate_medians(skill_data, skill_labels, bp_skill)

    skill_out = os.path.join(out_dir, f"{MODE}_farm_level_skill_metrics_boxplot.png")
    plt.savefig(skill_out, dpi=300)
    plt.close()

    # ==========================
    # Error metrics
    # ==========================
    error_metrics = [
        # c for c in ["Bias_mean", "MSE_mean", "RMSE_mean", "ubRMSE_mean", "unc_mean_mean"]
        c for c in ["Bias_mean", "MSE_mean", "RMSE_mean", "ubRMSE_mean"]
        if c in farm_df.columns
    ]

    error_data = [farm_df[c].dropna().values for c in error_metrics]
    error_labels = [prettify_metric_label(c) for c in error_metrics]

    plt.figure(figsize=figsize_error)
    bp_error = plt.boxplot(error_data, labels=error_labels, showfliers=True)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)  # important for bias
    plt.ylabel("Error metric value (m³/m³)")
    plt.title("Farm-level distribution of error metrics (mean)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    print("\nMedian values for Error metrics:")
    print_and_annotate_medians(error_data, error_labels, bp_error)

    error_out = os.path.join(out_dir, f"{MODE}_farm_level_error_metrics_boxplot.png")
    plt.savefig(error_out, dpi=300)
    plt.close()

    print("Saved plots:")
    # print(f"  Skill metrics: {skill_out}")
    # print(f"  Error metrics: {error_out}")



# ===============================
# Run as script
# ===============================
if __name__ == "__main__":

    metrics_csv = f"{OUT_DIR}/{MODE}_metrics_per_site_mc_test.csv"   # INPUT
    out_csv = f"{OUT_DIR}/farmlevel/{MODE}_metrics_per_farm_mc.csv" # OUTPUT

    farm_df = compute_farm_level_metrics(metrics_csv, out_csv)

    plot_farm_level_boxplots(
        farm_df,
        out_dir=f"{OUT_DIR}/farmlevel"
    )

    
