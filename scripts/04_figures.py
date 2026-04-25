# =============================================================================
# NOTEBOOK 04 — THESIS FIGURES
# Thesis: Event-Conditioned Reliability Analysis of Indoor LoRaWAN Links
# Author: Vincent Mwenda Mworia | University of Siegen | EMINENT Programme
#
# Purpose : Produce all publication-quality figures for Chapter 7.
#           Every figure saved as PDF (for Overleaf) and PNG (for preview).
# Thesis  : Figures 7.1 – 7.5 used in Chapter 7 (Results and Analysis)
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. PATHS & GLOBAL STYLE
# ---------------------------------------------------------------------------
DATA_DIR = Path("../data")
FIG_DIR  = Path("../figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV = DATA_DIR / "events.csv"

# IEEE two-column figure style — serif font, tight spacing
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# One colour per device — consistent across all figures
DEVICE_COLORS = {
    "ED0": "#1F4E79",
    "ED1": "#2E75B6",
    "ED2": "#C55A11",
    "ED3": "#375623",
    "ED4": "#7030A0",
    "ED5": "#833C00",
}
DEVICES = ["ED0", "ED1", "ED2", "ED3", "ED4", "ED5"]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def save_fig(name: str):
    """Save as both PDF (Overleaf) and PNG (preview)."""
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close()
    print(f"  Saved: {name}.pdf  |  {name}.png")


def pdr(grp: pd.DataFrame) -> float:
    rx    = len(grp)
    lost  = grp["loss"].sum()
    total = rx + lost
    return rx / total * 100 if total > 0 else np.nan


def rolling_pdr(device_df: pd.DataFrame, window: str = "30min") -> pd.Series:
    """Time-based rolling PDR for a single device."""
    d = device_df.copy().set_index("time").sort_index()
    d["rx1"] = 1
    rx_roll  = d["rx1"].rolling(window, min_periods=5).sum()
    tx_roll  = d["total_tx"].rolling(window, min_periods=5).sum()
    return (rx_roll / tx_roll * 100).clip(0, 100)

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
print("Loading events dataset …")
df = pd.read_csv(IN_CSV)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values(["device_id", "time"]).reset_index(drop=True)
print(f"  Rows: {len(df):,}")

# ---------------------------------------------------------------------------
# FIGURE 1 — Rolling PDR time-series for all 6 devices
#             (Thesis Figure 7.1 — Rolling PDR time series, Section 7.1.2)
# ---------------------------------------------------------------------------
print("\nFigure 1: Rolling PDR over time …")

campaign_pdr = len(df) / (len(df) + df["loss"].sum()) * 100

fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)

for ax, dev in zip(axes, DEVICES):
    d = df[df["device_id"] == dev]
    rpdr = rolling_pdr(d)
    ax.plot(rpdr.index, rpdr.values,
            linewidth=0.5, color=DEVICE_COLORS[dev], alpha=0.85)
    ax.axhline(y=campaign_pdr, color="red", linewidth=0.6,
               linestyle="--", alpha=0.6)
    ax.set_ylabel(dev, fontsize=9, rotation=0, ha="right", va="center")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

axes[-1].set_xlabel("Date (UTC)")
fig.suptitle(
    "Rolling 30-minute PDR per device — full 34-week campaign\n"
    f"(red dashed line = campaign mean PDR = {campaign_pdr:.1f}%)",
    fontsize=10, y=1.01,
)
plt.tight_layout()
save_fig("fig_7_1_rolling_pdr")

# ---------------------------------------------------------------------------
# FIGURE 2 — CO2 vs rolling PDR dual-axis plot (ED3 as representative device)
#             (Thesis Figure 7.2 — Occupancy proxy vs reliability, Section 7.4)
# ---------------------------------------------------------------------------
print("Figure 2: CO2 vs rolling PDR (ED3) …")

d_ed3 = df[df["device_id"] == "ED3"].copy().set_index("time").sort_index()

fig, ax1 = plt.subplots(figsize=(12, 4))

rpdr = rolling_pdr(df[df["device_id"] == "ED3"], window="2h")
ax1.plot(rpdr.index, rpdr.values,
         color="#1F4E79", linewidth=0.5, alpha=0.8, label="PDR (%)")
ax1.set_ylabel("PDR (%)", color="#1F4E79", fontsize=10)
ax1.set_ylim(0, 100)
ax1.tick_params(axis="y", labelcolor="#1F4E79")

ax2 = ax1.twinx()
co2_roll = d_ed3["co2"].rolling("2h", min_periods=5).mean()
ax2.plot(co2_roll.index, co2_roll.values,
         color="#C55A11", linewidth=0.6, alpha=0.75, label="CO₂ (ppm)")
ax2.set_ylabel("CO₂ (ppm)", color="#C55A11", fontsize=10)
ax2.tick_params(axis="y", labelcolor="#C55A11")

# Threshold lines
ax2.axhline(y=700, color="#C55A11", linewidth=0.5, linestyle=":", alpha=0.6)
ax2.axhline(y=500, color="#375623", linewidth=0.5, linestyle=":", alpha=0.6)

ax1.set_xlabel("Date (UTC)")
ax1.set_title("ED3: 2-hour rolling PDR vs CO₂ concentration (occupancy proxy)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
plt.tight_layout()
save_fig("fig_7_2_co2_vs_pdr")

# ---------------------------------------------------------------------------
# FIGURE 3 — Event-conditioned PDR: CO2 tier grouped bar chart
#             (Thesis Figure 7.3 — Table 7.5 visualised, Section 7.4)
# ---------------------------------------------------------------------------
print("Figure 3: PDR by CO2 tier …")

co2_order  = ["background", "moderate", "high"]
tier_colors = ["#1D9E75", "#378ADD", "#E24B4A"]

co2_pdr_data = (
    df[df["e1_co2_tier"].notna()]
    .groupby(["device_id", "e1_co2_tier"], observed=True)
    .apply(pdr)
    .reset_index()
)
co2_pdr_data.columns = ["device_id", "co2_tier", "pdr_pct"]
co2_pdr_data["co2_tier"] = pd.Categorical(
    co2_pdr_data["co2_tier"], categories=co2_order, ordered=True
)

pivot = co2_pdr_data.pivot(index="device_id", columns="co2_tier", values="pdr_pct")
pivot = pivot[co2_order]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(pivot))
width = 0.25

for i, (tier, color) in enumerate(zip(co2_order, tier_colors)):
    ax.bar(x + i * width, pivot[tier], width,
           label=tier.capitalize(), color=color,
           edgecolor="white", linewidth=0.5)

ax.set_xticks(x + width)
ax.set_xticklabels(pivot.index, rotation=0)
ax.set_xlabel("End Device")
ax.set_ylabel("PDR (%)")
ax.set_ylim(45, 75)
ax.set_title("Event-conditioned PDR by CO₂ tier (occupancy proxy)")
ax.legend(title="CO₂ tier", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
save_fig("fig_7_3_co2_tier_pdr")

# ---------------------------------------------------------------------------
# FIGURE 4 — SF × CO2 interaction heatmap
#             (Thesis Figure 7.4 — Novel joint finding, Section 7.6.2)
# ---------------------------------------------------------------------------
print("Figure 4: SF tier × CO2 tier heatmap (novel finding) …")

joint = (
    df[df["e1_co2_tier"].notna() & df["e14_sf_tier"].notna()]
    .groupby(["e14_sf_tier", "e1_co2_tier"], observed=True)
    .apply(pdr)
    .reset_index()
)
joint.columns = ["sf_tier", "co2_tier", "pdr_pct"]

co2_order_j = ["background", "moderate", "high"]
sf_order_j  = ["low_sf", "high_sf"]

hmap = joint.pivot(index="sf_tier", columns="co2_tier", values="pdr_pct")
hmap = hmap.reindex(index=sf_order_j, columns=co2_order_j)

fig, ax = plt.subplots(figsize=(7, 3))
vmin = hmap.values.min() - 2
vmax = hmap.values.max() + 2
im = ax.imshow(hmap.values, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

ax.set_xticks(range(len(co2_order_j)))
ax.set_xticklabels([t.capitalize() for t in co2_order_j], rotation=20, ha="right")
ax.set_yticks(range(len(sf_order_j)))
ax.set_yticklabels(["Low SF (7–8)", "High SF (9–10)"])

for i in range(len(sf_order_j)):
    for j in range(len(co2_order_j)):
        val = hmap.values[i, j]
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if val < (vmin + vmax) / 2 else "black")

plt.colorbar(im, ax=ax, label="PDR (%)")
ax.set_title("Joint PDR: SF tier × CO₂ tier — SF–occupancy interaction")
ax.set_xlabel("CO₂ tier (occupancy proxy)")
ax.set_ylabel("SF tier")
plt.tight_layout()
save_fig("fig_7_4_sf_co2_heatmap")

# ---------------------------------------------------------------------------
# FIGURE 5 — Logistic regression odds ratio plot
#             (Thesis Figure 7.5 — Table 7.8 visualised, Section 7.7)
#             Requires result_table_7_8.csv produced by 03_analysis.py
# ---------------------------------------------------------------------------
print("Figure 5: Odds ratio plot (loss risk) …")

t78_path = DATA_DIR / "result_table_7_8.csv"

if t78_path.exists():
    t78 = pd.read_csv(t78_path)
    # Keep only non-device rows sorted by loss OR
    t78 = t78.sort_values("loss_OR", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#C55A11" if v > 1 else "#1F4E79" for v in t78["loss_OR"]]
    y_pos  = range(len(t78))

    ax.barh(y_pos, t78["loss_OR"] - 1, left=1,
            color=colors, edgecolor="white", linewidth=0.5, height=0.6)
    ax.errorbar(
        t78["loss_OR"], y_pos,
        xerr=[t78["loss_OR"] - t78["loss_CI_lo"],
              t78["loss_CI_hi"] - t78["loss_OR"]],
        fmt="none", color="black", capsize=3, linewidth=0.8,
    )
    ax.axvline(x=1, color="black", linewidth=0.9, linestyle="-")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(t78["condition"], fontsize=8)
    ax.set_xlabel("Odds Ratio  (OR > 1 = increased loss risk)")
    ax.set_title("Logistic regression: packet-loss risk by observed condition (RQ3)")
    plt.tight_layout()
    save_fig("fig_7_5_odds_ratios")
else:
    print(f"  SKIPPED — {t78_path} not found. Run 03_analysis.py first.")

# ---------------------------------------------------------------------------
# FIGURE 6 — PDR by spreading factor (bar chart with ToA annotation)
#             (Thesis Figure 7.6 — Table 7.6 visualised, Section 7.6.1)
# ---------------------------------------------------------------------------
print("Figure 6: PDR by spreading factor …")

toa_ms = {7: 71.9, 8: 133.6, 9: 246.8, 10: 452.6}
sf_vals = [7, 8, 9, 10]
sf_pdr  = [pdr(df[df["e13_sf"] == sf]) for sf in sf_vals]
sf_cols = ["#1D9E75", "#378ADD", "#EF9F27", "#E24B4A"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(sf_vals, sf_pdr, color=sf_cols, edgecolor="white",
              linewidth=0.5, width=0.6)

for bar, sf, p_ in zip(bars, sf_vals, sf_pdr):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{p_:.1f}%\n(ToA {toa_ms[sf]:.0f} ms)",
            ha="center", va="bottom", fontsize=8)

ax.set_xticks(sf_vals)
ax.set_xticklabels([f"SF {s}" for s in sf_vals])
ax.set_xlabel("Spreading Factor")
ax.set_ylabel("PDR (%)")
ax.set_ylim(45, 75)
ax.set_title("Event-conditioned PDR by spreading factor (all devices, full campaign)")
plt.tight_layout()
save_fig("fig_7_6_sf_pdr")

# ---------------------------------------------------------------------------
print(f"\nAll figures saved to: {FIG_DIR.resolve()}")
print("Include in Overleaf with: \\includegraphics[width=\\columnwidth]{figures/fig_7_1_rolling_pdr.pdf}")
