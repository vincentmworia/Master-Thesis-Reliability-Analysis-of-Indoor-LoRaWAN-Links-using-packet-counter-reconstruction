# =============================================================================
# NOTEBOOK 03 — STATISTICAL ANALYSIS
# Thesis: Event-Conditioned Reliability Analysis of Indoor LoRaWAN Links
# Author: Vincent Mwenda Mworia | University of Siegen | EMINENT Programme
#
# Purpose : Answer all three research questions using statistical methods.
#           Every print block maps directly to a table in Chapter 7.
# Thesis  : Implements Chapter 7 (Results and Analysis) in full.
#           RQ1 → Section 7.1–7.2
#           RQ2 → Section 7.3–7.6
#           RQ3 → Section 7.7 (logistic regression, Table 7.8)
# =============================================================================

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. PATHS & HELPERS
# ---------------------------------------------------------------------------
DATA_DIR = Path("../data")
IN_CSV   = DATA_DIR / "events.csv"
OUT_DIR  = DATA_DIR          # results CSVs saved alongside data

def pdr(grp: pd.DataFrame) -> float:
    """Compute PDR for a group: received / (received + reconstructed lost)."""
    rx    = len(grp)
    lost  = grp["loss"].sum()
    total = rx + lost
    return rx / total * 100 if total > 0 else np.nan

def mannwhitney(a: np.ndarray, b: np.ndarray):
    """Two-sided Mann–Whitney U test. Returns (U, p, rank_biserial_r)."""
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r    = 1 - (2 * u) / (len(a) * len(b))   # rank-biserial correlation
    return u, p, r

def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
print("Loading events dataset …")
df = pd.read_csv(IN_CSV)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values(["device_id", "time"]).reset_index(drop=True)
print(f"  Rows: {len(df):,}  |  Devices: {sorted(df['device_id'].unique())}")

# ===========================================================================
# RQ1 — RECONSTRUCTION RESULTS  (Section 7.1, Table 7.1)
# ===========================================================================
section("RQ1 — GLOBAL PDR PER DEVICE  (Table 7.1)")

rows = []
for dev, g in df.groupby("device_id"):
    rx    = len(g)
    lost  = int(g["loss"].sum())
    total = rx + lost
    p     = rx / total * 100
    rows.append({"device": dev, "received": rx, "recon_lost": lost,
                 "total_tx": total, "pdr_pct": round(p, 2)})

t71 = pd.DataFrame(rows)
print(t71.to_string(index=False))
total_row = t71[["received", "recon_lost", "total_tx"]].sum()
print(f"\n  Campaign total: {total_row['received']:,} received  |  "
      f"{total_row['recon_lost']:,} lost  |  "
      f"PDR = {total_row['received']/total_row['total_tx']*100:.2f}%")
t71.to_csv(OUT_DIR / "result_table_7_1.csv", index=False)

# ===========================================================================
# RQ1 — BURST LOSS DISTRIBUTION  (Section 7.2, Table 7.2)
# ===========================================================================
section("RQ1 — BURST LOSS DISTRIBUTION  (Table 7.2)")

t72 = (
    df["e16_loss_type"]
    .value_counts()
    .reindex(["no_loss", "isolated", "small_burst", "large_burst"])
    .reset_index()
)
t72.columns = ["loss_type", "count"]
t72["pct"] = (t72["count"] / len(df) * 100).round(1)
print(t72.to_string(index=False))
t72.to_csv(OUT_DIR / "result_table_7_2.csv", index=False)

# ===========================================================================
# RQ2 — WEEKDAY vs WEEKEND  (Section 7.3.1, Table 7.3)
# ===========================================================================
section("RQ2 — WEEKDAY vs WEEKEND PDR  (Table 7.3)")

rows = []
for dev, g in df.groupby("device_id"):
    wk  = g[g["e2_is_weekday"] == 1]
    we  = g[g["e2_is_weekday"] == 0]
    p_wk = pdr(wk);  p_we = pdr(we)
    _, p_val, r = mannwhitney(wk["total_tx"].values, we["total_tx"].values)
    rows.append({"device": dev,
                 "weekday_pdr": round(p_wk, 2),
                 "weekend_pdr": round(p_we, 2),
                 "diff_pp":     round(p_wk - p_we, 2),
                 "p_value":     round(p_val, 4),
                 "r_biserial":  round(r, 3)})

t73 = pd.DataFrame(rows)
print(t73.to_string(index=False))
t73.to_csv(OUT_DIR / "result_table_7_3.csv", index=False)

# ===========================================================================
# RQ2 — TIME-OF-DAY PDR  (Section 7.3.2)
# ===========================================================================
section("RQ2 — TIME-OF-DAY PDR PROFILE")

tod_pdr = (
    df.groupby("e4_time_of_day", observed=True)
    .apply(pdr)
    .reset_index()
)
tod_pdr.columns = ["time_of_day", "pdr_pct"]
tod_pdr["pdr_pct"] = tod_pdr["pdr_pct"].round(2)
print(tod_pdr.to_string(index=False))

# ===========================================================================
# RQ2 — SEASONAL PDR AND SF TIER PDR  (Section 7.3.3 & 7.6, Table 7.4)
# ===========================================================================
section("RQ2 — SEASONAL PDR and SF TIER PDR  (Table 7.4)")

season_pdr = (
    df[df["e6_season"].notna()]
    .groupby("e6_season", observed=True)
    .apply(pdr)
    .reset_index()
)
season_pdr.columns = ["season", "pdr_pct"]
season_pdr["pdr_pct"] = season_pdr["pdr_pct"].round(2)
print("\n  Seasonal PDR:")
print(season_pdr.to_string(index=False))

sf_tier_pdr = (
    df.groupby("e14_sf_tier", observed=True)
    .apply(pdr)
    .reset_index()
)
sf_tier_pdr.columns = ["sf_tier", "pdr_pct"]
sf_tier_pdr["pdr_pct"] = sf_tier_pdr["pdr_pct"].round(2)
print("\n  SF Tier PDR:")
print(sf_tier_pdr.to_string(index=False))

t74 = pd.concat([season_pdr.rename(columns={"season": "condition"}),
                  sf_tier_pdr.rename(columns={"sf_tier": "condition"})])
t74.to_csv(OUT_DIR / "result_table_7_4.csv", index=False)

# ===========================================================================
# RQ2 — CO2 TIER PDR  (Section 7.4, Table 7.5)
# ===========================================================================
section("RQ2 — CO2 TIER PDR  (Table 7.5)")

co2_pdr = (
    df.groupby("e1_co2_tier", observed=True)
    .apply(lambda g: pd.Series({
        "pdr_pct":    round(pdr(g), 2),
        "burst_rate": round(g["e16_loss_type"].isin(["small_burst", "large_burst"]).mean() * 100, 1),
        "n_pct":      round(len(g) / len(df) * 100, 1),
    }))
    .reset_index()
)
co2_pdr.columns = ["co2_tier", "pdr_pct", "burst_rate_pct", "sample_pct"]
print(co2_pdr.to_string(index=False))
co2_pdr.to_csv(OUT_DIR / "result_table_7_5.csv", index=False)

# ===========================================================================
# RQ2 — SF PDR AND BURST RATE  (Section 7.6.1, Table 7.6)
# ===========================================================================
section("RQ2 — SF PDR AND BURST RATE  (Table 7.6)")

toa_map = {7: 71.9, 8: 133.6, 9: 246.8, 10: 452.6}

sf_pdr = (
    df.groupby("e13_sf", observed=True)
    .apply(lambda g: pd.Series({
        "toa_ms":     toa_map.get(int(g["e13_sf"].iloc[0]), np.nan),
        "pdr_pct":    round(pdr(g), 2),
        "burst_rate": round(g["e16_loss_type"].isin(["small_burst", "large_burst"]).mean() * 100, 1),
    }))
    .reset_index()
)

# Mann-Whitney vs SF7 baseline
sf7_tx = df[df["e13_sf"] == 7]["total_tx"].values
for sf in [8, 9, 10]:
    sf_tx = df[df["e13_sf"] == sf]["total_tx"].values
    _, p_val, _ = mannwhitney(sf7_tx, sf_tx)
    sf_pdr.loc[sf_pdr["e13_sf"] == sf, "p_vs_sf7"] = f"<0.001" if p_val < 0.001 else f"{p_val:.4f}"

sf_pdr.loc[sf_pdr["e13_sf"] == 7, "p_vs_sf7"] = "baseline"
print(sf_pdr.to_string(index=False))
sf_pdr.to_csv(OUT_DIR / "result_table_7_6.csv", index=False)

# ===========================================================================
# RQ2 — JOINT SF TIER × CO2 TIER  (Section 7.6.2, Table 7.7)
# ===========================================================================
section("RQ2 — JOINT SF TIER × CO2 TIER  (Table 7.7)")

joint = (
    df[df["e1_co2_tier"].notna() & df["e14_sf_tier"].notna()]
    .groupby(["e14_sf_tier", "e1_co2_tier"], observed=True)
    .apply(pdr)
    .reset_index()
)
joint.columns = ["sf_tier", "co2_tier", "pdr_pct"]
joint["pdr_pct"] = joint["pdr_pct"].round(2)
pivot = joint.pivot(index="sf_tier", columns="co2_tier", values="pdr_pct")
print(pivot.to_string())

# Interaction: PDR gap between low/high SF under each CO2 condition
for tier in ["background", "moderate", "high"]:
    if tier in pivot.columns:
        low_  = pivot.loc["low_sf",  tier] if "low_sf"  in pivot.index else np.nan
        high_ = pivot.loc["high_sf", tier] if "high_sf" in pivot.index else np.nan
        print(f"  SF gap under {tier:12s}: {low_ - high_:.1f} pp")

joint.to_csv(OUT_DIR / "result_table_7_7.csv", index=False)

# ===========================================================================
# RQ2 — OTHER EVENTS: PM2.5, PRESSURE, HUMIDITY  (Section 7.5)
# ===========================================================================
section("RQ2 — PM2.5 SPIKE, PRESSURE DROP, HUMIDITY TIER")

for event_col, event_label in [
    ("e7_pm25_spike",    "PM2.5 Spike (E7)"),
    ("e10_pressure_drop","Pressure Drop (E10)"),
]:
    inside  = df[df[event_col] == 1]
    outside = df[df[event_col] == 0]
    p_in  = pdr(inside);   p_out = pdr(outside)
    _, p_val, r = mannwhitney(inside["total_tx"].values, outside["total_tx"].values)
    print(f"\n  {event_label}:")
    print(f"    Inside event  PDR: {p_in:.2f}%")
    print(f"    Outside event PDR: {p_out:.2f}%")
    print(f"    Δ = {p_in - p_out:.2f} pp  |  p = {p_val:.4f}  |  r = {r:.3f}")

print("\n  Humidity Tier PDR:")
hum_pdr = (
    df[df["e11_humidity_tier"].notna()]
    .groupby("e11_humidity_tier", observed=True)
    .apply(pdr)
    .round(2)
)
print(hum_pdr.to_string())

# ===========================================================================
# RQ3 — LOGISTIC REGRESSION: LOSS RISK AND BURST-LOSS RISK  (Table 7.8)
# ===========================================================================
section("RQ3 — LOGISTIC REGRESSION ODDS RATIOS  (Table 7.8)")

# ---- Outcome variables
df["y_loss"]  = (df["loss"] > 0).astype(int)         # any loss in interval
df["y_burst"] = (df["loss"] >= 3).astype(int)         # burst loss (B=3)

# ---- Predictors: encode categorical events as dummies
#      Reference categories chosen to match thesis narrative
feature_spec = {
    "e1_co2_tier":      ("background",),
    "e2_is_weekday":    None,
    "e3_co2_rising":    None,
    "e5_office_hours":  None,
    "e4_time_of_day":   ("night",),
    "e6_season":        ("winter",),
    "e7_pm25_spike":    None,
    "e10_pressure_drop":None,
    "e11_humidity_tier":("dry",),
    "e14_sf_tier":      ("low_sf",),
    "e17_rssi_tier":    ("strong",),
}

cat_cols = [c for c, v in feature_spec.items() if v is not None]
bin_cols = [c for c, v in feature_spec.items() if v is None]

# Ensure string types for categorical columns
for c in cat_cols:
    df[c] = df[c].astype(str)
for c in bin_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# Device fixed effects (reference = ED0)
device_dummies = pd.get_dummies(df["device_id"], prefix="dev", drop_first=True)

# Build feature matrix
X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
X_bin = df[bin_cols].copy()
X = pd.concat([X_cat, X_bin, device_dummies], axis=1).fillna(0).astype(float)

feature_names = X.columns.tolist()

# ---- Helper: bootstrap 95% CI for odds ratios
def fit_lr_with_ci(X: pd.DataFrame, y: pd.Series, n_boot: int = 500):
    """Fit logistic regression and return OR with bootstrap 95% CIs."""
    model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    model.fit(X, y)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    coefs = model.coef_[0]

    boot_coefs = []
    for _ in range(n_boot):
        X_b, y_b = resample(X, y, stratify=y, random_state=None)
        m = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
        m.fit(X_b, y_b)
        boot_coefs.append(m.coef_[0])

    boot_coefs = np.array(boot_coefs)
    ci_lo = np.exp(np.percentile(boot_coefs, 2.5, axis=0))
    ci_hi = np.exp(np.percentile(boot_coefs, 97.5, axis=0))
    ors   = np.exp(coefs)

    result = pd.DataFrame({
        "feature":  feature_names,
        "OR":       ors.round(3),
        "CI_lo":    ci_lo.round(3),
        "CI_hi":    ci_hi.round(3),
        "coef":     coefs.round(4),
    }).sort_values("OR", ascending=False)

    return result, auc

print("\n  Fitting loss-risk model (outcome: any loss) …")
loss_results, auc_loss = fit_lr_with_ci(X, df["y_loss"])
print(f"  AUC (loss risk) = {auc_loss:.3f}")

# Filter to non-device rows for clean reporting
mask = ~loss_results["feature"].str.startswith("dev_")
print("\n  LOSS RISK odds ratios (excluding device fixed effects):")
print(loss_results[mask][["feature", "OR", "CI_lo", "CI_hi"]].to_string(index=False))

print("\n  Fitting burst-risk model (outcome: loss ≥ 3) …")
burst_results, auc_burst = fit_lr_with_ci(X, df["y_burst"])
print(f"  AUC (burst risk) = {auc_burst:.3f}")
print("\n  BURST RISK odds ratios (excluding device fixed effects):")
print(burst_results[mask][["feature", "OR", "CI_lo", "CI_hi"]].to_string(index=False))

# Merge into single table matching thesis Table 7.8
t78 = loss_results[mask][["feature", "OR", "CI_lo", "CI_hi"]].copy()
t78.columns = ["condition", "loss_OR", "loss_CI_lo", "loss_CI_hi"]
t78_b = burst_results[mask][["feature", "OR", "CI_lo", "CI_hi"]].copy()
t78_b.columns = ["condition", "burst_OR", "burst_CI_lo", "burst_CI_hi"]
t78_full = t78.merge(t78_b, on="condition")
t78_full.to_csv(OUT_DIR / "result_table_7_8.csv", index=False)
print(f"\n  Saved Table 7.8 → {OUT_DIR / 'result_table_7_8.csv'}")

print("\nAll analysis complete. Run 04_figures.py to generate thesis figures.")
