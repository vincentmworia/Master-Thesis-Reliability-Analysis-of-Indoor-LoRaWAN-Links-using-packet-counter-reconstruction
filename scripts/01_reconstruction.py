# =============================================================================
# NOTEBOOK 01 — PACKET COUNTER RECONSTRUCTION
# Thesis: Event-Conditioned Reliability Analysis of Indoor LoRaWAN Links
# Author: Vincent Mwenda Mworia | University of Siegen | EMINENT Programme
#
# Purpose : Reconstruct missing packets from p_count gaps per device.
#           Produces data/reconstructed.csv used by all subsequent notebooks.
# Thesis  : Implements Section 4.2 (Def. 1) and Section 5.1–5.2
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. PATHS
# ---------------------------------------------------------------------------
DATA_DIR = Path("../data")
RAW_CSV  = DATA_DIR / "3_cleaned_dataset_per_device.csv"
OUT_CSV  = DATA_DIR / "reconstructed.csv"

# ---------------------------------------------------------------------------
# 1. LOAD & SORT
#    - Parse timestamps as timezone-aware UTC (avoids DST ambiguity).
#    - Sort per device by time — InfluxDB export order is not guaranteed.
# ---------------------------------------------------------------------------
print("Loading dataset …")
df = pd.read_csv(RAW_CSV)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values(["device_id", "time"]).reset_index(drop=True)

print(f"  Rows loaded : {len(df):,}")
print(f"  Devices     : {sorted(df['device_id'].unique())}")
print(f"  Campaign    : {df['time'].min().date()} → {df['time'].max().date()}")

# ---------------------------------------------------------------------------
# 2. PER-DEVICE RECONSTRUCTION  (Definition 1, Section 4.2)
#
#   For consecutive received packets i and i+1 from the same device:
#       Δ = p_count[i+1] − p_count[i]
#       loss[i] = max(Δ − 1, 0)   if Δ > 0  (normal interval)
#       loss[i] = 0                if Δ ≤ 0  (counter reset — excluded)
#
#   Columns added:
#       loss     : reconstructed missing packets in this interval (int ≥ 0)
#       is_reset : True when counter decreased (device reboot)
#       total_tx : 1 (received) + loss (inferred missing) = transmission attempts
# ---------------------------------------------------------------------------
def reconstruct(grp: pd.DataFrame) -> pd.DataFrame:
    """Apply reconstruction formula to one device's sorted reception records."""
    grp = grp.copy().reset_index(drop=True)
    delta = grp["p_count"].diff()           # Δ between consecutive rows

    is_reset = delta < 0                    # counter decreased → reboot
    loss     = (delta - 1).clip(lower=0)   # Δ − 1 = missing packets
    loss[is_reset] = 0                      # reset intervals: loss unknown → 0
    loss.iloc[0]   = 0                      # first row has no prior interval

    grp["loss"]     = loss.astype(int)
    grp["is_reset"] = is_reset.fillna(False)
    grp["total_tx"] = 1 + grp["loss"]      # 1 received + N inferred lost
    return grp


print("\nRunning per-device reconstruction …")
df = df.groupby("device_id", group_keys=False).apply(reconstruct)

# ---------------------------------------------------------------------------
# 3. VERIFICATION SUMMARY  (expected values from thesis Table 7.1)
# ---------------------------------------------------------------------------
print("\n=== RECONSTRUCTION SUMMARY ===")
print(f"{'Device':<8} {'Received':>10} {'Recon.Lost':>12} {'Total TX':>10} "
      f"{'PDR (%)':>9} {'Resets':>7}")
print("-" * 62)

campaign_rx = campaign_lost = 0
for dev, g in df.groupby("device_id"):
    rx    = len(g)
    lost  = int(g["loss"].sum())
    total = rx + lost
    pdr   = rx / total * 100
    resets = int(g["is_reset"].sum())
    campaign_rx   += rx
    campaign_lost += lost
    print(f"{dev:<8} {rx:>10,} {lost:>12,} {total:>10,} {pdr:>9.2f} {resets:>7}")

print("-" * 62)
campaign_total = campaign_rx + campaign_lost
campaign_pdr   = campaign_rx / campaign_total * 100
print(f"{'ALL':<8} {campaign_rx:>10,} {campaign_lost:>12,} "
      f"{campaign_total:>10,} {campaign_pdr:>9.2f}")

# ---------------------------------------------------------------------------
# 4. p_count vs f_count DIVERGENCE  (Section 5.2.1, Table 5.2)
#
#    p_count grows on every TX attempt; f_count grows only on reception.
#    Divergence ≈ cumulative packet loss — independent cross-check.
# ---------------------------------------------------------------------------
print("\n=== p_count vs f_count DIVERGENCE (Table 5.2) ===")
div = (
    df.groupby("device_id")
    .agg(
        p_min=("p_count", "min"), p_max=("p_count", "max"),
        f_min=("f_count", "min"), f_max=("f_count", "max"),
    )
    .reset_index()
)
div["p_range"]    = div["p_max"] - div["p_min"]
div["f_range"]    = div["f_max"] - div["f_min"]
div["divergence"] = div["p_range"] - div["f_range"]
div["pct"]        = (div["divergence"] / div["p_range"] * 100).round(1)

print(div[["device_id", "p_range", "f_range", "divergence", "pct"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. LARGE-GAP AUDIT  (Section 5.2 — intervals > 1000 consecutive losses)
#    These likely represent device outages or server disconnections.
# ---------------------------------------------------------------------------
large_gaps = df[df["loss"] > 1000][["device_id", "time", "loss"]].copy()
print(f"\n=== LARGE GAPS (loss > 1000) ===")
if large_gaps.empty:
    print("  None found.")
else:
    print(large_gaps.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------------------------
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}  ({len(df):,} rows, {len(df.columns)} columns)")
print("Done. Run 02_events.py next.")
