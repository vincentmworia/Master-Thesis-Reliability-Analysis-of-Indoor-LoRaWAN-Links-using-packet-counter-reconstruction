# =============================================================================
# NOTEBOOK 02 — EVENT DEFINITION AND DETECTION
# Thesis: Event-Conditioned Reliability Analysis of Indoor LoRaWAN Links
# Author: Vincent Mwenda Mworia | University of Siegen | EMINENT Programme
#
# Purpose : Add all 18 event indicator columns to the reconstructed dataset.
#           Produces data/events.csv used by all analysis and figure notebooks.
# Thesis  : Implements Chapter 6 (Event Definition and Detection) in full.
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. PATHS
# ---------------------------------------------------------------------------
DATA_DIR = Path("../data")
IN_CSV   = DATA_DIR / "reconstructed.csv"
OUT_CSV  = DATA_DIR / "events.csv"

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
print("Loading reconstructed dataset …")
df = pd.read_csv(IN_CSV)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values(["device_id", "time"]).reset_index(drop=True)
print(f"  Rows: {len(df):,}  |  Devices: {sorted(df['device_id'].unique())}")

# ---------------------------------------------------------------------------
# 2. TEMPORAL EVENTS  (Section 6.2)
#
#    All time operations use local German time (UTC+1 winter, UTC+2 summer).
#    pandas tz_convert handles the DST transitions automatically.
# ---------------------------------------------------------------------------
local_time = df["time"].dt.tz_convert("Europe/Berlin")

df["hour"]      = local_time.dt.hour
df["dayofweek"] = local_time.dt.dayofweek   # 0 = Monday … 6 = Sunday
df["month"]     = local_time.dt.month

# E2: Weekday vs Weekend  (Section 6.1.2)
# 1 = weekday (Mon–Fri), 0 = weekend (Sat–Sun)
df["e2_is_weekday"] = (df["dayofweek"] < 5).astype(int)

# E4: Time-of-Day Segment  (Section 6.2.1)
# night=0–6, morning=7–10, peak=11–17, evening=18–23
df["e4_time_of_day"] = pd.cut(
    df["hour"],
    bins=[-1, 6, 10, 17, 23],
    labels=["night", "morning", "peak", "evening"],
)

# E5: Office Hours  (Section 6.2.2)
# Weekday AND hour in [8, 18)
df["e5_office_hours"] = (
    (df["e2_is_weekday"] == 1) & df["hour"].between(8, 17)
).astype(int)

# E6: Season  (Section 6.2.3)
# Campaign spans: Sep 2024 → May 2025  (autumn, winter, spring)
season_map = {9: "autumn", 10: "autumn", 11: "autumn",
              12: "winter",  1: "winter",   2: "winter",
               3: "spring",  4: "spring",   5: "spring"}
df["e6_season"] = df["month"].map(season_map)

# ---------------------------------------------------------------------------
# 3. OCCUPANCY EVENTS — CO2-BASED  (Section 6.1)
# ---------------------------------------------------------------------------

# E1: CO2 Concentration Tier  (Section 6.1.1)
# background ≤ 500 ppm | moderate 500–700 | high > 700
df["e1_co2_tier"] = pd.cut(
    df["co2"],
    bins=[0, 500, 700, 10_000],
    labels=["background", "moderate", "high"],
)

# E3: CO2 Rising Transition  (Section 6.1.3)
# First difference > 20 ppm per 60-second interval → people arriving
df["co2_delta"]   = df.groupby("device_id")["co2"].diff()
df["e3_co2_rising"] = (df["co2_delta"] > 20).astype(int)

# ---------------------------------------------------------------------------
# 4. AIR QUALITY EVENTS — PM2.5  (Section 6.3)
# ---------------------------------------------------------------------------

# E7: PM2.5 Spike — device-specific 90th percentile  (Section 6.3.1)
# Avoids penalising devices in naturally dustier locations
pm25_q90 = df.groupby("device_id")["pm25"].transform(lambda x: x.quantile(0.90))
df["e7_pm25_spike"] = (df["pm25"] > pm25_q90).astype(int)

# E8: PM2.5 Tier — absolute WHO thresholds  (Section 6.3.2)
df["e8_pm25_tier"] = pd.cut(
    df["pm25"],
    bins=[-0.01, 2, 10, 10_000],
    labels=["clean", "moderate", "elevated"],
)

# ---------------------------------------------------------------------------
# 5. ATMOSPHERIC EVENTS  (Section 6.4)
# ---------------------------------------------------------------------------

# E9: Pressure Tier — campaign quartiles  (Section 6.4.1)
df["e9_pressure_tier"] = pd.qcut(
    df["pressure"], q=4,
    labels=["low", "medium_low", "medium_high", "high"],
    duplicates="drop",
)

# E10: Pressure Drop — HVAC / door event  (Section 6.4.2)
# First difference < −0.5 hPa per interval
df["pressure_delta"] = df.groupby("device_id")["pressure"].diff()
df["e10_pressure_drop"] = (df["pressure_delta"] < -0.5).astype(int)

# E11: Humidity Tier — absolute bands  (Section 6.4.3)
df["e11_humidity_tier"] = pd.cut(
    df["humidity"],
    bins=[0, 40, 55, 70, 101],
    labels=["dry", "normal", "humid", "very_humid"],
)

# E12: Temperature Tier — campaign quartiles  (Section 6.4.4)
df["e12_temp_tier"] = pd.qcut(
    df["temperature"], q=4,
    labels=["cold", "cool", "warm", "hot"],
    duplicates="drop",
)

# ---------------------------------------------------------------------------
# 6. TRANSMISSION CONFIGURATION EVENTS  (Section 6.5)
# ---------------------------------------------------------------------------

# E13: Spreading Factor — categorical  (Section 6.5.1)
# SF column already present; rename for clarity
df["e13_sf"] = df["SF"].astype(int)

# E14: SF Tier — low (7–8) vs high (9–10)  (Section 6.5.2)
# Splits on ToA boundary: low ≤ 134 ms, high ≥ 247 ms
df["e14_sf_tier"] = df["e13_sf"].apply(
    lambda sf: "low_sf" if sf in (7, 8) else "high_sf"
)

# E15: Time-on-Air Class  (Section 6.5.3)
# Derived from SF; expressed in ms for collision-risk interpretation
TOA_MS = {7: 71.9, 8: 133.6, 9: 246.8, 10: 452.6}
df["e15_toa_ms"] = df["e13_sf"].map(TOA_MS)
df["e15_toa_class"] = df["e14_sf_tier"].map(
    {"low_sf": "short_toa", "high_sf": "long_toa"}
)

# ---------------------------------------------------------------------------
# 7. BURST LOSS EVENT  (Section 6.6)
# ---------------------------------------------------------------------------

# E16: Loss Type Classification  (Section 6.6.1)
# no_loss=0 | isolated=1 | small_burst=2–4 | large_burst ≥ 5
df["e16_loss_type"] = pd.cut(
    df["loss"],
    bins=[-1, 0, 1, 4, 10_000_000],
    labels=["no_loss", "isolated", "small_burst", "large_burst"],
)

# ---------------------------------------------------------------------------
# 8. SIGNAL CONTEXT EVENTS  (Section 6.7)
# ---------------------------------------------------------------------------

# E17: Pre-Loss RSSI Tier  (Section 6.7.1)
# Uses RSSI of the packet immediately before the loss interval
# Tiers: strong ≥ −70 dBm | moderate −90 to −70 | weak < −90
df["e17_rssi_tier"] = pd.cut(
    df["rssi"],
    bins=[-200, -90, -70, 0],
    labels=["weak", "moderate", "strong"],
)

# E18: ESP Tier  (Section 6.7.2)
# Campaign-quartile tiers on the Effective Signal Power column
df["e18_esp_tier"] = pd.qcut(
    df["esp"], q=3,
    labels=["low_esp", "medium_esp", "high_esp"],
    duplicates="drop",
)

# ---------------------------------------------------------------------------
# 9. SUMMARY — dataset statistics for Chapter 6
# ---------------------------------------------------------------------------
print("\n=== EVENT COLUMN COVERAGE ===")
event_cols = [c for c in df.columns if c.startswith("e") and "_" in c]
print(f"  Total event columns added: {len(event_cols)}")

print("\n=== CO2 STATISTICS (Section 6.1.1) ===")
weekday_co2 = df[df["e2_is_weekday"] == 1]["co2"]
weekend_co2 = df[df["e2_is_weekday"] == 0]["co2"]
print(f"  Weekday CO2 : mean={weekday_co2.mean():.0f} ppm  std={weekday_co2.std():.0f} ppm")
print(f"  Weekend CO2 : mean={weekend_co2.mean():.0f} ppm  std={weekend_co2.std():.0f} ppm")

print("\n=== CO2 TIER DISTRIBUTION ===")
print(df["e1_co2_tier"].value_counts(normalize=True).mul(100).round(1)
      .rename("pct").to_string())

print("\n=== WEEKDAY vs WEEKEND SPLIT ===")
vc = df["e2_is_weekday"].value_counts(normalize=True).mul(100).round(1)
print(f"  Weekday: {vc.get(1, 0):.1f}%  |  Weekend: {vc.get(0, 0):.1f}%")

print("\n=== SF DISTRIBUTION ===")
print(df["e13_sf"].value_counts(normalize=True).mul(100).sort_index()
      .rename("pct").to_string())

print("\n=== PM2.5 DEVICE 90th PERCENTILE THRESHOLDS ===")
print(df.groupby("device_id")["pm25"].quantile(0.90).round(2).to_string())

# ---------------------------------------------------------------------------
# 10. SAVE
# ---------------------------------------------------------------------------
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}  ({len(df):,} rows, {len(df.columns)} columns)")
print("Done. Run 03_analysis.py next.")
