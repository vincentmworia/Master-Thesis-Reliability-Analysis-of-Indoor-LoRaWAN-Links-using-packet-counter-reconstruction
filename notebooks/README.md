# Event-Conditioned Reliability Analysis of Indoor LoRaWAN Links Using Packet-Counter Reconstruction

**Master's Thesis** — Vincent Mwenda Mworia (Matrikel 1827244)  
EMINENT Programme, University of Siegen  
Submitted 05 August 2026

**Supervisors**  
Prof. Dr. Kristof Van Laerhoven · Nahshon Mokua Obiri, M.Sc.  
Ubiquitous Computing Group, University of Siegen

---

## Overview

This repository contains the analysis code for a master's thesis that uses packet-counter reconstruction to separate firmware-level drops, infrastructure-outage losses, and radio-link losses in a six-device, single-gateway indoor LoRaWAN deployment observed over one year. The reconstructed link reliability is then compared across seven event conditions and modeled with logistic regression.

The analysis uses the published dataset by Obiri and Van Laerhoven (2025), available at [Zenodo record 19089760](https://zenodo.org/records/19089760).

## Dataset

The input file is `3.cleaned_dataset_per_device.csv` (349.5 MB) from the Zenodo record above. Place it in a `data/` directory at the repository root before running the notebooks.

**Citation**  
N. M. Obiri and K. Van Laerhoven, "A Comprehensive Indoor LoRaWAN Dataset Integrating Environmental, Radio, and Structural Parameters," *IEEE Access*, vol. 13, 2025. DOI: [10.1109/ACCESS.2025.3569164](https://doi.org/10.1109/ACCESS.2025.3569164)

## Notebooks

The analysis is split across three Jupyter notebooks, designed to be run in order.

| Notebook | Purpose | Key outputs |
|----------|---------|-------------|
| `01_Packet-Counter_Reconstruction.ipynb` | Sorts packets, computes counter increments, reconstructs firmware drops and transmission losses, detects shared infrastructure outages, and computes the three PDR metrics. | `1_reconstructed.csv`, outage summary, reconciliation audit |
| `02_event_conditions.ipynb` | Defines the seven event variables (E1 CO2 tier, E2 office hours, E3 PM2.5 tier, E4 temperature tier, E5 humidity tier, E6 spreading factor, E7 ESP tier) and appends them to the reconstructed dataset. | `2_events.csv`, event distribution figures |
| `03_statistical_analysis.ipynb` | Computes per-event PDR profiles and block-bootstrap confidence intervals (RQ2), fits a single logistic regression model for transmission-loss occurrence (RQ3), and produces forest plots and fill-in sheets. | `3_results.csv`, `3_odds_ratios.csv`, all RQ2/RQ3 figures |

## Event Taxonomy

| ID | Event | Analytical conditions |
|----|-------|----------------------|
| E1 | CO2 tier | Background, moderate, high |
| E2 | Office hours | Office hours, off-hours |
| E3 | PM2.5 tier | Clean, moderate, elevated |
| E4 | Temperature tier | Cool, moderate, warm |
| E5 | Humidity tier | Dry, normal, humid |
| E6 | Spreading factor | SF7 through SF12 |
| E7 | ESP tier | Poor, moderate, strong |

## Key Results

- **PDR_system = 83.77%** (all loss sources) vs **PDR_link = 97.83%** (radio-link losses only, outside inferred outages)
- Firmware-level drops (8.10%) and infrastructure outages (6.48%) account for most missing packets
- Elevated PM2.5 produces the largest supported reliability reduction (ΔPDR = −20.01 pp) and the highest adjusted odds ratio (OR = 12.52)
- Per-device PDR_link spans only 1.81 pp across 8–40 m, with the most distant device the most reliable
- Loss-occurrence model AUC = 0.691 ± 0.031 (five-fold block-grouped cross-validation)

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib, scipy, scikit-learn
- Jupyter Notebook or JupyterLab

## Reproduction

1. Download `3.cleaned_dataset_per_device.csv` from the Zenodo record and place it in `data/`.
2. Run the three notebooks in order (01 → 02 → 03).
3. Each notebook reads the output of the previous one from `data/` and writes its own outputs there.
4. Figures are saved to `Figures/` with filenames matching the thesis figure references.

## License

The analysis code in this repository is provided for academic reproducibility. The underlying dataset is published under its own license at the Zenodo record linked above.
