# DiffPMV-HVAC

This is the source code repository for the Applied Energy paper:

> **"Occupant-Centric HVAC Control using End-to-end Differentiable Framework for Optimal Thermal Comfort and Energy Efficiency"**
>by HuiSeong Lee, Kiwon Um, and JungHyun Han.

---

## Overview

This repository implements an end-to-end differentiable HVAC control optimization framework that directly neutralizes the Predicted Mean Vote (PMV) thermal comfort index via gradient-based optimization.

The key idea is to integrate 3D airflow and temperature simulations and the PMV evaluation into a single differentiable pipeline using automatic differentiation ([PhiFlow](https://github.com/tum-pbs/PhiFlow)). 
This allows exact gradients with respect to the HVAC control variables — airflow speed, supply temperature, and outlet angle.

---


## Installation


### Setting Up the Environment

Clone the repository and create the conda environment from the provided `environment.yml`:

```bash
git clone https://github.com/medialab-ku/DiffPMV-HVAC
cd PMV_DP
conda env create -f environment.yml
conda activate DiffPMV
```

### Key Dependencies

| Package | Version |
|---|---|
| [PhiFlow](https://github.com/tum-pbs/PhiFlow) | 2.5.3 |
| [PyTorch](https://pytorch.org/get-started/locally/) | 2.7.1+cu128 |
| Gymnasium | 1.2.2 |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) | 2.7.0 |

with CUDA-capable GPU (tested with CUDA 11.8 / 12.8).


---

## Project Structure

```
PMV_DP/
├── src/
│   ├── __init__.py          # Core dataclasses and shared imports
│   ├── config.py            # code configuration (MODIFY HERE)
│   ├── env.py               # Environment setup
│   ├── forward.py           # Differentiable forward pass
│   ├── losses.py            # Loss function
│   ├── main.py              # OPTIMIZATION / SIMULATION
│   ├── setting_exporter.py  # Exports case settings and initial control variables
│   │
│   ├── MPC.py, MPC_run.py   # for MPC baseline
│   ├── RL.py, RL_run.py     # for RL baseline
│   │
│   ├── Cases/               # Scenario configuration files
│   └── Results/             # execution results (auto-generated, timestamped)
└── environment.yml
```

---

## Experiments

Three cases from the paper are provided under `src/Cases/`.

To regenerate case files:

```bash
python -m src.setting_exporter Case1
python -m src.setting_exporter Case2
python -m src.setting_exporter Case3
```

or you can generate new case file by adding new function at `setting_exporter.py`.

---

## Running

### Configuration

Before running, edit the `## MODIFY HERE ##` block in [src/config.py](src/config.py):

```python
## MODIFY HERE ##

setting_fileName = "Case1"      # Case to run
mode = "OPTIMIZATION"           # Run mode: "OPTIMIZATION" or "SIMULATION"
```

- **`OPTIMIZATION`**: Runs end-to-end differentiable optimization to find optimal HVAC control variables.
- **`SIMULATION`**: Runs a forward simulation only with the initial control variables (The results of RBC baseline can be obtained from this, see [src/setting_exporter.py](src/setting_exporter.py)).

### Per-Case Settings (`setting_exporter.py`)

All case-specific parameters - such as room size, geometry, occupants configuration, hyper parameters, and loss function - are defined in [src/setting_exporter.py](src/setting_exporter.py) and exported to `.yaml` / `.pt` files. 


### Proposed Method

```bash
python -m src.main
```

Results are saved under `src/Results/<timestamp>/`.

### Baselines

```bash
# MPC baseline
python -m src.MPC_run

# RL baseline
python -m src.RL_run

# D-PDE baseline
python -m src.main
```
For D-PDE baseline, you must modify the `mode` into `D-PDE_OPT` or `D-PDE_SIM` in [src/config.py](src/config.py) .

---

## Note on Reproducibility

The code in this repository has been refactored for better readability and generalization after the paper's publication. 
As a result, running this code may yield minor numerical differences compared to the exact figures reported in the paper. 
However, the overall trends and the main conclusions of our research remain strictly consistent.


