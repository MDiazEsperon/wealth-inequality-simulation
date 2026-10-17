# Economic Wealth Inequality & Gini Index Simulation (Monte Carlo)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=matplotlib&logoColor=black)](https://matplotlib.org/)

An agent-based, stochastic Monte Carlo computational model designed to simulate wealth accumulation dynamics, occupational income volatility, progressive fiscal policy effects, and temporal evolution of the **Gini coefficient** across an artificial economy.

---

## Table of Contents
- [Overview](#overview)
- [Theoretical & Mathematical Framework](#theoretical--mathematical-framework)
  - [Labor Market & Agent Classes](#labor-market--agent-classes)
  - [Stochastic Wealth Evolution (Geometric Brownian Motion)](#stochastic-wealth-evolution)
  - [Cost of Living & Fiscal Policy](#cost-of-living--fiscal-policy)
  - [Vectorized Gini Index Formulation](#vectorized-gini-index-formulation)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
  - [Interactive Mode](#interactive-mode)
  - [Automated CLI Mode](#automated-cli-mode)
- [Results & Visualizations](#results--visualizations)
- [License](#license)
- [Author](#author)

---

## Overview

Income and wealth inequality dynamics are governed by non-linear interactions among labor market stratification, return-on-capital variance, baseline subsistence costs, and redistributive tax systems. 

This repository implements a discrete-time, agent-based Monte Carlo engine simulating populations of economic agents over multi-year horizons. By aggregating independent stochastic simulation paths, the framework computes mean trajectory paths and 25th–75th interquartile confidence bands for economic inequality as measured by the Gini index.

---

## Theoretical & Mathematical Framework

### Labor Market & Agent Classes
The economy samples agents across four primary occupational archetypes with empirical stylized facts regarding median salary and annual volatility:

| Occupation | Base Annual Salary ($Y_0$) | Volatility ($\sigma$) | Economic Profile |
| :--- | :---: | :---: | :--- |
| **Farmer** | \$25,000 | 0.05 | Low variance, subsistence base |
| **Merchant** | \$70,000 | 0.10 | Moderate variance, trade-exposed |
| **Engineer** | \$120,000 | 0.03 | High stability, technical human capital |
| **Businessman** | \$250,000 | 0.15 | High return skewness and capital volatility |

---

### Stochastic Wealth Evolution
Each agent's accumulated wealth $W_{i, t}$ evolves annually according to an income-augmented Geometric Brownian Motion (GBM) with Ito-drift correction:

$$
W_{i, t} = \max\left(0, \; Y_{i, \text{eff}} + W_{i, t-1} \cdot \exp\left(\sigma_i \epsilon_{i, t} - \frac{\sigma_i^2}{2}\right) - C_i\right)
$$

where:
- $Y_{i, \text{eff}}$ is the net income after tax deductions.
- $\sigma_i$ is the occupational volatility parameter.
- $\epsilon_{i, t} \sim \mathcal{N}(0, 1)$ represents idiosyncratic macroeconomic and investment shocks.
- $C_i$ represents subsistence cost of living.
- A non-negativity constraint $\max(0, \cdot)$ prevents negative terminal wealth (insolvency floor).

---

### Cost of Living & Fiscal Policy
Subsistence expenditures scale non-linearly with income:
- $Y < \$20,000 \implies C = \$10,000$
- $\$20,000 \le Y < \$50,000 \implies C = \$15,000$
- $\$50,000 \le Y < \$100,000 \implies C = \$30,000$
- $Y \ge \$100,000 \implies C = \$75,000$

When progressive taxation is activated:
- Tier 1 (\$20k–\$50k): 5% effective tax rate ($\times 0.95$)
- Tier 2 (\$50k–\$100k): 25% effective tax rate ($\times 0.75$)
- Tier 3 ($\ge \$100k$): 55% effective tax rate ($\times 0.45$)

---

### Vectorized Gini Index Formulation
The Gini coefficient is calculated at each simulation step across the wealth vector $\mathbf{w}$ sorted in ascending order ($w_1 \le w_2 \le \dots \le w_n$):

$$
G = \frac{\sum_{i=1}^n (2i - n - 1) w_i}{n \sum_{i=1}^n w_i}
$$

The computation runs in $\mathcal{O}(n \log n)$ time using fully vectorized NumPy array operations.

---

## Key Features

- **Vectorized Monte Carlo Engine**: Fast execution of large populations ($n \ge 10,000$) over decades of simulated time.
- **Parametric Policy Testing**: Directly compare laissez-faire vs. progressive redistributive tax regimes.
- **Statistical Envelopes**: Visualizes mean trajectories alongside 25th–75th interquartile range (IQR) bands.
- **Dual Interface**: Interactive CLI prompts for exploratory analysis, alongside a fully automated flags interface (`argparse`) for batch experiments.

---

## Project Structure

```
wealth-inequality-simulation/
├── simulation.py        # Core simulation engine, agent classes, and plotting
├── requirements.txt     # Python dependencies (NumPy, Matplotlib)
├── LICENSE              # MIT License
└── README.md            # Model documentation & mathematical formulation
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/MDiazEsperon/wealth-inequality-simulation.git
cd wealth-inequality-simulation

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### Interactive Mode
Run the script without arguments to be guided through the interactive parameter setup:
```bash
python simulation.py
```
```
Population number?: 1000
Years taken?: 50
Enable taxes? (Yes/No): Yes
Rounds taken?: 100
```

### Automated CLI Mode
Provide parameters directly via command-line flags:
```bash
# Run simulation with taxes enabled
python simulation.py --population 2000 --years 40 --rounds 50 --taxes --output simulation.png

# Run simulation without taxes
python simulation.py --population 2000 --years 40 --rounds 50 --no-taxes --output simulation_notax.png
```

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-p, --population` | Number of economic agents | Interactive prompt |
| `-y, --years` | Simulation duration in years | Interactive prompt |
| `-r, --rounds` | Monte Carlo independent replications | Interactive prompt |
| `--taxes / --no-taxes` | Toggle progressive taxation | Interactive prompt |
| `-o, --output` | Filename for exported plot | `simulation.png` |

---

## Results & Visualizations

When executed, the simulation outputs high-resolution charts depicting the distribution of inequality trajectories over the simulated timeline:

- **Individual Paths**: Translucent traces showing stochastic volatility across individual Monte Carlo runs.
- **Mean Trajectory**: Bold line highlighting the expected long-run inequality equilibrium.
- **Interquartile Band**: Shaded region showing the 50% central probability envelope (25th to 75th percentiles).

---

## License

This project is open-source under the [MIT License](LICENSE).

---

## Author

**Miguel Díaz Esperón**  
- GitHub: [@MDiazEsperon](https://github.com/MDiazEsperon)  
- Repository: [wealth-inequality-simulation](https://github.com/MDiazEsperon/wealth-inequality-simulation)
