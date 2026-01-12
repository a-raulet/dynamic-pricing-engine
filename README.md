# Dynamic Pricing Engine

**Surge Pricing Analysis for Ride-Sharing Platforms**

A comprehensive data science project exploring dynamic pricing strategies used by Uber and Lyft, progressing from classical econometrics to reinforcement learning.

## Project Overview

This project analyzes surge pricing patterns in ride-sharing data from Boston, MA, demonstrating:

1. **Exploratory Data Analysis** - Understanding when and why surge pricing occurs
2. **Price Elasticity Modeling** - Bayesian estimation of demand sensitivity
3. **Contextual Bandits** - Real-time price optimization with exploration/exploitation
4. **Reinforcement Learning** - Long-term revenue optimization with Q-learning

## Dataset

**Uber & Lyft Cab Prices (Boston, MA)**
- Source: [Kaggle](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices)
- ~700K rides from November 2018
- Includes weather data, surge multipliers, and ride details

## Quick Start

```bash
# Install dependencies
poetry install

# Download dataset (requires Kaggle CLI)
make data

# Start Jupyter notebook
make notebook
```

## Project Structure

```
dynamic-pricing-engine/
├── data/
│   ├── raw/                    # Original Kaggle dataset
│   └── processed/              # Cleaned and enriched data
├── notebooks/
│   ├── 01_eda_and_story.ipynb          # EDA & business narrative
│   ├── 02_price_elasticity_bayesian.ipynb  # Bayesian elasticity model
│   ├── 03_contextual_bandits.ipynb     # Thompson Sampling
│   └── 04_reinforcement_learning.ipynb # Q-learning agent
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Pricing models
│   ├── simulation/             # RL environment
│   └── visualization/          # Publication-quality plots
├── tests/                      # Test suite
└── outputs/figures/            # Exported visualizations
```

## Key Findings

*To be updated as analysis progresses.*

## Tech Stack

- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Bayesian Modeling**: PyMC, ArviZ
- **Machine Learning**: scikit-learn

## Author

Arnaud

---

*Part of a data science portfolio focused on marketing analytics and pricing strategy.*
