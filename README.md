# Dynamic Pricing Engine

**Business problem.** A ride-hailing platform must choose a surge multiplier in real time. Too low leaves revenue on the table, too high loses riders. How price-sensitive is demand, and can an algorithm learn a better pricing policy than fixed rules?

**Key result.**
- Price elasticity, Bayesian estimate on 693k Boston rides (Nov 2018): −0.007 (94% HDI −0.029 to +0.016). The interval includes zero: in this dataset, demand shows no measurable response to surge.
- In a simulator built on that estimate, a contextual Thompson Sampling policy earns 42.7% more revenue than random pricing over 5,000 rounds (regret 1,052); decaying epsilon-greedy reaches the lowest regret (782).
- Tabular Q-learning underperforms the random baseline (−4.9%), while a simple heuristic gains +2.1%.

Because the simulator inherits a near-zero elasticity, it rewards higher surge almost mechanically; the bandit results show the method, not a deployable price level.

**Stack.** Python 3.11–3.12, PyMC and ArviZ, NumPy, pandas, scikit-learn, Streamlit, Poetry.

**How to run.**
```bash
make install           # poetry install
make notebook          # run notebooks 01 → 04 in order; they write outputs/*.json
poetry run streamlit run streamlit_app.py
make test
```

The Kaggle data is already in `data/`; `make data` re-downloads it (Kaggle CLI required).

---

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
