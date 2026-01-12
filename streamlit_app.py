"""Dynamic Pricing Engine - Streamlit Dashboard."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.models.bandits import (
    Context,
    EpsilonGreedy,
    UCB,
    ThompsonSampling,
    ContextualThompsonSampling,
    run_bandit_experiment,
)
from src.simulation.environment import SurgePricingEnvironment, RideSharingMDP

# Page config
st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
ROOT = Path(__file__).parent
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"


@st.cache_data
def load_results():
    """Load pre-computed results."""
    results = {}

    elasticity_path = OUTPUTS_DIR / "elasticity_results.json"
    if elasticity_path.exists():
        with open(elasticity_path) as f:
            results["elasticity"] = json.load(f)

    bandit_path = OUTPUTS_DIR / "bandit_results.json"
    if bandit_path.exists():
        with open(bandit_path) as f:
            results["bandits"] = json.load(f)

    rl_path = OUTPUTS_DIR / "rl_results.json"
    if rl_path.exists():
        with open(rl_path) as f:
            results["rl"] = json.load(f)

    return results


@st.cache_data
def load_data():
    """Load processed data if available."""
    parquet_path = DATA_DIR / "processed" / "rides_with_weather.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return None


def main():
    # Sidebar navigation
    st.sidebar.title("🚗 Dynamic Pricing")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "💰 Price Simulator", "🎰 Bandit Comparison", "🤖 RL Simulation", "📈 Data Explorer"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **About**
        Surge pricing analysis for ride-sharing platforms using:
        - Bayesian elasticity estimation
        - Multi-armed bandits
        - Reinforcement learning
        """
    )

    # Load data
    results = load_results()

    if page == "📊 Dashboard":
        show_dashboard(results)
    elif page == "💰 Price Simulator":
        show_price_simulator(results)
    elif page == "🎰 Bandit Comparison":
        show_bandit_comparison()
    elif page == "🤖 RL Simulation":
        show_rl_simulation()
    elif page == "📈 Data Explorer":
        show_data_explorer()


def show_dashboard(results):
    """Main dashboard with key findings."""
    st.title("Dynamic Pricing Engine Dashboard")
    st.markdown("### Key Findings from Surge Pricing Analysis")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    if "elasticity" in results:
        elasticity = results["elasticity"]
        col1.metric(
            "Price Elasticity",
            f"{elasticity['mean_elasticity']:.3f}",
            help="Mean elasticity estimate (negative = demand decreases with price)"
        )
        col2.metric(
            "95% HDI",
            f"[{elasticity['elasticity_hdi_low']:.2f}, {elasticity['elasticity_hdi_high']:.2f}]",
            help="94% Highest Density Interval"
        )

    if "bandits" in results:
        bandits = results["bandits"]
        col3.metric(
            "Best Algorithm",
            bandits["best_algorithm"],
            help="Algorithm with highest cumulative reward"
        )
        best_reward = bandits["final_rewards"][bandits["best_algorithm"]]
        col4.metric(
            "Best Revenue",
            f"${best_reward:,.0f}",
            help="Total revenue from best algorithm"
        )

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Algorithm Performance")
        if "bandits" in results:
            fig, ax = plt.subplots(figsize=(8, 5))
            rewards = results["bandits"]["final_rewards"]
            algos = list(rewards.keys())
            values = list(rewards.values())
            colors = ['#2ecc71' if a == results["bandits"]["best_algorithm"] else '#3498db' for a in algos]

            bars = ax.barh(algos, values, color=colors)
            ax.set_xlabel("Cumulative Revenue ($)")
            ax.set_title("Final Revenue by Algorithm")

            for bar, val in zip(bars, values):
                ax.text(val + 1000, bar.get_y() + bar.get_height()/2,
                       f'${val:,.0f}', va='center', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.subheader("Cumulative Regret")
        if "bandits" in results:
            fig, ax = plt.subplots(figsize=(8, 5))
            regrets = results["bandits"]["final_regrets"]
            algos = list(regrets.keys())
            values = list(regrets.values())

            colors = ['#e74c3c' if v > 10000 else '#f39c12' if v > 1000 else '#2ecc71' for v in values]
            bars = ax.barh(algos, values, color=colors)
            ax.set_xlabel("Cumulative Regret")
            ax.set_title("Final Regret by Algorithm (lower is better)")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Interpretation
    st.markdown("---")
    st.subheader("Interpretation")

    st.markdown("""
    **Key Insights:**

    1. **Near-Zero Elasticity**: The estimated elasticity is close to zero, suggesting demand is
       relatively inelastic to surge pricing in this dataset. This could indicate:
       - Customers accept surge pricing during high-demand periods
       - The surge levels in the data are within an acceptable range

    2. **Contextual Bandits Win**: The Contextual Thompson Sampling algorithm outperforms others,
       demonstrating that **context matters** for optimal pricing decisions.

    3. **Exploration-Exploitation Trade-off**: Epsilon-Greedy with decay performs well, showing
       that starting with exploration and gradually exploiting is effective.
    """)


def show_price_simulator(results):
    """Interactive price simulator."""
    st.title("💰 Dynamic Price Simulator")
    st.markdown("Simulate revenue under different surge pricing scenarios")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")

        base_price = st.slider("Base Price ($)", 5.0, 30.0, 15.0, 0.5)
        surge_multiplier = st.slider("Surge Multiplier", 1.0, 3.0, 1.5, 0.1)

        st.markdown("---")
        st.markdown("**Context**")

        hour = st.slider("Hour of Day", 0, 23, 18)
        is_weekend = st.checkbox("Weekend")
        is_raining = st.checkbox("Raining")
        base_demand = st.slider("Base Demand Level", 0.1, 1.0, 0.5, 0.05)

        # Elasticity parameter
        if "elasticity" in results:
            elasticity = results["elasticity"]["mean_elasticity"]
        else:
            elasticity = -0.8

        elasticity = st.slider("Price Elasticity", -2.0, 0.0, float(elasticity), 0.05)

    with col2:
        st.subheader("Revenue Projection")

        # Calculate demand and revenue
        final_price = base_price * surge_multiplier
        demand_multiplier = surge_multiplier ** elasticity

        # Context adjustments
        context_boost = 1.0
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            context_boost *= 1.3
        if is_raining:
            context_boost *= 1.2
        if is_weekend and (hour >= 22 or hour <= 2):
            context_boost *= 1.4

        adjusted_demand = base_demand * demand_multiplier * context_boost
        expected_rides = adjusted_demand * 100
        expected_revenue = expected_rides * final_price

        # Display metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Final Price", f"${final_price:.2f}")
        m2.metric("Expected Rides", f"{expected_rides:.0f}")
        m3.metric("Expected Revenue", f"${expected_revenue:.0f}")

        # Revenue curve
        st.markdown("---")
        st.markdown("**Revenue vs. Surge Multiplier**")

        surge_range = np.linspace(1.0, 3.0, 50)
        revenues = []
        for s in surge_range:
            dm = s ** elasticity
            adj_d = base_demand * dm * context_boost
            rev = adj_d * 100 * base_price * s
            revenues.append(rev)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(surge_range, revenues, 'b-', linewidth=2)
        ax.axvline(x=surge_multiplier, color='red', linestyle='--', label=f'Current: {surge_multiplier}x')

        optimal_idx = np.argmax(revenues)
        optimal_surge = surge_range[optimal_idx]
        ax.axvline(x=optimal_surge, color='green', linestyle='--', label=f'Optimal: {optimal_surge:.2f}x')

        ax.fill_between(surge_range, revenues, alpha=0.3)
        ax.set_xlabel("Surge Multiplier")
        ax.set_ylabel("Expected Revenue ($)")
        ax.set_title("Revenue Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

        if optimal_surge != surge_multiplier:
            improvement = ((revenues[optimal_idx] - expected_revenue) / expected_revenue) * 100
            st.info(f"💡 Optimal surge is **{optimal_surge:.2f}x** (potential {improvement:.1f}% revenue increase)")


def show_bandit_comparison():
    """Interactive bandit algorithm comparison."""
    st.title("🎰 Multi-Armed Bandit Comparison")
    st.markdown("Run experiments to compare pricing algorithms in real-time")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Experiment Settings")

        n_rounds = st.slider("Number of Rounds", 100, 5000, 1000, 100)
        base_elasticity = st.slider("Base Elasticity", -1.5, -0.2, -0.8, 0.1)
        random_seed = st.number_input("Random Seed", 1, 1000, 42)

        surge_levels = st.multiselect(
            "Surge Levels",
            [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
            default=[1.0, 1.25, 1.5, 2.0, 2.5]
        )

        if len(surge_levels) < 2:
            st.warning("Select at least 2 surge levels")
            return

        run_experiment = st.button("🚀 Run Experiment", type="primary")

    with col2:
        if run_experiment:
            with st.spinner("Running bandit experiment..."):
                # Setup environment
                env = SurgePricingEnvironment(
                    surge_levels=sorted(surge_levels),
                    base_elasticity=base_elasticity,
                    random_seed=random_seed,
                )

                # Setup algorithms
                n_arms = len(surge_levels)
                algorithms = {
                    "Random": EpsilonGreedy(n_arms, epsilon=1.0, epsilon_decay=1.0, random_seed=random_seed),
                    "ε-Greedy (0.1)": EpsilonGreedy(n_arms, epsilon=0.1, epsilon_decay=1.0, random_seed=random_seed),
                    "ε-Greedy (decay)": EpsilonGreedy(n_arms, epsilon=0.3, epsilon_decay=0.995, random_seed=random_seed),
                    "UCB": UCB(n_arms, c=2.0, random_seed=random_seed),
                    "Thompson Sampling": ThompsonSampling(n_arms, random_seed=random_seed),
                    "Contextual TS": ContextualThompsonSampling(n_arms, context_dim=6, random_seed=random_seed),
                }

                # Run experiment
                results = run_bandit_experiment(env, algorithms, n_rounds=n_rounds, random_seed=random_seed)

                # Plot results
                st.subheader("Cumulative Rewards")
                fig, ax = plt.subplots(figsize=(10, 5))

                for name, data in results.items():
                    ax.plot(data["rewards"], label=name, alpha=0.8)

                ax.set_xlabel("Round")
                ax.set_ylabel("Cumulative Revenue ($)")
                ax.legend(loc="upper left")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

                # Regret plot
                st.subheader("Cumulative Regret")
                fig, ax = plt.subplots(figsize=(10, 5))

                for name, data in results.items():
                    ax.plot(data["regrets"], label=name, alpha=0.8)

                ax.set_xlabel("Round")
                ax.set_ylabel("Cumulative Regret")
                ax.legend(loc="upper left")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

                # Summary table
                st.subheader("Final Results")
                summary_data = {
                    "Algorithm": [],
                    "Final Revenue": [],
                    "Final Regret": [],
                }
                for name, data in results.items():
                    summary_data["Algorithm"].append(name)
                    summary_data["Final Revenue"].append(f"${data['rewards'][-1]:,.0f}")
                    summary_data["Final Regret"].append(f"{data['regrets'][-1]:,.0f}")

                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        else:
            st.info("👈 Configure settings and click 'Run Experiment' to start")


def show_rl_simulation():
    """Reinforcement learning simulation."""
    st.title("🤖 RL Pricing Simulation")
    st.markdown("Simulate a full day of dynamic pricing with the MDP environment")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Environment Settings")

        base_price = st.slider("Base Fare ($)", 5.0, 25.0, 15.0, 1.0, key="rl_base")
        base_elasticity = st.slider("Base Elasticity", -1.5, -0.2, -0.8, 0.1, key="rl_elast")
        surge_fatigue = st.slider("Surge Fatigue", 0.0, 0.3, 0.1, 0.02)

        st.markdown("---")
        st.subheader("Strategy")

        strategy = st.radio(
            "Pricing Strategy",
            ["Fixed Surge", "Demand-Based", "Random"],
        )

        if strategy == "Fixed Surge":
            fixed_surge = st.slider("Fixed Surge Level", 1.0, 2.5, 1.25, 0.25)

        run_sim = st.button("▶️ Run Simulation", type="primary")

    with col2:
        if run_sim:
            with st.spinner("Simulating full day..."):
                # Create environment
                env = RideSharingMDP(
                    base_price=base_price,
                    base_elasticity=base_elasticity,
                    surge_fatigue=surge_fatigue,
                    random_seed=42,
                )

                state = env.reset()
                done = False

                while not done:
                    if strategy == "Fixed Surge":
                        # Find closest surge level
                        action = min(range(env.n_actions),
                                   key=lambda i: abs(env.surge_levels[i] - fixed_surge))
                    elif strategy == "Demand-Based":
                        # Higher surge during high demand
                        if state.demand_level == 2:
                            action = 3  # 2.0x
                        elif state.demand_level == 1:
                            action = 1  # 1.25x
                        else:
                            action = 0  # 1.0x
                    else:
                        action = np.random.randint(env.n_actions)

                    state, reward, done, info = env.step(action)

                # Get history
                history = env.get_history_dataframe()
                summary = env.get_episode_summary()

                # Display metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Revenue", f"${summary['total_reward']:,.0f}")
                m2.metric("Total Rides", f"{summary['total_rides']:,}")
                m3.metric("Avg Surge", f"{summary['avg_surge']:.2f}x")
                m4.metric("Avg Price", f"${summary['avg_price']:.2f}")

                # Hourly breakdown
                st.subheader("Hourly Performance")

                hourly = history.groupby("hour").agg({
                    "reward": "sum",
                    "rides": "sum",
                    "surge": "mean",
                    "final_demand": "mean",
                }).reset_index()

                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                # Revenue by hour
                axes[0, 0].bar(hourly["hour"], hourly["reward"], color="#3498db")
                axes[0, 0].set_xlabel("Hour")
                axes[0, 0].set_ylabel("Revenue ($)")
                axes[0, 0].set_title("Revenue by Hour")

                # Rides by hour
                axes[0, 1].bar(hourly["hour"], hourly["rides"], color="#2ecc71")
                axes[0, 1].set_xlabel("Hour")
                axes[0, 1].set_ylabel("Rides")
                axes[0, 1].set_title("Rides by Hour")

                # Surge by hour
                axes[1, 0].plot(hourly["hour"], hourly["surge"], 'o-', color="#e74c3c")
                axes[1, 0].set_xlabel("Hour")
                axes[1, 0].set_ylabel("Avg Surge")
                axes[1, 0].set_title("Average Surge by Hour")
                axes[1, 0].set_ylim(0.9, 2.6)

                # Demand by hour
                axes[1, 1].fill_between(hourly["hour"], hourly["final_demand"], alpha=0.5, color="#9b59b6")
                axes[1, 1].plot(hourly["hour"], hourly["final_demand"], 'o-', color="#9b59b6")
                axes[1, 1].set_xlabel("Hour")
                axes[1, 1].set_ylabel("Demand Level")
                axes[1, 1].set_title("Demand by Hour")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.info("👈 Configure settings and click 'Run Simulation' to start")

            # Show demand curve explanation
            st.markdown("---")
            st.subheader("Demand Pattern")

            env = RideSharingMDP()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.fill_between(range(24), env.hourly_demand, alpha=0.5, color="#3498db")
            ax.plot(range(24), env.hourly_demand, 'o-', color="#3498db")
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Demand Level")
            ax.set_title("Typical Daily Demand Pattern")
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3)

            # Annotate rush hours
            ax.annotate("Morning\nRush", xy=(8, 0.9), ha="center", fontsize=9)
            ax.annotate("Evening\nRush", xy=(18, 0.95), ha="center", fontsize=9)
            ax.annotate("Night", xy=(3, 0.2), ha="center", fontsize=9)

            st.pyplot(fig)
            plt.close()


def show_data_explorer():
    """Explore the raw data."""
    st.title("📈 Data Explorer")

    df = load_data()

    if df is None:
        st.warning("No processed data found. Run the notebooks first to generate data.")
        st.markdown("""
        ```bash
        poetry install
        make data
        make notebook
        ```
        """)
        return

    st.subheader(f"Dataset: {len(df):,} rides")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        if "cab_type" in df.columns:
            cab_types = st.multiselect("Cab Type", df["cab_type"].unique().tolist(),
                                       default=df["cab_type"].unique().tolist())
            df = df[df["cab_type"].isin(cab_types)]

    with col2:
        if "surge_multiplier" in df.columns:
            surge_range = st.slider("Surge Range",
                                   float(df["surge_multiplier"].min()),
                                   float(df["surge_multiplier"].max()),
                                   (1.0, 2.0))
            df = df[(df["surge_multiplier"] >= surge_range[0]) &
                   (df["surge_multiplier"] <= surge_range[1])]

    with col3:
        sample_size = st.slider("Sample Size", 1000, min(50000, len(df)), 5000)

    # Sample data
    df_sample = df.sample(min(sample_size, len(df)))

    # Stats
    st.markdown("---")
    st.subheader("Summary Statistics")

    if "surge_multiplier" in df.columns and "price" in df.columns:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Price", f"${df['price'].mean():.2f}")
        col2.metric("Avg Surge", f"{df['surge_multiplier'].mean():.2f}x")
        col3.metric("% Surge > 1", f"{(df['surge_multiplier'] > 1).mean()*100:.1f}%")
        if "distance" in df.columns:
            col4.metric("Avg Distance", f"{df['distance'].mean():.1f} mi")

    # Visualizations
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Time Analysis", "Raw Data"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            if "surge_multiplier" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                df_sample["surge_multiplier"].hist(bins=30, ax=ax, color="#3498db", edgecolor="white")
                ax.set_xlabel("Surge Multiplier")
                ax.set_ylabel("Frequency")
                ax.set_title("Surge Multiplier Distribution")
                st.pyplot(fig)
                plt.close()

        with col2:
            if "price" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                df_sample["price"].hist(bins=50, ax=ax, color="#2ecc71", edgecolor="white")
                ax.set_xlabel("Price ($)")
                ax.set_ylabel("Frequency")
                ax.set_title("Price Distribution")
                st.pyplot(fig)
                plt.close()

    with tab2:
        if "hour" in df.columns and "surge_multiplier" in df.columns:
            hourly_surge = df.groupby("hour")["surge_multiplier"].mean()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(hourly_surge.index, hourly_surge.values, color="#e74c3c")
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Average Surge")
            ax.set_title("Average Surge by Hour")
            ax.set_xticks(range(0, 24, 2))
            st.pyplot(fig)
            plt.close()

    with tab3:
        st.dataframe(df_sample.head(100), use_container_width=True)


if __name__ == "__main__":
    main()
