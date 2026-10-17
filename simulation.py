"""
Economic Wealth Inequality & Gini Index Simulation (Monte Carlo)
Author: Miguel Díaz Esperón
License: MIT
"""

import argparse
import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


class Worker:
    """Represents an economic agent with an occupation, salary, volatility, and wealth."""

    def __init__(self, job: str, salary: float, volatility: float) -> None:
        self.job = job
        self.salary = float(salary)
        self.volatility = float(volatility)
        self.wealth = 0.0

    def __str__(self) -> str:
        return f"Job: {self.job}, Salary: {self.salary}, Volatility: {self.volatility}, Wealth: {self.wealth:.2f}"


# Occupational labor market parameters: base salary and annual return/income volatility
JOB_MARKET: Dict[str, Dict[str, float]] = {
    "Farmer": {"salary": 25000.0, "volatility": 0.05},
    "Merchant": {"salary": 70000.0, "volatility": 0.10},
    "Engineer": {"salary": 120000.0, "volatility": 0.03},
    "Businessman": {"salary": 250000.0, "volatility": 0.15},
}


def compute_gini(wealth: np.ndarray, population_size: int) -> float:
    """
    Computes the Gini coefficient for a given wealth distribution vector.

    Formula:
        G = sum((2*i - n - 1) * w_i) / (n * sum(w_i))
    where w is sorted in ascending order and i is 1-indexed.
    """
    sorted_wealth = np.sort(wealth)
    total_wealth = np.sum(sorted_wealth)
    if total_wealth == 0:
        return 0.0

    index = np.arange(1, population_size + 1)
    gini = np.sum((2 * index - population_size - 1) * sorted_wealth) / (
        population_size * total_wealth
    )
    return float(gini)


def generate_population(n: int) -> List[Worker]:
    """Generates an initial population of n workers with uniformly sampled occupations."""
    job_titles = list(JOB_MARKET.keys())
    population: List[Worker] = []
    for _ in range(n):
        job = random.choice(job_titles)
        profile = JOB_MARKET[job]
        population.append(Worker(job, profile["salary"], profile["volatility"]))
    return population


def simulate_round(population: List[Worker], enable_taxes: bool) -> float:
    """
    Simulates one annual fiscal cycle for the population and returns the Gini coefficient.

    Wealth dynamics model:
        W_t = max(0, Y_eff + W_{t-1} * exp(sigma * eps - sigma^2 / 2) - CostOfLiving)
    """
    n = len(population)
    salaries = np.array([w.salary for w in population], dtype=float)
    volatilities = np.array([w.volatility for w in population], dtype=float)
    wealth = np.array([w.wealth for w in population], dtype=float)

    # Cost of living brackets based on gross income
    cost_of_living = np.zeros(n)
    cost_of_living[salaries < 20000] = 10000.0
    cost_of_living[(salaries >= 20000) & (salaries < 50000)] = 15000.0
    cost_of_living[(salaries >= 50000) & (salaries < 100000)] = 30000.0
    cost_of_living[salaries >= 100000] = 75000.0

    # Progressive taxation policy
    effective_salaries = salaries.copy()
    if enable_taxes:
        tax_multipliers = np.ones(n)
        tax_multipliers[(salaries >= 20000) & (salaries < 50000)] = 0.95
        tax_multipliers[(salaries >= 50000) & (salaries < 100000)] = 0.75
        tax_multipliers[salaries >= 100000] = 0.45
        effective_salaries *= tax_multipliers

    # Stochastic wealth evolution with Geometric Brownian Motion drift adjustment
    shocks = np.random.normal(0.0, 1.0, n)
    gbm_growth = np.exp(volatilities * shocks - (volatilities**2) / 2.0)
    updated_wealth = effective_salaries + (wealth * gbm_growth) - cost_of_living
    updated_wealth = np.maximum(0.0, updated_wealth)

    # Synchronize state back to Worker objects
    for i, worker in enumerate(population):
        worker.wealth = float(np.round(updated_wealth[i], 2))

    return compute_gini(updated_wealth, n)


def run_monte_carlo(
    population_size: int, years: int, enable_taxes: bool, rounds: int
) -> np.ndarray:
    """Executes Monte Carlo rounds and returns a 2D array of Gini trajectories (rounds x years)."""
    all_ginis: List[List[float]] = []
    for _ in range(rounds):
        population = generate_population(population_size)
        trajectory: List[float] = []
        for _ in range(years):
            gini = simulate_round(population, enable_taxes)
            trajectory.append(gini)
        all_ginis.append(trajectory)
    return np.array(all_ginis)


def plot_simulation(
    all_ginis: np.ndarray, years: int, output_path: str = "simulation.png"
) -> None:
    """Renders and saves publication-quality visualization of the Monte Carlo simulation."""
    plt.figure(figsize=(12, 7), dpi=300)

    for i in range(len(all_ginis)):
        plt.plot(
            range(1, years + 1),
            all_ginis[i],
            alpha=0.35,
            linewidth=0.8,
            label="Simulation rounds" if i == 0 else None,
        )

    percentile_25 = np.percentile(all_ginis, 25, axis=0)
    percentile_75 = np.percentile(all_ginis, 75, axis=0)
    mean_trajectory = np.mean(all_ginis, axis=0)

    plt.fill_between(
        range(1, years + 1),
        percentile_25,
        percentile_75,
        alpha=0.35,
        color="black",
        label="25th-75th Percentile Band",
    )
    plt.plot(
        range(1, years + 1),
        mean_trajectory,
        color="black",
        linewidth=2.5,
        label="Mean Trajectory",
    )

    plt.title("Monte Carlo Wealth Inequality Simulation (Gini Index Over Time)", fontsize=14, fontweight="bold")
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Gini Coefficient", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Simulation plot successfully saved to '{output_path}'.")


def get_user_data_interactive() -> Tuple[int, int, bool, int]:
    """Interactively prompts user for simulation parameters."""
    while True:
        try:
            population = int(input("Population number?: "))
            if population <= 0:
                print("Invalid answer. Please enter a positive integer.")
            else:
                break
        except ValueError:
            print("Invalid answer. Please enter an integer.")

    while True:
        try:
            years = int(input("Years taken?: "))
            if years <= 0:
                print("Invalid answer. Please enter a positive integer.")
            else:
                break
        except ValueError:
            print("Invalid answer. Please enter an integer.")

    while True:
        answer = input("Enable taxes? (Yes/No): ").strip().capitalize()
        if answer == "Yes":
            taxes = True
            break
        elif answer == "No":
            taxes = False
            break
        else:
            print("Invalid answer. Please type 'Yes' or 'No'.")

    while True:
        try:
            rounds = int(input("Rounds taken?: "))
            if rounds <= 0:
                print("Invalid answer. Please enter a positive integer.")
            else:
                break
        except ValueError:
            print("Invalid answer. Please enter an integer.")

    return population, years, taxes, rounds


def parse_arguments() -> argparse.Namespace:
    """Parses optional command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Economic Wealth Inequality & Gini Index Monte Carlo Simulation"
    )
    parser.add_argument("--population", "-p", type=int, default=None, help="Population size")
    parser.add_argument("--years", "-y", type=int, default=None, help="Number of simulated years")
    parser.add_argument("--rounds", "-r", type=int, default=None, help="Number of Monte Carlo iterations")
    parser.add_argument(
        "--taxes",
        dest="taxes",
        action="store_true",
        default=None,
        help="Enable progressive taxation",
    )
    parser.add_argument(
        "--no-taxes",
        dest="taxes",
        action="store_false",
        help="Disable progressive taxation",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="simulation.png",
        help="Destination path for generated plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if (
        args.population is not None
        and args.years is not None
        and args.rounds is not None
        and args.taxes is not None
    ):
        population = args.population
        years = args.years
        taxes = args.taxes
        rounds = args.rounds
    else:
        population, years, taxes, rounds = get_user_data_interactive()

    print(
        f"\nStarting simulation: Population={population}, Years={years}, Taxes={'Enabled' if taxes else 'Disabled'}, Rounds={rounds}..."
    )
    ginis = run_monte_carlo(population, years, taxes, rounds)
    plot_simulation(ginis, years, args.output)


if __name__ == "__main__":
    main()
