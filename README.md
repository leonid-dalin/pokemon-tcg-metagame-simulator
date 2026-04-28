# Pokémon TCG Metagame Simulator

This project models competitive **Pokémon Trading Card Game (TCG)** tournaments and long-term metagame shifts. It bridges tournament mathematics with evolutionary game theory to figure out which decks actually win events and which ones define the format over time.

The simulator runs on two main engines:
- **Monte Carlo Bracket Engine🏆** Evaluates short-term tournament success by simulating up to 250,000 iterations. It includes official Play! Pokémon Variant 5 structures, seeded Top Cut pairings, and a Parabolic Tie Convergence model that mirrors real-world match-point decay.
- **Replicator Dynamics Engine🧬** Predicts the Evolutionary Stable State (ESS) over thousands of generations. It finds the equilibrium point where deck frequencies settle, revealing the true shape of the format.

Currently, only the **Monte Carlo Bracket Engine** is fed into the Streamlit dashboard which provides strategic insights from two perspectives: macro-level archetype dominance and individual Expected Value (EV) for surviving the Day 2 bubble.

|         Main Dashboard & Controls         |          Head-to-Head Comparator          |
|:-----------------------------------------:|:-----------------------------------------:|
|        ![Main](docs/img/main.png)         |     ![H2H](docs/img/head-to-head.png)     |
|          **Top Recommendations**          |        **Simulation Diagnostics**         |
| ![Recs](docs/img/top-recommendations.png) | ![Diag](docs/img/simulation_complete.png) |

---

## 🚀 Features

### 🎲 Stochastic Tournament Modelling (Monte Carlo)
*   **High-fidelity Bracket Engine:** The simulator now uses a compiled, multithreaded Rust extension (`tcg_engine`) that entirely bypasses the Python Global Interpreter Lock (GIL). It can run up to a million full tournament iterations in a matter of seconds.
*   **Parabolic Tie Convergence**: Simulates the roughly ~15% real-world match-point bleed. It uses a parabolic probability curve to force ties in mathematically close matchups, remaining completely blind to subjective metrics like deck speed.
*   **Heuristic Rematch Prevention and Attrition:** The engine actively prevents players from hitting the same opponent twice in Swiss rounds using a 1-deep lookahead heuristic, and *can* drop players who accumulate three losses.

### 📊 Advanced Competitive Analytics
*   **Meta & Power Scoring:** Decks receive 0–100 rankings based on their win rates against the field (Power Score) and their overall format dominance (Meta Score). The mathematical framework for these metrics is directly inspired by the **[Vicious Syndicate Data Reaper](https://www.vicioussyndicate.com/drr/faq-data-reaper-report/)** methodology.
*   **Vectorised field constraints.** A high-speed NumPy water-filling algorithm enforces exact, minimum, or maximum field share constraints. You can use the "Omission Split" utility to mathematically normalise the remaining unconstrained archetypes proportionally to historical data.
*  **Clustering and RPS detection:** The project groups archetypes by their strategic matchup shape using K-Means and Silhouette Optimisation. Network graphs help identify non-transitive Rock-Paper-Scissors cycles.

### 💻 Containerized Microservice Architecture
> _I completely decoupled the architecture to stop the heavy math from freezing the API._
*   **Defensive Gateway:** FastAPI validates payloads and uses `slowapi` for distributed rate limiting. It enforces strict OWASP security headers.
*   **Headless worker:** A Redis-backed Huey task runner isolates the array calculations. It streams granular progress updates back to the UI via Server-Sent Events (SSE).
* **Autonomous scraping:** A daily cron job automatically fetches live HTML from LimitlessTCG. It parses the DOM, runs the data through strict Pydantic thermodynamic purity checks, and updates the local JSON store.
* **Distributed tracing:** The entire suite uses OpenTelemetry (OTLP gRPC) for span tracing and `structlog` for machine-readable JSON logging.

### 🧬 Metagame Evolution (Replicator Dynamics)
* **Evolutionary Stability:** Models the format using an Optimistic Multiplicative Weights Update (MWU). It applies gradient extrapolation and ambient entropy regularisation to prevent limit cycles and force convergence into a Nash Equilibrium.
* **Extinction and Reintroduction:** Simulates innovation by letting poorly performing decks die out while randomly reintroducing extinct decks to challenge the established order.

---

## 📦 Installation

> **⚠️ You must use Docker Compose** as the application now relies on a Redis instance for rate limiting and SSE state management, so running it purely locally without Redis will cause the solver and streams to stall.

1. **Clone the repository:**
```bash
git clone https://github.com/leonid-dalin/pokemon-tcg-metagame-simulator.git
cd pokemon-tcg-metagame-simulator
```
2. **Build and launch the services:**
```bash
docker-compose up --build -d
```

This boots three containers:
- the FastAPI gateway (`api` on port 8000),
- the Huey task runner (`worker`), 
- and the Streamlit frontend (`ui` on port 8501).

---

## 🛠️ Usage

### Running the Web Dashboard

Open `http://localhost:8501` in your browser to access the Streamlit UI.

The simulator also supports headless execution through the CLI for batch experiments.

## 🧬 Engine 1: Long-Term Metagame Evolution (ESS)

Predict the Evolutionary Stable State over thousands of generations using Replicator Dynamics.

```bash
python -m src.ui.cli -i data/input/ea_input.json -g 10_000 --mode replicator
```

## 🏆 Engine 2: Immediate Tournament Solver (Monte Carlo)

Run a quick tournament simulation directly from the terminal to fetch Expected Value (EV).

```bash
python -m src.ui.cli -i data/input/ea_input.json --predict --players 512
```

## Key Command-Line Arguments

| Short Flag | Long Flag                | Description                                                                       | Default               | Type    |
|------------|--------------------------|-----------------------------------------------------------------------------------|-----------------------|---------|
| `-i`       | `--input`                | Path to the matchup data JSON file (e.g., `input/ea_input.json`).                 | `input/ea_input.json` | `str`   |
| `-o`       | `--output`               | Directory to save simulation results, plots, and logs. Created if missing.        | `output/`             | `str`   |
| `-M`       | `--mode`                 | Simulation dynamics engine: `replicator` (ESS) or `tournament` (Agent).           | `replicator`          | `str`   |
| `-g`       | `--gens`                 | Maximum number of generations/epochs to simulate.                                 | `1000`                | `int`   |
| `-m`       | `--min-games`            | Minimum required match volume to include an archetype in the baseline.            | `100`                 | `int`   |
| `-e`       | `--extinction-threshold` | Metagame frequency drop-off point where a deck is considered mathematically dead. | `1e-10`               | `float` |
| `-N`       | `--noise`                | Scale of Gaussian noise injected into generation payoffs.                         | `0.0`                 | `float` |
| `-I`       | `--intro-prob`           | Stochastic probability per generation of a rogue/extinct deck re-entering.        | `0.0`                 | `float` |
| `-s`       | `--seed`                 | Fixed RNG seed for perfectly reproducible experiments.                            | `1312`                | `int`   |
|            | `--tournament-style`     | Bracket execution logic: `pure_swiss` (fast) or `championship_series` (BO3).      | `pure_swiss`          | `str`   |
|            | `--no-plot`              | Suppress generation of interactive HTML Plotly graphs (headless runs).            | `False`               | `bool`  |
| `-C`       | `--cluster`              | Enable Silhouette-optimized K-Means strategic archetype clustering.               | `False`               | `bool`  |
| `-l`       | `--log-level`            | Terminal logging verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`.               | `INFO`                | `str`   |
| `-b`       | `--batch`                | Enable batch execution mode for automated parameter sweeping.                     | `False`               | `bool`  |
| `-c`       | `--batch-config`         | Path to the JSON configuration file defining batch parameters.                    | `None`                | `str`   |
|            | `--predict`              | Bypass evolution entirely and run the static Monte Carlo EV solver.               | `False`               | `bool`  |
| `-P`       | `--players`              | Expected total field size for the upcoming event (Prediction Mode).               | `32`                  | `int`   |
|            | `--meta`                 | Comma-separated custom field constraints (e.g., "DeckA:0.2").                     | `""`                  | `str`   |

> **🚫 Validation:** Invalid values (e.g., `--intro-prob 1.5` or negative `--noise`) will cause immediate, descriptive `argparse` errors.
     

---

## 📂 Output

After running the simulation, a uniquely named directory (e.g., `YYYYMMDD_HHmmss_replicator_gens1000K_CONV@2017410`) will be created within your specified output directory. This directory will contain the following files:

*   `ess_equilibrium.csv`: The final state of the metagame, listing each deck's frequency, activity status, and extinction history.
*   `final_tiers.json`: A tier list (T0, T0.5, T1, T2, T3, T4) generated based on the **final state** of the metagame.
*   `simulation.txt`: A detailed log file of the simulation run, including convergence metrics, tier list summaries, and clustering results.
*   `metagame_evolution.html`: An interactive Plotly graph showing the frequency of each deck over time.
*   `matchup_heatmap.html`: An interactive heatmap of the win rate matrix.
*   `matchup_network.html`: An interactive network graph visualising deck matchups and identified RPS cycles.
*   `deck_similarity.json`: Pairwise strategic similarity matrix using Pearson Correlation.
	
---

## 🤝 Contributing

Contributions are welcome! :D Please open an issue to discuss a feature or bug, and submit a pull request for any changes.

---

## 🙏 Acknowledgements

This project stands on the shoulders of giants and would not be possible without the foundational work and data provided by several key members of the competitive metagaming community.

*   **Limitless TCG**: An enormous shoutout to the team at [Limitless](https://limitlesstcg.com/). This entire project is powered by the comprehensive and meticulously maintained Pokémon TCG data scraped from their website. Their platform is an indispensable resource for the global Pokémon TCG community. If you use their tools (and you should!), please consider supporting them on [Patreon](https://patreon.com/limitlesstcg) to ensure they can continue their fantastic work.

*   **Vicious Syndicate**: I personally owe a profound debt to [Vicious Syndicate](https://www.vicioussyndicate.com/) for pioneering rigorous, community-driven data analytics in the digital card game space. Their *Data Reaper Reports* and podcast set the global gold standard for metagame analysis, serving as an indispensable foundation for my own education in competitive modelling. This project’s Power & Meta Scoring metrics are directly inspired by their standardised 0–100 normalisation methodology. Beyond the math, their commitment to transparency and data integrity has been a guiding light for this project’s philosophy and my own development as a human being. :) 

*   **Dominic Calkosz & HearthNash**: A massive thank you to **Dominic** for his groundbreaking research on game-theoretic metagame analysis in Hearthstone. His project, [HearthNash](https://dominic-calkosz.com/HearthNash), was a direct inspiration for applying evolutionary dynamics and Nash equilibrium concepts to TCG metagames. His academic approach provided the theoretical bedrock for this simulator.

*   **FPL Analytics Community:** My journey was also profoundly shaped by the Discord community of the **FPL Analytics Community**. Engaging in their discussions, learning from their approach to data, and even experimenting around with meta-solvers such as **[Solio](https://fpl.solioanalytics.com/)** and **[FPL Review](https://fplreview.com/)** provided the foundational inspiration for this project. The concepts of "solving" a dynamic, stochastic game and the importance of probabilistic forecasting were directly translated into the **Monte Carlo tournament engine** and **EV prediction logic** used here. Their insights into data-driven decision-making were invaluable to my development as a developer and analyst.

This project is a synthesis of their collective efforts, and I am deeply grateful for the ecosystems they have created, and the bricks they've placed. 🙇 So, **_thank you_** for existing. 

---

## 📜 Licence

<a href="https://github.com/leonid-dalin/pokemon-tcg-metagame-simulator/">Pokémon TCG Metagame Evolution Simulator</a> © 2025 by <a href="https://github.com/leonid-dalin/">Leonid Dalin</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a><br>
<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;">

The use of any content in this repository for training any artificial intelligence (AI) model, or for any form of AI to remix, adapt, or build upon my works, especially without my explicit permission, is strictly prohibited.