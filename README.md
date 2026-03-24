# Pokémon TCG Metagame Simulator

This project is a high-fidelity analytical suite designed to model both the immediate, stochastic reality of competitive **Pokémon Trading Card Game (TCG)** tournaments and the long-term evolutionary trends of its metagame.

The simulator bridges rigorous tournament mathematics with evolutionary game theory to provide a complete picture of the TCG metagame:
- **Monte Carlo Bracket Engine🏆** Evaluates short-term tournament success by simulating up to 1,000,000 iterations. It natively supports official Play! Pokémon Variant #5 structures, seeded Top Cut pairings, and a Parabolic Tie Convergence model that mathematically mirrors real-world BO3 match-point decay.
- **Replicator Dynamics Engine🧬** Predicts the Evolutionary Stable State (ESS) by iterating through thousands of generations. It identifies the "unexploitable" equilibrium point where deck frequencies stabilize, revealing which archetypes truly define a format over time.

The former is integrated with a Streamlit dashboard, vectorized Swiss metrics, and dynamic Z-score evaluations, it provides deep strategic insights from two perspectives: macro-level archetype dominance and individual Expected Value (EV) for surviving the "Day 2 bubble"

| Main Dashboard & Controls | Head-to-Head Comparator  |
| :---: | :---: |
| ![Main](docs/img/main.png) | ![H2H](docs/img/head-to-head.png) |
| **Top Recommendations** | **Simulation Diagnostics** |
| ![Recs](docs/img/top-recommendations.png) | ![Diag](docs/img/simulation_complete.png) |

---

## 🚀 Features

### 🎲 Stochastic Tournament Modelling (Monte Carlo)
*   **High-Fidelity Bracket Engine** which simulates up to 1,000,000 full tournament iterations (Swiss + Single-Elimination Playoffs) using parallelised NumPy workers.
*   **Parabolic Tie Convergence** which is able to model the ~15% real-world "match-point bleed" using a parabolic probability curve to ensure close matchups correctly result in ties (1 point).
*   **Heuristic Rematch Prevention** helps actively preventing players from hitting the same opponent twice during Swiss rounds.
*   **X-3 Drop Logic** to simulate real-world tournament attrition by dropping players from the bracket once they accumulate three losses.

### 📊 Advanced Competitive Analytics
*   **Meta & Power Scoring.** Normalised 0–100 rankings evaluate decks based on win rates against the field (Power Score) and their overall format dominance (Meta Score). The mathematical framework for these metrics is directly inspired by the **[Vicious Syndicate Data Reaper](https://www.vicioussyndicate.com/drr/faq-data-reaper-report/)** methodology.
*   **Vectorised Metagame Constraints.** A high-speed, pure NumPy water-filling algorithm strictly enforces user-defined exact, minimum, or maximum field share constraints.
*   **Dynamic Clustering & RPS.** Uses K-Means with Silhouette Optimisation and `StandardScaler` to accurately group archetypes based on their strategic matchup shape. Network graphing helps identify non-transitive Rock-Paper-Scissors cycles.

### 💻 Professional Streamlit Interface
*   **Interactive Dashboard.** A sortable, tool-tipped dashboard with instant perspective toggles to seamlessly swap data between Individual EV and Macro Impact without recalculating.
*   **Head-to-Head Comparator.** A side-by-side "A vs. B" tool featuring native colour-coding to highlight favourable (≥ 55%) and unfavourable (≤ 45%) matchups across the predicted field.
*   **Limitless HTML Ingestion.** A native parser extracts metagame shares directly from Limitless Labs exports to instantly pre-populate simulation constraints.

### 🧬 Metagame Evolution (Replicator Dynamics)
*   **Evolutionary Stability.** Models the "Game Theory" of a metagame using Replicator Dynamics. Successful decks grow in frequency proportional to their meta-weighted performance.
*   **Deck Extinction & Reintroduction.** Simulates innovation and meta-shifts. Poorly performing decks can die out, while "rogue" decks can be randomly reintroduced to challenge the established equilibrium.

---

## 📦 Installation

0. **Get [Python 3.12](https://www.python.org/downloads/)**

    **Windows:**
    ```PowerShell
    winget install -e --id Python.Python.3.12
    ```
    *Note: Restart your terminal after installation.*

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/leonid-dalin/pokemon-tcg-metagame-simulator.git
    cd pokemon-tcg-metagame-simulator
    ```

2.  **Set up a virtual environment (Recommended):**

    **Windows:**
    ```bash
    python3.12 -m venv venv
    venv\Scripts\activate
    ```
    **Linux:**
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Install all required Python packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🛠️ Usage

The simulator requires a JSON input file containing the matchup data between decks. You can either create this file manually or use the provided scraper to generate it from HTML files.

### Option 1: Using the Data Scraper

The `scraper.py` script can convert HTML matchup data, specifically from LimitlessTCG, into the `ea_input.json` format required by this tool.

1.  Place your HTML files in a directory named `data/matchups`
2.  Run the scraper:
    ```bash
    python -m src.scraper
    ```
3. The script will process the files and, by default, create `ea_input.json` inside the `data/input` directory, which you can then use with the main simulator.

### Option 2: Manual Input Data

Your input file should be a JSON object. An example structure is provided in `data/input/ea_input.json`.

**Input Data Format (`ea_input.json`)**

```json
{
  "archetypes": [
    "Blissey",
    "Dragapult Dusknoir",
    "Gholdengo"
  ],
  "win_rate_matrix": {
    "Blissey": {
      "Blissey": {
        "win_rate": 0.5,
        "match_count": 0
      },
      "Dragapult Dusknoir": {
        "win_rate": 0.5675675675675675,
        "match_count": 74
      },
      "Gholdengo": {
        "win_rate": 0.5588235294117647,
        "match_count": 68
      }
    },
    "Dragapult Dusknoir": {
      "Blissey": {
        "win_rate": 0.43243243243243246,
        "match_count": 74
      },
      "Dragapult Dusknoir": {
        "win_rate": 0.5,
        "match_count": 0
      },
      "Gholdengo": {
        "win_rate": 0.47058823529411764,
        "match_count": 85
      }
    },
    "Gholdengo": {
      "Blissey": {
        "win_rate": 0.4411764705882353,
        "match_count": 68
      },
      "Dragapult Dusknoir": {
        "win_rate": 0.5294117647058824,
        "match_count": 85
      },
      "Gholdengo": {
        "win_rate": 0.5,
        "match_count": 0
      }
    }
  }
}
```

## 🧬 Engine 1: Long-Term Metagame Evolution (ESS)

Use this mode to predict the **Evolutionary Stable State** (the point where a metagame becomes unexploitable) over thousands of generations using Replicator Dynamics.

```bash
python -m src.main -i data/input/ea_input.json -g 10_000 --mode replicator
```

## 🏆 Engine 2: Immediate Tournament Solver (Monte Carlo)

Use this mode for **Expected Value (EV)** calculations. It evaluates specific tournament equity by simulating up to 1,000,000 brackets based on given data.

### Option A: Interactive Dashboard (Recommended)

The Streamlit UI provides a dual-perspective dashboard for real-time equity evaluation and "Day 2 bubble" analysis.

```bash
streamlit run src/interfaces/app.py
```

### Option B: CLI Prediction Mode

Run a quick tournament simulation directly from the terminal.
```bash
python -m src.main -i data/input/ea_input.json --predict --players 512 
```

## Key Command-Line Arguments

| Short Flag | Long Flag               | Description                                                                 | Default             | Type    |
| ---------- | ----------------------- | --------------------------------------------------------------------------- | ------------------- | ------- |
| `-i`       | `--input`               | Path to the matchup data JSON file (e.g., `input/ea_input.json`).           | `input/ea_input.json` | `str`   |
| `-o`       | `--output`              | Directory to save simulation results, plots, and logs. Created if missing.   | `output/`            | `str`   |
| `-M`       | `--mode`                | Simulation dynamics engine: `replicator` (ESS) or `tournament` (Agent).     | `replicator`        | `str`   |
| `-g`       | `--gens`                | Maximum number of generations/epochs to simulate.                           | `1000`              | `int`   |
| `-m`       | `--min-games`           | Minimum required match volume to include an archetype in the baseline.      | `100`               | `int`   |
| `-e`       | `--extinction-threshold`| Metagame frequency drop-off point where a deck is considered mathematically dead. | `1e-10`             | `float` |
| `-N`       | `--noise`               | Scale of Gaussian noise injected into generation payoffs.                   | `0.0`               | `float` |
| `-I`       | `--intro-prob`          | Stochastic probability per generation of a rogue/extinct deck re-entering.  | `0.0`               | `float` |
| `-s`       | `--seed`                | Fixed RNG seed for perfectly reproducible experiments.                      | `1312`              | `int`   |
|            | `--tournament-style`    | Bracket execution logic: `pure_swiss` (fast) or `championship_series` (BO3).| `pure_swiss`        | `str`   |
|            | `--no-plot`             | Suppress generation of interactive HTML Plotly graphs (headless runs).      | `False`             | `bool`  |
| `-C`       | `--cluster`             | Enable Silhouette-optimized K-Means strategic archetype clustering.         | `False`             | `bool`  |
| `-l`       | `--log-level`           | Terminal logging verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`.         | `INFO`              | `str`   |
| `-b`       | `--batch`               | Enable batch execution mode for automated parameter sweeping.               | `False`             | `bool`  |
| `-c`       | `--batch-config`        | Path to the JSON configuration file defining batch parameters.              | `None`              | `str`   |
|            | `--predict`             | Bypass evolution entirely and run the static Monte Carlo EV solver.         | `False`             | `bool`  |
| `-P`       | `--players`             | Expected total field size for the upcoming event (Prediction Mode).         | `32`                | `int`   |
|            | `--meta`                | Comma-separated custom field constraints (e.g., "DeckA:0.2").               | `""`                | `str`   |

> **🚫 Validation:** Invalid values (e.g., `--intro-prob 1.5` or negative `--noise`) will cause immediate, descriptive `argparse` errors.
     

---

## 📂 Output

After running the simulation, a uniquely named directory (e.g., `YYYYMMDD_HHmmss_replicator_gens1000K_CONV@2017410`) will be created within your specified output directory. This directory will contain the following files:

*   `ess_equilibrium.csv`: The final state of the metagame, listing each deck's frequency, activity status, and extinction history.
*   `final_tiers.json`: A tier list (T0, T0.5, T1, T2, T3, T4) generated based on the **final state** of the metagame.
*   `simulation.txt`: A detailed log file of the simulation run, including convergence metrics, tier list summaries, and clustering results.
*   `metagame_evolution.html`: An interactive Plotly graph showing the frequency of each deck over time.
*   `matchup_heatmap.html`: An interactive heatmap of the win rate matrix.
*   `matchup_network.html`: An interactive network graph visualizing deck matchups and identified RPS cycles.
*   `metagame_history_full.csv`: (If enabled) A full record of the metagame state at every generation, for deep analysis.

	
---

## 📈 Example Results | Replicator Dynamics

**(Based on BLK/WHT Standard 2025 Data)** After running the simulation for **2,017,509 generations** using the **Replicator Dynamics** model, the metagame reached a stable equilibrium. Below are the key insights derived from the final state.

> **Note**: The simulation intelligently filters out extinct decks for similarity and clustering analysis, ensuring results reflect only the *active, relevant* metagame.

---

### 🏆 Final State Tier List

This tier list reflects the **endgame meta**, prioritizing decks with high win rates against the final field and strong presence.

| Tier | Deck              | Win Rate | Metagame Share |
| :--- | :---------------- | :------- | :------------- |
| **T0.5** | **Gholdengo**     | 52.51%   | 5.13%          |
| **T0.5** | **Joltik Box**    | 50.22%   | **34.67%**     |
| **T1** | Dragapult Dusknoir| 50.67%   | 11.15%         |
| **T1** | Ogerpon           | 51.15%   | 5.54%          |
| **T1** | Gardevoir         | 49.47%   | 15.58%         |
| **T1** | Crustle           | **54.09%**   | 0.02%          |
| **T1** | Miraidon          | 49.23%   | 11.82%         |


✅ **Joltik Box** is the metagame king by sheer volume, commanding over a third of the final meta. **Gholdengo** boasts with the highest win rate, but a lower metagame share. **Crustle**, while nearly extinct, boasts the highest win rate, indicating it's a powerful but niche counter-strategy.

> **43 unique Rock-Paper-Scissors cycles** were identified (e.g., `Blissey → Festival Lead → Raging Bolt Ogerpon → Blissey`), indicating a healthy, non-transitive metagame with no single unbeatable deck.

> The most dominant deck against an even field at the start was **Blissey** (55.18%), showcasing how the meta can shift dramatically over time.

---

### 🧩 Strategic Archetype Clusters

Decks were grouped into clusters based on the similarity of their matchup profiles. This reveals hidden strategic families:

*   **Cluster 0 (Midrange):** `['Dragapult Dusknoir', 'Gardevoir', 'Dragapult Charizard', 'Ogerpon']`
*   **Cluster 1 (Tempo):** `['Blissey', 'Gholdengo', 'Joltik Box', 'Miraidon']`
*   **Cluster 2 (Niche):** `['Conkeldurr']`

---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss a feature or bug, and submit a pull request for any changes.

---

## 🙏 Acknowledgements

This project stands on the shoulders of giants and would not be possible without the foundational work and data provided by several key members of the competitive metagaming community.

*   **Limitless TCG**: An enormous shoutout to the team at [Limitless](https://limitlesstcg.com/). This entire project is powered by the comprehensive and meticulously maintained Pokémon TCG data scraped from their website. Their platform is an indispensable resource for the global Pokémon TCG community. If you use their tools (and you should!), please consider supporting them on [Patreon](https://patreon.com/limitlesstcg) to ensure they can continue their fantastic work.

*   **Vicious Syndicate**: I personally owe a profound debt to [Vicious Syndicate](https://www.vicioussyndicate.com/) for pioneering rigorous, community-driven data analytics in the digital card game space. Their *Data Reaper Reports* and podcast set the global gold standard for metagame analysis, serving as an indispensable foundation for my own education in competitive modeling. This project’s Power & Meta Scoring metrics are directly inspired by their standardized 0–100 normalization methodology. Beyond the math, their commitment to transparency and data integrity has been a guiding light for this project’s philosophy and my own development as a human being. :) 

*   **Dominic Calkosz & HearthNash**: A massive thank you to **Dominic** for his groundbreaking research on game-theoretic metagame analysis in Hearthstone. His project, [HearthNash](https://dominic-calkosz.com/HearthNash), was a direct inspiration for applying evolutionary dynamics and Nash equilibrium concepts to TCG metagames. His academic approach provided the theoretical bedrock for this simulator.

*   **FPL Analytics Community:** My journey was also profoundly shaped by the Discord community of the **FPL Analytics Community**. Engaging in their discussions, learning from their approach to data, and even experimenting around with meta-solvers such as **[Solio](https://fpl.solioanalytics.com/)** and **[FPL Review](https://fplreview.com/)** provided the foundational inspiration for this project. The concepts of "solving" a dynamic, stochastic game and the importance of probabilistic forecasting were directly translated into the **Monte Carlo tournament engine** and **EV prediction logic** used here. Their insights into data-driven decision-making were invaluable to my development as a developer and analyst.

This project is a synthesis of their collective efforts, and I am deeply grateful for the ecosystems they have created, and the bricks they've placed. 🙇 So, **_thank you_** for existing. 

---

## 📜 License

<a href="https://github.com/leonid-dalin/pokemon-tcg-metagame-simulator/">Pokémon TCG Metagame Evolution Simulator</a> © 2025 by <a href="https://github.com/leonid-dalin/">Leonid Dalin</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a><br>
<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;">

The use of any content in this repository for training any artificial intelligence (AI) model, or for any form of AI to remix, adapt, or build upon my works, especially without my explicit permission, is strictly prohibited.