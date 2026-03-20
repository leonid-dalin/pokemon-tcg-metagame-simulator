# Pokémon TCG Metagame Simulator

This project is a high-fidelity analytical suite designed to model both the immediate, stochastic reality of competitive **Pokémon Trading Card Game (TCG)** tournaments and the long-term evolutionary trends of its metagame.

Built on a **Dual-Engine Architecture**, the simulator bridges rigorous tournament mathematics with evolutionary game theory. The **Monte Carlo Bracket Engine** evaluates immediate tournament equity by simulating up to 1,000,000 full tournament iterations—accounting for official Play! Pokémon structures (Variant #5), seeded Top Cut pairings, and a parabolic tie convergence model that mathematically mirrors real-world Best-of-Three (BO3) match-point bleed. Complementing this is the **Replicator Dynamics Engine**, which predicts the Evolutionary Stable State (ESS), identifying the equilibrium point where deck frequencies become unexploitable over time.

Integrated with an interactive Streamlit dashboard, vectorized Swiss metrics (SoS, OMW), and dynamic Z-score evaluations, the simulator provides deep strategic insights from two distinct perspectives: macro-level archetype dominance and your individual Expected Value (EV) of surviving the "Day 2 bubble."

---

## 🚀 Features

### 🎲 Stochastic Tournament Modeling (Monte Carlo)
*   **High-Fidelity Bracket Engine:** Simulates up to 1,000,000 full tournament iterations (Swiss + Single-Elimination Playoffs) using parallelized NumPy workers.
*   **Official TPCi Variant #5 Support:** Automatically applies official Play! Pokémon round counts, match-point cutoffs (e.g., 16 or 19 points), and seeded Top Cut brackets based on player volume (up to 8,192 players).
*   **BETA: Parabolic Tie Convergence:** Models the ~15% real-world "match-point bleed" using a parabolic probability curve—ensuring that close matchups correctly result in ties (1 point) rather than binary win/loss outcomes.

### 📊 Advanced Competitive Analytics
*   **Dual-POV Composite Scoring:** A 0–100 normalized ranking using Z-Score standardization mapped to a sigmoid curve. Evaluates decks based on four pillars (Win Rate, Day 2, Top 8, Win Probability) from either a **Macro Impact** (Tournament Share) or **Player EV** (Individual Odds) perspective.
*   **Vectorized Metagame Constraints:** A high-speed, pure NumPy water-filling algorithm that strictly enforces user-defined exact, minimum, or maximum field share constraints.
*   **Swiss Logic Integration:** Vectorized calculation of **Strength of Schedule (SoS)** and **Opponent's Match Win % (OMW)**.
*   **Dynamic Clustering & RPS:** Uses K-Means with Silhouette Optimization and `StandardScaler` to accurately group archetypes based on their strategic matchup "shape", alongside network graphing to identify non-transitive Rock-Paper-Scissors cycles.

### 💻 Professional Streamlit Interface
*   **Interactive Dashboard:** A sortable, tool-tipped dashboard with instant **Perspective Toggles** to seamlessly swap data and recommendations between Individual EV and Macro Impact without recalculating.
*   **Head-to-Head Comparator:** A side-by-side "A vs. B" tool featuring native color-coding to highlight favorable ($\ge 55\%$) and unfavorable ($\le 45\%$) matchups across the predicted field.
*   **Execution Console:** A live status debugger providing real-time transparency into engine data loading, constraint resolution, and parallel core progress.
*   **Limitless HTML Ingestion:** Native parser to extract metagame shares directly from Limitless Labs exports to instantly pre-populate simulation constraints.

### 🧬 Metagame Evolution (Replicator Dynamics)
*   **Evolutionary Stability:** Models the "Game Theory" of a metagame using Replicator Dynamics, where successful decks grow in frequency proportional to their meta-weighted performance.
*   **Deck Extinction & Reintroduction:** Simulates innovation and meta-shifts. Poorly performing decks can die out, while "rogue" decks can be randomly reintroduced to challenge the established equilibrium.

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

### Running the Simulation

Execute the main script with your desired parameters:

```bash
python -m src.main -i data/input/ea_input.json -g 1_000_000 --mode replicator
```

## Running the Streamlit Web App (Recommended)

The easiest way to use the predictor is via the Streamlit app.

```bash
streamlit run src/interfaces/app.py
```

This will launch a local web server and open the app in your browser, allowing you to get recommendations and analyze the meta interactively.

#### Key Command-Line Arguments

| Short Flag | Long Flag               | Description                                                                 | Default             | Type    |
| ---------- | ----------------------- | --------------------------------------------------------------------------- | ------------------- | ------- |
| `-i`       | `--input`               | Path to the matchup data JSON file (e.g., `input/ea_input.json`).           | `input/ea_input.json` | `str`   |
| `-o`       | `--output`              | Directory to save simulation results, plots, and logs. Created if missing.   | `output/`            | `str`   |
| `-m`       | `--mode`                | Simulation dynamics mode: `replicator` (default) or `tournament`.           | `replicator`        | `str`   |
| `-g`       | `--gens`                | Maximum number of generations to simulate. Must be > 0.                     | `100`               | `int`   |
| `-M`       | `--min-games`           | Min games required to include a deck in the simulation.         			 | `700`              | `int` |
| `-e`       | `--extinction-threshold`| Deck frequency threshold for extinction (decks below this may vanish).      | `0.005`             | `float` |
| `-N`       | `--noise`               | Scale of stochastic noise injected into replicator dynamics. ≥ 0.0.         | `0.02`              | `float` |
| `-p`       | `--intro-prob`          | Probability per generation to reintroduce a previously extinct deck.        | `0.002`             | `float` |
| `-s`       | `--seed`                | RNG seed for reproducible simulations.                                      | `1312`              | `int`   |
| `-P`       | `--no-plot`             | Disable generation of interactive HTML plots (e.g., for headless runs).     | `False`             | `bool`  |
| `-C`       | `--cluster`             | Enable post-simulation deck archetype clustering analysis.                  | `False`             | `bool`  |
| `-l`       | `--log-level`           | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`.                  | `INFO`              | `str`   |
| `-b`       | `--batch`               | Run batch experiments (requires `--batch-config`).                          | `False`             | `bool`  |
| `-c`       | `--batch-config`        | Path to JSON file defining batch experiment parameters.                     | `None`              | `str`   |
|			 | `--predict`			   | Run metagame prediction instead of simulation.								 | `False`			   | `bool`	 |
|			 | `--players`			   | Expected tournament size for prediction mode.								 | `32`			   	   | `int`	 |
|			 | `--meta`			   	   | User-specified meta for prediction (e.g., "DeckA:0.2").					 | `""`			   	   | `str`	 |


> **🚫 Validation:** Invalid values (e.g., `--intro-prob 1.5`) will cause immediate, descriptive errors. 
     

---

## 📂 Output

After running the simulation, a uniquely named directory (e.g., `YYYYMMDD_HHmmss_replicator_gens1000K_CONV@2017410`) will be created within your specified output directory. This directory will contain the following files:

*   `ess_equilibrium.csv`: The final state of the metagame, listing each deck's frequency, activity status, and extinction history.
*   `final_tiers.json`: A tier list (S, A, B, C, D) generated based on the **final state** of the metagame.
*   `all_time_tiers.json`: A tier list generated based on the deck's performance over the **entire simulation history**.
*   `simulation.txt`: A detailed log file of the simulation run, including convergence metrics, tier list summaries, and clustering results.
*   `metagame_evolution.html`: An interactive Plotly graph showing the frequency of each deck over time.
*   `matchup_heatmap.html`: An interactive heatmap of the win rate matrix.
*   `matchup_network.html`: An interactive network graph visualizing deck matchups and identified RPS cycles.
*   `metagame_history_full.csv`: (If enabled) A full record of the metagame state at every generation, for deep analysis.

	
---

## 📈 Example Results (Based on BLK/WHT Standard 2025 Data) | Replicator Dynamics

After running the simulation for **2,017,509 generations** using the **Replicator Dynamics** model, the metagame reached a stable equilibrium. Below are the key insights derived from the final state.

> **Note**: The simulation intelligently filters out extinct decks for similarity and clustering analysis, ensuring results reflect only the *active, relevant* metagame.

---

### 🏆 Final State Tier List

This tier list reflects the **endgame meta**, prioritizing decks with high win rates against the final field and strong presence.

| Tier | Deck              | Win Rate | Metagame Share |
| :--- | :---------------- | :------- | :------------- |
| **A** | **Gholdengo**     | 52.51%   | 5.13%          |
| **A** | **Joltik Box**    | 50.22%   | **34.67%**     |
| **B** | Dragapult Dusknoir| 50.67%   | 11.15%         |
| **B** | Ogerpon           | 51.15%   | 5.54%          |
| **B** | Gardevoir         | 49.47%   | 15.58%         |
| **B** | Crustle           | **54.09%**   | 0.02%          |
| **B** | Miraidon          | 49.23%   | 11.82%         |


✅ **Joltik Box** is the metagame king by sheer volume, commanding over a third of the final meta. **Gholdengo** boasts with the highest win rate, but a lower metagame share. **Crustle**, while nearly extinct, boasts the highest win rate, indicating it's a powerful but niche counter-strategy.

---

### 📈 All-Time Performance Tier List

This tier list rewards **consistency and overall impact** across the *entire simulation*, not just the final snapshot.

| Tier | Deck              | Meta Impact Score | All-Time Presence |
| :--- | :---------------- | :---------------- | :---------------- |
| **S** | **Gardevoir**     | **0.1086**        | **21.01%**        |
| **S** | Gholdengo         | 0.0357            | 10.57%            |
| **S** | Blissey           | 0.0375            | 9.30%             |
| **A** | Dragapult Dusknoir| 0.0286            | 8.15%             |
| **A** | Joltik Box        | 0.0906            | 17.84%            |
| **A** | Miraidon          | 0.0190            | 6.00%             |

👑 **Gardevoir** dominates the all-time list, proving its long-term consistency and impact, even if its final win rate was not the highest. This highlights the difference between a deck that is powerful at the end and a deck that has been a major force throughout the metagame's evolution.

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

*   **Dominic Calkosz & HearthNash**: A massive thank you to **Dominic** for his groundbreaking research on game-theoretic metagame analysis in Hearthstone. His project, [HearthNash](https://dominic-calkosz.com/HearthNash), was a direct inspiration for applying evolutionary dynamics and Nash equilibrium concepts to TCG metagames. His academic approach provided the theoretical bedrock for this simulator.

*   **Vicious Syndicate**: I personally owe a profound debt to [Vicious Syndicate](https://www.vicioussyndicate.com/) for pioneering rigorous, community-driven data analytics in the digital card game space. Their weekly Data Reaper Reports have set the gold standard for metagame analysis in Hearthstone. Their commitment to transparency, data collection, and community engagement has been a guiding light for this project's philosophy, and for myself.

*   **FPL Analytics Community:** My journey was also profoundly shaped by the Discord community of the **FPL Analytics Community**. Engaging in their discussions, learning from their approach to data, and even playing around with meta-solvers for the Fantasy Premier League game (mainly **[Solio](https://fpl.solioanalytics.com/)** and **[FPL Review](https://fplreview.com/)**), was a massive source of inspiration. They helped me further my understanding of analysis, expected value, and the very concept of building a solver for a competitive game, insights that were invaluable to this project.

This simulator is a synthesis of their collective efforts, and I am deeply grateful for the ecosystems they have created. 🙇 So, **_thank you_** for existing. 

---

## 📜 License

<a href="https://github.com/leonid-dalin/pokemon-tcg-metagame-simulator/">Pokémon TCG Metagame Evolution Simulator</a> © 2025 by <a href="https://github.com/leonid-dalin/">Leonid Dalin</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a><br>
<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" style="max-width: 0.75em;max-height:1em;margin-left: .2em;">

The use of any content in this repository for training any artificial intelligence (AI) model, or for any form of AI to remix, adapt, or build upon my works, especially without my explicit permission, is strictly prohibited.