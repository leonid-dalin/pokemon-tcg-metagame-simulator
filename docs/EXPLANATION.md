# System Theory and Design

This document is an explanation of the core concepts, mathematical assumptions, and architectural philosophy driving the Pokémon TCG Metagame Simulator. Its goal is to provide a theoretical understanding of how the system models tournament environments and evolutionary game theory.

> ⚠️ **Caution:** This project is undergoing active refactoring. Expect frequent changes to the architecture, and verify information with the latest source code. I'll do my best to keep it updated or have it updated ASAP, but bear in mind this is a 1-person passion project.

---

## The Simulation Philosophy

To maintain high-speed parallel performance and prevent statistical noise from overwhelming the signal, the simulation engines operate under a specific set of constrained assumptions:

* **Perfect Equal Skill (Flat Elo):** The simulator assumes all pilots/players are of perfectly equal, "average" skill level. The engine pulls statically from the provided win-rate matrix and does not inject Elo-style distributions. A Top-10 global player and a Day 1 casual are treated identically by the math. 
* **Matchup Determinism:**  Basically, **luck** does **NOT** exist. While tournament bracket pairings are stochastic (RNG-driven), actual match results are resolved purely as a weighted coin flip based on the archetype matchup data. There is no simulation of bricked hands or top-decks; the gathered win rates are treated as **absolute** in their nature.
* **Speed-Agnostic Tie Convergence:** The Parabolic Tie Convergence feature forces match-point bleed (ties) based purely on the mathematical closeness of the matchup (e.g., a 50/50 matchup ties more than an 80/20 matchup). It is entirely blind to deck archetype speed to avoid introducing subjective "magic numbers" regarding a deck's piloting difficulty or pacing.

> Do they time waste for a tie because they're a `Gholdengho` player, because they're versing a difficult match-up, or because that's the *right* decision to do Competitively by leading first game 1-0?

Nobody knows. And **ignorance is bliss.** 
* **No Mid-Tournament Teching:** A deck's strategic profile remains frozen for the duration of the tournament or generation. The engine does not account for players altering tech cards in response to expected Day 2 shifts. *If only I had data on Mulligan WR, Draw WR, Played WR, etc. ...*
* **Structured Variance:** The Monte Carlo engine simulates tournament realities like BO3 math conversions via $3p^2 - 2p^3$, time-limit Tie Convergence, and X-3 Drop Logic. However, it assumes perfectly randomized pairings within equivalent point brackets using mergesort.
* **Thermodynamic Data Purity (Zero-Sum):** Real-world matchup data is mathematically forced into a strict environment where the diagonal of the win matrix (mirror matches) is exactly 0.5.

--- 

## The Microservice Topography

The simulator utilizes a containerized, three-tier architecture orchestrated via Docker Compose. This decoupled approach safely separates lightweight user interactions from computationally heavy, highly parallelized mathematical operations.

```text
┌─────────────────┐       HTTP / REST       ┌──────────────────┐
│                 │  (1) POST /api/v1/predict                   │
│  Streamlit UI   ├────────────────────────►│  FastAPI Server  │
│  (Thin Client)  │                         │  (Entry Point)   │
│                 │◄────────────────────────┤                  │
└─────────────────┘  (4) Return Task ID     └────────┬─────────┘
        ▲                                            │
        │ (5) Poll GET /api/v1/tasks/{task_id}       │ (2) Enqueue Job
        │                                            ▼
┌───────┴─────────┐                       ┌──────────────────┐
│                 │◄──── (3) Pull Job ────│                  │
│   Huey Worker   │                       │  SQLite Broker   │
│ (Heavy Compute) │───── (6) Write Res ──►│ (tcg_tasks.db)   │
│                 │                       │                  │
└─────────────────┘                       └──────────────────┘
```

1. `api` **(FastAPI Server)**: Serves as the gateway. Validates incoming simulation requests using strict Pydantic data contracts (`src.api.models`) and pushes tasks to the queue.
2. `worker` **(Huey Worker):** A headless background processor. It listens for tasks, executes the Monte Carlo engine and static solver logic, and writes results back to a lightweight SQLite backend.
3. `ui` **(Streamlit UI)**: The visual frontend located in `app.py`. It gathers user parameters, dispatches them to the API via HTTP, polls for results, and renders the resulting Plotly charts and Pandas dataframes.
