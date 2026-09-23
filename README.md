# Pokemon TCG Metagame Simulator

<p align="center">
  <a href="https://github.com/leonid-dalin/pokemon-tcg-metagame-simulator/actions/workflows/tests.yml"><img src="https://github.com/leonid-dalin/pokemon-tcg-metagame-simulator/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/Docker%20Compose-supported-2496ED?logo=docker&logoColor=white" alt="Docker Compose supported">
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-AGPL--3.0--or--later-blue" alt="AGPL-3.0-or-later licence"></a>
</p>

A Python and Rust simulator for Pokemon TCG metagame analysis. It provides two local workflows:

- Replicator dynamics for long-run metagame evolution
- Monte Carlo tournament simulation for short-run event outcomes

The Docker Compose deployment runs a FastAPI gateway, a Huey worker, a Streamlit UI, Redis, Jaeger, and a shared data volume

## 🖼️ App showcase

| Main dashboard and controls | Head-to-head comparator |
|:---:|:---:|
| ![Main dashboard](docs/img/main.png) | ![Head-to-head comparator](docs/img/head-to-head.png) |
| **Top recommendations** | **Simulation diagnostics** |
| ![Top recommendations](docs/img/top-recommendations.png) | ![Simulation complete](docs/img/simulation_complete.png) |

## 🚀 Quick start

Requirements: Docker Desktop with Compose support

```bash
git clone https://github.com/leonid-dalin/pokemon-tcg-metagame-simulator.git
cd pokemon-tcg-metagame-simulator
docker compose up -d --build
```

Open the dashboard at `http://localhost:8501` or the API documentation at `http://localhost:8000/docs/`

The Compose stack exposes:

- `ui` on port `8501`
- `api` on port `8000`
- `redis` on port `6379`
- `jaeger` on ports `16686`, `4317`, `4318`, `5778`, and `9411`
- `worker` and `init-data` as internal services

Stop the stack with:

```bash
docker compose down
```

## 💻 Run the CLI

Install the Python dependencies and build the Rust extension with the commands in [the local development guide](docs/how-to/local-development.md). Then run a simulation from the repository root:

```bash
python -m src.ui.cli -i data/input/ea_input.json --mode replicator --gens 10000
```

Run the static tournament predictor with the checked-in four-archetype example:

```bash
python -m src.ui.cli --input examples/predict_input.json --predict --players 512 --no-plot
```

The CLI writes timestamped directories under `output/` unless `--output` changes the destination. See [CLI reference](docs/reference/cli.md) for all supported arguments and [CLI guide](docs/how-to/run-cli.md) for complete examples

## 📚 Documentation

| You need to | Read |
| --- | --- |
| Learn the main workflow | [Quickstart tutorial](docs/tutorial/quickstart.md) |
| Run the CLI | [CLI how-to](docs/how-to/run-cli.md) |
| Run the Docker Compose stack | [Compose how-to](docs/how-to/run-compose.md) |
| Enable Limitless and BDIF analytics | [BDIF how-to](docs/how-to/enable-bdif.md) |
| Set up local development | [Local development how-to](docs/how-to/local-development.md) |
| Check CLI flags and defaults | [CLI reference](docs/reference/cli.md) |
| Integrate with the API | [API reference](docs/reference/api.md) |
| Check configuration values | [Configuration reference](docs/reference/configuration.md) |
| Understand the service layout | [Architecture explanation](docs/explanation/architecture.md) |
| Understand the models and evidence rules | [Analytics explanation](docs/explanation/analytics.md) |
| Understand the system design | [System design](docs/EXPLANATION.md) |
| Find source entry points | [Code reference](docs/REFERENCE.md) |
| See what changed | [Changelog](docs/CHANGELOGS.md) |
| Read acknowledgements for Limitless TCG and other contributors | [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) |

## 🧾 Project documents

| You need to | Read |
| --- | --- |
| Contribute code or data | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Read community standards | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| Report a security issue | [SECURITY.md](SECURITY.md) |
| Read the project licence | [LICENSE](LICENSE) |
| Check dependency and data notices | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

## 🗃️ Data files

The CLI default input is `data/input/ea_input.json`. The file contains archetype names and a win-rate matrix. The loader enforces the configured minimum match count and supplies matchup details for posterior analytics when those details exist

Limitless ingestion is disabled by default. It writes `data/limitless.db`, `data/input/limitless_input.json`, and, when enough observations exist, `data/input/limitless_model_input.json`. BDIF card-model reporting is also disabled by default. See [BDIF analytics](docs/how-to/enable-bdif.md)

## 🧪 Tests

Use the project environment and run:

```bash
python -m pytest -q
```

The Docker build compiles `src/tournament/tcg_engine` with Maturin before installing the wheel

## 📄 Licence

Copyright (C) 2025 Leonid Dalin. This project is licensed under [GNU AGPL-3.0-or-later](LICENSE)

The use of any content in this repository for training any artificial intelligence (AI) model without my explicit consent is strictly prohibited.
