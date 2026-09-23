# Set up local development

Use the local environment for tests, CLI runs, and Rust extension work. Docker remains the supported path for the full multi-service application

## Create an environment

Python 3.12 is the tested project interpreter. From the repository root:

```bash
python -m venv .venv
```

Activate it, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Build the Rust extension:

```bash
python -m maturin develop --release --manifest-path src/tournament/tcg_engine/Cargo.toml
```

## Run tests

```bash
python -m pytest -q
```

For a faster focused run:

```bash
python -m pytest tests/api tests/core tests/tournament tests/worker -q
```

## Run quality checks

The repository QA harness includes:

```bash
PYTHONPATH=. python tools/qa-harness/test_quality_audit.py --tests tests --package src
PYTHONPATH=. python tools/qa-harness/contract_probe.py --spec tools/qa-harness/contract_spec_bdif.py
PYTHONPATH=. python tools/qa-harness/flag_matrix.py --spec tools/qa-harness/flag_spec_bdif.py
PYTHONPATH=. python tools/qa-harness/interval_sanity.py --spec tools/qa-harness/interval_spec_h1.py
PYTHONPATH=. python tools/qa-harness/dead_symbol_scan.py --src src --tests tests
```

Run mutation sweeps only from a committed clean tree:

```bash
PYTHONPATH=. python tools/qa-harness/mutation_sweep.py --repo . --mutations tools/qa-harness/mutations.minmatches.json --tests tests --pythonpath .
PYTHONPATH=. python tools/qa-harness/mutation_sweep.py --repo . --mutations tools/qa-harness/mutations.pr5-remediated.json --tests tests --pythonpath .
```

On Windows Bash, put `PYTHONPATH=.` before the command. This project uses the repository's active Python environment, so replace `python` with its absolute interpreter path when the shell is not activated
