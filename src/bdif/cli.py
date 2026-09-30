import argparse
import importlib
import json
import logging
import os
import sys
from typing import Any

from pydantic import ValidationError

from src.api.models import PredictionRequest
from src.core.config import OUTPUT_DIR, RNG_SEED
from src.core.logger import setup_structured_logging, logger
from src.core.telemetry import setup_telemetry

EVIDENCE_UNAVAILABLE = {
    "ingest": {"disabled", "failed", "partial"},
    "refit": {"missing", "insufficient observations", "not identifiable"},
}


def _meta_spec(value: str) -> dict[str, float]:
    result = {}
    if not value:
        return result
    for part in value.split(","):
        if ":" not in part:
            raise argparse.ArgumentTypeError("--meta entries must use deck:share format")
        deck, share = part.rsplit(":", 1)
        try:
            parsed = float(share)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--meta shares must be numbers") from exc
        if not deck.strip() or not 0 <= parsed <= 1:
            raise argparse.ArgumentTypeError("--meta entries require a deck and share between 0 and 1")
        result[deck.strip()] = parsed
    return result


def _ingest_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--limit must be an integer") from exc
    if not 1 <= parsed <= 1000:
        raise argparse.ArgumentTypeError("--limit must be between 1 and 1000")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BDIF data and report commands")
    parser.add_argument("-l", "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show local BDIF data status")
    ingest = commands.add_parser("ingest", help="Ingest bounded BDIF data")
    ingest.add_argument("--limit", type=_ingest_limit, default=None, help="Tournaments to fetch, 1 to 1000")
    commands.add_parser("refit", help="Refit the local card model")
    report = commands.add_parser("report", help="Generate a tournament report")
    report.add_argument("-i", "--input", default=None)
    report.add_argument("-o", "--output", default=OUTPUT_DIR)
    report.add_argument("--seed", type=int, default=RNG_SEED)
    report.add_argument("-P", "--players", type=int, default=256)
    report.add_argument("--tournament-style", choices=("pure_swiss", "championship_series"), default="pure_swiss")
    report.add_argument("--meta", type=_meta_spec, default={})
    report.add_argument("--panel", type=lambda text: [deck.strip() for deck in text.split(",") if deck.strip()], default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    setup_structured_logging()
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.log_level))
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=getattr(logging, args.log_level), force=True)
    setup_telemetry("tcg-bdif-cli")
    try:
        service = importlib.import_module("src.bdif.service")
        if args.command == "status":
            result = service.bdif_status()
        elif args.command == "ingest":
            result = service.run_ingestion(limit=args.limit)
        elif args.command == "refit":
            result = service.refit_card_model()
        else:
            if not 4 <= args.players <= 8192:
                _parser().error("--players must be between 4 and 8192")
            input_path = args.input or service.simulation_input_path()
            if not os.path.isfile(input_path):
                _parser().error(f"input file not found: {input_path}")
            os.makedirs(args.output, exist_ok=True)
            from src.core.data import load_matchup_data
            deck_names, matrix, _ = load_matchup_data(input_path)
            request = PredictionRequest(
                job_id="bdif_cli_report", total_players=args.players,
                user_meta_spec=args.meta, tournament_style=args.tournament_style,
                deck_names=deck_names, matchup_matrix=matrix.tolist(),
                bdif_panel_decks=args.panel,
            )
            result = service.run_prediction(request, seed=args.seed, input_path=input_path)
        print(json.dumps(result, sort_keys=True, default=lambda value: value.value if hasattr(value, "value") else str(value)))
        return _exit_code(args.command, result)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("bdif_command_failed", command=args.command, error=str(exc), exc_info=False)
        for handler in root_logger.handlers:
            handler.flush()
        return 1


def _exit_code(command: str, result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    if command == "report":
        mc_results = result.get("mc_results")
        if not isinstance(mc_results, dict):
            return 0
        ranked_metrics = mc_results.get("ranked_metrics")
        return 3 if not isinstance(ranked_metrics, dict) or not ranked_metrics else 0
    statuses = EVIDENCE_UNAVAILABLE.get(command, set())
    return 3 if result.get("status") in statuses else 0


if __name__ == "__main__":
    sys.exit(main())
