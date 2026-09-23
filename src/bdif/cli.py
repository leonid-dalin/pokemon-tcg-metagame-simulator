import argparse
import importlib
import json
import logging
import os
import sys
from typing import Any

from pydantic import ValidationError

from src.api.models import PredictionRequest
from src.core.config import INPUT_DATA, OUTPUT_DIR, RNG_SEED
from src.core.logger import setup_structured_logging, logger
from src.core.telemetry import setup_telemetry


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BDIF data and report commands")
    parser.add_argument("-l", "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("status", "Show local BDIF data status"), ("ingest", "Ingest bounded BDIF data"), ("refit", "Refit the local card model")):
        commands.add_parser(name, help=help_text)
    report = commands.add_parser("report", help="Generate a tournament report")
    report.add_argument("-i", "--input", default=INPUT_DATA)
    report.add_argument("-o", "--output", default=OUTPUT_DIR)
    report.add_argument("--seed", type=int, default=RNG_SEED)
    report.add_argument("-P", "--players", type=int, default=256)
    report.add_argument("--tournament-style", choices=("pure_swiss", "championship_series"), default="pure_swiss")
    report.add_argument("--meta", type=_meta_spec, default={})
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
            result = service.run_ingestion()
        elif args.command == "refit":
            result = service.refit_card_model()
        else:
            if not os.path.isfile(args.input):
                _parser().error(f"input file not found: {args.input}")
            if not 4 <= args.players <= 8192:
                _parser().error("--players must be between 4 and 8192")
            os.makedirs(args.output, exist_ok=True)
            from src.core.data import load_matchup_data
            deck_names, matrix, _ = load_matchup_data(args.input)
            request = PredictionRequest(
                job_id="bdif_cli_report", total_players=args.players,
                user_meta_spec=args.meta, tournament_style=args.tournament_style,
                deck_names=deck_names, matchup_matrix=matrix.tolist(),
            )
            result = service.run_prediction(request)
        print(json.dumps(result, sort_keys=True, default=lambda value: value.value if hasattr(value, "value") else str(value)))
        if args.command == "report" and _evidence_unavailable(result):
            return 3
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("bdif_command_failed", command=args.command, error=str(exc), exc_info=False)
        for handler in root_logger.handlers:
            handler.flush()
        return 1


def _evidence_unavailable(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    mc_results = result.get("mc_results")
    if not isinstance(mc_results, dict):
        return False
    if mc_results.get("insufficient_data"):
        return True
    for addon_group in ("best60_recommendations", "h1_report"):
        addon = mc_results.get(addon_group)
        if not isinstance(addon, dict):
            continue
        if addon.get("status") in {"failed", "unavailable", "insufficient_data"}:
            return True
        if any(
            isinstance(value, dict) and value.get("status") in {"failed", "unavailable", "insufficient_data"}
            for value in addon.values()
        ):
            return True
    posterior = mc_results.get("posterior")
    interval_status = posterior.get("interval_status") if isinstance(posterior, dict) else None
    return isinstance(interval_status, str) and interval_status.lower() in {"failed", "unavailable", "insufficient_data"}


if __name__ == "__main__":
    sys.exit(main())
