#!/usr/bin/env python3
# cli.py | Command Line Interface entry point, experiment batching, logging control, reproducibility

from __future__ import annotations
import json
import logging
import time
import os
import csv
from typing import cast, Any, Optional, Literal

import structlog
from opentelemetry import trace

from src.api.models import PredictionRequest, PrecisionTier
from src.core.config import *
from src.core.data import (
    load_matchup_data,
    cluster_decks_by_matchup_profile,
    compute_deck_dominance,
)
from src.core.types import SimulationConfig
from src.core.logger import setup_structured_logging
from src.core.telemetry import setup_telemetry
from src.evolution.engine import find_evolutionary_stable_state
from src.evolution.analysis import (
    compute_convergence_metrics,
    generate_final_state_tier_list,
    compute_matchup_cycles,
    compute_deck_similarity,
)
from src.evolution.plotting import (
    plot_metagame_evolution_interactive,
    plot_matchup_heatmap_interactive,
    plot_matchup_network,
)
from src.ui.cli_args import parse_args, Args
from src.tournament.solver import predict_best_decks


logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


# ----------------------------
# Single Experiment Runner
# ----------------------------
def run_single_experiment(args: Args, config_override: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Run a single metagame simulation experiment with distributed tracing."""
    with tracer.start_as_current_span("run_single_experiment") as span:
        log_level = getattr(logging, args.log_level.upper())

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_str = args.mode
        gen_str = f"gens{args.gens // 1000}K" if args.gens >= 1000 else f"gens{args.gens}"

        base_output_name = f"{timestamp}_{mode_str}_{gen_str}"
        base_output_dir = os.path.join(args.output, base_output_name)

        experiment_id = "default"
        if config_override:
            experiment_id = config_override.get("experiment_id", "default")
            base_output_dir = os.path.join(base_output_dir, experiment_id)

        os.makedirs(base_output_dir, exist_ok=True)

        # Attach experiment-specific JSON file handler
        log_file = os.path.join(base_output_dir, "simulation_trace.jsonl")
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)

        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)

        logger.info("experiment_initialised", experiment_id=experiment_id, output_dir=base_output_dir)
        logger.debug("experiment_parameters", mode=args.mode, gens=args.gens, noise=args.noise)

        mode_literal = cast(Literal["replicator", "tournament"], args.mode)
        style_literal = cast(Literal["pure_swiss", "championship_series"],
                             getattr(args, "tournament_style", "pure_swiss"))
        derived_mutation_rate = getattr(args, "mutation_rate", MUTATION_RATE)

        sim_config = SimulationConfig(
            mode=mode_literal,
            max_generations=args.gens,
            extinction_threshold=args.extinction_threshold,
            stability_threshold=args.stability_threshold,
            convergence_window=args.convergence_window,
            max_inactive_generations=args.max_inactive_generations,
            use_bayesian_winrates=args.use_bayesian_winrates,
            tournament_size=args.tournament_size,
            num_tournaments_per_gen=args.num_tournaments_per_gen,
            num_rounds=args.num_rounds,
            use_multiproc=args.use_multiproc,
            seed=args.seed,
            mutation_rate=derived_mutation_rate,
            noise_scale=args.noise,
            selection_pressure=args.selection_pressure,
            tournament_style=style_literal,
        )

        span.set_attribute("experiment.id", experiment_id)
        span.set_attribute("simulation.mode", mode_literal)

        derived_min_games = getattr(args, "min_games", MIN_GAMES)
        deck_names, win_matrix, matchup_details = load_matchup_data(args.input, derived_min_games)
        if not deck_names:
            logger.error("missing_deck_data", detail="No reliable decks loaded. Aborting.")
            return {}

        compute_deck_dominance(win_matrix, deck_names)

        if args.cluster:
            cluster_decks_by_matchup_profile(win_matrix, deck_names, n_clusters="auto")

        history_file_path = os.path.join(base_output_dir, "metagame_history_full.csv")

        # Trace the core evolutionary simulation logic
        start_time = time.time()
        with tracer.start_as_current_span("find_evolutionary_stable_state"):
            results, history, extinction_gens = find_evolutionary_stable_state(
                deck_names=deck_names,
                win_matrix=win_matrix,
                matchup_details=matchup_details,
                config=sim_config,
                history_file_path=history_file_path,
            )
        sim_duration = time.time() - start_time
        logger.info("simulation_engine_completed", duration_seconds=sim_duration)

        with tracer.start_as_current_span("compute_convergence_metrics"):
            convergence_metrics = compute_convergence_metrics(history)
            final_conv_gen = convergence_metrics["convergence_generation"]

        if final_conv_gen is not None:
            conv_status = f"CONV@{final_conv_gen}"
        else:
            conv_status = "NOCONV"

        logger.info("convergence_evaluated", status=conv_status, generation=final_conv_gen)

        final_output_name = f"{timestamp}_{mode_str}_{gen_str}_{conv_status}"
        if config_override and experiment_id != "default":
            final_output_name = f"{final_output_name}_{experiment_id}"

        file_handler.close()
        final_output_dir = os.path.join(args.output, final_output_name)

        try:
            if base_output_dir != final_output_dir:
                if os.path.exists(final_output_dir):
                    counter = 1
                    while os.path.exists(f"{final_output_dir}_{counter}"):
                        counter += 1
                    final_output_dir = f"{final_output_dir}_{counter}"
                os.rename(base_output_dir, final_output_dir)
                logger.info("output_directory_renamed", target=final_output_dir)
        except Exception as e:
            logger.error("directory_rename_failed", error=str(e), exc_info=True)
            final_output_dir = base_output_dir

        output_dir = final_output_dir
        csv_path = os.path.join(output_dir, "ess_equilibrium.csv")
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    cast(Any, f),
                    fieldnames=[
                        "deck",
                        "frequency",
                        "is_active",
                        "generations_inactive",
                        "extinction_generation",
                    ],
                )
                writer.writeheader()
                for r in results:
                    writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
            logger.info("results_persisted", path=csv_path)
        except (IOError, OSError) as e:
            logger.error("csv_write_failed", error=str(e))

        # Trace post-simulation analysis
        with tracer.start_as_current_span("post_simulation_analysis"):
            final_tiers = generate_final_state_tier_list(deck_names, history, win_matrix)
            with open(os.path.join(output_dir, "final_tiers.json"), "w", encoding="utf-8") as f:
                json.dump(final_tiers, cast(Any, f), indent=2)

            final_active_mask = [r["is_active"] for r in results]
            cycles = compute_matchup_cycles(win_matrix, deck_names, final_active_mask=final_active_mask)

            similarity = compute_deck_similarity(win_matrix, deck_names, final_active_mask=final_active_mask)
            with open(os.path.join(output_dir, "deck_similarity.json"), "w", encoding="utf-8") as f:
                json.dump(similarity.tolist(), cast(Any, f), indent=2)

        if not getattr(args, "no_plot", False):
            with tracer.start_as_current_span("generate_plotly_visualisations"):
                plot_metagame_evolution_interactive(
                    history,
                    deck_names,
                    extinction_gens,
                    save_path=os.path.join(output_dir, "metagame_evolution.html"),
                )
                tier_order = []
                for tier in TIER_ORDER:
                    tier_order.extend([deck["deck"] for deck in final_tiers.get(tier, [])])

                plot_matchup_heatmap_interactive(
                    win_matrix,
                    deck_names,
                    tier_order,
                    save_path=os.path.join(output_dir, "matchup_heatmap.html"),
                )
                plot_matchup_network(
                    win_matrix,
                    deck_names,
                    cycles,
                    metagame_history=history,
                    save_path=os.path.join(output_dir, "matchup_network.html"),
                )

        metadata = {
            "experiment_id": experiment_id,
            "duration_seconds": sim_duration,
            "final_active_decks": len([r for r in results if r["is_active"]]),
            "convergence_generation": final_conv_gen,
            "top_deck": (max(results, key=lambda x: x["frequency"])["deck"] if results else None),
            "output_dir": output_dir,
            "parameters": {
                "mode": args.mode,
                "gens": args.gens,
                "noise": args.noise,
                "extinction_threshold": args.extinction_threshold,
                "mutation_rate": derived_mutation_rate,
            },
        }
        return metadata


# ----------------------------
# Batch Experiment Runner
# ----------------------------
def run_batch_experiments(args: Args):
    """Run multiple experiments with different configurations wrapped in a single parent trace."""
    with tracer.start_as_current_span("run_batch_experiments") as batch_span:
        batch_config_path = args.batch_config
        if not batch_config_path or not os.path.exists(batch_config_path):
            logger.error("batch_config_missing", path=batch_config_path)
            return

        with open(batch_config_path, "r", encoding="utf-8") as f:
            batch_config = json.load(f)

        experiments = batch_config.get("experiments", [])
        base_output = args.output
        results_summary = []

        batch_span.set_attribute("batch.total_experiments", len(experiments))

        for i, exp_config in enumerate(experiments):
            exp_id = exp_config.get("experiment_id", f"exp{i + 1:03d}")
            logger.info("starting_batch_iteration", index=i + 1, total=len(experiments), experiment_id=exp_id)

            args_dict = args._asdict()

            if exp_config:
                valid_keys = args_dict.keys()
                for key, value in exp_config.items():
                    if key in valid_keys:
                        args_dict[key] = value
                    else:
                        logger.warning("ignored_unknown_parameter", parameter=key)

            args_dict["output"] = os.path.join(base_output, "batch")
            exp_args = Args(**args_dict)

            metadata = run_single_experiment(exp_args, config_override=exp_config)
            results_summary.append(metadata)

        summary_path = os.path.join(base_output, "batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results_summary, cast(Any, f), indent=4)

        logger.info("batch_execution_complete", summary_path=summary_path)


# ----------------------------
# Main Entry Point
# ----------------------------
def main():
    """Main entry point for the simulator, initialising standard telemetry."""
    setup_structured_logging()
    setup_telemetry("tcg-cli")

    args = parse_args()

    # --- Prediction Mode ---
    if args.predict:
        with tracer.start_as_current_span("cli_static_prediction"):
            user_meta = {}
            meta_str = getattr(args, "meta", "")
            if meta_str:
                for part in meta_str.split(","):
                    if ":" in part:
                        deck, prop = part.split(":", 1)
                        try:
                            user_meta[deck.strip()] = float(prop.strip())
                        except ValueError:
                            continue
            try:
                deck_names, win_matrix, _ = load_matchup_data(args.input, MIN_GAMES)

                request_payload = PredictionRequest(
                    job_id="cli_static_predict",
                    total_players=args.players,
                    user_meta_spec=user_meta,
                    tournament_style=getattr(args, "tournament_style", "pure_swiss"),
                    precision_tier=PrecisionTier.STANDARD,
                    match_format="BO3",
                    deck_names=deck_names,
                    matchup_matrix=win_matrix.tolist()
                )

                result = predict_best_decks(
                    request=request_payload
                )
                recs = result["recommendations"]

                for i, r in enumerate(recs, 1):
                    # Wire confidence to the global MIN_GAMES threshold
                    sample_support = float(r['sample_support'])
                    is_confident = r["confidence"] >= 0.6 and sample_support >= MIN_GAMES
                    conf_status = "✅ High" if is_confident else "⚠️ Low"

                    logger.info(
                        "prediction_generated",
                        rank=i,
                        deck=r['deck'],
                        expected_win_rate=f"{r['expected_win_rate']:.2%}",
                        meta_share=f"{r['meta_share']:.2%}",
                        confidence_status=conf_status,
                        sample_support=sample_support
                    )
                return
            except Exception as e:
                import sys
                logger.error("prediction_failed", error=str(e), exc_info=True)
                sys.exit(1)

    # --- Batch or Single Mode ---
    if getattr(args, "batch", False):
        run_batch_experiments(args)
    else:
        run_single_experiment(args)


if __name__ == "__main__":
    main()