#!/usr/bin/env python3
# main.py — CLI entry point, experiment batching, logging control, reproducibility.

from __future__ import annotations
import json
import logging
import time
import os
from typing import cast, Dict, Any, Optional, Literal, TextIO

# Local modules
from .data import (
    load_matchup_data,
    cluster_decks_by_matchup_profile,
    compute_deck_dominance,
)
from .simulation import find_evolutionary_stable_state
from .analysis import (
    compute_convergence_metrics,
    generate_final_state_tier_list,
    generate_all_time_tier_list,
    compute_matchup_cycles,
    compute_deck_similarity,
)
from .plotting import (
    plot_metagame_evolution_interactive,
    plot_matchup_heatmap_interactive,
    plot_matchup_network,
)
from .config import *
from .cli_args import parse_args, Args
from .predictor import predict_best_decks
from .simulation_config import SimulationConfig


# ----------------------------
# Single Experiment Runner
# ----------------------------
def run_single_experiment(args: Args, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a single metagame simulation experiment."""
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Construct the Unique Output Directory ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_str = args.mode
    gen_str = f"gens{args.gens // 1000}K" if args.gens >= 1000 else f"gens{args.gens}"

    base_output_name = f"{timestamp}_{mode_str}_{gen_str}"
    base_output_dir = os.path.join(args.output, base_output_name)

    # Safely get experiment_id
    experiment_id = "default"
    if config_override:
        experiment_id = config_override.get("experiment_id", "default")
        # For batch runs, create a subdirectory for each experiment
        base_output_dir = os.path.join(base_output_dir, experiment_id)

    # Create the initial directory
    os.makedirs(base_output_dir, exist_ok=True)

    # Log to file
    log_file = os.path.join(base_output_dir, "simulation.txt")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Avoid adding handlers repeatedly in batch mode
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    logging.info(f"🚀 Starting experiment '{experiment_id}'. Output: {base_output_dir}")
    logging.info(f"Parameters: mode={args.mode}, gens={args.gens}, noise={args.noise}")

    # --- Build SimulationConfig directly ---
    min_generations = max(1, int(args.gens * MIN_GENERATIONS_PROP))
    mode_literal = cast(Literal["replicator", "tournament"], args.mode)

    sim_config = SimulationConfig(
        mode=mode_literal,
        max_generations=args.gens,
        min_generations=min_generations,
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
        dynamic_deck_intro_prob=args.intro_prob,
        mutation_floor=args.mutation_floor,
        noise_scale=args.noise,
        selection_pressure=args.selection_pressure,
    )
    logging.debug(f"Simulation Config: {sim_config}")

    # Load data
    deck_names, win_matrix, matchup_details = load_matchup_data(args.input, args.min_games)
    if not deck_names:
        logging.error("No reliable decks loaded. Aborting.")
        return {}

    # Dominance diagnostic
    compute_deck_dominance(win_matrix, deck_names)

    # Clustering
    if args.cluster:
        cluster_decks_by_matchup_profile(win_matrix, deck_names, n_clusters=5)

    history_file_path = os.path.join(base_output_dir, "metagame_history_full.csv")

    # Run simulation
    start_time = time.time()
    results, history, extinction_gens = find_evolutionary_stable_state(
        deck_names=deck_names,
        win_matrix=win_matrix,
        matchup_details=matchup_details,
        config=sim_config,
        history_file_path=history_file_path,
    )
    sim_duration = time.time() - start_time
    logging.info(f"⏱️  Simulation completed in {sim_duration:.2f} seconds")

    convergence_metrics = compute_convergence_metrics(history)
    final_conv_gen = convergence_metrics["convergence_generation"]
    if final_conv_gen is not None:
        conv_status = f"CONV@{final_conv_gen}"
    else:
        conv_status = "NOCONV"
    logging.info(f"📈 Convergence at generation: {final_conv_gen if final_conv_gen is not None else 'Not Converged'}")

    # Construct final directory name
    final_output_name = f"{timestamp}_{mode_str}_{gen_str}_{conv_status}"
    if config_override and experiment_id != "default":
        final_output_name = f"{final_output_name}_{experiment_id}"

    file_handler.close()
    final_output_dir = os.path.join(args.output, final_output_name)

    # Rename the directory
    try:
        if os.path.exists(final_output_dir):
            # If final dir somehow exists, add a counter
            counter = 1
            while os.path.exists(f"{final_output_dir}_{counter}"):
                counter += 1
            final_output_dir = f"{final_output_dir}_{counter}"
        os.rename(base_output_dir, final_output_dir)
        logging.info(f"✅ Output directory renamed to: {final_output_dir}")
    except Exception as e:
        logging.error(f"Failed to rename output directory: {e}")
        final_output_dir = base_output_dir  # Fallback to original

    output_dir = final_output_dir

    # Save results
    csv_path = os.path.join(output_dir, "ess_equilibrium.csv")
    try:
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                cast(TextIO, f),
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
        logging.info(f"✅ Results saved to {csv_path}")
    except (IOError, OSError) as e:
        logging.error(f"Failed to write CSV: {e}")

    # Tier lists
    final_tiers = generate_final_state_tier_list(deck_names, history, win_matrix)
    all_time_tiers = generate_all_time_tier_list(deck_names, history, win_matrix)
    with open(os.path.join(output_dir, "final_tiers.json"), "w", encoding="utf-8") as f:
        json.dump(final_tiers, cast(TextIO, f), indent=2)
    with open(os.path.join(output_dir, "all_time_tiers.json"), "w", encoding="utf-8") as f:
        json.dump(all_time_tiers, cast(TextIO, f), indent=2)

    # Matchup analysis
    cycles = compute_matchup_cycles(win_matrix, deck_names)

    # Create and pass final_active_mask to compute_deck_similarity
    # Create a list indicating if deck is active in the FINAL state
    final_active_mask = [r["is_active"] for r in results]
    similarity = compute_deck_similarity(win_matrix, deck_names, final_active_mask=final_active_mask)
    with open(os.path.join(output_dir, "deck_similarity.json"), "w", encoding="utf-8") as f:
        json.dump(similarity.tolist(), cast(TextIO,f), indent=2)

    # Plotting
    if not args.no_plot:
        plot_metagame_evolution_interactive(
            history,
            deck_names,
            extinction_gens,
            save_path=os.path.join(output_dir, "metagame_evolution.html"),
        )
        tier_order = []
        for tier in "SABCD":
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

    # Return metadata for batch runs
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
            "intro_prob": args.intro_prob,
        },
    }
    return metadata


# ----------------------------
# Batch Experiment Runner
# ----------------------------
def run_batch_experiments(args: Args):
    """Run multiple experiments with different configurations."""
    batch_config_path = args.batch_config
    if not batch_config_path or not os.path.exists(batch_config_path):
        logging.error(f"Batch config file not found: {batch_config_path}")
        return

    with open(batch_config_path, "r", encoding="utf-8") as f:
        batch_config = json.load(f)

    experiments = batch_config.get("experiments", [])
    base_output = args.output
    results_summary = []

    for i, exp_config in enumerate(experiments):
        exp_id = exp_config.get("experiment_id", f"exp{i + 1:03d}")
        logging.info(f"--- Starting Batch Experiment {i + 1}/{len(experiments)}: {exp_id} ---")

        # Start with the base arguments from the command line
        args_dict = args._asdict()

        # Update them with any overrides from the current experiment's config
        if exp_config:
            valid_keys = args_dict.keys()
            for key, value in exp_config.items():
                if key in valid_keys:
                    args_dict[key] = value
                else:
                    logging.warning(f"Ignoring unknown parameter '{key}' in batch config.")

        # Set the main output directory for all batch experiments
        args_dict["output"] = os.path.join(base_output, "batch")

        # Create a new immutable Args object for this specific run
        exp_args = Args(**args_dict)

        metadata = run_single_experiment(exp_args, config_override=exp_config)
        results_summary.append(metadata)

    # Save batch summary
    summary_path = os.path.join(base_output, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, cast(TextIO, f), indent=4)
    logging.info(f"📊 Batch summary saved to {summary_path}")


# ----------------------------
# Main Entry Point
# ----------------------------
def main():
    """Main entry point for the simulator."""
    args = parse_args()

    # --- Prediction Mode ---
    if args.predict:
        user_meta = {}
        meta_str = args.meta
        if meta_str:
            for part in meta_str.split(","):
                if ":" in part:
                    deck, prop = part.split(":", 1)
                    try:
                        user_meta[deck.strip()] = float(prop.strip())
                    except ValueError:
                        continue
        try:
            result = predict_best_decks(
                user_meta_spec=user_meta,
                total_players=args.players,
                min_sample_threshold=10,
            )
            recs = result["recommendations"]

            for i, r in enumerate(recs, 1):
                conf = "✅ High" if r["confidence"] >= 0.6 else "⚠️ Low"
                print(f"{i}. {r['deck']}")
                print(f"   Expected Win Rate: {r['expected_win_rate']:.2%}")
                print(f"   Meta Share: {r['meta_share']:.2%}")
                print(f"   Confidence: {conf} ({r['sample_support']:.1f} avg matches vs meta)")
                print()
            return
        except Exception as e:
            import sys
            logging.error(f"Prediction failed: {e}")
            sys.exit(1)

    # --- Batch or Single Mode ---
    if args.batch:
        run_batch_experiments(args)
    else:
        run_single_experiment(args)


if __name__ == "__main__":
    main()