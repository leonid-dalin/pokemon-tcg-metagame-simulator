# engine.py | Replicator & Tournament generations
from __future__ import annotations
import time
import numpy as np
import csv
import structlog
from opentelemetry import trace
from typing import List, Any, Optional

# Optional modules
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None
try:
    import multiprocessing as mp

    MULTIPROC_AVAILABLE = True
except ImportError:
    MULTIPROC_AVAILABLE = False
    mp = None

from src.core.config import *
from src.core.data import safe_normalize
from src.core.types import SimulationConfig
from src.tournament.solver import get_variant_5_structure

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


# ----------------------------
# Tournament Simulation Workers
# ----------------------------
def _pure_swiss_worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a standard, fast, single-phase Swiss tournament using fast array slicing.
    """
    field_indices, config, win_matrix, matchup_details, rng_seed = args
    local_rng = np.random.default_rng(rng_seed)

    num_players = len(field_indices)
    player_wins = np.zeros(num_players, dtype=int)
    active_players = np.arange(num_players)

    num_rounds = config["num_rounds"]
    use_bayesian = config["use_bayesian_winrates"]
    deck_names = config["deck_names"]
    n_decks = len(deck_names)
    wins_per_deck = np.zeros(n_decks, dtype=float)
    matches_per_deck = np.zeros(n_decks, dtype=float)

    for _ in range(num_rounds):
        # Fast Swiss pairing logic
        local_rng.shuffle(active_players)
        order = np.argsort(-player_wins[active_players], kind='mergesort')
        sorted_active = active_players[order]

        p1s = sorted_active[0::2]
        p2s = sorted_active[1::2]
        if len(p1s) > len(p2s):
            p1s = p1s[:-1]
        if len(p1s) == 0:
            continue

        p1_indices = field_indices[p1s]
        p2_indices = field_indices[p2s]

        if use_bayesian:
            win_probs = np.zeros(len(p1s))
            for idx, (d1, d2) in enumerate(zip(p1_indices, p2_indices)):
                mu = matchup_details.get((deck_names[d1], deck_names[d2]), {"win_rate": 0.5, "match_count": 0})
                wr, mc = mu["win_rate"], mu["match_count"]

                if mc > 0:
                    win_probs[idx] = local_rng.beta(wr * mc + 1, (1 - wr) * mc + 1)
                else:
                    win_probs[idx] = 0.5
        else:
            win_probs = win_matrix[p1_indices, p2_indices]

        p1_wins = local_rng.random(len(p1s)) < win_probs
        player_wins[p1s[p1_wins]] += 1
        player_wins[p2s[~p1_wins]] += 1
        np.add.at(matches_per_deck, p1_indices, 1)
        np.add.at(matches_per_deck, p2_indices, 1)

    np.add.at(wins_per_deck, field_indices, player_wins)

    return wins_per_deck, matches_per_deck


def _championship_series_worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a full 2-Day Play! Pokémon Regional/International (Variant #5).
    Incorporates Parabolic Tie Convergence and vectorized memory allocation.
    """
    field_indices, config, win_matrix, matchup_details, rng_seed = args
    local_rng = np.random.default_rng(rng_seed)

    num_players = len(field_indices)
    n_decks = len(config["deck_names"])

    d1_rounds, cut_points, d2_rounds, top_cut = get_variant_5_structure(num_players)
    max_rounds = d1_rounds + d2_rounds

    # BO3 Conversion Formula: P_bo3 = 3p^2 - 2p^3
    bo3_win_matrix = 3 * (win_matrix ** 2) - 2 * (win_matrix ** 3)

    match_points = np.zeros(num_players, dtype=int)

    # Pre-allocated 2D array for speed instead of Python lists
    opponents_matrix = np.full((num_players, max_rounds), -1, dtype=int)
    rounds_played = np.zeros(num_players, dtype=int)

    active_players = np.arange(num_players)
    global_tie_rate = config.get("global_tie_rate", 0.15)

    def simulate_swiss_phase(rounds_to_play, current_active):
        for _ in range(rounds_to_play):
            if len(current_active) < 2: break

            local_rng.shuffle(current_active)
            order = np.argsort(-match_points[current_active], kind='mergesort')
            sorted_active = current_active[order]

            sw_p1s = sorted_active[0::2]
            sw_p2s = sorted_active[1::2]
            if len(sw_p1s) > len(sw_p2s): sw_p1s = sw_p1s[:-1]

            p1_decks = field_indices[sw_p1s]
            p2_decks = field_indices[sw_p2s]
            win_probs = bo3_win_matrix[p1_decks, p2_decks]

            # Parabolic Tie Convergence
            tie_probs = global_tie_rate * 4.0 * win_probs * (1.0 - win_probs)
            rolls = local_rng.random(len(sw_p1s))

            p1_win_thresh = np.clip(win_probs - (tie_probs / 2.0), 0.0, 1.0)
            tie_thresh = np.clip(p1_win_thresh + tie_probs, 0.0, 1.0)

            p1_wins_mask = rolls < p1_win_thresh
            p2_wins_mask = rolls >= tie_thresh
            ties_mask = ~(p1_wins_mask | p2_wins_mask)

            match_points[sw_p1s[p1_wins_mask]] += 3
            match_points[sw_p2s[p2_wins_mask]] += 3
            match_points[sw_p1s[ties_mask]] += 1
            match_points[sw_p2s[ties_mask]] += 1

            # Vectorized history tracking
            opponents_matrix[sw_p1s, rounds_played[sw_p1s]] = sw_p2s
            opponents_matrix[sw_p2s, rounds_played[sw_p2s]] = sw_p1s
            rounds_played[sw_p1s] += 1
            rounds_played[sw_p2s] += 1

    # Phase 1 (Day 1)
    simulate_swiss_phase(d1_rounds, active_players)

    # The Cut
    day2_mask = match_points >= cut_points
    day2_players = np.where(day2_mask)[0]

    # Phase 2 (Day 2)
    simulate_swiss_phase(d2_rounds, day2_players)

    # Tiebreakers & Top Cut

    if len(day2_players) > 0 and top_cut > 0:
        owp = np.zeros(num_players)
        for i in day2_players:
            rp = rounds_played[i]
            if rp == 0: continue
            opps = opponents_matrix[i, :rp]
            opp_points = match_points[opps]
            opp_rp = rounds_played[opps]
            # Avoid division by zero, enforce 0.25 floor per handbook
            opp_win_pcts = np.clip(opp_points / (np.maximum(opp_rp, 1) * 3), 0.25, 1.0)
            owp[i] = np.mean(opp_win_pcts)

        top_order = np.lexsort((-owp[day2_players], -match_points[day2_players]))
        top_players = day2_players[top_order][:top_cut]

        # Single Elimination Playoffs (No ties)
        standings = top_players.copy()
        while len(standings) > 1:
            next_round = []
            tc_p1s = np.array(standings[0::2])
            tc_p2s = np.array(standings[1::2])
            if len(tc_p1s) > len(tc_p2s): tc_p1s = tc_p1s[:-1]

            tc_p1_decks = field_indices[tc_p1s]
            tc_p2_decks = field_indices[tc_p2s]
            p1_wins = local_rng.random(len(tc_p1s)) < bo3_win_matrix[tc_p1_decks, tc_p2_decks]

            next_round.extend(tc_p1s[p1_wins])
            next_round.extend(tc_p2s[~p1_wins])
            standings = np.array(next_round)

            # Grant extra equivalent points for advancing in Top Cut
            match_points[next_round] += 3
            rounds_played[tc_p1s] += 1
            rounds_played[tc_p2s] += 1

    # Pure scaling based on actual match points acquired
    wins_equiv = match_points / 3.0
    matches_equiv = rounds_played.astype(float)

    wins_per_deck = np.zeros(n_decks, dtype=float)
    matches_per_deck = np.zeros(n_decks, dtype=float)
    np.add.at(wins_per_deck, field_indices, wins_equiv)
    np.add.at(matches_per_deck, field_indices, matches_equiv)

    return wins_per_deck, matches_per_deck


def run_tournament_generation(
        current_freq: np.ndarray,
        deck_names: List[str],
        win_matrix: np.ndarray,
        matchup_details: Dict[Tuple[str, str], Dict[str, Any]],
        config: Dict[str, Any],
        rng: np.random.Generator,
        pool: Optional[Any] = None,
) -> np.ndarray:
    n_decks = len(deck_names)
    tasks = []
    tournament_style = config.get("tournament_style", "pure_swiss")

    task_config = {
        "num_rounds": config["num_rounds"],
        "use_bayesian_winrates": config["use_bayesian_winrates"],
        "deck_names": deck_names,
        "selection_pressure": config["selection_pressure"],
        "tournament_style": tournament_style,
        "global_tie_rate": config.get("global_tie_rate", 0.15)
    }

    for _ in range(config["num_tournaments_per_gen"]):
        field_indices = rng.choice(n_decks, size=config["tournament_size"], p=current_freq)
        task_rng_seed = rng.integers(1 << 60)
        tasks.append((field_indices, task_config, win_matrix, matchup_details, task_rng_seed))

    deck_wins = np.zeros(n_decks)
    deck_matches = np.zeros(n_decks)

    worker_func = _championship_series_worker if tournament_style == "championship_series" else _pure_swiss_worker

    if pool is not None and len(tasks) > 1:
        import typing
        active_pool = typing.cast(Any, pool)
        for wins, matches in active_pool.imap_unordered(worker_func, tasks):
            deck_wins += wins
            deck_matches += matches
    else:
        for task in tasks:
            wins, matches = worker_func(task)
            deck_wins += wins
            deck_matches += matches

    with np.errstate(divide="ignore", invalid="ignore"):
        payoffs = np.where(deck_matches > 0, deck_wins / deck_matches, 0.5)

    new_freq = current_freq * np.exp(config["selection_pressure"] * (payoffs - 0.5))
    return safe_normalize(new_freq)


# ----------------------------
# Exponential Replicator Dynamic (MWU) with Entropy Regularization
# ----------------------------
def update_replicator_dynamics(
        current_freq: np.ndarray,
        win_matrix: np.ndarray,
        selection_pressure: float = SELECTION_PRESSURE,
        mutation_rate: float = MUTATION_RATE,
        noise_scale: float = NOISE_SCALE,
        rng: Optional[np.random.Generator] = None,
        last_payoffs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
        Executes the Multiplicative Weights Update (MWU).
        By extrapolating the gradient using the previous generation's payoffs,
        this algorithm completely dampens the zero-sum limit cycles (orbits)
        inherent to TCG metagames, forcing rapid point-wise convergence to the ESS.
        """
    n = len(current_freq)

    # 1. Calculate Current Expected Value (EV)
    payoffs = win_matrix @ current_freq

    # 2. Gradient Extrapolation
    if last_payoffs is not None:
        optimistic_payoffs = 2.0 * payoffs - last_payoffs
    else:
        optimistic_payoffs = payoffs

    # 3. Dynamic Centering
    avg_payoff = current_freq @ optimistic_payoffs

    # 4. Base Exponential Gradient
    exponent = selection_pressure * (optimistic_payoffs - avg_payoff)

    # 5. Stochastic Environmental Volatility
    if noise_scale > 0.0 and rng is not None:
        exponent += rng.normal(0, noise_scale, size=n)

    # 6. Weights Update Step
    growth = np.exp(exponent)
    raw_next_freq = current_freq * growth

    s = raw_next_freq.sum()
    if s > 0:
        raw_next_freq /= s
    else:
        raw_next_freq = np.ones(n) / n

    # 7. Ambient Uniform Mutation (Entropy Regularization)
    new_freq: np.ndarray = np.asarray((1.0 - mutation_rate) * raw_next_freq + (mutation_rate / n), dtype=float)
    return safe_normalize(new_freq), payoffs


# ----------------------------
# Evolutionary Stable State Solver
# ----------------------------
# src/evolution/engine.py
def find_evolutionary_stable_state(
        deck_names: List[str],
        win_matrix: np.ndarray,
        matchup_details: dict,
        config: SimulationConfig,
        history_file_path: Optional[str] = None,
) -> tuple[List[dict], List[np.ndarray], dict]:
    with tracer.start_as_current_span("find_evolutionary_stable_state") as ess_span:
        n = len(deck_names)
        current_freq = np.full(n, 1.0 / n)
        history: List[np.ndarray] = [current_freq.copy()]

        extinction_threshold = config.extinction_threshold
        extinct_decks: set[int] = set()
        inactive_counts = np.zeros(n, dtype=int)
        extinction_gens: dict[int, int] = {}

        logger.info("starting_simulation", mode=config.mode, max_gens=config.max_generations)

        pool = None
        safe_cores = 1
        if config.mode == "tournament" and config.use_multiproc and MULTIPROC_AVAILABLE:
            if mp is not None:
                with tracer.start_as_current_span("initialise_multiprocessing_pool"):
                    safe_cores = max(1, mp.cpu_count() // 2)
                    pool = mp.Pool(processes=safe_cores)
                    logger.info("multiprocessing_enabled", cores=safe_cores)
        history_file_handle = None
        history_writer = None
        if history_file_path:
            history_file_handle = open(history_file_path, "w", newline="", encoding="utf-8")
            history_writer = csv.writer(history_file_handle)
            history_writer.writerow(["Generation"] + deck_names)
            history_writer.writerow([0] + [f"{x:.6f}" for x in current_freq])

        start_time = time.time()
        gen_iterator = range(config.max_generations)
        if TQDM_AVAILABLE and tqdm:
            gen_iterator = tqdm(gen_iterator, desc=f"Simulating ({config.mode})", unit="gen")

        try:
            with tracer.start_as_current_span("evolutionary_generation_loop"):
                for gen in gen_iterator:
                    if config.mode == "replicator":
                        payoffs = win_matrix @ current_freq
                    else:
                        active_indices = [i for i in range(n) if i not in extinct_decks]
                        if len(active_indices) < 2:
                            logger.warning("simulation_aborted", reason="Less than 2 active decks remain.")
                            break

                        worker_func = (
                            _championship_series_worker
                            if config.tournament_style == "championship_series"
                            else _pure_swiss_worker
                        )

                        with tracer.start_as_current_span("execute_tournament_batch"):
                            payoffs = np.zeros(n)
                            tasks = []
                            for i in range(config.num_tournaments_per_gen):
                                seed = (
                                    config.seed + gen * config.num_tournaments_per_gen + i
                                    if config.seed is not None else None
                                )
                                tasks.append((active_indices, config.__dict__, win_matrix, matchup_details, seed))

                            if pool is not None:
                                # Logic unified using the hoisted worker_func
                                chunk_size = max(1, len(tasks) // (safe_cores * 2))
                                results = pool.imap_unordered(worker_func, tasks, chunksize=chunk_size)
                                for local_payoffs, _ in results:
                                    payoffs += local_payoffs
                            else:
                                for task in tasks:
                                    local_payoffs, _ = worker_func(task)
                                    payoffs += local_payoffs

                            payoffs /= config.num_tournaments_per_gen

                    if config.noise_scale > 0:
                        noise = np.random.normal(0, config.noise_scale, n)
                        payoffs += noise

                    avg_payoff = current_freq @ payoffs
                    growth_rates = (payoffs - avg_payoff) * config.selection_pressure

                    next_freq = current_freq + current_freq * growth_rates
                    next_freq = np.clip(next_freq, 0, None)

                    for i in range(n):
                        if i in extinct_decks:
                            if np.random.random() < config.mutation_rate:
                                next_freq[i] = config.mutation_rate
                                extinct_decks.remove(i)
                                inactive_counts[i] = 0
                            else:
                                next_freq[i] = 0.0
                        else:
                            if next_freq[i] < extinction_threshold:
                                next_freq[i] = 0.0
                                inactive_counts[i] += 1
                                if inactive_counts[i] >= config.max_inactive_generations:
                                    extinct_decks.add(i)
                                    if i not in extinction_gens:
                                        extinction_gens[i] = gen
                                        logger.debug("deck_extinction", deck=deck_names[i], generation=gen)
                            else:
                                inactive_counts[i] = 0

                    if np.sum(next_freq) > 0:
                        next_freq = safe_normalize(next_freq)
                    else:
                        next_freq = np.full(n, 1.0 / n)

                    delta = np.max(np.abs(next_freq - current_freq))
                    current_freq = next_freq

                    if gen % 10 == 0:
                        history.append(current_freq.copy())
                        if history_writer:
                            history_writer.writerow([gen + 1] + [f"{x:.6f}" for x in current_freq])

                    is_kinetically_stable = delta < config.stability_threshold

                    current_payoffs = win_matrix @ current_freq
                    avg_payoff = current_freq @ current_payoffs
                    max_advantage = float(np.max(current_payoffs - avg_payoff).item())

                    is_nash_equilibrium = max_advantage < NASH_EQUILIBRIUM

                    if is_kinetically_stable and is_nash_equilibrium:
                        logger.info("ess_reached", generation=gen + 1)
                        break

        except KeyboardInterrupt:
            logger.warning("simulation_interrupted")
        finally:
            if pool is not None:
                import typing
                active_pool = typing.cast(Any, pool)
                active_pool.close()
                active_pool.join()
            if history_file_handle:
                history_file_handle.close()

            if len(history) == 0 or not np.array_equal(history[-1], current_freq):
                history.append(current_freq.copy())

            sim_duration = time.time() - start_time
            logger.info("simulation_complete", duration_seconds=sim_duration)
            ess_span.set_attribute("simulation.duration", sim_duration)

        results = []
        for i in range(n):
            results.append(
                {
                    "deck": deck_names[i],
                    "frequency": float(current_freq[i]),
                    "is_active": current_freq[i] > extinction_threshold,
                    "generations_inactive": int(inactive_counts[i]),
                    "extinction_generation": extinction_gens.get(i),
                }
            )

        return results, history, extinction_gens
