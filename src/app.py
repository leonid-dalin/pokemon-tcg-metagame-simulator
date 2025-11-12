# src/app.py
import streamlit as st
import os
import sys
import numpy as np
from typing import List, cast

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.predictor import (
    predict_best_decks,
    UserMetaSpec,
    MetaValue,
    swiss_rounds_from_players,
    InferenceMode,
)
from src.data import load_matchup_data
from src.config import *
from src.simulation import find_evolutionary_stable_state
from src.simulation_config import SimulationConfig

@st.cache_data
def get_valid_deck_names() -> List[str]:
    input_path = os.path.join(INPUT_DIR, "ea_input.json")
    deck_names, _, _ = load_matchup_data(input_path, MIN_GAMES)
    return sorted(deck_names)


@st.cache_data
def load_full_win_matrix():
    input_path = os.path.join(INPUT_DIR, "ea_input.json")
    deck_names, win_matrix, _ = load_matchup_data(input_path, MIN_GAMES)
    return deck_names, win_matrix


# <--- OPTIMIZATION: Cache the expensive "pro" meta simulation ---
@st.cache_data
def get_pro_meta() -> np.ndarray:
    """
    Runs the full simulation once and caches the result for "pro" mode.
    """
    input_path = os.path.join(INPUT_DIR, "ea_input.json")
    deck_names, win_matrix, matchup_details = load_matchup_data(input_path, MIN_GAMES)

    sim_config = SimulationConfig(
        mode="replicator",
        max_generations=MAX_GENERATIONS,
        min_generations=int(MAX_GENERATIONS * MIN_GENERATIONS_PROP),
        extinction_threshold=EXTINCTION_THRESHOLD,
        stability_threshold=STABILITY_THRESHOLD,
        convergence_window=CONVERGENCE_WINDOW,
        max_inactive_generations=MAX_INACTIVE_GENERATIONS,
        use_bayesian_winrates=USE_BAYESIAN_WINRATES,
        tournament_size=TOURNAMENT_SIZE,
        num_tournaments_per_gen=NUM_TOURNAMENTS_PER_GEN,
        num_rounds=NUM_ROUNDS,
        use_multiproc=USE_MULTIPROC,
        seed=RNG_SEED,
        dynamic_deck_intro_prob=DYNAMIC_DECK_INTRO_PROB,
        mutation_floor=MUTATION_FLOOR,
        noise_scale=NOISE_SCALE,
        selection_pressure=SELECTION_PRESSURE,
    )
    results, _, _ = find_evolutionary_stable_state(
        deck_names=deck_names,
        win_matrix=win_matrix,
        matchup_details=matchup_details,
        config=sim_config,
    )
    fallback_meta = np.array([r["frequency"] for r in results])
    return fallback_meta


# --- End of Optimization ---

def main():
    st.set_page_config(page_title="TCG Metagame Predictor", layout="wide")
    st.title("🏆 TCG Metagame Predictor")

    # Sidebar
    st.sidebar.header("🏟️ Tournament Settings")
    players = st.sidebar.number_input(
        "Number of Players", min_value=2, max_value=256, value=32, step=1, format="%d"
    )
    swiss_rounds = swiss_rounds_from_players(players)
    st.sidebar.info(f"Swiss Rounds: **{swiss_rounds}**")

    input_mode = st.sidebar.radio(
        "Input Mode",
        ["Players", "Percentage"],
        help="Enter **raw player counts** or **percentages**."
    )

    inference_mode = st.sidebar.radio(
        "Inference Mode",
        ["pro", "casual"],
        format_func=lambda x: "Pro (Idealized Meta)" if x == "pro" else "Casual (Uniform Fill)",
        help="**Pro**: Uses simulation equilibrium. **Casual**: Uniform fill."
    )

    st.header("🃏 Your Deck (Optional)")
    my_deck = st.selectbox(
        "Simulate your expected performance",
        [""] + get_valid_deck_names(),
        key="my_deck"
    )

    st.header("📊 Declare Known Meta")

    # Dynamic deck input
    if "deck_inputs" not in st.session_state:
        st.session_state.deck_inputs = []

    # Add Deck button at TOP-LEFT
    if st.button("➕ Add Deck", key="add_deck_top"):
        st.session_state.deck_inputs.append({})

    deck_names = get_valid_deck_names()
    user_meta: UserMetaSpec = {}
    total_min = total_max = 0.0
    seen_decks = set()

    # Build dynamic inputs
    for idx in range(len(st.session_state.deck_inputs)):
        with st.container():
            st.divider()
            cols = st.columns([2, 1, 1])
            with cols[0]:
                available = [d for d in deck_names if d not in seen_decks]
                if not available:
                    st.warning("No more decks to add.")
                    break
                deck = st.selectbox(
                    "Deck", options=available, key=f"deck_{idx}", label_visibility="collapsed"
                )
                seen_decks.add(deck)

            with cols[1]:
                spec_type = st.radio(
                    "Type", ["Exact", "Min/Max"], key=f"type_{idx}", horizontal=True, label_visibility="collapsed"
                )

            with cols[2]:
                if input_mode == "Players":
                    max_val = players
                    step = 1
                    suffix = " players"
                else:
                    max_val = 1.0
                    step = 0.01
                    suffix = "%"

                if spec_type == "Exact":
                    if input_mode == "Players":
                        val = st.number_input(
                            f"Exact{suffix}",
                            min_value=0,
                            max_value=max_val,
                            value=min(2, max_val),
                            step=step,
                            key=f"val_{idx}",
                            format="%d" if step == 1 else "%.3f",
                        )
                        prop = val / players if players > 0 else 0.0
                    else:
                        val = st.number_input(
                            f"Exact{suffix}",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.1,
                            step=step,
                            key=f"val_{idx}",
                            format="%.3f"
                        )
                        prop = val
                    if prop > 0:
                        user_meta[deck] = prop
                        total_min += prop
                        total_max += prop

                else:  # Min/Max
                    c1, c2 = st.columns(2)
                    if input_mode == "Players":
                        min_ui = c1.number_input("Min", min_value=0, max_value=max_val, value=0, step=step,
                                                 key=f"min_{idx}", format="%d")
                        max_ui = c2.number_input("Max", min_value=0, max_value=max_val, value=min(3, max_val),
                                                 step=step, key=f"max_{idx}", format="%d")
                        min_prop = min_ui / players if players > 0 else 0.0
                        max_prop = max_ui / players if players > 0 else 0.0
                    else:
                        min_ui = c1.number_input("Min", min_value=0.0, max_value=1.0, value=0.05, step=step,
                                                 key=f"min_{idx}", format="%.3f")
                        max_ui = c2.number_input("Max", min_value=0.0, max_value=1.0, value=0.15, step=step,
                                                 key=f"max_{idx}", format="%.3f")
                        min_prop = min_ui
                        max_prop = max_ui

                    if min_prop >= max_prop:
                        st.error("Min must be < Max")
                    else:
                        user_meta[deck] = {"min": min_prop, "max": max_prop}
                        total_min += min_prop
                        total_max += max_prop

            # Delete button
            if st.button("🗑️", key=f"del_{idx}"):
                st.session_state.deck_inputs.pop(idx)
                st.rerun()

    if total_min > 1.0:
        st.error(f"❌ Minimum total ({total_min:.1%}) > 100%")
        st.stop()

    if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):

        # <--- OPTIMIZATION: Get the cached "pro" meta if needed
        fallback_pro = None
        if inference_mode == "pro":
            with st.spinner("Loading 'Pro' meta... (runs simulation on first load)"):
                fallback_pro = get_pro_meta()
        # --- End of Optimization ---

        with st.spinner("Analyzing metagame..."):
            try:
                result = predict_best_decks(
                    user_meta_spec=user_meta,
                    total_players=players,
                    min_sample_threshold=10,
                    inference_mode=cast(InferenceMode, inference_mode),
                    fallback_meta_pro=fallback_pro  # <--- Pass the cached meta
                )
            except Exception as e:
                st.exception(e)
                st.stop()

        # Load full data for matchup details
        full_deck_names, full_win_matrix = load_full_win_matrix()
        deck_to_idx = {name: i for i, name in enumerate(full_deck_names)}

        # Tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "🚫 Decks to Avoid", "🔍 Full Metagame"])

        with tab1:
            st.subheader("Top 3 Recommended Decks")
            for i, r in enumerate(result["recommendations"], 1):
                deck = r["deck"]
                st.markdown(f"### {i}. {deck}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Expected Win Rate", f"{r['expected_win_rate']:.2%}")
                col2.metric("Inferred Meta Share", f"{r['meta_share']:.2%}")
                col3.metric("Confidence", "✅ High" if r["confidence"] >= 0.6 else "⚠️ Low")
                col4.metric("Avg Match Samples", f"{r['sample_support']:.1f}")

                # Matchup breakdown (only vs decks with ≥2% meta share)
                if deck in deck_to_idx:
                    idx = deck_to_idx[deck]
                    favorable = []
                    threats = []
                    for opp_name in full_deck_names:
                        if opp_name == deck:
                            continue
                        opp_share = result["full_meta"].get(opp_name, 0)
                        if opp_share < 0.02:  # Only show vs relevant decks
                            continue
                        opp_idx = deck_to_idx[opp_name]
                        wr = full_win_matrix[idx, opp_idx]
                        if wr >= 0.55:
                            favorable.append((opp_name, wr))
                        elif wr <= 0.45:
                            threats.append((opp_name, wr))

                    if favorable:
                        fav_text = ", ".join([f"{d} ({wr:.0%})" for d, wr in favorable])
                        st.success(f"✅ Favorable vs: {fav_text}")
                    if threats:
                        thr_text = ", ".join([f"{d} ({wr:.0%})" for d, wr in threats])
                        st.warning(f"⚠️ Threats: {thr_text}")

            # My Deck Simulation
            if my_deck and my_deck in result["metrics_per_deck"]:
                st.divider()
                st.subheader("🃏 Your Deck Performance")
                my_metrics = result["metrics_per_deck"][my_deck]

                expected_wr = my_metrics["expected_win_rate"]
                undefeated_prob = my_metrics["undefeated_probability"]
                sos = my_metrics["sos"]
                omw = my_metrics["omw"]
                sample_support = my_metrics["sample_support"]

                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Expected Win Rate", f"{expected_wr:.2%}")
                col2.metric(f"Chance to Go {result['swiss_rounds']}-0", f"{undefeated_prob:.2%}")
                col3.metric("Strength of Schedule (SoS)", f"{sos:.2f}")
                col4.metric("Opponent's Avg Win Rate (OMW)", f"{omw:.2%}")

                # Additional context
                st.caption(f"Based on {sample_support:.1f} avg match samples against the predicted meta.")
                if undefeated_prob > 0.1:
                    st.info(
                        f"Your deck has a **{undefeated_prob:.2%}** chance of going undefeated in {result['swiss_rounds']} rounds.")
                elif undefeated_prob > 0.01:
                    st.warning(
                        f"Your deck has a **{undefeated_prob:.2%}** chance of going undefeated. A perfect run is unlikely.")
                else:
                    st.error(
                        f"Your deck has a **{undefeated_prob:.2%}** chance of going undefeated. Focus on a strong record instead of perfection.")

        with tab2:
            st.subheader("Decks to Avoid")
            for i, r in enumerate(result["avoid"], 1):
                deck = r["deck"]
                st.markdown(f"### {i}. {deck}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Win Rate", f"{r['expected_win_rate']:.2%}")
                col2.metric("Inferred Meta Share", f"{r['meta_share']:.2%}")
                col3.metric("Avg Match Samples", f"{r['sample_support']:.1f}")

                # Matchup breakdown for AVOIDED decks
                if deck in deck_to_idx:
                    idx = deck_to_idx[deck]
                    unfavorable = []
                    threats = []
                    for opp_name in full_deck_names:
                        if opp_name == deck:
                            continue
                        opp_share = result["full_meta"].get(opp_name, 0)
                        if opp_share < 0.02:  # Only show vs relevant decks
                            continue
                        opp_idx = deck_to_idx[opp_name]
                        wr = full_win_matrix[idx, opp_idx]
                        # For avoided decks, show what *beats them* (their threats)
                        if wr <= 0.45:  # Opp beats this deck
                            threats.append((opp_name, wr))
                        elif wr >= 0.55:  # This deck beats opp (but it's avoided)
                            unfavorable.append((opp_name, wr))

                    if threats:
                        thr_text = ", ".join([f"{d} ({wr:.0%})" for d, wr in threats])
                        st.warning(f"❌ Beaten by: {thr_text}")
                    if unfavorable:
                        unfav_text = ", ".join([f"{d} ({wr:.0%})" for d, wr in unfavorable])
                        st.info(f"🟢 Beats: {unfav_text}")

        with tab3:
            st.subheader("Full Inferred Metagame")
            for deck, prop in sorted(result["full_meta"].items(), key=lambda x: -x[1]):
                if prop > 0.001:
                    st.text(f"{deck}: {prop:.2%}")

        st.divider()
        st.caption(
            "Note: Matchups with <10 games are uncertain. "
            "Favorable: ≥55% WR vs decks with ≥2% meta share. Threats: ≤45% WR."
        )


if __name__ == "__main__":
    main()