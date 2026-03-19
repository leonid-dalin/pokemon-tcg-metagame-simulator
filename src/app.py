import streamlit as st
import os
import sys
import uuid
import json
import re
import numpy as np
import pandas as pd
from typing import List, cast, Tuple, Dict, Any

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.predictor import predict_best_decks, UserMetaSpec, MatchFormat, swiss_rounds_from_players
from src.data import load_matchup_data
from src.config import INPUT_DIR, MIN_GAMES, TIER_S_THRESHOLD, TIER_A_THRESHOLD, TIER_B_THRESHOLD, TIER_C_THRESHOLD
from src.simulation import get_variant_5_structure
from src.monte_carlo import run_monte_carlo_analytics

@st.cache_data(show_spinner=False)
def get_valid_deck_names() -> List[str]:
    input_path = os.path.join(INPUT_DIR, "ea_input.json")
    deck_names, _, _ = load_matchup_data(input_path, MIN_GAMES)
    return sorted(deck_names)

@st.cache_data(show_spinner=False)
def load_full_win_matrix():
    input_path = os.path.join(INPUT_DIR, "ea_input.json")
    deck_names, win_matrix, _ = load_matchup_data(input_path, MIN_GAMES)
    return deck_names, win_matrix

# --- HTML Parser for Limitless Labs ---
def parse_limitless_html(html_str: str, valid_decks: List[str]) -> Tuple[Dict[str, int], int, int]:
    """Parses Limitless Labs HTML for deck representation."""
    deck_script_match = re.search(r'<script type="application/json" data-sveltekit-fetched data-url="[^"]*/tcg/decks[^"]*">(.*?)</script>', html_str, re.DOTALL)
    
    if not deck_script_match:
        raise ValueError("Could not locate the internal JSON data block in the uploaded HTML.")
        
    wrapper_json = json.loads(deck_script_match.group(1))
    body_json = json.loads(wrapper_json.get("body", "{}"))
    decks = body_json.get("message", [])
    
    if not decks:
        raise ValueError("Deck array was empty in the parsed JSON.")

    parsed_meta = {}
    wildcard_players = 0
    total_players = 0

    valid_decks_lower = {d.lower(): d for d in valid_decks}

    for d in decks:
        name = d.get("name", "Unknown")
        players = int(d.get("players", 0))
        total_players += players
        
        if name.lower() in valid_decks_lower:
            canonical_name = valid_decks_lower[name.lower()]
            parsed_meta[canonical_name] = parsed_meta.get(canonical_name, 0) + players
        else:
            wildcard_players += players

    return parsed_meta, total_players, wildcard_players

# --- Composite Score Generator ---
def calculate_ultimate_score(res: Dict[str, Any], mc_res: Dict[str, Any], d2_rounds: int, top_cut: int) -> Dict[str, Any]:
    full_deck_names = list(res["metrics_per_deck"].keys())
    active_decks = [d for d in full_deck_names if res["metrics_per_deck"][d]["meta_share"] >= 0.001]
    
    if not active_decks: 
        return res
    
    wrs = np.array([res["metrics_per_deck"][d]["expected_win_rate"] for d in active_decks])
    meta_shares = np.array([res["metrics_per_deck"][d]["meta_share"] for d in active_decks])
    
    # Raw individual probabilities
    raw_day2s = np.array([mc_res.get(d, {}).get("day2_conversion", 0) for d in active_decks])
    raw_top8s = np.array([mc_res.get(d, {}).get("top_cut_conversion", 0) for d in active_decks])
    raw_wins = np.array([mc_res.get(d, {}).get("win_probability", 0) for d in active_decks])

    def z_score(arr):
        std = np.std(arr)
        return (arr - np.mean(arr)) / std if std > 0 else np.zeros_like(arr)

    z_wrs = z_score(wrs)

    def compute_final_score(z_d2, z_t8, z_w):
        if d2_rounds > 0 and top_cut > 0:
            raw_z = 0.20 * z_wrs + 0.25 * z_d2 + 0.25 * z_t8 + 0.30 * z_w
        elif top_cut > 0:
            raw_z = 0.30 * z_wrs + 0.30 * z_t8 + 0.40 * z_w
        else:
            raw_z = z_wrs
        return (np.tanh(raw_z) + 1.0) / 2.0 * 100.0

    # 1. Player POV (Individual EV)
    scores_player = compute_final_score(z_score(raw_day2s), z_score(raw_top8s), z_score(raw_wins))
    
    # 2. Archetype POV (Macro Impact)
    scores_archetype = compute_final_score(
        z_score(raw_day2s * meta_shares), 
        z_score(raw_top8s * meta_shares), 
        z_score(raw_wins * meta_shares)
    )
    
    for idx, d in enumerate(active_decks):
        res["metrics_per_deck"][d]["score_player"] = float(scores_player[idx])
        res["metrics_per_deck"][d]["score_archetype"] = float(scores_archetype[idx])
    
    return res

# --- State Management ---
def init_session_state():
    defaults = {
        "meta_rows": [], "prediction_result": None, "mc_result": None,
        "rec_limit": 3, "avoid_limit": 3, "imported_players": 256
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def add_meta_row(deck="", spec_type="Exact", val=10.0, is_locked=False): 
    st.session_state.meta_rows.append({"id": str(uuid.uuid4()), "deck": deck, "spec_type": spec_type, "val": val, "is_locked": is_locked})
def delete_meta_row(row_id: str): st.session_state.meta_rows = [r for r in st.session_state.meta_rows if r["id"] != row_id]
def clear_all_rows(): st.session_state.meta_rows = []
def load_more_recs(): st.session_state.rec_limit += 3
def load_more_avoids(): st.session_state.avoid_limit += 3

def get_tier(score: float) -> str:
    if score >= TIER_S_THRESHOLD * 100: return "S"
    if score >= TIER_A_THRESHOLD * 100: return "A"
    if score >= TIER_B_THRESHOLD * 100: return "B"
    if score >= TIER_C_THRESHOLD * 100: return "C"
    return "D"

# --- Main Application ---
def main():
    st.set_page_config(page_title="TCG Metagame Predictor", page_icon="🏆", layout="wide")
    init_session_state()
    
    st.title("🏆 TCG Metagame Predictor")
    st.markdown("Predict meta-weighted performance using real-world empirical data and strict Monte Carlo bracket simulations.")

    # ==========================================
    # SIDEBAR: TOURNAMENT & ENGINE SETTINGS
    # ==========================================    
    with st.sidebar:
        st.header("🏟️ Tournament Settings")
        with st.container(border=True):
            tourney_structure = st.radio("Tournament Structure", ["Championship Series", "Pure Swiss"], index=0)
            players = st.number_input("Number of Players", min_value=2, max_value=8192, value=st.session_state.imported_players, step=1)
            
            if tourney_structure == "Championship Series":
                d1_rounds, cut_points, d2_rounds, top_cut = get_variant_5_structure(players)
                total_swiss = d1_rounds + d2_rounds
                st.caption("**Championship Series Details**")
                st.markdown(f"- **Day 1**: {d1_rounds} Rounds\n- **Day 2 Cut**: {cut_points} Match Pts\n- **Day 2**: {d2_rounds} Rounds\n- **Playoffs**: Top {top_cut}")
            else:
                d1_rounds = swiss_rounds_from_players(players)
                cut_points, d2_rounds, top_cut = 999, 0, (8 if players >= 8 else 0)
                total_swiss = d1_rounds
                st.caption("**Pure Swiss Details**")
                st.markdown(f"- **Swiss**: {total_swiss} Rounds")
                if top_cut > 0: st.markdown(f"- **Playoffs**: Top {top_cut}")

        st.header("⚙️ Engine Parameters")
        with st.container(border=True):
            mc_iterations = st.selectbox("Monte Carlo Iterations", options=[1000, 10000, 100000, 1000000], format_func=lambda x: f"{x:,}", index=1)
            match_format = st.radio("Match Format", ["BO1", "BO3"], index=1)
            
            min_sample_threshold = st.slider("Matchup Minimum Games", min_value=1, max_value=100, value=10, step=1)
            input_mode = st.radio("Constraint Mode", ["Percentage", "Raw Players"])
            is_raw = input_mode == "Raw Players"

            st.divider()
            st.markdown("**🧪 BETA Features**")
            use_tie_convergence = st.toggle("Enable Win-Rate Tie Convergence", value=True, help="Mathematically simulates real-world match timeouts using a parabolic curve based on matchup closeness.")
            global_tie_rate = st.slider("Global Tie Rate (%)", min_value=0.0, max_value=30.0, value=15.0, step=0.5, disabled=not use_tie_convergence) / 100.0

    # ==========================================
    # MAIN UI: META CONSTRAINTS & IMPORT
    # ==========================================
    st.subheader("📊 Metagame Constraints")
    
    with st.expander("📁 Import Limitless Labs HTML", expanded=False):
        uploaded_file = st.file_uploader("Upload a saved HTML file from Limitless Labs...", type=["html", "htm"])
        if uploaded_file is not None:
            if st.button("Extract Data & Populate Constraints", type="secondary"):
                try:
                    parsed_meta, total_import_players, wildcard_players = parse_limitless_html(uploaded_file.getvalue().decode("utf-8"), get_valid_deck_names())
                    st.session_state.imported_players = total_import_players
                    clear_all_rows()
                    for deck_name, count in parsed_meta.items():
                        add_meta_row(deck=deck_name, spec_type="Exact", val=(count / total_import_players) * 100.0 if total_import_players > 0 else 0.0, is_locked=True)
                    st.success(f"✅ Successfully imported {len(parsed_meta)} recognized decks ({total_import_players} total players).")
                    if wildcard_players > 0: st.warning(f"⚠️ {wildcard_players} players were using unrecognized/rogue decks. Handled via rescaling.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse file: {e}")

    user_meta: UserMetaSpec = {}
    total_min = 0.0
    deck_names = get_valid_deck_names()
    seen_decks = set()

    with st.expander("🛠️ Active Custom Constraints", expanded=True):
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("➕ Add Constraint"): add_meta_row()
        with c2:
            if st.button("🗑️ Clear All"): clear_all_rows(); st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)
        constraint_cols = st.columns(2)
        
        for idx, row in enumerate(st.session_state.meta_rows):
            row_id = row["id"]
            target_col = constraint_cols[idx % 2]
            with target_col:
                with st.container(border=True):
                    cols = st.columns([3, 2, 3, 0.8])
                    with cols[0]:
                        available = [d for d in deck_names if d not in seen_decks or d == row.get("deck")]
                        if not available: break
                        default_idx = available.index(row["deck"]) if row.get("deck") in available else 0
                        deck = st.selectbox("Deck", options=available, index=default_idx, key=f"deck_{row_id}", label_visibility="collapsed")
                        seen_decks.add(deck)
                        st.session_state.meta_rows[idx]["deck"] = deck
                    with cols[1]:
                        spec_type = st.radio("Type", ["Exact", "Range"], index=0 if row.get("spec_type")=="Exact" else 1, key=f"type_{row_id}", horizontal=True, label_visibility="collapsed")
                        st.session_state.meta_rows[idx]["spec_type"] = spec_type
                    with cols[2]:
                        if spec_type == "Exact":
                            if is_raw:
                                default_val = int(row.get("val", 10.0)) if not row.get("is_locked") else int((row.get("val", 10)/100)*players)
                                val_ui = st.number_input("Players", min_value=0, max_value=int(players), value=min(default_val, int(players)), step=1, key=f"val_{row_id}")
                                prop = val_ui / players if players > 0 else 0.0
                            else:
                                val_ui = st.number_input("Percent (%)", min_value=0.0, max_value=100.0, value=float(row.get("val", 10.0)), step=0.1, format="%.2f", key=f"val_{row_id}")
                                prop = val_ui / 100.0
                            if prop > 0: user_meta[deck] = prop; total_min += prop
                        else:
                            if is_raw:
                                val = st.slider("Range (Players)", 0, int(players), (0, min(15, int(players))), key=f"slider_{row_id}")
                                min_prop, max_prop = val[0] / players if players > 0 else 0.0, val[1] / players if players > 0 else 0.0
                            else:
                                val = st.slider("Range (%)", 0.0, 100.0, (0.0, 15.0), step=0.1, format="%.1f%%", key=f"slider_{row_id}")
                                min_prop, max_prop = val[0] / 100.0, val[1] / 100.0
                            user_meta[deck] = {"min": min_prop, "max": max_prop}; total_min += min_prop
                    with cols[3]:
                        st.button("🗑️", key=f"del_{row_id}", on_click=delete_meta_row, args=(row_id,))

    if total_min > 1.0: st.error(f"❌ Minimum total across all constraints ({total_min:.1%}) exceeds 100%. Please adjust your values before predicting."); st.stop()

    st.divider()

    # ==========================================
    # EXECUTION & DATAFRAME RESULTS
    # ==========================================
    if st.button("🚀 Generate Metagame & Predict", type="primary", width="stretch", disabled=(total_min > 1.0)):
        with st.status("⚙️ Initializing Engine...", expanded=True) as status:
            try:
                full_deck_names, full_win_matrix = load_full_win_matrix()
                res = predict_best_decks(user_meta_spec=user_meta, total_players=players, min_sample_threshold=min_sample_threshold, match_format=cast(MatchFormat, match_format))
                
                mc_progress = st.progress(0, text="Dispatching parallel workers...")
                def mc_progress_callback(completed, total):
                    pct = int((completed / total) * 100)
                    mc_progress.progress(completed / total, text=f"Processing Monte Carlo brackets... {pct}% ({completed}/{total} cores complete)")

                mc_res = run_monte_carlo_analytics(
                    deck_names=full_deck_names, win_matrix=full_win_matrix, meta_distribution=res["full_meta"],
                    d1_rounds=d1_rounds, cut_points=cut_points, d2_rounds=d2_rounds, top_cut=top_cut,
                    players=players, iterations=mc_iterations, match_format=match_format,
                    progress_callback=mc_progress_callback, use_tie_convergence=use_tie_convergence, global_tie_rate=global_tie_rate
                )
                
                res = calculate_ultimate_score(
                    res, 
                    mc_res, 
                    d2_rounds, 
                    top_cut, 
                )
                status.update(label="✅ Simulation Complete!", state="complete", expanded=False)
                
                st.session_state.prediction_result = res
                st.session_state.mc_result = mc_res
                st.session_state.rec_limit = 3
                st.session_state.avoid_limit = 3
            except Exception as e:
                status.update(label="❌ Simulation Failed", state="error", expanded=True); st.exception(e); st.stop()

    # --- Render Results if Available ---
    if st.session_state.prediction_result is not None and st.session_state.mc_result is not None:
        res = st.session_state.prediction_result
        mc_res = st.session_state.mc_result
        
        full_deck_names, full_win_matrix = load_full_win_matrix()
        deck_to_idx = {name: i for i, name in enumerate(full_deck_names)}

        st.subheader("📊 Interactive Dashboard & Tournament Odds")
        
        # The View Toggle sits right above the data
        view_col1, view_col2 = st.columns([1, 3])
        with view_col1:
            odds_view = st.radio(
                "Odds Perspective", 
                ["Player (Individual EV)", "Archetype (Macro Impact)"], 
                index=0, 
                help="Switch between individual player odds (If I play this deck...) and macro performance (% of the bracket dominating)."
            )
        
        # Dynamically sort based on the chosen view
        score_key = "score_player" if odds_view == "Player (Individual EV)" else "score_archetype"
        
        active_decks = [d for d in full_deck_names if res["metrics_per_deck"][d]["meta_share"] >= 0.001]
        all_decks_sorted = sorted(active_decks, key=lambda d: res["metrics_per_deck"][d][score_key], reverse=True)
        
        data = []
        for i, deck in enumerate(all_decks_sorted, 1):
            metrics = res["metrics_per_deck"][deck]
            mc_metrics = mc_res.get(deck, {"day2_conversion": 0, "top_cut_conversion": 0, "win_probability": 0, "day2_share": 0, "top_cut_share": 0})
            
            row_data = {
                "#": i,
                "Tier": get_tier(metrics[score_key]),
                "Deck": deck,
                "Score": metrics[score_key],
                "Type": "🔒 User" if deck in user_meta else "📈 Base",
                "Exp. WR %": metrics["expected_win_rate"] * 100,
                "SoS %": metrics["sos"] * 100,
                "OMW %": metrics["omw"] * 100
            }
            
            meta_share = metrics["meta_share"] * 100
            
            if odds_view == "Player (Individual EV)":
                row_data["Meta Share %"] = meta_share
                if d2_rounds > 0: row_data["Day 2 (Player) %"] = mc_metrics.get("day2_conversion", 0) * 100
                if top_cut > 0:
                    row_data["Top 8 (Player) %"] = mc_metrics.get("top_cut_conversion", 0) * 100
                    row_data["Win (Player) %"] = mc_metrics.get("win_probability", 0) * 100
            else:
                row_data["Day 1 Share %"] = meta_share
                if d2_rounds > 0: row_data["Day 2 Share %"] = mc_metrics.get("day2_share", 0) * 100
                if top_cut > 0:
                    row_data["Top 8 Share %"] = mc_metrics.get("top_cut_share", 0) * 100
                    row_data["Winner Share %"] = mc_metrics.get("win_probability", 0) * meta_share * players
                
            data.append(row_data)
            
        df = pd.DataFrame(data)

        col_config = {
            "#": st.column_config.NumberColumn(width="small"),
            "Tier": st.column_config.TextColumn(width="small", help="S-Tier (>90), A-Tier (>70), B-Tier (>50)."),
            "Deck": st.column_config.TextColumn(width="medium"),
            "Score": st.column_config.NumberColumn(format="%.2f", help="Ultimate Composite Score (0-100)."),
            "Type": st.column_config.TextColumn(help="Forced by user constraint (🔒) or simulated baseline (📈)."),
            "Exp. WR %": st.column_config.NumberColumn(format="%.2f %%", help="Expected Win Rate against the entire predicted field."),
            "SoS %": st.column_config.NumberColumn(format="%.2f %%", help="Strength of Schedule. Lower means an easier run."),
            "OMW %": st.column_config.NumberColumn(format="%.2f %%", help="Opponent's Match Win %. Primary tiebreaker.")
        }
        
        if odds_view == "Player (Individual EV)":
            col_config["Meta Share %"] = st.column_config.NumberColumn(format="%.2f %%")
            if d2_rounds > 0: col_config["Day 2 (Player) %"] = st.column_config.NumberColumn(format="%.2f %%")
            if top_cut > 0:
                col_config["Top 8 (Player) %"] = st.column_config.NumberColumn(format="%.2f %%")
                col_config["Win (Player) %"] = st.column_config.NumberColumn(format="%.2f %%")
        else:
            col_config["Day 1 Share %"] = st.column_config.NumberColumn(format="%.2f %%")
            if d2_rounds > 0: col_config["Day 2 Share %"] = st.column_config.NumberColumn(format="%.2f %%")
            if top_cut > 0:
                col_config["Top 8 Share %"] = st.column_config.NumberColumn(format="%.2f %%")
                col_config["Winner Share %"] = st.column_config.NumberColumn(format="%.2f %%")

        st.caption("Click any column header to sort. Hover over headers for detailed metric definitions. Click `#` to reset to default sorting.")
        st.dataframe(df, width="stretch", hide_index=True, column_config=col_config)
        st.divider()

        # --- Head-to-Head Field Comparator ---
        st.subheader("⚔️ Head-to-Head Field Comparator")
        st.markdown("Compare odds and matchups between two decks across the predicted field.")
        
        valid_meta_decks = [d["Deck"] for d in data]
        c1, c2 = st.columns(2)
        deck_a = c1.selectbox("Primary Deck (Baseline)", valid_meta_decks, index=0)
        deck_b = c2.selectbox("Comparison Deck (Challenger)", valid_meta_decks, index=1 if len(valid_meta_decks)>1 else 0)

        if deck_a and deck_b:
            da_idx, db_idx = deck_to_idx[deck_a], deck_to_idx[deck_b]
            da_metrics, db_metrics = res["metrics_per_deck"][deck_a], res["metrics_per_deck"][deck_b]
            da_mc, db_mc = mc_res.get(deck_a, {}), mc_res.get(deck_b, {})

            metrics_cols = st.columns(5)
            metrics_cols[0].metric("Ultimate Score", f"{da_metrics[score_key]:.2f}", f"{da_metrics[score_key] - db_metrics[score_key]:.2f} vs {deck_b}")
            metrics_cols[1].metric("Exp. Win Rate", f"{da_metrics['expected_win_rate']:.2%}", f"{da_metrics['expected_win_rate'] - db_metrics['expected_win_rate']:.2%} vs {deck_b}")
            
            if d2_rounds > 0:
                metrics_cols[2].metric("Day 2 Odds", f"{da_mc.get('day2_conversion',0):.2%}", f"{da_mc.get('day2_conversion',0) - db_mc.get('day2_conversion',0):.2%} vs {deck_b}")
            if top_cut > 0:
                metrics_cols[3].metric("Top 8 Odds", f"{da_mc.get('top_cut_conversion',0):.2%}", f"{da_mc.get('top_cut_conversion',0) - db_mc.get('top_cut_conversion',0):.2%} vs {deck_b}")
                metrics_cols[4].metric("Win Odds", f"{da_mc.get('win_probability',0):.2%}", f"{da_mc.get('win_probability',0) - db_mc.get('win_probability',0):.2%} vs {deck_b}")

            st.markdown("#### Matchups vs Top Metagame")
            
            top_field = [d for d in data if res["metrics_per_deck"][d["Deck"]]["meta_share"] >= 0.03] 
            comp_data = []
            for field_deck in top_field:
                opp_name = field_deck["Deck"]
                opp_idx = deck_to_idx[opp_name]
                wr_a = full_win_matrix[da_idx, opp_idx] * 100
                wr_b = full_win_matrix[db_idx, opp_idx] * 100
                comp_data.append({
                    "Opponent": opp_name,
                    "Field Share": res["metrics_per_deck"][opp_name]["meta_share"], # Keep as float for styling, format in config
                    f"{deck_a} WR": wr_a,
                    f"{deck_b} WR": wr_b,
                    "Advantage": f"{deck_a}" if wr_a > wr_b else f"{deck_b}" if wr_b > wr_a else "Tie"
                })
            
            comp_df = pd.DataFrame(comp_data)
            
            # --- PANDAS STYLER FOR COLOR CODING ---
            def highlight_winrates(val):
                if isinstance(val, (int, float)):
                    if val >= 55.0:
                        return 'background-color: rgba(46, 204, 113, 0.2)' # Soft Green
                    elif val <= 45.0:
                        return 'background-color: rgba(231, 76, 60, 0.2)' # Soft Red
                return ''

            styled_comp_df = comp_df.style.map(highlight_winrates, subset=[f"{deck_a} WR", f"{deck_b} WR"])

            st.dataframe(
                styled_comp_df, 
                width="stretch", 
                hide_index=True, 
                column_config={
                    "Field Share": st.column_config.NumberColumn(format="%.2f %%"),
                    f"{deck_a} WR": st.column_config.NumberColumn(format="%.2f %%"),
                    f"{deck_b} WR": st.column_config.NumberColumn(format="%.2f %%")
                }
            )

        st.divider()

        # Generate Dynamic Recommendations based on active view
        recommendations = [{"deck": d, **res["metrics_per_deck"][d]} for d in all_decks_sorted]
        avoids = [{"deck": d, **res["metrics_per_deck"][d]} for d in all_decks_sorted[::-1]]

        best_day2 = max(active_decks, key=lambda d: mc_res.get(d, {}).get("day2_conversion", 0)) if d2_rounds > 0 else None
        best_top8 = max(active_decks, key=lambda d: mc_res.get(d, {}).get("top_cut_conversion", 0)) if top_cut > 0 else None
        best_win = max(active_decks, key=lambda d: mc_res.get(d, {}).get("win_probability", 0)) if top_cut > 0 else None
        best_wr = max(active_decks, key=lambda d: res["metrics_per_deck"][d]["expected_win_rate"])

        # --- Recommendations & Avoids ---
        col_rec, col_avoid = st.columns(2)
        with col_rec:
            st.markdown(f"### 🔥 Top Recommendations ({odds_view})")
            visible_recs = recommendations[:st.session_state.rec_limit]
            
            for i, r in enumerate(visible_recs, 1):
                deck = r["deck"]
                
                tags = []
                if deck == best_win: tags.append("🏆 Best to Win Event")
                elif deck == best_top8: tags.append("🏅 Best for Top 8")
                elif deck == best_day2: tags.append("🛡️ Safest Day 2")
                if deck == best_wr: tags.append("📈 Highest Raw Win Rate")
                
                with st.container(border=True):
                    st.markdown(f"#### {i}. {deck}")
                    if tags:
                        st.markdown(" ".join([f"`{t}`" for t in tags]))
                        
                    if deck in deck_to_idx:
                        idx = deck_to_idx[deck]
                        favorable, threats = [], []
                        for opp_name, opp_share in res["full_meta"].items():
                            if opp_name == deck or opp_share < 0.02: 
                                continue
                            wr = full_win_matrix[idx, deck_to_idx[opp_name]]
                            if wr >= 0.55: favorable.append((opp_name, wr))
                            elif wr <= 0.45: threats.append((opp_name, wr))

                        if favorable: 
                            st.success(f"✅ Favored vs: {', '.join([f'{d} ({wr:.0%})' for d, wr in favorable])}")
                        if threats: 
                            st.error(f"⚠️ Weak vs: {', '.join([f'{d} ({wr:.0%})' for d, wr in threats])}")

            if st.session_state.rec_limit < len(recommendations):
                st.button("⬇️ Load more...", key="btn_load_recs", on_click=load_more_recs, width="stretch")

        with col_avoid:
            st.markdown(f"### 🚫 Decks to Avoid ({odds_view})")
            visible_avoids = avoids[:st.session_state.avoid_limit]
            
            for i, r in enumerate(visible_avoids, 1):
                deck = r["deck"]
                with st.container(border=True):
                    st.markdown(f"#### {i}. {deck}")
                    if deck in deck_to_idx:
                        idx = deck_to_idx[deck]
                        favorable, threats = [], []
                        for opp_name, opp_share in res["full_meta"].items():
                            if opp_name == deck or opp_share < 0.02: 
                                continue
                            wr = full_win_matrix[idx, deck_to_idx[opp_name]]
                            if wr >= 0.55: favorable.append((opp_name, wr))
                            elif wr <= 0.45: threats.append((opp_name, wr))

                        if threats: 
                            st.error(f"⚠️ Weak vs: {', '.join([f'{d} ({wr:.0%})' for d, wr in threats])}")
                        if favorable: 
                            st.success(f"✅ Favored vs: {', '.join([f'{d} ({wr:.0%})' for d, wr in favorable])}")

            if st.session_state.avoid_limit < len(avoids):
                st.button("⬇️ Load more...", key="btn_load_avoids", on_click=load_more_avoids, width="stretch")

if __name__ == "__main__":
    main()