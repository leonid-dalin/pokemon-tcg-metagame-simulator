# app.py | Streamlit dashboard
import streamlit as st
import os
import sys
import uuid
import json
import re
import numpy as np
import pandas as pd
from typing import List, cast, Tuple, Dict, Any
from pathlib import Path

# Resolve the project root
project_root = str(Path(__file__).resolve().parents[2])

# Inject it into the system path so Python can find the 'src' module
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data import load_matchup_data
from src.core.config import INPUT_DIR, MIN_GAMES, aggressive_colorscale, tier_thresholds
from src.tournament.monte_carlo import run_monte_carlo_analytics
from src.tournament.solver import predict_best_decks, UserMetaSpec, MatchFormat, swiss_rounds_from_players, get_variant_5_structure

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

def get_tier(expected_wr: float) -> str:
    """
    Tier assignments based on rigorous TCG win-rate thresholds.
    S-Tier: > 52% | A-Tier: > 50% | B-Tier: > 47% | C-Tier: < 47%
    """
    assigned = False
    for tier, threshold in tier_thresholds.items():
        if expected_wr >= threshold: 
            return tier
            break
    if not assigned:
        return "D"

# --- Main Application ---
def main():
    st.set_page_config(page_title="TCG Metagame Solver", page_icon="🏆", layout="wide")
    init_session_state()
    
    st.title("🏆 TCG Metagame Solver")
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
            
            use_drop_feature = st.toggle("Enable X-3 Drop Logic", value=False, help="Simulates players dropping from the tournament after accumulating 3 losses.")

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
                    progress_callback=mc_progress_callback, use_tie_convergence=use_tie_convergence, global_tie_rate=global_tie_rate,
                    use_drop_feature=use_drop_feature
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

        st.info("""
        **How to read the Scores:**
        * **Power Score (0-100):** A deck's normalized win rate against the field. `100` represents the highest win rate in the format. `0` represents the baseline floor of viability.
        * **Meta Score (0-100):** The average of a deck's Power Score and its Popularity (Frequency Score). It measures a deck's true dominance and distance to the "theoretical best deck."
        """)
        
        view_col1, view_col2 = st.columns([1, 3])
        with view_col1:
            odds_view = st.radio(
                "Bracket Perspective", 
                ["Player (Micro)", "Archetype (Macro)"], 
                index=0, 
                help="Switch between individual player odds (If I play this deck...) and macro performance (% of the bracket dominating)."
            )
        
        sort_key = "base_meta_score"
        
        active_decks = [d for d in full_deck_names if res["metrics_per_deck"][d]["meta_share"] >= 0.001]
        all_decks_sorted = sorted(active_decks, key=lambda d: res["metrics_per_deck"][d][sort_key], reverse=True)
        
        # Calculate Day 2 Expected Win Rates dynamically
        day2_share_vec = np.zeros(len(full_deck_names))
        for i, d_name in enumerate(full_deck_names):
            day2_share_vec[i] = mc_res.get(d_name, {}).get("day2_share", 0)
            
        if np.sum(day2_share_vec) > 0:
            day2_share_vec /= np.sum(day2_share_vec)

        day2_expected_wrs = full_win_matrix @ day2_share_vec 
        day2_wr_dict = {d: day2_expected_wrs[deck_to_idx[d]] for d in active_decks}

        data = []
        for i, deck in enumerate(all_decks_sorted, 1):
            metrics = res["metrics_per_deck"][deck]
            mc_metrics = mc_res.get(deck, {"day2_conversion": 0, "top_cut_conversion": 0, "win_probability": 0, "day2_share": 0, "top_cut_share": 0})
            
            row_data = {
                "#": i,
                "Tier": get_tier(metrics["expected_win_rate"]),
                "Deck": deck,
                "Meta Score": metrics["base_meta_score"],
                "Power Score": metrics["power_score"],
                "Type": "🔒 User" if deck in user_meta else "📈 Base",
                "Exp. WR %": metrics["expected_win_rate"] * 100,
            }
            
            if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
                row_data["Day 2 Exp. WR %"] = day2_wr_dict[deck] * 100
            
            meta_share = metrics["meta_share"] * 100
            
            if odds_view == "Player (Micro)":
                row_data["Day 1 Share %"] = meta_share
                if d2_rounds > 0: row_data["Day 2 Conv. Rate %"] = mc_metrics.get("day2_conversion", 0) * 100
                if top_cut > 0:
                    row_data["Top 8 Conv. Rate %"] = mc_metrics.get("top_cut_conversion", 0) * 100
                    row_data["Win Event %"] = mc_metrics.get("win_probability", 0) * 100
            else:
                row_data["Day 1 Share %"] = meta_share
                if d2_rounds > 0: row_data["Day 2 Meta Share %"] = mc_metrics.get("day2_share", 0) * 100
                if top_cut > 0:
                    row_data["Top 8 Meta Share %"] = mc_metrics.get("top_cut_share", 0) * 100
                    row_data["Winner Share %"] = mc_metrics.get("win_probability", 0) * meta_share * players
                
            data.append(row_data)

        df = pd.DataFrame(data)

        col_config = {
            "#": st.column_config.NumberColumn(width="small"),
            "Tier": st.column_config.TextColumn(width="small", help="S-Tier (≥52% WR), A-Tier (≥50% WR), B-Tier (≥47% WR), C-Tier (≥44% WR)."),
            "Deck": st.column_config.TextColumn(width="medium"),
            "Meta Score": st.column_config.NumberColumn(format="%.1f", help="Average of Power and Popularity. High scores mean heavily dominant."),
            "Power Score": st.column_config.NumberColumn(format="%.1f", help="0-100 normalization of Win Rate. 100 is the best performing deck."),
            "Type": st.column_config.TextColumn(help="Forced by user constraint (🔒) or simulated baseline (📈)."),
            "Exp. WR %": st.column_config.NumberColumn(format="%.2f %%", help="Expected Win Rate against the entire predicted field."),
        }
        
        if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
            col_config["Day 2 Exp. WR %"] = st.column_config.NumberColumn(format="%.2f %%", help="Expected Win Rate against the condensed Day 2 meta.")

        if odds_view == "Player (Micro)":
            col_config["Day 1 Share %"] = st.column_config.NumberColumn(format="%.2f %%")
            if d2_rounds > 0: col_config["Day 2 Conv. Rate %"] = st.column_config.NumberColumn(format="%.2f %%")
            if top_cut > 0:
                col_config["Top 8 Conv. Rate %"] = st.column_config.NumberColumn(format="%.2f %%")
                col_config["Win Event %"] = st.column_config.NumberColumn(format="%.2f %%")
        else:
            col_config["Day 1 Share %"] = st.column_config.NumberColumn(format="%.2f %%")
            if d2_rounds > 0: col_config["Day 2 Meta Share %"] = st.column_config.NumberColumn(format="%.2f %%")
            if top_cut > 0:
                col_config["Top 8 Meta Share %"] = st.column_config.NumberColumn(format="%.2f %%")
                col_config["Winner Share %"] = st.column_config.NumberColumn(format="%.2f %%")

        st.caption("Click any column header to sort. Hover over headers for detailed metric definitions.")
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

            m_row1 = st.columns(4)
            m_row1[0].metric("Meta Score", f"{da_metrics['base_meta_score']:.1f}", f"{da_metrics['base_meta_score'] - db_metrics['base_meta_score']:.1f} vs {deck_b}")
            m_row1[1].metric("Power Score", f"{da_metrics['power_score']:.1f}", f"{da_metrics['power_score'] - db_metrics['power_score']:.1f} vs {deck_b}")
            m_row1[2].metric("Exp. Win Rate", f"{da_metrics['expected_win_rate']:.2%}", f"{da_metrics['expected_win_rate'] - db_metrics['expected_win_rate']:.2%} vs {deck_b}")
            if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
                m_row1[3].metric("Day 2 Exp. WR", f"{day2_wr_dict.get(deck_a, 0):.2%}", f"{day2_wr_dict.get(deck_a, 0) - day2_wr_dict.get(deck_b, 0):.2%} vs {deck_b}")

            m_row2 = st.columns(3)
            if d2_rounds > 0:
                m_row2[0].metric("Day 2 Odds", f"{da_mc.get('day2_conversion',0):.2%}", f"{da_mc.get('day2_conversion',0) - db_mc.get('day2_conversion',0):.2%} vs {deck_b}")
            if top_cut > 0:
                m_row2[1].metric("Top 8 Odds", f"{da_mc.get('top_cut_conversion',0):.2%}", f"{da_mc.get('top_cut_conversion',0) - db_mc.get('top_cut_conversion',0):.2%} vs {deck_b}")
                m_row2[2].metric("Win Odds", f"{da_mc.get('win_probability',0):.2%}", f"{da_mc.get('win_probability',0) - db_mc.get('win_probability',0):.2%} vs {deck_b}")

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
                    "Field Share": res["metrics_per_deck"][opp_name]["meta_share"],
                    f"{deck_a} WR": wr_a,
                    f"{deck_b} WR": wr_b,
                    "Advantage": f"{deck_a}" if wr_a > wr_b else f"{deck_b}" if wr_b > wr_a else "Tie"
                })
            
            comp_df = pd.DataFrame(comp_data)
            
            def highlight_winrates(val):
                if not isinstance(val, (int, float)): return ''
                
                norm_val = val / 100.0
                
                target_rgb = aggressive_colorscale[0][1]
                for threshold, color in aggressive_colorscale:
                    if norm_val >= threshold:
                        target_rgb = color
                    else:
                        break
                        
                nums = re.findall(r'\d+', target_rgb)
                return f'background-color: rgba({nums[0]}, {nums[1]}, {nums[2]}, 0.25)'

            styled_comp_df = comp_df.style.map(highlight_winrates, subset=[f"{deck_a} WR", f"{deck_b} WR"])

            st.dataframe(
                styled_comp_df, width="stretch", hide_index=True, 
                column_config={
                    "Field Share": st.column_config.NumberColumn(format="%.2f %%"),
                    f"{deck_a} WR": st.column_config.NumberColumn(format="%.2f %%"),
                    f"{deck_b} WR": st.column_config.NumberColumn(format="%.2f %%")
                }
            )

        st.divider()

        # --- Dynamic Actionable Intelligence ---
        # 1. Top Threats are defined strictly by format dominance (Meta Score)
        top_threats = sorted(active_decks, key=lambda d: res["metrics_per_deck"][d]["base_meta_score"], reverse=True)[:3]
        
        # 2. Recommendations are defined strictly by EV / pilot profitability (Power Score)
        recs_sorted_by_power = sorted(active_decks, key=lambda d: res["metrics_per_deck"][d]["power_score"], reverse=True)
        recommendations = [{"deck": d, **res["metrics_per_deck"][d]} for d in recs_sorted_by_power]
        avoids = [{"deck": d, **res["metrics_per_deck"][d]} for d in recs_sorted_by_power[::-1]]

        best_day2 = max(active_decks, key=lambda d: mc_res.get(d, {}).get("day2_conversion", 0)) if d2_rounds > 0 else None
        best_top8 = max(active_decks, key=lambda d: mc_res.get(d, {}).get("top_cut_conversion", 0)) if top_cut > 0 else None
        best_win = max(active_decks, key=lambda d: mc_res.get(d, {}).get("win_probability", 0)) if top_cut > 0 else None
        best_wr = recs_sorted_by_power[0]
        best_day2_predator = max(day2_wr_dict, key=day2_wr_dict.get) if d2_rounds > 0 and sum(day2_share_vec) > 0 else None

        # --- Tabbed Intelligence Dashboard ---
        tab_rec, tab_threat, tab_avoid = st.tabs(["🔥 Top Recommendations", "🚨 Top Threats", "🚫 Decks to Avoid"])
        
        with tab_threat:
            st.markdown("### 🚨 The Meta Dictators")
            st.caption("These decks have the highest Meta Scores. They combine high win rates with massive popularity and warp the tournament around them. You MUST respect them.")
            for i, deck in enumerate(top_threats, 1):
                metrics = res["metrics_per_deck"][deck]
                with st.container(border=True):
                    c_title, c_stats = st.columns([2, 1])
                    c_title.markdown(f"#### {i}. {deck}")
                    c_stats.markdown(f"**Meta Score:** `{metrics['base_meta_score']:.1f}` | **Day 1 Share:** `{metrics['meta_share']:.1%}`")
                    
                    idx = deck_to_idx[deck]
                    favorable, threats = [], []
                    for opp_name in active_decks:
                        if opp_name == deck: continue
                        wr = full_win_matrix[idx, deck_to_idx[opp_name]]
                        if wr >= 0.55: favorable.append(opp_name)
                        elif wr <= 0.45: threats.append(opp_name)
                    
                    st.error(f"**This deck pushes these out of the meta:** {', '.join(favorable[:5])}...")
                    st.success(f"**It is heavily countered by:** {', '.join(threats[:5])}...")

        with tab_rec:
            st.markdown("### 🔥 Best EV (Expected Value)")
            st.caption("These decks are ranked strictly by Power Score. They mathematically yield the highest win rates against the predicted field.")
            visible_recs = recommendations[:st.session_state.rec_limit]
            
            for i, r in enumerate(visible_recs, 1):
                deck = r["deck"]
                metrics = res["metrics_per_deck"][deck]
                
                tags = []
                if deck == best_win: tags.append("🏆 Best to Win Event")
                elif deck == best_top8: tags.append("🏅 Best for Top 8")
                elif deck == best_day2: tags.append("🛡️ Safest Day 2")
                if deck == best_wr: tags.append("📈 Highest Raw Win Rate")
                if deck == best_day2_predator: tags.append("🧱 Day 2 Predator")
                if metrics["meta_share"] >= 0.10 and metrics["expected_win_rate"] < 0.49: tags.append("🪤 Overplayed Trap")
                if deck not in top_threats and full_win_matrix[deck_to_idx[deck], deck_to_idx[top_threats[0]]] > 0.55:
                    tags.append(f"💡 Rogue Meta Breaker (Beats {top_threats[0]})")
                        
                with st.container(border=True):
                    st.markdown(f"#### {i}. {deck}")
                    if tags:
                        st.markdown(" ".join([f"`{t}`" for t in tags]))
                    
                    st.markdown(f"**Power Score:** `{metrics['power_score']:.1f}` | **Exp. WR:** `{metrics['expected_win_rate']:.2%}`")
                    
                    # --- Explicitly cross-reference against the Top Threats ---
                    st.markdown("##### Performance vs Top Threats:")
                    threat_cols = st.columns(len(top_threats))
                    for t_idx, threat_deck in enumerate(top_threats):
                        wr_vs_threat = full_win_matrix[deck_to_idx[deck], deck_to_idx[threat_deck]]
                        color = "🟢" if wr_vs_threat >= 0.55 else "🔴" if wr_vs_threat <= 0.45 else "🟡"
                        threat_cols[t_idx].markdown(f"{color} **{threat_deck}**: `{wr_vs_threat:.0%}`")

            if st.session_state.rec_limit < len(recommendations):
                st.button("⬇️ Load more...", key="btn_load_recs", on_click=load_more_recs, width="stretch")

        with tab_avoid:
            st.markdown("### 🚫 Negative Expected Value")
            st.caption("These decks are ranked by the lowest Power Scores. They are mathematically unfavored against the predicted field.")
            visible_avoids = avoids[:st.session_state.avoid_limit]
            
            for i, r in enumerate(visible_avoids, 1):
                deck = r["deck"]
                metrics = res["metrics_per_deck"][deck]
                with st.container(border=True):
                    st.markdown(f"#### {i}. {deck}")
                    st.markdown(f"**Power Score:** `{metrics['power_score']:.1f}` | **Exp. WR:** `{metrics['expected_win_rate']:.2%}`")
                    
                    idx = deck_to_idx[deck]
                    threats = []
                    for opp_name, opp_share in res["full_meta"].items():
                        if opp_name == deck or opp_share < 0.02: continue
                        if full_win_matrix[idx, deck_to_idx[opp_name]] <= 0.45: 
                            threats.append(opp_name)

                    if threats: 
                        st.error(f"⚠️ Farmed by: {', '.join(threats[:5])}...")

            if st.session_state.avoid_limit < len(avoids):
                st.button("⬇️ Load more...", key="btn_load_avoids", on_click=load_more_avoids, width="stretch")

if __name__ == "__main__":
    main()