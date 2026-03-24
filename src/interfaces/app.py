# app.py | Streamlit dashboard
import streamlit as st
import os
import sys
import uuid
import json
import re
import numpy as np
import pandas as pd
from typing import List, cast, Tuple, Dict
from pathlib import Path

# Resolve the project root
project_root = str(Path(__file__).resolve().parents[2])

# Inject it into the system path so Python can find the 'src' module
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data import load_matchup_data
from src.core.config import INPUT_DIR, MIN_GAMES, WIN_THRESHOLD, aggressive_colorscale, TIER_THRESHOLDS, TIER_1_THRESHOLD
from src.tournament.monte_carlo import run_monte_carlo_analytics
from src.tournament.solver import predict_best_decks, UserMetaSpec, MatchFormat, swiss_rounds_from_players, get_variant_5_structure
from src.evolution.plotting import plot_metagame_scatter, plot_head_to_head_radar

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
    T0 (≥52.5%), T0.5 (≥50%), T1 (≥47.5%), T2 (≥45%), T3 (≥42.5%), T4 (≥0%).
    """
    for tier, threshold in TIER_THRESHOLDS.items():
        if expected_wr >= threshold: 
            return tier
    return "T4" 

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
                st.caption("**Championship Series Details**")
                st.markdown(f"- **Day 1**: {d1_rounds} Rounds\n- **Day 2 Cut**: {cut_points} Match Pts\n- **Day 2**: {d2_rounds} Rounds\n- **Playoffs**: Top {top_cut}")
            else:
                d1_rounds = swiss_rounds_from_players(players)
                cut_points, d2_rounds, top_cut = 999, 0, (8 if players >= 8 else 0)
                st.caption("**Pure Swiss Details**")
                st.markdown(f"- **Swiss**: {d1_rounds} Rounds")
                if top_cut > 0: st.markdown(f"- **Playoffs**: Top {top_cut}")

        st.header("⚙️ Engine Parameters")
        with st.container(border=True):
            mc_iterations = st.selectbox("Monte Carlo Iterations", options=[1000, 10_000, 100_000, 1_000_000], format_func=lambda x: f"{x:,}", index=1)
            match_format = st.radio("Match Format", ["BO1", "BO3"], index=1)
            
            min_sample_threshold = st.slider("Matchup Minimum Games", min_value=1, max_value=100, value=10, step=1)
            input_mode = st.radio("Constraint Mode", ["Percentage", "Raw Players"])
            is_raw = input_mode == "Raw Players"

            st.divider()
            st.markdown("**🧪 BETA Features**")
            use_tie_convergence = st.toggle("Enable Win-Rate Tie Convergence", value=True, help="Mathematically simulates real-world match timeouts using a parabolic curve based on matchup closeness.")
            global_tie_rate = st.slider("Global Tie Rate (%)", min_value=0.0, max_value=30.0, value=15.0, step=0.5, disabled=not use_tie_convergence) / 100.0
            
            use_drop_feature = st.toggle("Enable X-3 Drop Logic", value=False, help="Simulates players dropping from the tournament after accumulating 3 losses.")
            allow_negative_power = st.toggle("Show Negative Power Scores", value=False, help="When disabled, deeply unviable decks (Power Score < 0) are completely hidden from the Metagame Scatter Plot to keep the view clean. Toggle this ON to view them at their true negative depths.")

    # ==========================================
    # MAIN UI: META CONSTRAINTS & IMPORT
    # ==========================================
    st.subheader("📊 Metagame Constraints")
    
    if "import_msg" in st.session_state:
        st.success(st.session_state.import_msg)
        del st.session_state.import_msg
    if "import_warn" in st.session_state:
        st.warning(st.session_state.import_warn)
        del st.session_state.import_warn

    with st.expander("📁 Import Limitless Labs HTML", expanded=True):
        uploaded_file = st.file_uploader("Upload a saved HTML file from Limitless Labs...", type=["html", "htm"])
        if uploaded_file is not None:
            if st.button("Extract Data & Populate Constraints", type="secondary"):
                try:
                    parsed_meta, total_import_players, wildcard_players = parse_limitless_html(uploaded_file.getvalue().decode("utf-8"), get_valid_deck_names())
                    st.session_state.imported_players = total_import_players
                    clear_all_rows()
                    for deck_name, count in parsed_meta.items():
                        add_meta_row(deck=deck_name, spec_type="Exact", val=(count / total_import_players) * 100.0 if total_import_players > 0 else 0.0, is_locked=True)
                    st.session_state.import_msg = f"✅ Successfully imported {len(parsed_meta)} recognized decks ({total_import_players} total players)."
                    if wildcard_players > 0: st.warning(f"⚠️ {wildcard_players} players were using unrecognized/rogue decks. Handled via rescaling.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error parsing HTML file: {str(e)}")
                    st.exception(e)
                    st.stop()

    user_meta: UserMetaSpec = {}
    total_min = 0.0
    deck_names = get_valid_deck_names()
    seen_decks = set()

    with st.expander("🛠️ Active Custom Constraints", expanded=False):
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
                                val_slider = st.slider("Range (Players)", 0, int(players), (0, min(15, int(players))), key=f"slider_{row_id}")
                                min_prop, max_prop = val_slider[0] / players if players > 0 else 0.0, val_slider[1] / players if players > 0 else 0.0
                            else:
                                val_slider = st.slider("Range (%)", 0.0, 100.0, (0.0, 15.0), step=0.1, format="%.1f%%",
                                                       key=f"slider_{row_id}")
                                min_prop, max_prop = val_slider[0] / 100.0, val_slider[1] / 100.0

                            user_meta[deck] = {"min": min_prop, "max": max_prop}; total_min += min_prop
                    with cols[3]:
                        st.button("🗑️", key=f"del_{row_id}", on_click=delete_meta_row, args=(row_id,))

    if total_min > 1.0: st.error(f"❌ Minimum total across all constraints ({total_min:.1%}) exceeds 100%. Please adjust your values before predicting."); st.stop()

    st.divider()

    # ==========================================
    # EXECUTION
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

    # ==========================================
    # DATAFRAME RESULTS
    # ==========================================
    if st.session_state.prediction_result is not None and st.session_state.mc_result is not None:
        res = st.session_state.prediction_result
        mc_res = st.session_state.mc_result
        
        full_deck_names, full_win_matrix = load_full_win_matrix()
        deck_to_idx = {name: i for i, name in enumerate(full_deck_names)}

        st.subheader("📊 Dashboard & Tournament Odds")

        st.info("""
        **How to read the Scores:**
        * **Power Score (≤100):** A deck's normalized win rate against the field. `100` represents the highest win rate. Scores can be negative if a deck performs drastically below the baseline floor of viability.
        * **Frequency Score (0-100):** A deck's normalized popularity.
        * **Meta Score (≤100):** The average of Power Score and Frequency Score. It measures a deck's true dominance and distance to the "theoretical best deck."
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
            
            meta_share = metrics["meta_share"]
            
            row_data = {
                "#": str(i),
                "Deck": deck,
                "Type": "🔒 User" if deck in user_meta else "📈 Base",
                "Tier": get_tier(metrics["expected_win_rate"]),
                "Power Score": round(metrics["power_score"], 2),
                "Freq Score": round(metrics["frequency_score"], 2),
                "Meta Score": round(metrics["base_meta_score"], 2),
                "Power Ranking (Day 1)": round(metrics["expected_win_rate"] * 100, 2),
                "Share % (Day 1)": round(meta_share * 100, 2),
            }

            if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
                row_data["Power Ranking (Day 2)"] = round(float(day2_wr_dict[deck]) * 100, 2)
            
            # --- Micro/Macro Odds View ---
            if odds_view == "Player (Micro)":
                if d2_rounds > 0: row_data["Conv. Rate % (Day 2)"] = round(mc_metrics.get("day2_conversion", 0) * 100, 2)
                if top_cut > 0:
                    row_data[f"Conv. Rate % (Top {top_cut})"] = round(mc_metrics.get("top_cut_conversion", 0) * 100, 2)
                    row_data["Win Event %"] = round(mc_metrics.get("win_probability", 0) * 100, 2)
            else:
                if d2_rounds > 0: row_data["Share % (Day 2)"] = round(mc_metrics.get("day2_share", 0) * 100, 2)
                if top_cut > 0:
                    row_data[f"Share % (Top {top_cut})"] = round(mc_metrics.get("top_cut_share", 0) * 100, 2)
                    row_data["Share % (Winner)"] = round(mc_metrics.get("win_probability", 0) * meta_share * players * 100, 2)
            
            data.append(row_data)

        df = pd.DataFrame(data)

        base_cols = ["#", "Deck", "Type", "Tier", "Power Score", "Freq Score", "Meta Score", "Power Ranking (Day 1)"]
        if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
            base_cols.append("Power Ranking (Day 2)")
        
        mc_cols = ["Share % (Day 1)"]
        if odds_view == "Player (Micro)":
            if d2_rounds > 0: mc_cols.append("Conv. Rate % (Day 2)")
            if top_cut > 0: mc_cols.extend([f"Conv. Rate % (Top {top_cut})", "Win Event %"])
        else:
            if d2_rounds > 0: mc_cols.append("Share % (Day 2)")
            if top_cut > 0: mc_cols.extend([f"Share % (Top {top_cut})", "Share % (Winner)"])
                
        final_column_order = base_cols + mc_cols

        col_config = {
            "#": st.column_config.TextColumn(width="small"),
            "Deck": st.column_config.TextColumn(width="medium"),
            "Type": st.column_config.TextColumn(width="small", help="Forced by user constraint (🔒) or simulated baseline (📈)."),
            "Tier": st.column_config.TextColumn(width="small", help="T0 (≥52.5%), T0.5 (≥50%), T1 (≥47.5%), T2 (≥45%), T3 (≥42.5%), T4 (≥0%)."), 
            "Power Score": st.column_config.NumberColumn(width="small", format="%.2f", help="Normalization of Win Rate. 100 is the best performing deck. Can be negative."),
            "Freq Score": st.column_config.NumberColumn(width="small", format="%.2f", help="Normalization of Popularity. 100 is the most played deck."),
            "Meta Score": st.column_config.NumberColumn(width="small", format="%.2f", help="Average of Power and Frequency. High scores mean heavily dominant."),
            "Power Ranking (Day 1)": st.column_config.NumberColumn(format="%.2f %%", help="Weighted Win Rate against the day 1 field."),
            "Power Ranking (Day 2)": st.column_config.NumberColumn(format="%.2f %%", help="Expected Win Rate against the condensed Day 2 meta."),
        }
        
        for c in mc_cols:
            col_config[c] = st.column_config.NumberColumn(format="%.2f %%")

        st.caption("Click any column header to sort. Hover over headers for detailed metric definitions.")
        st.dataframe(df[final_column_order], width="stretch", hide_index=True, column_config=col_config)
        st.divider()

        # --- Metagame Scatter Plot ---
        st.subheader("🌌 Metagame Scatter Plot")
        st.markdown("Visualizing the format. The ideal 'Best Deck' pulls towards the top right (100, 100)")
        fig_scatter = plot_metagame_scatter(df, allow_negative_power)
        st.plotly_chart(fig_scatter, width="stretch")
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
            m_row1[0].metric("Meta Score", f"{da_metrics['base_meta_score']:.2f}", f"{da_metrics['base_meta_score'] - db_metrics['base_meta_score']:.2f} vs {deck_b}")
            m_row1[1].metric("Power Score", f"{da_metrics['power_score']:.2f}", f"{da_metrics['power_score'] - db_metrics['power_score']:.2f} vs {deck_b}")
            m_row1[2].metric("Power Ranking (Day 1)", f"{da_metrics['expected_win_rate']:.2%}", f"{da_metrics['expected_win_rate'] - db_metrics['expected_win_rate']:.2%} vs {deck_b}")
            if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
                m_row1[3].metric("Power Ranking (Day 2)", f"{day2_wr_dict.get(deck_a, 0):.2%}", f"{day2_wr_dict.get(deck_a, 0) - day2_wr_dict.get(deck_b, 0):.2%} vs {deck_b}")

            m_row2 = st.columns(3)
            if d2_rounds > 0:
                m_row2[0].metric("Day 2 Odds", f"{da_mc.get('day2_conversion',0):.2%}", f"{da_mc.get('day2_conversion',0) - db_mc.get('day2_conversion',0):.2%} vs {deck_b}")
            if top_cut > 0:
                m_row2[1].metric(f"Top {top_cut} Odds", f"{da_mc.get('top_cut_conversion',0):.2%}", f"{da_mc.get('top_cut_conversion',0) - db_mc.get('top_cut_conversion',0):.2%} vs {deck_b}")
                m_row2[2].metric("Win Odds", f"{da_mc.get('win_probability',0):.2%}", f"{da_mc.get('win_probability',0) - db_mc.get('win_probability',0):.2%} vs {deck_b}")
            
            # --- Interactive Spider/Radar Chart ---
            st.markdown("#### 🕸️ Head-to-Head Stat Radar")

            global_max_meta = float(max(m["base_meta_score"] for m in res["metrics_per_deck"].values()))
            global_max_power = float(max(m["power_score"] for m in res["metrics_per_deck"].values()))
            global_max_pr1 = float(max(m["expected_win_rate"] for m in res["metrics_per_deck"].values())) * 100.0

            def safe_norm(v, max_val):
                return max(0.0, (v / max_val * 100.0)) if max_val > 0 else 0.0

            categories = ['Meta Score', 'Power Score', 'Power Ranking (D1)']
            
            da_vals_norm = [
                safe_norm(da_metrics['base_meta_score'], global_max_meta),
                safe_norm(da_metrics['power_score'], global_max_power),
                safe_norm(da_metrics['expected_win_rate'] * 100, global_max_pr1)
            ]
            db_vals_norm = [
                safe_norm(db_metrics['base_meta_score'], global_max_meta),
                safe_norm(db_metrics['power_score'], global_max_power),
                safe_norm(db_metrics['expected_win_rate'] * 100, global_max_pr1)
            ]
            
            da_texts = [
                f"Meta Score: {da_metrics['base_meta_score']:.2f}",
                f"Power Score: {da_metrics['power_score']:.2f}",
                f"Power Rank D1: {da_metrics['expected_win_rate']:.2%}"
            ]
            db_texts = [
                f"Meta Score: {db_metrics['base_meta_score']:.2f}",
                f"Power Score: {db_metrics['power_score']:.2f}",
                f"Power Rank D1: {db_metrics['expected_win_rate']:.2%}"
            ]

            if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
                global_max_pr2 = float(max(float(v) for v in day2_wr_dict.values()) * 100.0) if day2_wr_dict else 1.0
                categories.append('Power Ranking (D2)')
                da_vals_norm.append(safe_norm(day2_wr_dict.get(deck_a, 0) * 100, global_max_pr2))
                db_vals_norm.append(safe_norm(day2_wr_dict.get(deck_b, 0) * 100, global_max_pr2))
                da_texts.append(f"Power Rank D2: {day2_wr_dict.get(deck_a, 0):.2%}")
                db_texts.append(f"Power Rank D2: {day2_wr_dict.get(deck_b, 0):.2%}")
                
            if d2_rounds > 0:
                global_max_d2 = float(max(m.get("day2_conversion", 0) for m in mc_res.values()) * 100.0)
                categories.append('Day 2 Odds')
                da_vals_norm.append(safe_norm(da_mc.get('day2_conversion', 0) * 100, global_max_d2))
                db_vals_norm.append(safe_norm(db_mc.get('day2_conversion', 0) * 100, global_max_d2))
                da_texts.append(f"D2 Odds: {da_mc.get('day2_conversion', 0):.2%}")
                db_texts.append(f"D2 Odds: {db_mc.get('day2_conversion', 0):.2%}")

            if top_cut > 0:
                global_max_t8 = float(max(m.get("top_cut_conversion", 0) for m in mc_res.values()) * 100.0)
                global_max_win = float(max(m.get("win_probability", 0) for m in mc_res.values()) * 100.0)
                categories.append(f'Top {top_cut} Odds')
                da_vals_norm.append(safe_norm(da_mc.get('top_cut_conversion', 0) * 100, global_max_t8))
                db_vals_norm.append(safe_norm(db_mc.get('top_cut_conversion', 0) * 100, global_max_t8))
                da_texts.append(f"Top {top_cut} Odds: {da_mc.get('top_cut_conversion', 0):.2%}")
                db_texts.append(f"Top {top_cut} Odds: {db_mc.get('top_cut_conversion', 0):.2%}")
                categories.append('Win Odds')
                da_vals_norm.append(safe_norm(da_mc.get('win_probability', 0) * 100, global_max_win))
                db_vals_norm.append(safe_norm(db_mc.get('win_probability', 0) * 100, global_max_win))
                da_texts.append(f"Win Odds: {da_mc.get('win_probability', 0):.2%}")
                db_texts.append(f"Win Odds: {db_mc.get('win_probability', 0):.2%}")

            fig_radar = plot_head_to_head_radar(deck_a, deck_b, categories, da_vals_norm, db_vals_norm, da_texts, db_texts)
            st.plotly_chart(fig_radar, width="stretch")

            st.markdown("#### Matchups")
            
            top_field = [d for d in data if res["metrics_per_deck"][d["Deck"]]["meta_share"] >= 0.03] 
            comp_data = []
            for field_deck in top_field:
                opp_name = field_deck["Deck"]
                opp_idx = deck_to_idx[opp_name]
                wr_a = float(full_win_matrix[da_idx, opp_idx] * 100)
                wr_b = float(full_win_matrix[db_idx, opp_idx] * 100)
                comp_data.append({
                    "Opponent": opp_name,
                    "Share % (Day 1)": round(float(res["metrics_per_deck"][opp_name]["meta_share"]) * 100, 2),
                    f"{deck_a} WR": wr_a,
                    f"{deck_b} WR": wr_b,
                    "Advantage": f"{deck_a}" if wr_a > wr_b else f"{deck_b}" if wr_b > wr_a else "Tie"
                })
            
            comp_df = pd.DataFrame(comp_data)

            def highlight_winrates(v):
                if not isinstance(v, (int, float)): return ''
                norm_val = v / 100.0
                target_rgb = aggressive_colorscale[0][1]
                for threshold, co in aggressive_colorscale:
                    if norm_val >= threshold:
                        target_rgb = co
                    else:
                        break
                nums = re.findall(r'\d+', target_rgb)
                return f'background-color: rgba({nums[0]}, {nums[1]}, {nums[2]}, 0.55); color: #ffffff; font-weight: bold;'

            styled_comp_df = comp_df.style.map(highlight_winrates, subset=[f"{deck_a} WR", f"{deck_b} WR"])

            st.dataframe(
                styled_comp_df, width="stretch", hide_index=True, 
                column_config={
                    "Share % (Day 1)": st.column_config.NumberColumn(width="small", format="%.2f %%"),
                    f"{deck_a} WR": st.column_config.NumberColumn(format="%.2f %%"),
                    f"{deck_b} WR": st.column_config.NumberColumn(format="%.2f %%")
                }
            )

        st.divider()

        # --- Dynamic Actionable Intelligence ---
        top_threats = sorted([d for d in active_decks if res["metrics_per_deck"][d]["base_meta_score"] > 50], 
                             key=lambda d: res["metrics_per_deck"][d]["base_meta_score"], reverse=True)

        if d2_rounds > 0:
            ev_key = "day2_conversion"
            ev_label = "Day 2 Odds"
        elif top_cut > 0:
            ev_key = "top_cut_conversion"
            ev_label = f"Top {top_cut} Odds"
        else:
            ev_key = "power_score"
            ev_label = "Power Score"

        recs_sorted_by_ev = sorted(
            active_decks, 
            key=lambda d: (
                mc_res.get(d, {}).get(ev_key, 0) if ev_key != "power_score" else res["metrics_per_deck"][d]["power_score"], 
                res["metrics_per_deck"][d]["power_score"]
            ), 
            reverse=True
        )
        recommendations = [{"deck": d, **res["metrics_per_deck"][d]} for d in recs_sorted_by_ev]
        avoids = [{"deck": d, **res["metrics_per_deck"][d]} for d in recs_sorted_by_ev[::-1]]

        best_day2 = max(active_decks, key=lambda d: mc_res.get(d, {}).get("day2_conversion", 0)) if d2_rounds > 0 else None
        best_top8 = max(active_decks, key=lambda d: mc_res.get(d, {}).get("top_cut_conversion", 0)) if top_cut > 0 else None
        best_win = max(active_decks, key=lambda d: mc_res.get(d, {}).get("win_probability", 0)) if top_cut > 0 else None
        best_wr = max(active_decks, key=lambda d: res["metrics_per_deck"][d]["power_score"])
        best_day2_predator = max(day2_wr_dict.keys(), key=lambda k: float(day2_wr_dict[k])) if d2_rounds > 0 and np.sum(
            day2_share_vec) > 0 else None

        best_ev_val = mc_res.get(recs_sorted_by_ev[0], {}).get(ev_key, 0) if ev_key != "power_score" else res["metrics_per_deck"][recs_sorted_by_ev[0]]["power_score"]

        tab_rec, tab_threat, tab_avoid = st.tabs(["🔥 Top Recommendations", "🚨 Top Threats", "🚫 Decks to Avoid"])
        
        with tab_rec:
            st.markdown("### 🔥 Best EV (Expected Value)")
            st.caption(f"These decks are ranked strictly by **{ev_label}**. They mathematically yield the highest returns in this specific bracket structure.")
            visible_recs = recommendations[:st.session_state.rec_limit]
            
            for i, r in enumerate(visible_recs, 1):
                deck = r["deck"]
                metrics = res["metrics_per_deck"][deck]
                mc_metrics = mc_res.get(deck, {})
                
                tags = []
                if deck == best_win: tags.append("🏆 Best to Win Event")
                elif deck == best_top8: tags.append(f"🏅 Best for Top {top_cut}")
                elif deck == best_day2: tags.append("🛡️ Safest Day 2")
                if deck == best_wr: tags.append("📈 Highest Raw Win Rate")
                if deck == best_day2_predator: tags.append("🧱 Day 2 Predator")
                if metrics["meta_share"] >= 0.10 and metrics["expected_win_rate"] < TIER_1_THRESHOLD: tags.append("🪤 Overplayed Trap")
                if top_threats and deck not in top_threats and full_win_matrix[deck_to_idx[deck], deck_to_idx[top_threats[0]]] > WIN_THRESHOLD:
                    tags.append(f"💡 Rogue Meta Breaker (Beats {top_threats[0]})")
                    
                with st.container(border=True):
                    st.markdown(f"#### {i}. {deck}")
                    if tags:
                        st.markdown(" ".join([f"`{t}`" for t in tags]))
                    
                    # Calculate EV delta vs #1 spot
                    deck_ev_val = mc_metrics.get(ev_key, 0) if ev_key != "power_score" else metrics["power_score"]
                    delta = deck_ev_val - best_ev_val
                    
                    if i == 1 or delta == 0:
                        if ev_key == "power_score":
                            ev_str = f"**{ev_label}:** `{deck_ev_val:.2f}`"
                        else:
                            ev_str = f"**Tournament EV ({ev_label}):** `{deck_ev_val:.2%}`"
                    else:
                        if ev_key == "power_score":
                            ev_str = f"**{ev_label}:** `{deck_ev_val:.2f}` ({delta:.2f} from #1)"
                        else:
                            ev_str = f"**Tournament EV ({ev_label}):** `{deck_ev_val:.2%}` ({delta:.2%} from #1)"

                    st.markdown(f"{ev_str} | **Power Ranking (Day 1):** `{metrics['expected_win_rate']:.2%}`")
                    
                    st.markdown("##### Performance vs Top Threats:")
                    if top_threats:
                        threat_cols = st.columns(len(top_threats))
                        for t_idx, threat_deck in enumerate(top_threats):
                            wr_vs_threat = full_win_matrix[deck_to_idx[deck], deck_to_idx[threat_deck]]
                            color = "🟢" if wr_vs_threat >= WIN_THRESHOLD else "🔴" if wr_vs_threat <= (1 - WIN_THRESHOLD) else "🟡"
                            threat_cols[t_idx].markdown(f"{color} **{threat_deck}**: `{wr_vs_threat:.0%}`")
                    else:
                        st.caption("No dominant meta juggernaut detected in this field.")

            if st.session_state.rec_limit < len(recommendations):
                st.button("⬇️ Load more...", key="btn_load_recs", on_click=load_more_recs, width="stretch")
                
        with tab_threat:
            st.markdown("### 🚨 The Meta Juggernauts")
            st.caption("These decks have the highest Meta Scores (>50). They combine high win rates with massive popularity.")
            for i, deck in enumerate(top_threats, 1):
                metrics = res["metrics_per_deck"][deck]
                with st.container(border=True):
                    c_title, c_stats = st.columns([2, 1])
                    c_title.markdown(f"#### {i}. {deck}")
                    c_stats.markdown(f"**Meta Score:** `{metrics['base_meta_score']:.2f}` | **Day 1 Share:** `{metrics['meta_share']:.2%}`")
                    
                    idx = deck_to_idx[deck]
                    favorable_data, threat_data = [], []
                    
                    for opp_name in active_decks:
                        if opp_name == deck: continue
                        opp_share = res["full_meta"][opp_name]
                        if opp_share < 0.01: continue # 1% cutoff
                        
                        wr = full_win_matrix[idx, deck_to_idx[opp_name]]
                        if wr >= WIN_THRESHOLD: 
                            favorable_data.append((opp_name, opp_share))
                        elif wr <= (1 - WIN_THRESHOLD): 
                            threat_data.append((opp_name, opp_share))
                    
                    favorable_data.sort(key=lambda x: x[1], reverse=True)
                    threat_data.sort(key=lambda x: x[1], reverse=True)
                    
                    favorable_share = sum(share for _, share in favorable_data)
                    threats_share = sum(share for _, share in threat_data)
                    
                    favorable_strs = [f"{name} ({share:.2%})" for name, share in favorable_data]
                    threat_strs = [f"{name} ({share:.2%})" for name, share in threat_data]
                    
                    if favorable_strs:
                        st.error(f"**Pushes out (Total {favorable_share:.2%}):** {', '.join(favorable_strs)}")
                    if threat_strs:
                        st.success(f"**Countered by (Total {threats_share:.2%}):** {', '.join(threat_strs)}")

        with tab_avoid:
            st.markdown("### 🚫 Negative Expected Value")
            st.caption("These decks are mathematically unfavored against the predicted field. Do yourself a favour and **avoid.**")
            visible_avoids = avoids[:st.session_state.avoid_limit]
            
            for i, r in enumerate(visible_avoids, 1):
                deck = r["deck"]
                metrics = res["metrics_per_deck"][deck]
                with st.container(border=True):
                    st.markdown(f"#### {i}. {deck}")
                    st.markdown(f"**Power Score:** `{metrics['power_score']:.2f}` | **Power Ranking (Day 1):** `{metrics['expected_win_rate']:.2%}`")
                    
                    idx = deck_to_idx[deck]
                    threat_data = []
                    
                    for opp_name, opp_share in res["full_meta"].items():
                        if opp_name == deck or opp_share < 0.01: continue # 1% Cutoff
                        if full_win_matrix[idx, deck_to_idx[opp_name]] <= 0.45: 
                            threat_data.append((opp_name, opp_share))

                    threat_data.sort(key=lambda x: x[1], reverse=True)
                    threats_share = sum(share for _, share in threat_data)
                    threat_strs = [f"{name} ({share:.2%})" for name, share in threat_data]

                    if threat_strs: 
                        st.error(f"⚠️ **Farmed by (Total {threats_share:.2%}):** {', '.join(threat_strs)}")

            if st.session_state.avoid_limit < len(avoids):
                st.button("⬇️ Load more...", key="btn_load_avoids", on_click=load_more_avoids, width="stretch")
                
if __name__ == "__main__":
    main()