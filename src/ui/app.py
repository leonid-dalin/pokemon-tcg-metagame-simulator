# src/ui/app.py | Streamlit dashboard
import json
import structlog
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from streamlit.runtime.uploaded_file_manager import UploadedFile
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Resolve the project root
project_root = str(Path(__file__).resolve().parents[2])

# Inject it into the system path so Python can find the 'src' module
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api.models import PrecisionTier, RangeSpec, ExactSpec, PredictionRequest
from src.core.data import load_matchup_data
from src.core.config import NASH_EQUILIBRIUM, INPUT_DATA, MIN_GAMES, WIN_THRESHOLD, aggressive_colorscale, TIER_THRESHOLDS, \
    TIER_2_THRESHOLD
from src.tournament.solver import swiss_rounds_from_players, get_variant_5_structure
from src.evolution.plotting import plot_metagame_scatter, plot_head_to_head_radar
from src.core.logger import setup_structured_logging
from src.core.telemetry import setup_telemetry, tracer

if "temp_input_mode" not in st.session_state:
    st.session_state.temp_input_mode = "Exact"

# ==========================================
# Observability Initialisation
# ==========================================
@st.cache_resource
def initialise_observability():
    """
    Initialise tracing and logging once per Streamlit lifecycle to avoid provider conflicts.
    """
    setup_structured_logging()
    setup_telemetry("tcg-ui")
    RequestsInstrumentor().instrument()
    return structlog.get_logger()

app_logger = initialise_observability()

# ==========================================
# Helper Funcs
# ==========================================
TTL_TIMER = 600

@st.cache_data(show_spinner=False, ttl=TTL_TIMER)
def get_valid_deck_names() -> List[str]:
    input_path = os.path.join(str(INPUT_DATA))
    deck_names, _, _ = load_matchup_data(input_path, MIN_GAMES)
    return sorted(deck_names)


@st.cache_data(show_spinner=False, ttl=TTL_TIMER)
def load_full_win_matrix():
    input_path = os.path.join(str(INPUT_DATA))
    deck_names, win_matrix, matchup_details = load_matchup_data(input_path, MIN_GAMES)
    return deck_names, win_matrix, matchup_details


def parse_limitless_html(html_str: str, valid_decks: List[str]) -> Tuple[Dict[str, int], int, int]:
    soup = BeautifulSoup(html_str, "html.parser")

    script_tag = soup.find(
        "script",
        attrs={
            "type": "application/json",
            "data-sveltekit-fetched": True,
            "data-url": lambda u: u and "/tcg/decks" in u
        }
    )

    # Fallback heuristic if Limitless changes their Svelte URL routing
    if not script_tag:
        script_tags = soup.find_all("script", attrs={"type": "application/json"})
        for tag in script_tags:
            content = tag.string
            if content and isinstance(content, str) and '"players":' in content and '"wins":' in content:
                script_tag = tag
                break

    if not script_tag or not script_tag.string:
        raise ValueError("Could not locate the internal JSON data block in the uploaded HTML.")

    script_content = script_tag.string
    if not isinstance(script_content, str):
        raise ValueError("Script tag content is invalid or None.")
    wrapper_json = json.loads(script_content)
    if "body" in wrapper_json and isinstance(wrapper_json["body"], str):
        body_json = json.loads(wrapper_json["body"])
        decks = body_json.get("message", [])
    elif "message" in wrapper_json:
        decks = wrapper_json["message"]
    elif isinstance(wrapper_json, list):
        decks = wrapper_json
    else:
        decks = []

    if not decks:
        raise ValueError("Deck array was empty in the parsed JSON.")

    parsed_meta = {}
    wildcard_players = 0
    total_players = 0

    valid_decks_lower = {d.lower(): d for d in valid_decks}

    for d in decks:
        name = str(d.get("name", "Unknown"))
        players = int(d.get("players", 0))
        total_players += players

        if name.lower() in valid_decks_lower:
            canonical_name = valid_decks_lower[name.lower()]
            parsed_meta[canonical_name] = parsed_meta.get(canonical_name, 0) + players
        else:
            wildcard_players += players

    return parsed_meta, total_players, wildcard_players


def init_session_state():
    defaults = {
        "meta_rows": [], "prediction_result": None, "mc_result": None,
        "rec_limit": 3, "avoid_limit": 3, "imported_players": 256,
        "internal_input_mode": "Percentage",
        "temp_input_mode": "Percentage"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def add_meta_row(deck="", spec_type="Exact", val=10.0, is_locked=True, mode=None):
    if mode is None:
        mode = st.session_state.internal_input_mode

    st.session_state.meta_rows.append({
        "id": str(uuid.uuid4()),
        "deck": deck,
        "spec_type": spec_type,
        "val": val,
        "is_locked": is_locked,
        "mode": mode
    })


def delete_meta_row(row_id: str): st.session_state.meta_rows = [r for r in st.session_state.meta_rows if
                                                                r["id"] != row_id]


def clear_all_rows(): st.session_state.meta_rows = []


def load_more_recs(): st.session_state.rec_limit += 3


def load_more_avoids(): st.session_state.avoid_limit += 3


def handle_mode_change():
    new_mode = st.session_state.temp_input_mode
    old_mode = st.session_state.internal_input_mode
    if new_mode == old_mode: return

    players = st.session_state.get('imported_players', 256)

    for idx, row in enumerate(st.session_state.meta_rows):
        row_id = row["id"]
        raw_val = row.get("val", 1.0)
        is_list = isinstance(raw_val, (list, tuple))

        if new_mode == "Raw Players":
            if is_list:
                new_val = (int((float(raw_val[0]) / 100.0) * players), int((float(raw_val[1]) / 100.0) * players))
            else:
                new_val = int((float(raw_val) / 100.0) * players)
        else:
            if is_list:
                new_val = ((float(raw_val[0]) / players) * 100.0 if players > 0 else 0.0,
                           (float(raw_val[1]) / players) * 100.0 if players > 0 else 0.0)
            else:
                new_val = (float(raw_val) / players) * 100.0 if players > 0 else 0.0

        st.session_state.meta_rows[idx]["val"] = new_val
        st.session_state.meta_rows[idx]["mode"] = new_mode

        if is_list:
            st.session_state[f"slider_{row_id}"] = new_val
        else:
            st.session_state[f"val_{row_id}"] = new_val

    st.session_state.internal_input_mode = new_mode


def get_tier(expected_wr: float) -> str:
    """
    Tier assignments based on rigorous TCG win-rate thresholds.
    T0 (≥55.0%), T1 (≥52.5%), T2 (≥50.0%), T3 (≥47.5%), T4 (≥45.0%), T5 (≥0%).
    """
    for tier, threshold in TIER_THRESHOLDS.items():
        if expected_wr >= threshold:
            return tier
    return "T5"


# --- State Management ---
if 'expander_import_open' not in st.session_state:
    st.session_state.expander_import_open = True
if 'expander_constraints_open' not in st.session_state:
    st.session_state.expander_constraints_open = False


# ==========================================
# Main App
# ==========================================
def main():
    st.set_page_config(page_title="TCG Metagame Solver", page_icon="🏆", layout="wide")
    init_session_state()

    st.title("🏆 TCG Metagame Solver")
    st.markdown(
        "Predict meta-weighted performance using real-world empirical data and strict Monte Carlo bracket simulations.")

    # ==========================================
    # SIDEBAR: TOURNAMENT & ENGINE SETTINGS
    # ==========================================    
    with st.sidebar:
        st.header("🏟️ Tournament Settings")

        player_field = PredictionRequest.model_fields['total_players']
        tie_field = PredictionRequest.model_fields['global_tie_rate']
        sample_field = PredictionRequest.model_fields['min_sample_threshold']

        with st.container(border=True):
            tourney_structure = st.radio("Tournament Structure", ["Championship Series", "Pure Swiss"], index=0)
            players_raw = st.number_input(
                "Number of Players",
                min_value=next(int(m.ge) for m in player_field.metadata if hasattr(m, 'ge')),
                max_value=next(int(m.le) for m in player_field.metadata if hasattr(m, 'le')),
                value=int(st.session_state.get('imported_players', player_field.default)),
                step=1
            )
            players: int = int(players_raw) if players_raw is not None else player_field.default
            st.session_state.imported_players = players

            if tourney_structure == "Championship Series":
                d1_rounds, cut_points, d2_rounds, top_cut = get_variant_5_structure(players)
                st.caption("**Championship Series Details**")
                st.markdown(
                    f"- **Day 1**: {d1_rounds} Rounds\n- **Day 2 Cut**: {cut_points} Match Pts\n- **Day 2**: {d2_rounds} Rounds\n- **Playoffs**: Top {top_cut}")
            else:
                d1_rounds = swiss_rounds_from_players(players)
                cut_points, d2_rounds, top_cut = 999, 0, (8 if players >= 8 else 0)
                st.caption("**Pure Swiss Details**")
                st.markdown(f"- **Swiss**: {d1_rounds} Rounds")
                if top_cut > 0: st.markdown(f"- **Playoffs**: Top {top_cut}")

        st.header("⚙️ Engine Parameters")
        with st.container(border=True):
            selected_tier_value = st.selectbox(
                "Monte Carlo Precision (Speed Tier)",
                options=[tier.value for tier in PrecisionTier],
                index=2,  # Defaults to "3 - STANDARD"
                help="Higher tiers run more iterations for greater fidelity, but take longer to process."
            )


            match_format = st.radio("Match Format", ["BO1", "BO3"], index=1)

            mode_options = ["Percentage", "Raw Players"]
            mode_idx = mode_options.index(st.session_state.internal_input_mode)
            st.radio(
                "Constraint Mode",
                mode_options,
                index=mode_idx,
                key="temp_input_mode",
                on_change=handle_mode_change
            )
            is_raw = st.session_state.internal_input_mode == "Raw Players"
            use_tie_convergence = st.toggle(
                "Enable Win-Rate Tie Convergence",
                value=True,
                help="Mathematically simulates real-world match timeouts using a parabolic curve based on matchup closeness.")
            global_tie_rate = st.slider(
                "Global Tie Rate (%)",
                min_value=next(float(m.ge) for m in tie_field.metadata if hasattr(m, 'ge')) * 100.0,
                max_value=next(float(m.le) for m in tie_field.metadata if hasattr(m, 'le')) * 100.0,
                value=float(tie_field.default * 100.0),
                step=0.5,
                disabled=not use_tie_convergence
            ) / 100.0

            min_sample_threshold = st.slider(
                "Matchup Minimum Games",
                min_value=next(int(m.ge) for m in sample_field.metadata if hasattr(m, 'ge')),
                max_value=next(int(m.le) for m in sample_field.metadata if hasattr(m, 'le')),
                value=int(sample_field.default),
                step=1
            )

            st.divider()
            st.markdown("**🧪 BETA Features**")
            use_drop_feature = st.toggle("Enable X-3 Drop Logic", value=False,
                                         help="Simulates players dropping from the tournament after accumulating 3 losses.")
            allow_negative_power = st.toggle("Show Negative Power Scores", value=False,
                                             help="When disabled, deeply unviable decks (Power Score < 0) are completely hidden from the Metagame Scatter Plot to keep the view clean. Toggle this ON to view them at their true negative depths.")

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

    with st.expander("📁 Import Metagame Data", expanded=st.session_state.expander_import_open):
        uploaded_file = st.file_uploader(
            "Upload Limitless Matchup Export (.html) or saved JSON config",
            type=['html', 'htm', 'json'],
            accept_multiple_files=False,
            help="HTML imports number of players + custom states as per Limitless Labs format; JSON restores custom states."
        )

        if isinstance(uploaded_file, UploadedFile):
            is_json = uploaded_file.name.lower().endswith('.json')

            if is_json:
                if st.button("Apply Saved Configuration", type="primary", use_container_width=True):
                    try:
                        saved_rows = json.loads(uploaded_file.getvalue().decode("utf-8"))
                        clear_all_rows()

                        if isinstance(saved_rows, dict) and "constraints" in saved_rows:
                            st.session_state.internal_input_mode = saved_rows.get("input_mode", "Percentage")
                            rows_to_load = saved_rows["constraints"]
                        else:
                            rows_to_load = saved_rows

                        for row in rows_to_load:
                            add_meta_row(
                                deck=row.get("deck", ""),
                                spec_type=row.get("spec_type", "Exact"),
                                val=row.get("val", 1.0),
                                is_locked=True,
                                mode=row.get("mode", "Percentage")
                            )
                        st.success(f"✅ Loaded {len(saved_rows)} constraints.")
                        st.rerun()
                    except Exception as e:
                        app_logger.error("invalid_json_import", error=str(e), exc_info=True)
                        st.error("❌ Invalid JSON format.")

            else:
                if st.button("Extract Data & Populate Constraints", type="secondary", use_container_width=True):
                    try:
                        parsed_meta, total_import_players, wildcard_players = parse_limitless_html(
                            uploaded_file.getvalue().decode("utf-8"), get_valid_deck_names()
                        )
                        st.session_state.imported_players = total_import_players
                        clear_all_rows()
                        for deck_name, count in parsed_meta.items():
                            add_meta_row(
                                deck=deck_name,
                                spec_type="Exact",
                                val=(count / total_import_players) * 100.0 if total_import_players > 0 else 0.0,
                                is_locked=True
                            )
                        st.session_state.import_msg = f"✅ Successfully imported {len(parsed_meta)} recognized decks ({total_import_players} total players)."
                        if wildcard_players > 0:
                            st.session_state.import_warn = f"⚠️ {wildcard_players} players were using unrecognized/rogue decks. Handled via rescaling."
                        st.session_state.expander_import_open = False
                        st.session_state.expander_constraints_open = True
                        st.rerun()
                    except Exception as e:
                        app_logger.error("html_parsing_error", error=str(e), exc_info=True)
                        st.error("❌ Error parsing HTML file. Please verify the format.")
                        st.stop()

    user_meta: Dict[str, Union[float, ExactSpec, RangeSpec]] = {}
    total_min = 0.0
    deck_names = get_valid_deck_names()
    seen_decks = set()

    with st.expander("🛠️ Custom Metagame Constraints", expanded=st.session_state.expander_constraints_open):
        # Calculate live widget state
        current_locked_sum = 0.0
        for r in st.session_state.meta_rows:
            if r.get("spec_type") == "Exact":
                row_id = r["id"]
                v = st.session_state.get(f"val_{row_id}", r.get("val", 0.0))
                if st.session_state.internal_input_mode == "Raw Players":
                    current_locked_sum += float(v) / (players if players > 0 else 1)
                else:
                    current_locked_sum += float(v) / 100.0

        clamped_sum = max(0.0, min(current_locked_sum, 1.0))
        remaining_pct = (1.0 - clamped_sum) * 100.0

        st.progress(clamped_sum,
                    text=f"Metagame Allocation: {clamped_sum * 100:.1f}% Assigned | {remaining_pct:.1f}% Remaining")
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
        with col1:
            if st.button("➕ Add Constraint", use_container_width=True):
                add_meta_row()
        with col2:
            if st.button("💧 Auto-Fill Remaining", use_container_width=True,
                         help="Splits the remaining percentage among all other archetypes proportionally to their online play rates."):
                d_names, _, matchup_details = load_full_win_matrix()

                # 1. Calc empirical baseline
                counts = np.zeros(len(d_names), dtype=float)
                deck_to_idx = {name: i for i, name in enumerate(d_names)}
                for (d1, d2), details in matchup_details.items():
                    if d1 in deck_to_idx:
                        counts[deck_to_idx[d1]] += float(details.get("match_count", 0.0))

                s = float(np.sum(counts))
                baseline = np.asarray(counts / s if s > 0 else np.ones(len(d_names)) / len(d_names), dtype=float)

                # 2. Extract locked percentage
                locked_sum = 0.0
                locked_decks = set()
                for r in st.session_state.meta_rows:
                    if r.get("spec_type") == "Exact":
                        locked_decks.add(r["deck"])
                        val = r["val"]
                        if st.session_state.internal_input_mode == "Raw Players":
                            locked_sum += float(
                                val) / st.session_state.imported_players if st.session_state.imported_players > 0 else 0.0
                        else:
                            locked_sum += float(val) / 100.0

                remaining = max(0.0, 1.0 - locked_sum)

                # 3. Process unfixed mask
                unfixed_mask = np.array([d not in locked_decks for d in d_names], dtype=bool)
                if not np.any(unfixed_mask) or remaining <= 0.0:
                    st.rerun()
                unfixed_baseline = np.asarray(baseline[unfixed_mask], dtype=float)

                s_unfixed = float(np.sum(unfixed_baseline))
                unfixed_baseline = np.asarray(
                    unfixed_baseline / s_unfixed if s_unfixed > 0 else np.ones(len(unfixed_baseline)) / len(
                        unfixed_baseline), dtype=float)

                final_unfixed_props = np.asarray(unfixed_baseline * remaining, dtype=float)
                unfixed_decks = [d for i, d in enumerate(d_names) if unfixed_mask[i]]

                for i, d in enumerate(unfixed_decks):
                    prop = float(final_unfixed_props[i])
                    if prop < NASH_EQUILIBRIUM: continue  # ignore sub 0% stragglers

                    if st.session_state.internal_input_mode == "Raw Players":
                        val = int(prop * st.session_state.imported_players)
                        if val < 1: continue
                    else:
                        val = round(prop * 100.0, 2)

                    add_meta_row(deck=d, spec_type="Exact", val=val, is_locked=True)

                st.rerun()
        with col3:
            export_data = {
                "input_mode": st.session_state.internal_input_mode,
                "constraints": [
                    {
                        "deck": r["deck"],
                        "spec_type": r["spec_type"],
                        "val": r["val"],
                        "is_locked": True,
                        "mode": st.session_state.internal_input_mode,
                    }
                    for r in st.session_state.meta_rows
                ]
            }
            st.download_button(
                label="📥 Export JSON",
                data=json.dumps(export_data, indent=2),
                file_name="tcg_meta_config.json",
                mime="application/json",
                use_container_width=True,
                help="Save current constraints for future sessions."
            )
        with col4:
            if st.button("🗑️ Clear All", use_container_width=True):
                clear_all_rows()
                st.rerun()


        st.markdown("<br>", unsafe_allow_html=True)
        constraint_cols = st.columns(2)

        for idx, row in enumerate(st.session_state.meta_rows):
            row_id = str(row["id"])
            target_col = constraint_cols[idx % 2]
            with target_col:
                with st.container(border=True):
                    cols = st.columns([3, 2, 3, 0.8])
                    with cols[0]:
                        available = [d for d in deck_names if d not in seen_decks or d == row.get("deck")]
                        if not available: break
                        default_idx = available.index(row["deck"]) if row.get("deck") in available else 0
                        deck = st.selectbox("Deck", options=available, index=default_idx, key=f"deck_{row_id}",
                                            label_visibility="collapsed")
                        seen_decks.add(str(deck))
                        st.session_state.meta_rows[idx]["deck"] = str(deck)
                    with cols[1]:
                        spec_type = st.radio("Type", ["Exact", "Range"],
                                             index=0 if row.get("spec_type") == "Exact" else 1, key=f"type_{row_id}",
                                             horizontal=True, label_visibility="collapsed")
                        st.session_state.meta_rows[idx]["spec_type"] = str(spec_type)
                    with cols[2]:
                        raw_val_any = row.get("val", 1.0)

                        if isinstance(raw_val_any, (list, tuple)) and len(raw_val_any) >= 2:
                            r_min, r_max = float(raw_val_any[0]), float(raw_val_any[1])
                            e_val = float(raw_val_any[0])
                        elif isinstance(raw_val_any, (int, float)):
                            r_min, r_max = 0.0, float(raw_val_any)
                            e_val = float(raw_val_any)
                        else:
                            r_min, r_max, e_val = 0.0, 10.0, 10.0

                        if spec_type == "Exact":
                            if is_raw:
                                safe_int_val = max(0, min(int(e_val), int(players)))
                                val_ui_raw = st.number_input(
                                    "Players", min_value=0, max_value=int(players),
                                    value=safe_int_val, step=1, key=f"val_{row_id}"
                                )
                                val_ui_int = int(val_ui_raw) if val_ui_raw is not None else safe_int_val
                                prop = float(val_ui_int) / players if players > 0 else 0.0
                                st.session_state.meta_rows[idx]["val"] = val_ui_int
                            else:
                                safe_float_val = max(0.0, min(e_val, 100.0))
                                val_ui_raw = st.number_input(
                                    "Percent (%)", min_value=0.0, max_value=100.0,
                                    value=safe_float_val, step=0.1, format="%.2f", key=f"val_{row_id}"
                                )
                                val_ui_float = float(val_ui_raw) if val_ui_raw is not None else safe_float_val
                                prop = val_ui_float / 100.0
                                st.session_state.meta_rows[idx]["val"] = val_ui_float

                            if prop > 0:
                                user_meta[str(deck)] = prop
                                total_min += prop

                        else:  # Range Mode
                            if is_raw:
                                v_min = max(0, min(int(r_min), int(players)))
                                v_max = max(0, min(int(r_max), int(players)))
                                if v_min > v_max: v_min = v_max

                                val_slider_raw = st.slider(
                                    "Range (Players)", 0, int(players),
                                    value=(v_min, v_max), key=f"slider_{row_id}"
                                )
                                if val_slider_raw is not None and isinstance(val_slider_raw, tuple):
                                    min_prop = float(val_slider_raw[0]) / players if players > 0 else 0.0
                                    max_prop = float(val_slider_raw[1]) / players if players > 0 else 0.0
                                    st.session_state.meta_rows[idx]["val"] = val_slider_raw
                                else:
                                    min_prop, max_prop = 0.0, 0.0
                            else:
                                v_min_f = max(0.0, min(r_min, 100.0))
                                v_max_f = max(0.0, min(r_max, 100.0))
                                if v_min_f > v_max_f: v_min_f = v_max_f

                                val_slider_f = st.slider(
                                    "Range (%)", 0.0, 100.0,
                                    value=(v_min_f, v_max_f), step=0.1, format="%.1f%%", key=f"slider_{row_id}"
                                )
                                if val_slider_f is not None and isinstance(val_slider_f, tuple):
                                    min_prop = float(val_slider_f[0]) / 100.0
                                    max_prop = float(val_slider_f[1]) / 100.0
                                    st.session_state.meta_rows[idx]["val"] = val_slider_f
                                else:
                                    min_prop, max_prop = 0.0, 0.0

                            user_meta[str(deck)] = cast(RangeSpec, cast(object, {"min": min_prop, "max": max_prop}))
                    with cols[3]:
                        st.button("🗑️", key=f"del_{row_id}", on_click=delete_meta_row, args=(row_id,))

    if total_min > 1.0: st.error(
        f"❌ Minimum total across all constraints ({total_min:.1%}) exceeds 100%. Please adjust your values before predicting."); st.stop()

    st.divider()

    # ==========================================
    # EXECUTION
    # ==========================================
    if st.button("🚀 Generate Metagame & Predict", type="primary", width="stretch", disabled=(total_min > 1.0)):
        job_id = str(uuid.uuid4())
        with (tracer.start_as_current_span("streamlit_prediction_workflow") as ui_span):
            ui_span.set_attribute("job.id", job_id)
            ui_span.set_attribute("total_players", players)

            with (st.status("⚙️ Initializing Engine...", expanded=True) as status):
                st_progress_bar = st.progress(0, text="Booting Data Solver...")
                start_time = time.time()
                try:
                    _, win_matrix, _ = load_full_win_matrix()
                    # 1. Prepare the payload
                    payload = {
                        "job_id": job_id,
                        "deck_names": deck_names,
                        "matchup_matrix": win_matrix.tolist() if hasattr(win_matrix, "tolist") else win_matrix,
                        "tournament_style": str(tourney_structure).lower().replace(" ", "_"),
                        "match_format": match_format,
                        "total_players": players,
                        "user_meta_spec": user_meta,
                        "precision_tier": selected_tier_value,
                        "global_tie_rate": global_tie_rate,
                        "use_tie_convergence": use_tie_convergence,
                        "use_drop_feature": use_drop_feature,
                        "min_sample_threshold": min_sample_threshold,
                    }

                    # 2. POST to FastAPI
                    api_url = os.environ.get("API_URL", "http://localhost:8000/api/v1")
                    app_logger.info("dispatching_prediction_request", job_id=job_id, api_url=api_url)
                    response = requests.post(f"{api_url}/predict", json=payload)
                    response.raise_for_status()
                    task_id = response.json()["task_id"]

                    # 3. Connect to the SSE Stream
                    status.update(label="Establishing SSE Connection...", state="running", expanded=True)
                    is_complete = False
                    max_retries = 3
                    retries = 0

                    with tracer.start_as_current_span("sse_stream_polling"):
                        while retries < max_retries and not is_complete:
                            try:
                                with requests.get(f"{api_url}/tasks/{task_id}/stream?job_id={job_id}", stream=True,
                                                  timeout=45) as stream_response:
                                    stream_response.raise_for_status()

                                    for line in stream_response.iter_lines():
                                        if not line: continue
                                        decoded_line = line.decode('utf-8')

                                        if decoded_line.startswith("data:"):
                                            data_str = decoded_line[5:].strip()
                                            task_res = json.loads(data_str)

                                            if task_res["status"] == "processing":
                                                pct = task_res.get("progress", 0)
                                                elapsed = time.time() - start_time

                                                # ETA Math
                                                if pct > 0:
                                                    total_est = elapsed / (pct / 100.0)
                                                    eta = total_est - elapsed
                                                    status.update(label=f"Simulating... {pct}%", state="running", expanded=True)
                                                    st_progress_bar.progress(pct / 100.0,
                                                                             text=f"Processing Monte Carlo Brackets: {pct}% | ETA: {eta:.1f}s")
                                                else:
                                                    status.update(label="Solving Water-Filling Constraints...",
                                                                  state="running")

                                            elif task_res["status"] == "complete":
                                                st_progress_bar.progress(1.0,
                                                                         text="Simulation Complete! Rendering Data...")
                                                st.session_state.prediction_result = task_res["data"]["solver_results"]
                                                st.session_state.mc_result = task_res["data"]["mc_results"]
                                                status.update(label="✅ Run Complete!", state="complete", expanded=False)
                                                is_complete = True
                                                break

                                            elif task_res["status"] == "failed":
                                                raise Exception(
                                                    task_res.get("error", "Unknown background task failure."))

                                # If stream drops mid-process, it will break the loop and retry
                                if not is_complete:
                                    retries += 1
                                    time.sleep(2)

                            except requests.exceptions.RequestException:
                                retries += 1
                                if retries >= max_retries:
                                    break
                                time.sleep(2)

                        if not is_complete:
                            raise Exception("Lost connection to the backend server and exhausted all retry attempts.")

                except Exception as e:
                    app_logger.error("simulation_workflow_failed", error=str(e), exc_info=True)
                    ui_span.record_exception(e)
                    status.update(label="❌ Simulation Failed", state="error", expanded=True)
                    st.error("An internal error occurred during the simulation. Please consult the system logs.")
                    st.stop()

    # ==========================================
    # DATAFRAME RESULTS
    # ==========================================
    if st.session_state.prediction_result is not None and st.session_state.mc_result is not None:
        res = st.session_state.prediction_result
        mc_res = st.session_state.mc_result

        full_deck_names, full_win_matrix, _ = load_full_win_matrix()
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

        active_decks = [str(d) for d in full_deck_names if res["metrics_per_deck"][d]["meta_share"] >= 0.001]
        all_decks_sorted = sorted(active_decks, key=lambda d: float(res["metrics_per_deck"][d][sort_key]), reverse=True)

        # Calculate Day 2 Expected Win Rates dynamically
        day2_share_vec = np.zeros(len(full_deck_names))
        for i, d_name in enumerate(full_deck_names):
            day2_share_vec[i] = mc_res.get(str(d_name), {}).get("day2_share", 0)

        if np.sum(day2_share_vec) > 0:
            day2_share_vec /= np.sum(day2_share_vec)

        day2_expected_wrs = full_win_matrix @ day2_share_vec
        day2_wr_dict = {d: float(day2_expected_wrs[deck_to_idx[d]]) for d in active_decks}

        data = []
        for i, deck in enumerate(all_decks_sorted, 1):
            metrics = res["metrics_per_deck"][deck]
            mc_metrics = mc_res.get(deck, {"day2_conversion": 0, "top_cut_conversion": 0, "win_probability": 0,
                                           "day2_share": 0, "top_cut_share": 0})

            meta_share = float(metrics["meta_share"])

            row_data = {
                "#": str(i),
                "Deck": str(deck),
                "Type": "🔒 User" if deck in user_meta else "📈 Base",
                "Tier": get_tier(float(metrics["expected_win_rate"])),
                "Power Score": round(float(metrics["power_score"]), 2),
                "Freq Score": round(float(metrics["frequency_score"]), 2),
                "Meta Score": round(float(metrics["base_meta_score"]), 2),
                "Power Ranking (Day 1)": round(float(metrics["expected_win_rate"]) * 100, 2),
                "Share % (Day 1)": round(meta_share * 100, 2),
            }

            if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
                row_data["Power Ranking (Day 2)"] = round(float(day2_wr_dict[deck]) * 100, 2)

            # --- Micro/Macro Odds View ---
            if odds_view == "Player (Micro)":
                if d2_rounds > 0: row_data["Conv. Rate % (Day 2)"] = round(
                    float(mc_metrics.get("day2_conversion", 0)) * 100, 2)
                if top_cut > 0:
                    row_data[f"Conv. Rate % (Top {top_cut})"] = round(
                        float(mc_metrics.get("top_cut_conversion", 0)) * 100, 2)
                    row_data["Win Event %"] = round(float(mc_metrics.get("win_probability", 0)) * 100, 2)
            else:
                if d2_rounds > 0: row_data["Share % (Day 2)"] = round(float(mc_metrics.get("day2_share", 0)) * 100, 2)
                if top_cut > 0:
                    row_data[f"Share % (Top {top_cut})"] = round(float(mc_metrics.get("top_cut_share", 0)) * 100, 2)
                    row_data["Share % (Winner)"] = round(
                        float(mc_metrics.get("win_probability", 0)) * meta_share * players * 100, 2)

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
            "Type": st.column_config.TextColumn(width="small",
                                                help="Forced by user constraint (🔒) or simulated baseline (📈)."),
            "Tier": st.column_config.TextColumn(width="small",
                                                help="T0 (≥52.5%), T0.5 (≥50%), T1 (≥47.5%), T2 (≥45%), T3 (≥42.5%), T4 (≥0%)."),
            "Power Score": st.column_config.NumberColumn(width="small", format="%.2f",
                                                         help="Normalization of Win Rate. 100 is the best performing deck. Can be negative."),
            "Freq Score": st.column_config.NumberColumn(width="small", format="%.2f",
                                                        help="Normalization of Popularity. 100 is the most played deck."),
            "Meta Score": st.column_config.NumberColumn(width="small", format="%.2f",
                                                        help="Average of Power and Frequency. High scores mean heavily dominant."),
            "Power Ranking (Day 1)": st.column_config.NumberColumn(format="%.2f %%",
                                                                   help="Weighted Win Rate against the day 1 field."),
            "Power Ranking (Day 2)": st.column_config.NumberColumn(format="%.2f %%",
                                                                   help="Expected Win Rate against the condensed Day 2 meta."),
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

        valid_meta_decks = [str(d["Deck"]) for d in data]
        c1, c2 = st.columns(2)
        deck_a_raw = c1.selectbox("Primary Deck (Baseline)", valid_meta_decks, index=0)
        deck_b_raw = c2.selectbox("Comparison Deck (Challenger)", valid_meta_decks,
                                  index=1 if len(valid_meta_decks) > 1 else 0)

        if deck_a_raw is not None and deck_b_raw is not None:
            deck_a = str(deck_a_raw)
            deck_b = str(deck_b_raw)
            da_idx, db_idx = deck_to_idx[deck_a], deck_to_idx[deck_b]
            da_metrics, db_metrics = res["metrics_per_deck"][deck_a], res["metrics_per_deck"][deck_b]
            da_mc, db_mc = mc_res.get(deck_a, {}), mc_res.get(deck_b, {})

            m_row1 = st.columns(4)
            m_row1[0].metric("Meta Score", f"{float(da_metrics['base_meta_score']):.2f}",
                             f"{float(da_metrics['base_meta_score']) - float(db_metrics['base_meta_score']):.2f} vs {deck_b}")
            m_row1[1].metric("Power Score", f"{float(da_metrics['power_score']):.2f}",
                             f"{float(da_metrics['power_score']) - float(db_metrics['power_score']):.2f} vs {deck_b}")
            m_row1[2].metric("Power Ranking (Day 1)", f"{float(da_metrics['expected_win_rate']):.2%}",
                             f"{float(da_metrics['expected_win_rate']) - float(db_metrics['expected_win_rate']):.2%} vs {deck_b}")
            if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
                m_row1[3].metric("Power Ranking (Day 2)", f"{day2_wr_dict.get(deck_a, 0.0):.2%}",
                                 f"{day2_wr_dict.get(deck_a, 0.0) - day2_wr_dict.get(deck_b, 0.0):.2%} vs {deck_b}")
            m_row2 = st.columns(3)
            if d2_rounds > 0:
                m_row2[0].metric("Day 2 Odds", f"{float(da_mc.get('day2_conversion', 0)):.2%}",
                                 f"{float(da_mc.get('day2_conversion', 0)) - float(db_mc.get('day2_conversion', 0)):.2%} vs {deck_b}")
            if top_cut > 0:
                m_row2[1].metric(f"Top {top_cut} Odds", f"{float(da_mc.get('top_cut_conversion', 0)):.2%}",
                                 f"{float(da_mc.get('top_cut_conversion', 0)) - float(db_mc.get('top_cut_conversion', 0)):.2%} vs {deck_b}")
                m_row2[2].metric("Win Odds", f"{float(da_mc.get('win_probability', 0)):.2%}",
                                 f"{float(da_mc.get('win_probability', 0)) - float(db_mc.get('win_probability', 0)):.2%} vs {deck_b}")

            # --- Interactive Spider/Radar Chart ---
            st.markdown("#### 🕸️ Head-to-Head Stat Radar")

            global_max_meta = float(max(float(m["base_meta_score"]) for m in res["metrics_per_deck"].values()))
            global_max_power = float(max(float(m["power_score"]) for m in res["metrics_per_deck"].values()))
            global_max_pr1 = float(max(float(m["expected_win_rate"]) for m in res["metrics_per_deck"].values())) * 100.0

            def safe_norm(v: float, max_val: float) -> float:
                return max(0.0, (v / max_val * 100.0)) if max_val > 0 else 0.0

            categories = ['Meta Score', 'Power Score', 'Power Ranking (D1)']

            da_vals_norm = [
                safe_norm(float(da_metrics['base_meta_score']), global_max_meta),
                safe_norm(float(da_metrics['power_score']), global_max_power),
                safe_norm(float(da_metrics['expected_win_rate']) * 100, global_max_pr1)
            ]
            db_vals_norm = [
                safe_norm(float(db_metrics['base_meta_score']), global_max_meta),
                safe_norm(float(db_metrics['power_score']), global_max_power),
                safe_norm(float(db_metrics['expected_win_rate']) * 100, global_max_pr1)
            ]

            da_texts = [
                f"Meta Score: {float(da_metrics['base_meta_score']):.2f}",
                f"Power Score: {float(da_metrics['power_score']):.2f}",
                f"Power Rank D1: {float(da_metrics['expected_win_rate']):.2%}"
            ]
            db_texts = [
                f"Meta Score: {float(db_metrics['base_meta_score']):.2f}",
                f"Power Score: {float(db_metrics['power_score']):.2f}",
                f"Power Rank D1: {float(db_metrics['expected_win_rate']):.2%}"
            ]

            if d2_rounds > 0 and np.sum(day2_share_vec) > 0:
                global_max_pr2 = float(max(float(v) for v in day2_wr_dict.values()) * 100.0) if day2_wr_dict else 1.0
                categories.append('Power Ranking (D2)')
                da_vals_norm.append(safe_norm(day2_wr_dict.get(deck_a, 0.0) * 100, global_max_pr2))
                db_vals_norm.append(safe_norm(day2_wr_dict.get(deck_b, 0.0) * 100, global_max_pr2))
                da_texts.append(f"Power Rank D2: {day2_wr_dict.get(deck_a, 0.0):.2%}")
                db_texts.append(f"Power Rank D2: {day2_wr_dict.get(deck_b, 0.0):.2%}")

            if d2_rounds > 0:
                global_max_d2 = float(max(float(m.get("day2_conversion", 0)) for m in mc_res.values()) * 100.0)
                categories.append('Day 2 Odds')
                da_vals_norm.append(safe_norm(float(da_mc.get('day2_conversion', 0)) * 100, global_max_d2))
                db_vals_norm.append(safe_norm(float(db_mc.get('day2_conversion', 0)) * 100, global_max_d2))
                da_texts.append(f"D2 Odds: {float(da_mc.get('day2_conversion', 0)):.2%}")
                db_texts.append(f"D2 Odds: {float(db_mc.get('day2_conversion', 0)):.2%}")

            if top_cut > 0:
                global_max_t8 = float(max(float(m.get("top_cut_conversion", 0)) for m in mc_res.values()) * 100.0)
                global_max_win = float(max(float(m.get("win_probability", 0)) for m in mc_res.values()) * 100.0)
                categories.append(f'Top {top_cut} Odds')
                da_vals_norm.append(safe_norm(float(da_mc.get('top_cut_conversion', 0)) * 100, global_max_t8))
                db_vals_norm.append(safe_norm(float(db_mc.get('top_cut_conversion', 0)) * 100, global_max_t8))
                da_texts.append(f"Top {top_cut} Odds: {float(da_mc.get('top_cut_conversion', 0)):.2%}")
                db_texts.append(f"Top {top_cut} Odds: {float(db_mc.get('top_cut_conversion', 0)):.2%}")
                categories.append('Win Odds')
                da_vals_norm.append(safe_norm(float(da_mc.get('win_probability', 0)) * 100, global_max_win))
                db_vals_norm.append(safe_norm(float(db_mc.get('win_probability', 0)) * 100, global_max_win))
                da_texts.append(f"Win Odds: {float(da_mc.get('win_probability', 0)):.2%}")
                db_texts.append(f"Win Odds: {float(db_mc.get('win_probability', 0)):.2%}")

            fig_radar = plot_head_to_head_radar(deck_a, deck_b, categories, da_vals_norm, db_vals_norm, da_texts,
                                                db_texts)
            st.plotly_chart(fig_radar, width="stretch")

            st.markdown("#### Matchups")

            top_field = [d for d in data if float(res["metrics_per_deck"][str(d["Deck"])]["meta_share"]) >= 0.03]
            comp_data = []
            for field_deck in top_field:
                opp_name = str(field_deck["Deck"])
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

            if not comp_df.empty:
                def highlight_winrates(v: Any) -> str:
                    if not isinstance(v, (int, float)): return ''
                    norm_val = float(v) / 100.0
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
            else:
                st.info("No common opponents found exceeding the 3% field threshold.")

        st.divider()

        # --- Dynamic Actionable Intelligence ---
        top_threats = sorted([d for d in active_decks if float(res["metrics_per_deck"][d]["base_meta_score"]) > 50],
                             key=lambda d: float(res["metrics_per_deck"][d]["base_meta_score"]), reverse=True)

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
                float(mc_res.get(str(d), {}).get(ev_key, 0)) if ev_key != "power_score" else float(
                    res["metrics_per_deck"][str(d)]["power_score"]),
                float(res["metrics_per_deck"][str(d)]["power_score"])
            ),
            reverse=True
        )
        recommendations = [{"deck": str(d), **res["metrics_per_deck"][str(d)]} for d in recs_sorted_by_ev]
        avoids = [{"deck": str(d), **res["metrics_per_deck"][str(d)]} for d in recs_sorted_by_ev[::-1]]

        best_day2 = str(max(active_decks, key=lambda d: float(mc_res.get(str(d), {}).get("day2_conversion", 0)))) if (
                d2_rounds > 0 and active_decks) else None
        best_top8 = str(
            max(active_decks, key=lambda d: float(mc_res.get(str(d), {}).get("top_cut_conversion", 0)))) if (
                top_cut > 0 and active_decks) else None
        best_win = str(max(active_decks, key=lambda d: float(mc_res.get(str(d), {}).get("win_probability", 0)))) if (
                top_cut > 0 and active_decks) else None
        best_wr = str(max(active_decks, key=lambda d: float(
            res["metrics_per_deck"][str(d)]["power_score"]))) if active_decks else None
        best_day2_predator = str(max(day2_wr_dict.keys(), key=lambda k: float(day2_wr_dict[str(k)]))) if (
                d2_rounds > 0 and np.sum(day2_share_vec) > 0 and day2_wr_dict) else None

        best_ev_val = float(mc_res.get(str(recs_sorted_by_ev[0]), {}).get(ev_key, 0) if ev_key != "power_score" else
                            res["metrics_per_deck"][str(recs_sorted_by_ev[0])][
                                "power_score"]) if recs_sorted_by_ev else 0.0
        tab_rec, tab_threat, tab_avoid = st.tabs(["🔥 Top Recommendations", "🚨 Top Threats", "🚫 Decks to Avoid"])

        with tab_rec:
            st.markdown("### 🔥 Best EV")
            st.caption(
                f"These decks are ranked strictly by **{ev_label}**. They mathematically yield the highest returns in this specific bracket structure.")
            visible_recs = recommendations[:st.session_state.rec_limit]

            for i, r in enumerate(visible_recs, 1):
                deck = str(r["deck"])
                metrics = res["metrics_per_deck"][deck]
                mc_metrics = mc_res.get(deck, {})

                tags = []
                if deck == best_win:
                    tags.append("🏆 Best to Win Event")
                elif deck == best_top8:
                    tags.append(f"🏅 Best for Top {top_cut}")
                elif deck == best_day2:
                    tags.append("🛡️ Safest Day 2")
                if deck == best_wr: tags.append("📈 Highest Raw Win Rate")
                if deck == best_day2_predator: tags.append("🧱 Day 2 Predator")
                if float(metrics["meta_share"]) >= 0.10 and float(
                        metrics["expected_win_rate"]) < TIER_2_THRESHOLD: tags.append("🪤 Overplayed Trap")
                if top_threats and deck not in top_threats and float(
                        full_win_matrix[deck_to_idx[deck], deck_to_idx[str(top_threats[0])]]) > WIN_THRESHOLD:
                    tags.append(f"💡 Rogue Meta Breaker (Beats {top_threats[0]})")

                with st.container(border=True):
                    st.markdown(f"#### {i}. {deck}")
                    if tags:
                        st.markdown(" ".join([f"`{t}`" for t in tags]))

                    # Calculate EV delta vs #1 spot
                    deck_ev_val = float(mc_metrics.get(ev_key, 0)) if ev_key != "power_score" else float(
                        metrics["power_score"])
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

                    st.markdown(f"{ev_str} | **Power Ranking (Day 1):** `{float(metrics['expected_win_rate']):.2%}`")

                    st.markdown("##### Performance vs Top Threats:")
                    if top_threats:
                        threat_cols = st.columns(len(top_threats))
                        for t_idx, threat_deck in enumerate(top_threats):
                            threat_deck_str = str(threat_deck)
                            wr_vs_threat = float(full_win_matrix[deck_to_idx[deck], deck_to_idx[threat_deck_str]])
                            color = "🟢" if wr_vs_threat >= WIN_THRESHOLD else "🔴" if wr_vs_threat <= (
                                    1 - WIN_THRESHOLD) else "🟡"
                            threat_cols[t_idx].markdown(f"{color} **{threat_deck_str}**: `{wr_vs_threat:.0%}`")
                    else:
                        st.caption("No dominant meta juggernaut detected in this field.")

            if st.session_state.rec_limit < len(recommendations):
                st.button("⬇️ Load more...", key="btn_load_recs", on_click=load_more_recs, width="stretch")

        with tab_threat:
            st.markdown("### 🚨 The Meta Juggernauts")
            st.caption(
                "These decks have the highest Meta Scores (>50). They combine high win rates with massive popularity.")
            for i, deck_raw in enumerate(top_threats, 1):
                deck = str(deck_raw)
                metrics = res["metrics_per_deck"][deck]
                with st.container(border=True):
                    c_title, c_stats = st.columns([2, 1])
                    c_title.markdown(f"#### {i}. {deck}")
                    c_stats.markdown(
                        f"**Meta Score:** `{float(metrics['base_meta_score']):.2f}` | **Day 1 Share:** `{float(metrics['meta_share']):.2%}`")

                    idx = deck_to_idx[deck]
                    favorable_data, threat_data = [], []

                    for opp_key in active_decks:
                        opp_name = str(opp_key)
                        if opp_name == deck: continue
                        opp_share = float(res["full_meta"][opp_name])
                        if opp_share < 0.01: continue  # 1% cutoff

                        wr = float(full_win_matrix[idx, deck_to_idx[opp_name]])
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
            st.markdown("### 🚫 Negative EV")
            st.caption(
                "These decks are mathematically unfavored against the predicted field. Do yourself a favour and **avoid.**")
            visible_avoids = avoids[:st.session_state.avoid_limit]

            for i, r in enumerate(visible_avoids, 1):
                deck = str(r["deck"])
                metrics = res["metrics_per_deck"][deck]
                with st.container(border=True):
                    st.markdown(f"#### {i}. {deck}")
                    st.markdown(
                        f"**Power Score:** `{float(metrics['power_score']):.2f}` | **Power Ranking (Day 1):** `{float(metrics['expected_win_rate']):.2%}`")

                    idx = deck_to_idx[deck]
                    threat_data = []

                    for opp_key, opp_share_raw in res["full_meta"].items():
                        opp_name = str(opp_key)
                        opp_share = float(opp_share_raw)
                        if opp_name == deck or opp_share < 0.01: continue  # 1% Cutoff
                        if float(full_win_matrix[idx, deck_to_idx[opp_name]]) <= 0.45:
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
