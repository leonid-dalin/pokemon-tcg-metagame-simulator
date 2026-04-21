# src/core/scraper.py
import concurrent.futures
import csv
import json
import os
import re
import requests
import structlog
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import List, Dict, Tuple, Set, Any, cast

from opentelemetry import trace
from src.core.config import MATCHUP_DIR, INPUT_DIR, INPUT_FILE
from src.core.logger import setup_structured_logging

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------
# Pre-compiled Regex Patterns
# ---------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_CHARS_RE = re.compile(r"[^\w\s]")


def normalize_archetype(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    name = name.replace("&", " and ")
    name = _WS_RE.sub(" ", name)
    name = _CHARS_RE.sub("", name)
    return name.strip()


def extract_deck_info_from_filename(filename: str) -> Tuple[str, str]:
    """
    Extracts archetype and format from the naming convention:
    Archetype - Format - Website
    """
    base_name = os.path.splitext(filename)[0]
    parts = [p.strip() for p in base_name.split(" - ")]

    if len(parts) >= 2:
        return parts[0], parts[1]

    raise ValueError(f"Could not parse filename under new convention: {filename}")


def get_deck_archetype(file_path: str, filename: str) -> Tuple[str, str]:
    try:
        return extract_deck_info_from_filename(filename)
    except Exception:
        # Fallback for legacy or misnamed files: parse the HTML infobox
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            infobox_elem = soup.find("div", class_="infobox")

            if isinstance(infobox_elem, Tag):
                name_elem = infobox_elem.find("div", class_="name")
                format_elem = infobox_elem.find("div", class_="format")

                deck_name = name_elem.get_text(strip=True) if isinstance(name_elem, Tag) else "Unknown Archetype"
                deck_format = format_elem.get_text(strip=True) if isinstance(format_elem, Tag) else "Unknown Standard"

                if deck_name != "Unknown Archetype":
                    return deck_name, deck_format

        raise ValueError(f"Archetype extraction failed for {filename}")


def _fetch_url(session: requests.Session, url: str) -> Tuple[str, BeautifulSoup | None]:
    """Helper method to encapsulate the HTTP request and initial BS4 parse for the thread pool."""
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return url, BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        logger.error("url_fetch_failed", url=url, error=str(e))
        return url, None


def fetch_live_matchup_data(target_urls: List[str], canonical_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Automates the HTTP requests to Limitless TCG to grab live HTML concurrently,
    bypassing the need for manual file downloads.
    """
    with tracer.start_as_current_span("fetch_live_limitless_data") as span:
        span.set_attribute("url_count", len(target_urls))
        all_matchups = []
        fetched_pages = []

        with requests.Session() as session:
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(_fetch_url, session, url) for url in target_urls]

                for future in concurrent.futures.as_completed(futures):
                    url, soup = future.result()
                    if soup is None:
                        continue

                    infobox_elem = soup.find("div", class_="infobox")
                    if not isinstance(infobox_elem, Tag):
                        print(f"  ! Skipping {url}: Could not find infobox.")
                        continue

                    name_elem = infobox_elem.find("div", class_="name")
                    if not isinstance(name_elem, Tag):
                        print(f"  ! Skipping {url}: Could not find deck name in infobox.")
                        continue

                    deck_archetype = name_elem.get_text(strip=True)
                    format_name = "Standard"

                    # Sequentially populate the canonical map to avoid odd thread-safety anomalies
                    norm_name = normalize_archetype(deck_archetype)
                    if norm_name not in canonical_map:
                        canonical_map[norm_name] = deck_archetype

                    fetched_pages.append((soup, deck_archetype, format_name))

        # Parse the matchup tables when the whitelist is fully populated
        for soup, deck_archetype, format_name in fetched_pages:
            try:
                with tracer.start_as_current_span("parse_matchup_soup") as sub_span:
                    sub_span.set_attribute("archetype", deck_archetype)
                    matchups = scrape_matchup_soup(soup, deck_archetype, format_name, canonical_map)
                    all_matchups.extend(matchups)
            except Exception as e:
                logger.error("matchup_parse_failed", archetype=deck_archetype, error=str(e))

        return all_matchups


def scrape_matchup_soup(soup: BeautifulSoup, deck_archetype: str, format_name: str, canonical_map: Dict[str, str]) -> \
List[Dict[str, Any]]:
    """
    Parses the matchup table from a BeautifulSoup object directly.
    Extracts Opponents, Matches, and W-L-T scores to build the win rate matrix.
    """
    excluded_opponents = {"bye", "unknown", "", "other"}

    table_elem = soup.find("table", class_="striped")
    if not isinstance(table_elem, Tag):
        raise ValueError(f"Matchups table missing or not a Tag for archetype: {deck_archetype}")

    matchups = []

    for row in table_elem.find_all("tr")[1:]:
        if not isinstance(row, Tag):
            continue

        opponent_str = str(row.get("data-name", "")).strip()
        if not opponent_str or opponent_str.lower() in excluded_opponents:
            continue

        norm_opponent = normalize_archetype(opponent_str)
        if norm_opponent not in canonical_map:
            continue

        opponent_archetype = canonical_map[norm_opponent]

        raw_attr = row.get("data-matches")
        if isinstance(raw_attr, list):
            matches_attr = raw_attr[0] if raw_attr else "0"
        else:
            matches_attr = raw_attr
        matches = int(matches_attr) if matches_attr is not None else 0

        wins = losses = ties = 0
        score_tds = row.find_all("td")
        if len(score_tds) > 3:
            score_text = score_tds[3].get_text(strip=True)
            # Parse the "W - L - T" string (e.g., "6 - 12 - 5")
            parts = [p.strip() for p in score_text.split("-") if p.strip()]

            try:
                if len(parts) >= 3:
                    wins, losses, ties = map(int, parts[:3])
                elif len(parts) == 2:
                    wins, losses = map(int, parts)
            except (ValueError, TypeError):
                pass

        if matches > 0:
            # Formula: (Wins + 0.5 * Ties) / Total Matches
            winrate = (wins + (0.5 * ties)) / matches
        else:
            winrate = 0.5

        matchups.append({
            "deck_archetype": deck_archetype,
            "format": format_name,
            "opponent_archetype": opponent_archetype,
            "total_matches": matches,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": winrate,
        })

    return matchups


def scrape_matchup_data(file_path: str, deck_archetype: str, format_name: str, canonical_map: Dict[str, str]) -> List[
    Dict[str, Any]]:
    """
    Reads a local HTML file and delegates DOM parsing to scrape_matchup_soup.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    return scrape_matchup_soup(soup, deck_archetype, format_name, canonical_map)


def build_complete_matchup_matrix(all_matchup_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    with tracer.start_as_current_span("build_matchup_matrix_logic"):
        unique_archetypes: Set[str] = set()
        for m in all_matchup_data:
            unique_archetypes.add(m["deck_archetype"])
            unique_archetypes.add(m["opponent_archetype"])

        valid_archetypes = sorted(list(unique_archetypes))

        # Initialise empty matrix
        matrix = {a: {b: {"win_rate": 0.5, "match_count": 0} for b in valid_archetypes} for a in valid_archetypes}

        for m in all_matchup_data:
            da = m["deck_archetype"]
            oa = m["opponent_archetype"]
            matrix[da][oa] = {
                "win_rate": m["win_rate"],
                "match_count": m["total_matches"],
            }

        # Mirror missing matchups
        for a in valid_archetypes:
            for b in valid_archetypes:
                if a == b:
                    matrix[a][b] = {"win_rate": 0.5, "match_count": 0}
                elif matrix[a][b]["win_rate"] == 0.5 and matrix[b][a]["win_rate"] != 0.5:
                    matrix[a][b] = {
                        "win_rate": 1.0 - matrix[b][a]["win_rate"],
                        "match_count": matrix[b][a]["match_count"],
                    }
        logger.info("matrix_reconstruction_complete", archetypes=len(valid_archetypes))
        return {"archetypes": valid_archetypes, "matchup_matrix": matrix}


def save_to_csv(data: List[Dict[str, Any]], input_path: str):
    if not data:
        return
    with open(input_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(cast(Any, f), fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def save_matrix_to_csv(matrix_data: Dict[str, Any], input_path: str):
    archetypes = matrix_data["archetypes"]
    matrix = matrix_data["matchup_matrix"]
    with open(input_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(cast(Any, f))
        writer.writerow([""] + archetypes)
        for a in archetypes:
            row = [a]
            for b in archetypes:
                wr = matrix[a][b]["win_rate"] * 100
                row.append(f"{wr:.4f}%")
            writer.writerow(row)


def main():
    """
    CLI entry point for manual scraping and local matrix reconstruction.
    All outputs migrated to structured logging (UK English).
    """
    setup_structured_logging()
    os.makedirs(INPUT_DIR, exist_ok=True)

    html_files = [f for f in os.listdir(MATCHUP_DIR) if f.lower().endswith((".htm", ".html"))]

    if not html_files:
        logger.warning("no_html_files_found", directory=MATCHUP_DIR)
        return

    logger.info("manual_scrape_initiated", directory=MATCHUP_DIR, file_count=len(html_files))

    file_archetypes = {}
    canonical_map = {}

    # Phase 1: Archetype Identification
    for f in html_files:
        file_path = os.path.join(MATCHUP_DIR, f)
        try:
            # Note: ensure get_deck_archetype is imported or defined locally
            archetype, format_name = get_deck_archetype(file_path, f)
            file_archetypes[f] = (archetype, format_name)

            norm_name = normalize_archetype(archetype)
            if norm_name and norm_name not in canonical_map:
                canonical_map[norm_name] = archetype
        except Exception as e:
            logger.error("file_identification_skipped", filename=f, error=str(e))

    logger.info("canonical_map_established", archetype_count=len(canonical_map))

    all_matchups = []
    processed_files = 0

    # Phase 2: Matchup Extraction
    for f in html_files:
        if f not in file_archetypes:
            continue

        deck_archetype, format_name = file_archetypes[f]
        file_path = os.path.join(MATCHUP_DIR, f)

        try:
            matchups = scrape_matchup_data(file_path, deck_archetype, format_name, canonical_map)
            all_matchups.extend(matchups)
            processed_files += 1
        except Exception as e:
            logger.error("matchup_extraction_failed", filename=f, archetype=deck_archetype, error=str(e))

    logger.info("processing_summary",
                processed_files=processed_files,
                total_files=len(html_files),
                total_matchups=len(all_matchups))

    # Phase 3: Matrix Reconstruction & Persistence
    if all_matchups:
        save_to_csv(all_matchups, os.path.join(INPUT_DIR, "all_matchups.csv"))

        # Wrapped in a span if telemetry is imported
        matrix_data = build_complete_matchup_matrix(all_matchups)
        save_matrix_to_csv(matrix_data, os.path.join(INPUT_DIR, "matchup_matrix.csv"))

        output_path = os.path.join(INPUT_DIR, INPUT_FILE)
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(
                {
                    "archetypes": matrix_data["archetypes"],
                    "win_rate_matrix": {
                        a: {b: matrix_data["matchup_matrix"][a][b] for b in matrix_data["archetypes"]}
                        for a in matrix_data["archetypes"]
                    },
                },
                cast(Any, f_out),
                indent=2,
            )

        sample = matrix_data["archetypes"][:3] if len(matrix_data["archetypes"]) >= 3 else matrix_data["archetypes"]
        logger.info("data_persistence_complete",
                    output_file=output_path,
                    total_archetypes=len(matrix_data['archetypes']),
                    sample_decks=sample)


if __name__ == "__main__":
    main()