import os
import json
import csv
import re
from bs4 import BeautifulSoup
from bs4.element import Tag
from collections import defaultdict
from .config import MATCHUP_DIR, INPUT_DIR

def normalize_archetype(name):
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r'[&]', ' and ', name)
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[^\w\s]', '', name)
    return name.strip()

def extract_deck_info_from_filename(filename):
    base_name = os.path.splitext(filename)[0]
    patterns = [
        r"^(.*?)\s*-\s*([^-\n]+?)\s*-\s*Matchups",
        r"^(.*?)\s*-\s*([^-\n]+?)\s*-"
    ]
    for pattern in patterns:
        match = re.search(pattern, base_name)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    parts = base_name.split('-', 1)
    if len(parts) > 1:
        return parts[0].strip(), parts[1].strip()
    raise ValueError(f"Could not parse filename: {filename}")

def get_deck_archetype(file_path, filename):
    try:
        return extract_deck_info_from_filename(filename)
    except Exception:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            infobox_elem = soup.find('div', class_='infobox')
            if infobox_elem and isinstance(infobox_elem, Tag):
                name_elem = infobox_elem.find('div', class_='name')
                if name_elem and isinstance(name_elem, Tag):
                    return name_elem.get_text(strip=True), "BLK/WHT Standard 2025"
            if soup.title and soup.title.string:
                title = soup.title.string
                match = re.search(r"^(.*?)(?:\s*-\s*|$)", title)
                if match:
                    return match.group(1).strip(), "BLK/WHT Standard 2025"
        raise ValueError(f"Archetype extraction failed for {filename}")

def scrape_matchup_data(file_path, deck_archetype, format_name, canonical_map):
    EXCLUDED_OPPONENTS = {'bye', 'unknown', ''}
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    table_elem = soup.find('table', class_='striped')
    if not table_elem or not isinstance(table_elem, Tag):
        raise ValueError(f"Matchups table missing or not a Tag in {file_path}")

    matchups = []
    for row in table_elem.find_all('tr')[1:]:
        if not isinstance(row, Tag):
            continue

        # Safely get opponent name
        raw_name = row.get('data-name')
        if raw_name:
            opponent_str = str(raw_name).strip()
        else:
            opponent_str = ""
        if not opponent_str or opponent_str.lower() in EXCLUDED_OPPONENTS:
            continue

        norm_opponent = normalize_archetype(opponent_str)
        if norm_opponent not in canonical_map:
            continue
        opponent_archetype = canonical_map[norm_opponent]

        # Safely parse matches
        raw_matches = row.get('data-matches')
        try:
            matches = int(raw_matches) if raw_matches and isinstance(raw_matches, str) else 0
        except (ValueError, TypeError):
            matches = 0

        # Safely parse winrate
        raw_winrate = row.get('data-winrate')
        try:
            winrate = float(raw_winrate) if raw_winrate and isinstance(raw_winrate, str) else 0.5
        except (ValueError, TypeError):
            winrate = 0.5

        # Extract score from <td>s
        score_tds = row.find_all('td')
        wins = losses = ties = 0
        if len(score_tds) > 3:
            score_text = score_tds[3].get_text(strip=True)
            parts = [p.strip() for p in score_text.split('-') if p.strip()]
            if len(parts) >= 3:
                try:
                    wins, losses, ties = map(int, parts[:3])
                except (ValueError, TypeError):
                    pass
            elif len(parts) == 2:
                try:
                    wins, losses = map(int, parts)
                except (ValueError, TypeError):
                    pass

            if wins + losses + ties != matches and matches > 0:
                print(f"⚠️  Mismatch: {deck_archetype} vs {opponent_archetype}: {wins}+{losses}+{ties} != {matches}")
                winrate = wins / matches

        matchups.append({
            'deck_archetype': deck_archetype,
            'format': format_name,
            'opponent_archetype': opponent_archetype,
            'total_matches': matches,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_rate': winrate,
            'win_percentage': f"{winrate * 100:.4f}%"
        })

    return matchups

def build_complete_matchup_matrix(all_matchup_data):
    match_counts = defaultdict(int)
    for m in all_matchup_data:
        match_counts[m['deck_archetype']] += m['total_matches']
        match_counts[m['opponent_archetype']] += m['total_matches']

    MIN_MATCHES_REQ = 700
    valid_archetypes = [a for a, cnt in match_counts.items() if cnt >= MIN_MATCHES_REQ]

    matrix = {
        a: {
            b: {"win_rate": 0.5, "match_count": 0} for b in valid_archetypes
        } for a in valid_archetypes
    }

    for m in all_matchup_data:
        da = m['deck_archetype']
        oa = m['opponent_archetype']
        if da in valid_archetypes and oa in valid_archetypes:
            matrix[da][oa] = {
                "win_rate": m['win_rate'],
                "match_count": m['total_matches']
            }

    # Mirror missing matchups
    for a in valid_archetypes:
        for b in valid_archetypes:
            if a == b:
                matrix[a][b] = {"win_rate": 0.5, "match_count": 0}
            elif matrix[a][b]["win_rate"] == 0.5 and matrix[b][a]["win_rate"] != 0.5:
                matrix[a][b] = {
                    "win_rate": 1 - matrix[b][a]["win_rate"],
                    "match_count": matrix[b][a]["match_count"]
                }

    return {
        'archetypes': valid_archetypes,
        'matchup_matrix': matrix
    }

def save_to_csv(data, input_path):
    if not data:
        return
    with open(input_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def save_matrix_to_csv(matrix_data, input_path):
    archetypes = matrix_data['archetypes']
    matrix = matrix_data['matchup_matrix']
    with open(input_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + archetypes)
        for a in archetypes:
            row = [a]
            for b in archetypes:
                wr = matrix[a][b]["win_rate"] * 100
                row.append(f"{wr:.2f}%")
            writer.writerow(row)

def main():
    matchups_dir = MATCHUP_DIR
    input_dir = INPUT_DIR
    os.makedirs(input_dir, exist_ok=True)

    html_files = [f for f in os.listdir(matchups_dir) if f.lower().endswith(('.htm', '.html'))]
    if not html_files:
        print(f"No HTML files found in {matchups_dir}")
        return

    print(f"Found {len(html_files)} matchup files")

    file_archetypes = {}
    canonical_map = {}

    for f in html_files:
        file_path = os.path.join(matchups_dir, f)
        try:
            archetype, format_name = get_deck_archetype(file_path, f)
            file_archetypes[f] = (archetype, format_name)
            norm_name = normalize_archetype(archetype)
            if norm_name and norm_name not in canonical_map:
                canonical_map[norm_name] = archetype
        except Exception as e:
            print(f"  ! Skipping {f}: {str(e)}")

    print(f"Identified {len(canonical_map)} canonical archetypes")

    all_matchups = []
    processed_files = 0

    for f in html_files:
        if f not in file_archetypes:
            continue
        deck_archetype, format_name = file_archetypes[f]
        file_path = os.path.join(matchups_dir, f)
        try:
            matchups = scrape_matchup_data(file_path, deck_archetype, format_name, canonical_map)
            all_matchups.extend(matchups)
            processed_files += 1
        except Exception as e:
            print(f"  ! Error in {f}: {str(e)}")

    print(f"\nProcessed {processed_files}/{len(html_files)} files")
    print(f"Total matchups: {len(all_matchups)}")

    unique_archetypes = set()
    for m in all_matchups:
        unique_archetypes.add(m['deck_archetype'])
        unique_archetypes.add(m['opponent_archetype'])

    print(f"Unique archetypes: {len(unique_archetypes)}")

    if all_matchups:
        save_to_csv(all_matchups, os.path.join(input_dir, "all_matchups.csv"))

        matrix_data = build_complete_matchup_matrix(all_matchups)
        save_matrix_to_csv(matrix_data, os.path.join(input_dir, "matchup_matrix.csv"))

        with open(os.path.join(input_dir, "ea_input.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'archetypes': matrix_data['archetypes'],
                'win_rate_matrix': {
                    a: {
                        b: matrix_data['matchup_matrix'][a][b]
                        for b in matrix_data['archetypes']
                    }
                    for a in matrix_data['archetypes']
                }
            }, f, indent=2)

        print(f"\nValid archetypes (≥700 matches): {len(matrix_data['archetypes'])}")
        sample = matrix_data['archetypes'][:3] if len(matrix_data['archetypes']) >= 3 else matrix_data['archetypes']
        print("Sample:", sample)

if __name__ == "__main__":
    main()