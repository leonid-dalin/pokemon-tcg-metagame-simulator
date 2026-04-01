use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand::distributions::WeightedIndex;
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
struct Player {
    id: usize,
    deck_idx: usize,
    match_points: i32,
    losses: i32,
    dropped: bool,
    opponents: Vec<usize>,
}

impl Player {
    fn new(id: usize, deck_idx: usize) -> Self {
        Self {
            id,
            deck_idx,
            match_points: 0,
            losses: 0,
            dropped: false,
            opponents: Vec::with_capacity(16), // Pre-allocate for standard tournament lengths
        }
    }
}

fn play_rounds(
    rounds: usize,
    players: &mut Vec<Player>,
    active_ids: &mut Vec<usize>,
    win_matrix: &[Vec<f64>],
    rng: &mut StdRng,
    use_tie_convergence: bool,
    global_tie_rate: f64,
    use_drop_feature: bool,
) {
    for _ in 0..rounds {
        if active_ids.len() < 2 {
            break;
        }

        // Shuffle active IDs to ensure random pairings among equal points
        for i in (1..active_ids.len()).rev() {
            let j = rng.gen_range(0..=i);
            active_ids.swap(i, j);
        }

        // Stable sort by match points descending
        active_ids.sort_by(|&a, &b| {
            players[b].match_points.cmp(&players[a].match_points)
        });

        // 1-Deep Lookahead Rematch Prevention
        let mut i = 0;
        while i + 1 < active_ids.len() {
            let p1_id = active_ids[i];
            let p2_id = active_ids[i + 1];

            if players[p1_id].opponents.contains(&p2_id) {
                if i + 3 < active_ids.len() {
                    let next_p2_id = active_ids[i + 2];
                    let next_p1_id = active_ids[i + 3]; // The guy who would pair with next_p2_id

                    if !players[p1_id].opponents.contains(&next_p2_id) &&
                       !players[next_p1_id].opponents.contains(&p2_id) {
                        active_ids.swap(i + 1, i + 2);
                    }
                }
            }
            i += 2;
        }

        // Resolve Matches
        let mut i = 0;
        while i + 1 < active_ids.len() {
            let p1_id = active_ids[i];
            let p2_id = active_ids[i + 1];

            let p1_deck = players[p1_id].deck_idx;
            let p2_deck = players[p2_id].deck_idx;
            let win_prob = win_matrix[p1_deck][p2_deck];

            let roll: f64 = rng.gen_range(0.0..1.0);

            players[p1_id].opponents.push(p2_id);
            players[p2_id].opponents.push(p1_id);

            if use_tie_convergence {
                let tie_prob = global_tie_rate * 4.0 * win_prob * (1.0 - win_prob);
                let p1_win_thresh = win_prob - (tie_prob / 2.0);
                let tie_thresh = p1_win_thresh + tie_prob;

                if roll < p1_win_thresh {
                    players[p1_id].match_points += 3;
                    players[p2_id].losses += 1;
                } else if roll >= tie_thresh {
                    players[p2_id].match_points += 3;
                    players[p1_id].losses += 1;
                } else {
                    players[p1_id].match_points += 1;
                    players[p2_id].match_points += 1;
                }
            } else {
                if roll < win_prob {
                    players[p1_id].match_points += 3;
                    players[p2_id].losses += 1;
                } else {
                    players[p2_id].match_points += 3;
                    players[p1_id].losses += 1;
                }
            }

            if use_drop_feature {
                if players[p1_id].losses >= 3 { players[p1_id].dropped = true; }
                if players[p2_id].losses >= 3 { players[p2_id].dropped = true; }
            }

            i += 2;
        }

        // Apply Drops
        if use_drop_feature {
            active_ids.retain(|&id| !players[id].dropped);
        }
    }
}

// Struct to hold the aggregated results from a thread
#[derive(Clone)]
struct MCResult {
    total_initial: Vec<usize>,
    total_day2: Vec<usize>,
    total_topcut: Vec<usize>,
    total_champ: Vec<usize>,
}

#[pyfunction]
#[pyo3(signature = (iterations, num_players, meta_distribution, win_matrix, d1_rounds, cut_points, d2_rounds, top_cut, base_seed, use_tie_convergence, global_tie_rate, use_drop_feature))]
fn run_parallel_monte_carlo(
    iterations: usize,
    num_players: usize,
    meta_distribution: Vec<f64>,
    win_matrix: Vec<Vec<f64>>,
    d1_rounds: usize,
    cut_points: i32,
    d2_rounds: usize,
    top_cut: usize,
    base_seed: u64,
    use_tie_convergence: bool,
    global_tie_rate: f64,
    use_drop_feature: bool,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)> {

    let n_decks = win_matrix.len();

    // Rayon parallel iterator across iterations
    let final_result = (0..iterations)
        .into_par_iter()
        .map(|iter_idx| {
            // Seed a deterministic RNG for this specific iteration
            let mut rng = StdRng::seed_from_u64(base_seed + iter_idx as u64);

            // Build local tracking arrays
            let mut local_initial = vec![0; n_decks];
            let mut local_day2 = vec![0; n_decks];
            let mut local_topcut = vec![0; n_decks];
            let mut local_champ = vec![0; n_decks];

            // 1. Assign decks based on meta distribution
            let dist = WeightedIndex::new(&meta_distribution).unwrap();
            let mut players: Vec<Player> = (0..num_players)
                .map(|i| {
                    let deck_idx = rng.sample(&dist);
                    local_initial[deck_idx] += 1;
                    Player::new(i, deck_idx)
                })
                .collect();

            let mut active_ids: Vec<usize> = (0..num_players).collect();

            // 2. Play Day 1
            play_rounds(d1_rounds, &mut players, &mut active_ids, &win_matrix, &mut rng, use_tie_convergence, global_tie_rate, use_drop_feature);

            // Calculate Day 2 players
            let mut day2_ids: Vec<usize> = players.iter()
                .filter(|p| p.match_points >= cut_points)
                .map(|p| p.id)
                .collect();

            if !day2_ids.is_empty() && d2_rounds > 0 {
                for &id in &day2_ids {
                    local_day2[players[id].deck_idx] += 1;
                }
            }

            // 3. Play Day 2
            if d2_rounds > 0 && day2_ids.len() > 1 {
                play_rounds(d2_rounds, &mut players, &mut day2_ids, &win_matrix, &mut rng, use_tie_convergence, global_tie_rate, use_drop_feature);
            }

            // 4. Calculate OWP and Top Cut Sorting
            let mut top_players: Vec<usize> = Vec::new();
            if top_cut > 0 {
                let pool_for_owp = if d2_rounds > 0 { &day2_ids } else { &active_ids };

                if !pool_for_owp.is_empty() {
                    let mut pool_with_owp: Vec<(usize, f64, i32)> = pool_for_owp.iter().map(|&id| {
                        let opps = &players[id].opponents;
                        let owp = if opps.is_empty() {
                            0.0
                        } else {
                            let mut sum_pct = 0.0;
                            for &opp_id in opps {
                                let opp = &players[opp_id];
                                let matches = opp.opponents.len().max(1) as f64;
                                let mut pct = (opp.match_points as f64) / (matches * 3.0);
                                if pct < 0.25 { pct = 0.25; }
                                if pct > 1.0 { pct = 1.0; }
                                sum_pct += pct;
                            }
                            sum_pct / (opps.len() as f64)
                        };
                        (id, owp, players[id].match_points)
                    }).collect();

                    // Sort by Match Points DESC, then OWP DESC
                    pool_with_owp.sort_by(|a, b| {
                        match b.2.cmp(&a.2) {
                            Ordering::Equal => b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal),
                            other => other,
                        }
                    });

                    let cut_size = top_cut.min(pool_with_owp.len());
                    top_players = pool_with_owp.iter().take(cut_size).map(|&t| t.0).collect();

                    for &id in &top_players {
                        local_topcut[players[id].deck_idx] += 1;
                    }
                }
            }

            // 5. Single Elimination Playoffs
            if !top_players.is_empty() {
                let mut standings = top_players;

                while standings.len() > 1 {
                    let half = standings.len() / 2;
                    let mut next_round = Vec::new();

                    let mut tc_p1s = standings[..half].to_vec();
                    let mut tc_p2s: Vec<usize> = standings[half..].to_vec();
                    tc_p2s.reverse(); // Fold the bracket (1v8, 2v7)

                    let mut unpaired = Vec::new();
                    if tc_p2s.len() > tc_p1s.len() {
                        unpaired.push(tc_p2s.pop().unwrap());
                    } else if tc_p1s.len() > tc_p2s.len() {
                        unpaired.push(tc_p1s.pop().unwrap());
                    }

                    for i in 0..tc_p1s.len() {
                        let p1_id = tc_p1s[i];
                        let p2_id = tc_p2s[i];
                        let p1_deck = players[p1_id].deck_idx;
                        let p2_deck = players[p2_id].deck_idx;

                        let win_prob = win_matrix[p1_deck][p2_deck];
                        let roll: f64 = rng.gen_range(0.0..1.0);

                        if roll < win_prob {
                            next_round.push(p1_id);
                        } else {
                            next_round.push(p2_id);
                        }
                    }

                    next_round.extend(unpaired);
                    standings = next_round;
                }

                if !standings.is_empty() {
                    local_champ[players[standings[0]].deck_idx] += 1;
                }
            }

            MCResult {
                total_initial: local_initial,
                total_day2: local_day2,
                total_topcut: local_topcut,
                total_champ: local_champ,
            }
        })
        .reduce(
            || MCResult {
                total_initial: vec![0; n_decks],
                total_day2: vec![0; n_decks],
                total_topcut: vec![0; n_decks],
                total_champ: vec![0; n_decks],
            },
            |mut acc, res| {
                for i in 0..n_decks {
                    acc.total_initial[i] += res.total_initial[i];
                    acc.total_day2[i] += res.total_day2[i];
                    acc.total_topcut[i] += res.total_topcut[i];
                    acc.total_champ[i] += res.total_champ[i];
                }
                acc
            },
        );

    Ok((
        final_result.total_initial,
        final_result.total_day2,
        final_result.total_topcut,
        final_result.total_champ,
    ))
}

#[pymodule]
fn tcg_engine(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_parallel_monte_carlo, m)?)?;
    Ok(())
}