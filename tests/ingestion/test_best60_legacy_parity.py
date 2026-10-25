from src.ingestion.best60 import build_best60


def test_legacy_small_corpus_keeps_legal_vote():
    row = {"event": "e0", "player": "p", "deck_id": "a", "placing": 1,
           "wins": 1, "losses": 1, "players": 16,
           "date": "2026-09-01T00:00:00Z",
           "decklist": {"trainer": [{"name": "Core", "count": 1}],
                        "energy": [{"name": "Grass Energy", "count": 59}]}}
    result = build_best60("A", [row], {}, {("e0", "p"): (1, 1)}, "novel",
                          games=(), field={"B": 1.0}, as_of="2026-10-01T00:00:00Z")
    assert result["total_copies"] == 60
    assert {entry["card"]: entry["copies"] for entry in result["cards"]} == {"Core": 1, "Grass Energy": 59}
    assert result["swaps"] == []
    assert result["observational"] is True
