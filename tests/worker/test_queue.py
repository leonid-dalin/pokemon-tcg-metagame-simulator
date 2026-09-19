from unittest.mock import MagicMock

import pytest

from src.worker import queue


@pytest.mark.unit
def test_daily_pipeline_reraises_scraper_failures(monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise RuntimeError("scraper failed")

    monkeypatch.setattr(queue, "discover_live_matchup_urls", lambda session: [])
    monkeypatch.setattr(queue, "fetch_live_matchup_data", fail_fetch)

    with pytest.raises(RuntimeError, match="scraper failed"):
        queue.automated_daily_pipeline.call_local()


@pytest.mark.unit
def test_daily_pipeline_uses_discovered_live_urls(monkeypatch):
    discovered_urls = [
        "https://play.limitlesstcg.com/decks/dragapult-ex/matchups?format=standard&rotation=2026&set=PBL",
        "https://play.limitlesstcg.com/decks/greavard/matchups?format=standard&rotation=2026&set=PBL",
    ]
    fetch_mock = MagicMock(return_value=[])

    monkeypatch.setattr(queue, "discover_live_matchup_urls", lambda session: discovered_urls)
    monkeypatch.setattr(queue, "fetch_live_matchup_data", fetch_mock)

    with pytest.raises(ValueError, match="Scraper returned zero matchups"):
        queue.automated_daily_pipeline.call_local()

    fetch_mock.assert_called_once_with(discovered_urls, {})


@pytest.mark.unit
def test_daily_pipeline_reraises_discovery_failures(monkeypatch):
    discovery_error = RuntimeError("discovery failed")
    fetch_mock = MagicMock(return_value=[])

    monkeypatch.setattr(
        queue,
        "discover_live_matchup_urls",
        lambda session: (_ for _ in ()).throw(discovery_error),
    )
    monkeypatch.setattr(queue, "fetch_live_matchup_data", fetch_mock)

    with pytest.raises(RuntimeError, match="discovery failed"):
        queue.automated_daily_pipeline.call_local()

    fetch_mock.assert_not_called()