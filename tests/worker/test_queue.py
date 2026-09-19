import pytest

from src.worker import queue


@pytest.mark.unit
def test_daily_pipeline_reraises_scraper_failures(monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise RuntimeError("scraper failed")

    monkeypatch.setattr(queue, "fetch_live_matchup_data", fail_fetch)

    with pytest.raises(RuntimeError, match="scraper failed"):
        queue.automated_daily_pipeline.call_local()