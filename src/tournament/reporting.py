from typing import Any, Dict


def build_bdif_report(
    monte_carlo_result: Dict[str, Any],
    best60_recommendations: Dict[str, Any],
    h1_report: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "metrics": monte_carlo_result["metrics"],
        "ranked_metrics": monte_carlo_result["ranked_metrics"],
        "insufficient_data": monte_carlo_result["insufficient_data"],
        "matchup_panel": monte_carlo_result["matchup_panel"],
        "best60_recommendations": best60_recommendations or {},
        "h1_report": h1_report or {},
    }