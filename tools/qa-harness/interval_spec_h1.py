from src.ingestion.model import fit_h1_misty_variant

OBSERVATIONS = (
    [{"misty": 1, "hammer_variant": 0, "result": 1}] * 30
    + [{"misty": 0, "hammer_variant": 0, "result": 0}] * 5
    + [{"misty": 1, "hammer_variant": 1, "result": 0}] * 10
    + [{"misty": 0, "hammer_variant": 1, "result": 1}] * 10
)


def _fit_unadjusted(data):
    report = fit_h1_misty_variant(data)
    block = report["without_variant"]
    return block["beta"], block["interval"]


def _fit_controlled(data):
    report = fit_h1_misty_variant(data)
    block = report["with_variant"]
    return block["beta"], block["interval"]


CASES = [
    {"name": "H1 Misty coefficient, no variant control", "data": OBSERVATIONS, "fit": _fit_unadjusted},
    {"name": "H1 Misty coefficient, Dedenne + Enhanced Hammer controlled", "data": OBSERVATIONS, "fit": _fit_controlled},
]
