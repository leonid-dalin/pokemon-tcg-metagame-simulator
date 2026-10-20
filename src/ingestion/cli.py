from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.ingestion.labs import LabsClient, LabsEvent, collect_event, import_bundles, validate_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import in-person Limitless Labs events into a snapshot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Fetch event bundles into a bundle directory")
    collect.add_argument("--event", action="append", required=True, help="LABS_ID:PUBLIC_ID:CITY:ISO_DATE, repeatable")
    collect.add_argument("--bundle-dir", required=True, help="Directory for fetched bundles")
    collect.add_argument("--min-delay", type=float, default=1.0, help="Minimum seconds between requests, shared across workers")
    collect.add_argument("--workers", type=int, default=4, help="Concurrent decklist fetches")

    validate = subparsers.add_parser("validate", help="Check fetched bundles against the store contract")
    validate.add_argument("--bundle-dir", required=True)

    apply = subparsers.add_parser("apply", help="Write verified bundles into a snapshot database")
    apply.add_argument("--database", required=True)
    apply.add_argument("--bundle-dir", required=True)
    apply.add_argument("--mapping", help="Archetype mapping file; defaults to the checked-in map")
    return parser


def _load_bundles(bundle_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(bundle_dir.glob("*/bundle.json"))
    if not paths:
        raise SystemExit(f"no bundles found under {bundle_dir}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def main() -> int:
    args = _parser().parse_args()

    if args.command == "collect":
        client = LabsClient(min_delay=args.min_delay, workers=args.workers)
        results = []
        for spec in args.event:
            event = LabsEvent.parse(spec)
            bundle = collect_event(client, event, Path(args.bundle_dir) / event.labs_id)
            results.append({"labs_id": event.labs_id, "public_id": event.public_id, "players": len(bundle["standings"]), "pairings": len(bundle["pairings"])})
        print(json.dumps({"collected": results}, ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "validate":
        reports = [validate_bundle(bundle) for bundle in _load_bundles(Path(args.bundle_dir))]
        print(json.dumps({"reports": reports}, ensure_ascii=False, sort_keys=True))
        return 0 if all(report["ok"] for report in reports) else 1

    deck_mapping = None
    if args.mapping:
        deck_mapping = {str(key): value for key, value in json.loads(Path(args.mapping).read_text(encoding="utf-8")).items()}
    counts = import_bundles(args.database, _load_bundles(Path(args.bundle_dir)), deck_mapping=deck_mapping)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
