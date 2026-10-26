"""Run management and stage orchestration for the offline Best-60 experiment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.core.files import relative_path

from .context import build_context
from .corpus import extract_corpus
from .protocol import ProtocolMismatch, lock_protocol, require_protocol
from .splits import _utc_date, split_events


class RunnerError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _selection_validation_views(
    corpus: dict[str, Any],
    nominal_ids: list[str],
    validation_dates: dict[str, Any],
    test_dates: dict[str, Any],
    embargo_days: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the pre-test selection embargo without changing the nominal split."""
    def utc_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    dated_test = [parsed for value in test_dates.values()
                  if (parsed := utc_datetime(value)) is not None]
    if not dated_test:
        raise RunnerError(3, "test split has no valid event dates")
    first_test_date = min(parsed.date() for parsed in dated_test)
    boundary = datetime.combine(first_test_date, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=embargo_days)
    validation_date_by_id = {event_id: utc_datetime(validation_dates.get(event_id)) for event_id in nominal_ids}
    active_ids = {event_id for event_id, parsed in validation_date_by_id.items()
                  if parsed is not None and parsed < boundary}
    nominal_set = set(nominal_ids)
    active_views = [row for row in corpus["pairing_views"]
                    if row["event"] in active_ids]
    metadata = {
        "nominal_event_ids": sorted(nominal_set),
        "active_event_ids": sorted(active_ids),
        "embargoed_event_ids": sorted(nominal_set - active_ids),
        "boundary_utc_exclusive": boundary.isoformat(),
        "first_test_date_utc": first_test_date.isoformat(),
    }
    return active_views, metadata


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(2, f"cannot read JSON file {path}: {exc}") from exc


def _code_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    files = {}
    package = root / "src" / "ingestion" / "best60_research"
    for path in sorted(package.glob("*.py")):
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    revision = "unknown"
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"revision": revision, "files": files, "sha256": _sha({"revision": revision, "files": files})}


def _output_is_ignored(repo: Path, output: Path) -> bool:
    check_path = str(output)
    if not output.exists():
        check_path = check_path.rstrip("/\\") + os.sep
    try:
        result = subprocess.run(["git", "check-ignore", "--quiet", check_path], cwd=repo,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunnerError(2, f"cannot verify that output is ignored: {exc}") from exc
    return result.returncode == 0


def _recorded_field_path(run: Path, recorded: str) -> Path | None:
    """Find a field file recorded by name, so no local path is written into the manifest."""
    candidate = Path(recorded)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for base in (Path.cwd(), Path(__file__).resolve().parents[3], run):
        anchored = base / candidate
        if anchored.is_file():
            return anchored
    return None


def _validate_paths(database: Path, output: Path, config: Path) -> None:
    if not database.is_file():
        raise RunnerError(2, f"database does not exist: {database}")
    if not config.is_file():
        raise RunnerError(2, f"config does not exist: {config}")
    try:
        output_resolved = output.resolve()
        repo = Path(__file__).resolve().parents[3]
        output_resolved.relative_to(repo)
    except ValueError:
        return
    except OSError:
        return
    if not _output_is_ignored(repo, output_resolved):
        raise RunnerError(2, "output inside the repository must be ignored by Git")


def _config(path: Path) -> tuple[dict[str, Any], str]:
    value = _read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RunnerError(2, "config must be a schema_version 1 JSON object")
    try:
        return value, _sha(value)
    except (TypeError, ValueError) as exc:
        raise RunnerError(2, f"config cannot be hashed: {exc}") from exc


def _run_directory(output: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise RunnerError(2, "resume must be a run ID, not a path")
    run = output.resolve() / run_id
    if not run.is_dir():
        raise RunnerError(2, f"run does not exist: {run_id}")
    return run


def _manifest(run: Path, source_hash: str, snapshot_fingerprint: str | None,
              config_hash: str) -> dict[str, Any]:
    value = _read_json(run / "manifest.json")
    if value.get("source_sha256") != source_hash:
        raise RunnerError(2, "source database fingerprint differs from this run")
    if value.get("snapshot_fingerprint") != snapshot_fingerprint:
        raise RunnerError(2, "source snapshot or WAL fingerprint differs from this run")
    if value.get("config_sha256") != config_hash:
        raise RunnerError(2, "config fingerprint differs from this run")
    current_code = _code_fingerprint()
    if current_code["sha256"] != value.get("code", {}).get("sha256"):
        raise RunnerError(2, "research code fingerprint differs from this run")
    stored_splits = _read_json(run / "splits.json")
    if stored_splits != value.get("splits") or _sha(stored_splits) != value.get("split_sha256"):
        raise RunnerError(2, "saved split IDs or dates do not match the audited fingerprint")
    return value


def _normalised_as_of(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunnerError(2, "--as-of must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _read_field(path: str | None) -> tuple[dict[str, float] | None, str | None]:
    if path is None:
        return None, None
    source = Path(path)
    if not source.is_file():
        raise RunnerError(2, f"field file does not exist: {source}")
    raw = _read_json(source)
    if not isinstance(raw, dict) or not raw:
        raise RunnerError(2, "field file must be a non-empty JSON mapping of archetype IDs to weights")
    try:
        values = {str(key): float(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise RunnerError(2, f"field file contains non-numeric weights: {exc}") from exc
    if any(not np.isfinite(value) or value < 0 for value in values.values()) or sum(values.values()) <= 0:
        raise RunnerError(2, "field weights must be finite, non-negative and have positive total mass")
    total = sum(values.values())
    return {key: value / total for key, value in sorted(values.items())}, hashlib.sha256(source.read_bytes()).hexdigest()


def _eligible_ids(splits: dict[str, list[str]], *, include_test: bool = False) -> list[str]:
    names = ("development", "validation", "test") if include_test else ("development", "validation")
    return sorted(event for name in names for event in splits[name])


def _read_metadata(database: Path, as_of: str | None = None) -> tuple[dict[str, Any], dict[str, list[str]]]:
    corpus = _extract(database, include_event_ids=[])
    events = corpus["events"]
    if as_of is not None:
        cutoff = datetime.fromisoformat(as_of)
        filtered = []
        for event in events:
            if not event.get("date"):
                continue
            event_date = datetime.fromisoformat(str(event["date"]).replace("Z", "+00:00"))
            if event_date.tzinfo is None:
                event_date = event_date.replace(tzinfo=timezone.utc)
            if event_date.astimezone(timezone.utc) <= cutoff:
                filtered.append(event)
        events = filtered
    if not events:
        raise RunnerError(3, "database contains no tournament events")
    try:
        splits = split_events(events)
    except ValueError as exc:
        raise RunnerError(3, f"event metadata cannot be split: {exc}") from exc
    if not splits["development"] or not splits["validation"] or not splits["test"]:
        raise RunnerError(3, "all three chronological splits need at least one event")
    return corpus, splits


def _check_budget(start: float, maximum: int | None) -> None:
    if maximum is not None and time.monotonic() - start >= maximum:
        raise RunnerError(4, "stage exceeded --max-seconds; completed checkpoints remain resumable")


def _extract(database: Path, **kwargs: Any) -> dict[str, Any]:
    try:
        return extract_corpus(database, **kwargs)
    except ValueError as exc:
        raise RunnerError(3, f"database invariant failed: {exc}") from exc


def run_stage(args: Any) -> dict[str, Any]:
    database, output, config_path = Path(args.database), Path(args.output), Path(args.config)
    _validate_paths(database, output, config_path)
    protocol, config_hash = _config(config_path)
    field_values, field_hash = _read_field(args.field_file)
    if args.max_seconds is not None and args.max_seconds < 1:
        raise RunnerError(2, "--max-seconds must be positive")
    start = time.monotonic()
    stage = args.stage
    as_of = _normalised_as_of(args.as_of)
    if stage == "audit":
        if args.resume:
            raise RunnerError(2, "audit creates a new run and does not accept --resume")
        metadata, splits = _read_metadata(database, as_of)
        _check_budget(start, args.max_seconds)
        included = _eligible_ids(splits)
        corpus = _extract(database, include_event_ids=included)
        if not corpus["lists"] or not corpus["pairings"]:
            raise RunnerError(3, "development and validation splits contain no extractable lists or pairings")
        split_payload = {name: {"event_ids": ids,
                                "dates": {event["id"]: event["date"] for event in metadata["events"]
                                          if event["id"] in set(ids)}}
                         for name, ids in splits.items()}
        split_hash = _sha(split_payload)
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        run = output.resolve() / run_id
        try:
            run.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RunnerError(2, f"run directory already exists: {run_id}") from exc
        manifest = {"schema_version": 1, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "database": relative_path(str(database)), "source_sha256": corpus["audit"]["source_sha256"],
                    "source_files": corpus["audit"].get("source_files", {}),
                    "snapshot_fingerprint": corpus["audit"].get("snapshot_fingerprint"),
                    "config": relative_path(str(config_path)), "config_sha256": config_hash,
                    "protocol": protocol, "split_sha256": split_hash, "splits": split_payload,
                    "included_event_ids": included, "code": _code_fingerprint(),
                    "archetype_id": args.archetype_id or "kangaskhan-bouffalant",
                    "field_file": Path(args.field_file).name if args.field_file else None,
                    "field_file_sha256": field_hash, "field_values": field_values,
                    "as_of": as_of}
        _atomic_json(run / "manifest.json", manifest)
        _atomic_json(run / "audit.json", {"source": corpus["audit"],
                    "extraction": {"lists": corpus["audit"]["lists"],
                                   "pairing_exclusions": corpus["audit"]["pairing_exclusions"],
                                   "pairing_rows_non_test": len(corpus["pairings"]),
                                   "eligible_pairing_views_non_test": len(corpus["pairing_views"])},
                    "test_outcomes_read": False, "test_event_count": len(splits["test"])})
        _atomic_json(run / "splits.json", split_payload)
        _atomic_json(run / "state.json", {"stage": "audited", "completed": ["audit"]})
        return {"status": "complete", "stage": stage, "run_id": run_id, "run_directory": relative_path(str(run))}

    if not args.resume:
        raise RunnerError(2, f"stage {stage} requires --resume RUN_ID")
    run = _run_directory(output, args.resume)
    initial_manifest = _read_json(run / "manifest.json")
    if as_of is not None and as_of != initial_manifest.get("as_of"):
        raise RunnerError(2, "--as-of differs from the audited run")
    as_of = initial_manifest.get("as_of")
    source_metadata, _ = _read_metadata(database, as_of)
    source_hash = source_metadata["audit"]["source_sha256"]
    manifest = _manifest(run, source_hash, source_metadata["audit"].get("snapshot_fingerprint"), config_hash)
    if args.archetype_id and args.archetype_id != manifest.get("archetype_id"):
        raise RunnerError(2, "archetype ID differs from the audited run")
    if args.field_file:
        if field_hash != manifest.get("field_file_sha256"):
            raise RunnerError(2, "field file differs from the audited run")
    elif manifest.get("field_file"):
        saved_path = _recorded_field_path(run, manifest["field_file"])
        if saved_path is None:
            raise RunnerError(2, "resume needs --field-file: the saved field file "
                                 f"{manifest['field_file']!r} is not in the working directory, "
                                 "the repository or the run directory")
        saved_values, saved_hash = _read_field(str(saved_path))
        if saved_hash != manifest.get("field_file_sha256") or saved_values != manifest.get("field_values"):
            raise RunnerError(2, "saved field file has changed since audit")
    state = _read_json(run / "state.json")
    completed = set(state.get("completed", []))
    if stage == "validate":
        if "audit" not in completed:
            raise RunnerError(2, "validation requires a completed audit stage")
        splits = {name: row["event_ids"] for name, row in manifest["splits"].items()}
        corpus = _extract(database, include_event_ids=_eligible_ids(splits))
        validation_dates = [manifest["splits"]["validation"]["dates"][event]
                            for event in splits["validation"] if manifest["splits"]["validation"]["dates"].get(event)]
        cutoff = min(validation_dates) if validation_dates else None
        if cutoff is None:
            raise RunnerError(3, "validation split has no valid event dates")
        context = build_context(corpus, cutoff=cutoff, embargo_days=int(protocol["time"]["primary_embargo_days"]),
                                min_slot_lists=int(protocol["features"]["slot_minimum_lists"]))
        validation_views, selection_window = _selection_validation_views(
            corpus,
            splits["validation"],
            manifest["splits"]["validation"]["dates"],
            manifest["splits"]["test"]["dates"],
            int(protocol["time"]["primary_embargo_days"]),
        )
        from .experiment import validate_experiment

        validation = validate_experiment(
            context, validation_views, protocol, archetype_id=manifest["archetype_id"],
            checkpoint_path=run / "validation-checkpoint.json", max_seconds=args.max_seconds,
        )
        selection_window["training_cutoff_utc"] = cutoff
        validation["run_cache_key"] = _sha({"source": manifest["source_sha256"],
                                             "snapshot": manifest["snapshot_fingerprint"],
                                             "split": manifest["split_sha256"], "config": config_hash,
                                             "code": manifest["code"]["sha256"], "cutoff": cutoff,
                                             "archetype_id": manifest["archetype_id"],
                                             "selection_window": selection_window})
        validation["selection_window"] = selection_window
        _atomic_json(run / "validation.json", validation)
        _check_budget(start, args.max_seconds)
        if validation["status"] in {"complete", "inconclusive", "not_identified"}:
            completed.add("validate")
        state_value = {"stage": "validated" if "validate" in completed else "audited",
                       "completed": sorted(completed)}
        if "validate" in completed:
            state_value["validation_sha256"] = hashlib.sha256((run / "validation.json").read_bytes()).hexdigest()
        _atomic_json(run / "state.json", state_value)
        return {"status": validation["status"], "stage": stage, "run_id": args.resume,
                "benchmarks": validation["benchmarks"]}

    if stage == "lock":
        if "validate" not in completed:
            raise RunnerError(2, "lock requires a completed validation stage")
        validation_file_hash = hashlib.sha256((run / "validation.json").read_bytes()).hexdigest()
        if state.get("validation_sha256") != validation_file_hash:
            raise RunnerError(2, "validation output changed since it completed")
        validation = _read_json(run / "validation.json")
        validation_dates = manifest["splits"]["validation"]["dates"]
        validation_ids = manifest["splits"]["validation"]["event_ids"]
        nominal_validation_dates = [validation_dates[event_id] for event_id in validation_ids
                                    if validation_dates.get(event_id)]
        cutoff = min(nominal_validation_dates) if nominal_validation_dates else None
        if cutoff is None:
            raise RunnerError(3, "validation split has no valid event dates")
        _, expected_selection_window = _selection_validation_views(
            {"pairing_views": []},
            validation_ids,
            validation_dates,
            manifest["splits"]["test"]["dates"],
            int(protocol["time"]["primary_embargo_days"]),
        )
        expected_selection_window["training_cutoff_utc"] = cutoff
        if validation.get("selection_window") != expected_selection_window:
            raise RunnerError(2, "validation selection window does not match audited split dates and embargo")
        expected_cache_key = _sha({"source": manifest["source_sha256"],
                                   "snapshot": manifest["snapshot_fingerprint"],
                                   "split": manifest["split_sha256"], "config": config_hash,
                                   "code": manifest["code"]["sha256"],
                                   "cutoff": cutoff,
                                   "archetype_id": manifest["archetype_id"],
                                   "selection_window": expected_selection_window})
        if validation.get("run_cache_key") != expected_cache_key:
            raise RunnerError(2, "validation output does not match the audited data, protocol, code and cutoff")
        payload = {"config_sha256": config_hash, "source_sha256": manifest["source_sha256"],
                   "snapshot_fingerprint": manifest.get("snapshot_fingerprint"),
                   "split_sha256": manifest["split_sha256"], "code": manifest["code"],
                   "splits": manifest["splits"], "archetype_id": manifest["archetype_id"],
                   "validation_sha256": hashlib.sha256((run / "validation.json").read_bytes()).hexdigest(),
                   "study_protocol": protocol}
        lock_hash = lock_protocol(run / "protocol-lock.json", payload)
        completed.add("lock")
        _atomic_json(run / "state.json", {"stage": "locked", "completed": sorted(completed),
                                          "protocol_lock_sha256": lock_hash})
        return {"status": "complete", "stage": stage, "run_id": args.resume, "protocol_lock_sha256": lock_hash}

    if stage in {"evaluate", "simulate", "report"}:
        if "lock" not in completed:
            raise RunnerError(2, f"{stage} requires a locked protocol")
        locked = _read_json(run / "protocol-lock.json")
        expected_lock = {"config_sha256": config_hash, "source_sha256": manifest["source_sha256"],
                         "snapshot_fingerprint": manifest.get("snapshot_fingerprint"),
                         "split_sha256": manifest["split_sha256"], "code": _code_fingerprint(),
                         "splits": manifest["splits"], "archetype_id": manifest["archetype_id"],
                         "validation_sha256": hashlib.sha256((run / "validation.json").read_bytes()).hexdigest(),
                         "study_protocol": protocol}
        require_protocol(run / "protocol-lock.json", expected_lock)
        if stage == "evaluate":
            if "evaluate" in completed:
                return {"status": "complete", "stage": stage, "run_id": args.resume}
            result = _evaluate_locked(database, manifest, protocol, manifest.get("field_values"),
                                      run / "evaluation-progress.json", args.max_seconds, stage_start=start)
            _atomic_json(run / "evaluation.json", result)
            _check_budget(start, args.max_seconds)
            if result.get("status") in {"complete", "inconclusive", "not_identified"}:
                completed.add("evaluate")
            _atomic_json(run / "state.json", {"stage": "evaluated" if "evaluate" in completed else "locked",
                                              "completed": sorted(completed),
                                              "protocol_lock_sha256": locked["protocol_hash"]})
            return {"status": result.get("status", "complete"), "stage": stage, "run_id": args.resume}
        if stage == "simulate":
            if "simulate" in completed:
                return {"status": "complete", "stage": stage, "run_id": args.resume}
            result = _simulate_locked(database, manifest, protocol, run, start, args.max_seconds)
            _atomic_json(run / "simulation.json", result)
            _check_budget(start, args.max_seconds)
            if result.get("status") in {"complete", "inconclusive", "not_identified"}:
                completed.add("simulate")
            _atomic_json(run / "state.json", {"stage": "simulated" if "simulate" in completed else "locked",
                                              "completed": sorted(completed),
                                              "protocol_lock_sha256": locked["protocol_hash"]})
            return {"status": result.get("status", "complete"), "stage": stage, "run_id": args.resume}
        summary = {"run_id": args.resume, "protocol_lock_sha256": locked["protocol_hash"],
                   "evaluation": _read_json(run / "evaluation.json") if (run / "evaluation.json").exists() else None,
                   "simulation": _read_json(run / "simulation.json") if (run / "simulation.json").exists() else None}
        if "evaluate" not in completed or "simulate" not in completed:
            raise RunnerError(4, "report requires completed evaluation and simulation stages")
        report_status = "inconclusive" if any(
            value.get("status") in {"inconclusive", "not_identified"}
            for value in (summary["evaluation"], summary["simulation"]) if value
        ) else "complete"
        summary["status"] = report_status
        _atomic_json(run / "report.json", summary)
        completed.add("report")
        _atomic_json(run / "state.json", {"stage": "reported", "completed": sorted(completed),
                                          "protocol_lock_sha256": locked["protocol_hash"]})
        return {"status": report_status, "stage": stage, "run_id": args.resume, "report": str(run / "report.json")}
    raise RunnerError(2, f"unknown stage: {stage}")



def _evaluate_locked(database: Path, manifest: dict[str, Any], protocol: dict[str, Any], field_override: dict[str, float] | None,
                     output_path: Path, maximum: int | None, *, stage_start: float) -> dict[str, Any]:
    splits = {name: row["event_ids"] for name, row in manifest["splits"].items()}
    included = _eligible_ids(splits, include_test=True)
    corpus = _extract(database, include_event_ids=included)
    test_ids = set(splits["test"])
    test_views = [row for row in corpus["pairing_views"] if row["event"] in test_ids]
    dates = [manifest["splits"]["test"]["dates"][event] for event in splits["test"]
             if manifest["splits"]["test"]["dates"].get(event)]
    if not dates:
        raise RunnerError(3, "test split has no dated events")
    context = build_context(corpus, cutoff=min(dates), embargo_days=int(protocol["time"]["primary_embargo_days"]),
                            min_slot_lists=int(protocol["features"]["slot_minimum_lists"]))
    validation = _read_json(output_path.with_name("validation.json"))
    from .experiment import evaluate_experiment

    remaining = None if maximum is None else max(0.001, float(maximum) - (time.monotonic() - stage_start))
    return evaluate_experiment(context, test_views, validation, protocol,
                               archetype_id=manifest["archetype_id"], field_override=field_override,
                               output_path=output_path, max_seconds=remaining)


def _simulate_locked(database: Path, manifest: dict[str, Any], protocol: dict[str, Any], run: Path,
                     start: float, maximum: int | None) -> dict[str, Any]:
    from .experiment import prepare_simulation_context
    from .simulation import run_simulation

    splits = {name: row["event_ids"] for name, row in manifest["splits"].items()}
    test_dates = manifest["splits"]["test"]["dates"]
    def utc_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    test_utc_dates = [parsed for event_id in splits["test"]
                      if (parsed := utc_datetime(test_dates.get(event_id))) is not None]
    test_blocks = sorted({_utc_date(value) for value in test_dates.values() if _utc_date(value) is not None})
    minimum_clusters = int(protocol.get("h1", {}).get("minimum_independent_event_clusters", 20))
    embargo_days = int(protocol["time"]["primary_embargo_days"])
    pretest_ids = set(splits["development"]) | set(splits["validation"])
    cutoff = min(test_utc_dates) if test_utc_dates else None
    if cutoff is None:
        raise RunnerError(3, "test split has no dated events")
    boundary = (cutoff - timedelta(days=embargo_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    training_blocks = sorted({
        block for event_id in pretest_ids
        if (parsed := utc_datetime(
            manifest["splits"]["development"]["dates"].get(event_id)
            if event_id in splits["development"] else manifest["splits"]["validation"]["dates"].get(event_id)
        )) is not None and parsed < boundary and (block := _utc_date(parsed)) is not None
    })
    n_training_blocks = len(training_blocks)
    if n_training_blocks >= 5:
        train_blocks = max(1, int(0.6 * n_training_blocks))
        selection_blocks = max(1, int(0.2 * n_training_blocks))
        measurement_upper_bound = max(0, n_training_blocks - train_blocks - selection_blocks)
        measurement_split_unit = "atomic_utc_date"
    else:
        # The simulation falls back to event-level randomisation below five date blocks.
        # Such a split cannot create more distinct measurement date clusters than exist.
        measurement_upper_bound = n_training_blocks
        measurement_split_unit = "event_fallback_bounded_by_date_blocks"
    if measurement_upper_bound < minimum_clusters:
        reason = (f"the pre-test training context has {n_training_blocks} atomic UTC date blocks; "
                  f"the configured 60/20/20 simulation split can provide at most "
                  f"{measurement_upper_bound} measurement date blocks, below the H1 minimum "
                  f"of {minimum_clusters}")
        replicates = int(protocol.get("simulation", {}).get("full_replicates_per_effect", 500))
        effects = protocol.get("simulation", {}).get("effects_log_odds", (0.0, 0.08, 0.2, 0.4))
        effect_status = {format(float(effect), ".12g"): {
            "status": "not_run_design_inconclusive", "effect": float(effect),
            "requested_replicates": replicates, "completed_replicates": 0,
        } for effect in effects}
        target_archetype = str(manifest["archetype_id"])
        return {
            "status": "inconclusive",
            "simulation_design_status": "inconclusive_design",
            "reason": reason,
            "preflight": {
                "measurement_event_count": len(splits["test"]),
                "test_split_date_block_count": len(test_blocks),
                "pretest_training_date_block_count": n_training_blocks,
                "measurement_date_block_upper_bound": measurement_upper_bound,
                "measurement_split_unit": measurement_split_unit,
                "measurement_date_blocks": training_blocks,
                "minimum_independent_clusters": minimum_clusters,
                "replicates_executed": 0,
                "test_outcomes_read": False,
            },
            "benchmarks": {
                "pooled": {"status": "inconclusive_design", "effects": dict(effect_status)},
                target_archetype: {"status": "inconclusive_design", "effects": dict(effect_status)},
            },
        }
    corpus = _extract(database, include_event_ids=_eligible_ids(splits))
    dated_test_events = [test_dates[event] for event in splits["test"] if test_dates.get(event)]
    if not dated_test_events:
        raise RunnerError(3, "test split has no dated events")
    context = build_context(corpus, cutoff=cutoff.isoformat(),
                            embargo_days=embargo_days,
                            min_slot_lists=int(protocol["features"]["slot_minimum_lists"]))
    simulation = protocol["simulation"]
    estimators = protocol["estimators"]
    replicates = int(simulation.get("full_replicates_per_effect", 500))
    effects = tuple(float(value) for value in simulation.get("effects_log_odds", (0.0, 0.08, 0.2, 0.4)))
    rank_grid = tuple(int(value) for value in simulation.get("full_rank_grid", estimators.get("fm_ranks", (2, 4, 8))))
    seed_grid = tuple(int(value) for value in simulation.get("full_seed_grid", estimators.get("fm_seeds", (1312, 1313, 1314))))
    penalties = tuple(float(value) for value in simulation.get(
        "full_interaction_penalty_grid", estimators.get("interaction_penalties", (1e-5, 1e-4, 1e-3))))
    if not effects or not rank_grid or not seed_grid or not penalties:
        raise RunnerError(2, "full simulation requires non-empty effects, ranks, seeds and penalties")
    configured_budget = float(simulation.get("full_wall_clock_seconds", 86400))
    stage_budget = configured_budget if maximum is None else min(configured_budget, float(maximum))
    deadline = start + stage_budget
    target_archetype = str(manifest["archetype_id"])
    benchmarks = {"pooled": prepare_simulation_context(context),
                  target_archetype: prepare_simulation_context(context, archetype_id=target_archetype)}
    results: dict[str, Any] = {}
    for benchmark, arrays in benchmarks.items():
        effect_results: dict[str, Any] = {}
        for effect in effects:
            effect_key = format(effect, ".12g")
            checkpoint = run / "simulation-checkpoints" / f"{benchmark}-{effect_key}.json"
            remaining = max(0.001, deadline - time.monotonic())
            result = run_simulation(
                arrays,
                replicates=replicates,
                base_seed=int(seed_grid[0]),
                effect=effect,
                mode="package",
                checkpoint=checkpoint,
                max_seconds=remaining,
                rank_grid=rank_grid,
                interaction_penalties=penalties,
                seed_grid=seed_grid,
                max_iter=int(estimators.get("max_iterations", 5000)),
            )
            effect_results[effect_key] = result
            if result.get("status") != "complete" or time.monotonic() >= deadline:
                break
        for effect in effects:
            effect_key = format(effect, ".12g")
            if effect_key not in effect_results:
                effect_results[effect_key] = {
                    "status": "incomplete", "effect": effect,
                    "requested_replicates": replicates, "completed_replicates": 0,
                    "checkpoint": str(run / "simulation-checkpoints" / f"{benchmark}-{effect_key}.json"),
                }
        results[benchmark] = {
            "status": "complete" if all(value.get("status") == "complete"
                                         and value.get("completed_replicates") == replicates
                                         for value in effect_results.values()) else "incomplete",
            "effects": effect_results,
            "scope": "pooled eligible views" if benchmark == "pooled" else "focal eligible views with within-pair weights normalised",
        }
        if time.monotonic() >= deadline:
            break
    for benchmark in ("pooled", target_archetype):
        if benchmark not in results:
            results[benchmark] = {
                "status": "incomplete",
                "scope": "pooled eligible views" if benchmark == "pooled" else "focal eligible views with within-pair weights normalised",
                "effects": {format(effect, ".12g"): {
                    "status": "incomplete", "effect": effect,
                    "requested_replicates": replicates, "completed_replicates": 0,
                    "checkpoint": str(run / "simulation-checkpoints" / f"{benchmark}-{format(effect, '.12g')}.json"),
                } for effect in effects},
            }
    status = "complete" if all(value["status"] == "complete" for value in results.values()) else "incomplete"
    return {
        "status": status,
        "mode": "package",
        "requested_replicates_per_effect": replicates,
        "effects_log_odds": list(effects),
        "rank_grid": list(rank_grid),
        "seed_grid": list(seed_grid),
        "interaction_penalties": list(penalties),
        "benchmarks": results,
        "simulation_context": {
            "cutoff": context["cutoff"], "embargo_days": context["embargo_days"],
            "training_event_count": context["coverage"]["training_event_count"],
            "training_pairings": context["coverage"]["eligible_pairing_count"],
            "event_block_count": len(context["event_blocks"]),
            "feature_count": len(context["slot_names"]),
        },
        "preflight": {
            "test_split_date_block_count": len(test_blocks),
            "pretest_training_date_block_count": n_training_blocks,
            "measurement_date_block_upper_bound": measurement_upper_bound,
            "measurement_split_unit": measurement_split_unit,
            "minimum_independent_clusters": minimum_clusters,
            "measurement_date_blocks": training_blocks,
        },
        "checkpoint_each_replicate": True,
    }
