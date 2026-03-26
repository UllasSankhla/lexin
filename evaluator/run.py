#!/usr/bin/env python3
"""Evaluator CLI — ingest transcripts, replay, evaluate, and generate suggestions.

Usage:
    python run.py [transcripts_dir] [--config PATH] [--skip-replay] [--verbose]

Arguments:
    transcripts_dir  Directory containing .txt or .csv transcript files.
                     Defaults to ./transcripts/

Options:
    --config PATH    Path to a Python module exporting MOCK_CONFIG dict.
                     Defaults to config/mock_config.py (MOCK_CONFIG).
    --skip-replay    Skip replay (re-use existing replay data if available).
    --verbose        Enable DEBUG logging.

The evaluator runs each transcript through the full pipeline:
  1. Ingest   — parse transcript into TestCase
  2. Replay   — drive the real agents with caller utterances (no API calls)
  3. Evaluate — compare AI responses to human responses turn-by-turn
  4. Synthesise — cluster findings and generate improvement suggestions
  5. Report   — write JSON + Markdown reports to ./reports/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

# ── Package root on path ──────────────────────────────────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Load .env before any data-plane imports touch os.environ ─────────────────
def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_load_dotenv(_HERE / ".env")

from evaluator.ingestion  import parse_transcript, build_test_case
from evaluator.replay     import replay_test_case
from evaluator.evaluation import evaluate_replay
from evaluator.synthesis  import synthesise_suggestions, build_batch_report
from evaluator.reporting  import (
    write_conversation_report_json,
    write_conversation_report_md,
    write_conversation_report_html,
    write_batch_report_json,
    write_batch_report_md,
    write_batch_report_html,
)

_PROCESSED_PATH = _HERE / "data" / "processed_hashes.json"

logger = logging.getLogger(__name__)


# ── Deduplication by file hash ────────────────────────────────────────────────

def _load_hashes() -> set[str]:
    if _PROCESSED_PATH.exists():
        try:
            return set(json.loads(_PROCESSED_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return set()


def _save_hashes(hashes: set[str]) -> None:
    _PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROCESSED_PATH.write_text(
        json.dumps(sorted(hashes), indent=2),
        encoding="utf-8",
    )


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_config(config_path: str | None) -> dict:
    if config_path:
        spec = importlib.util.spec_from_file_location("_eval_config", config_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "MOCK_CONFIG")
    # Default
    from config.mock_config import MOCK_CONFIG
    return MOCK_CONFIG


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AI receptionist transcripts and generate improvement suggestions."
    )
    parser.add_argument(
        "transcripts_dir",
        nargs="?",
        default=str(_HERE / "transcripts"),
        help="Directory containing transcript files (.txt or .csv)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a Python module exporting MOCK_CONFIG dict",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip the replay phase (not yet implemented — reserved for future caching)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    config = _load_config(args.config)
    transcripts_dir = Path(args.transcripts_dir)

    if not transcripts_dir.exists():
        logger.error("Transcripts directory not found: %s", transcripts_dir)
        sys.exit(1)

    transcript_files = sorted(
        p for p in transcripts_dir.iterdir()
        if p.suffix.lower() in (".txt", ".csv") and p.is_file()
    )

    if not transcript_files:
        logger.error("No .txt or .csv files found in %s", transcripts_dir)
        sys.exit(1)

    logger.info("Found %d transcript(s) in %s", len(transcript_files), transcripts_dir)

    processed_hashes = _load_hashes()
    reports = []
    skipped = 0

    for transcript_path in transcript_files:
        logger.info("── Processing: %s", transcript_path.name)

        # ── Ingest ────────────────────────────────────────────────────────────
        try:
            raw = parse_transcript(transcript_path)
        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", transcript_path.name, exc)
            skipped += 1
            continue

        if raw.file_hash in processed_hashes:
            logger.info("Skipping %s — already processed (hash match)", transcript_path.name)
            skipped += 1
            continue

        try:
            test_case = build_test_case(raw)
        except Exception as exc:
            logger.error("TestCase build failed for %s: %s", transcript_path.name, exc)
            skipped += 1
            continue

        if not test_case.paired_exchanges:
            logger.warning(
                "No paired exchanges in %s — skipping evaluation", transcript_path.name
            )
            skipped += 1
            continue

        logger.info(
            "Ingested %s | %d caller turns | %d pairs",
            transcript_path.name,
            len(test_case.caller_turns),
            len(test_case.paired_exchanges),
        )

        # ── Replay ────────────────────────────────────────────────────────────
        logger.info("Replaying %s through agent pipeline...", test_case.conversation_id)
        try:
            replay_result = replay_test_case(test_case, config)
        except Exception as exc:
            logger.error("Replay failed for %s: %s", transcript_path.name, exc, exc_info=True)
            skipped += 1
            continue

        if replay_result.error:
            logger.warning(
                "Replay completed with error for %s: %s",
                transcript_path.name, replay_result.error,
            )

        logger.info(
            "Replay done: %d turns replayed", len(replay_result.replay_turns)
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        logger.info("Evaluating %s...", test_case.conversation_id)
        try:
            report = evaluate_replay(replay_result, config)
        except Exception as exc:
            logger.error("Evaluation failed for %s: %s", transcript_path.name, exc)
            skipped += 1
            continue

        logger.info(
            "Evaluation done: %d findings | overall score=%.1f",
            len(report.findings), report.overall_quality_score,
        )

        reports.append(report)
        processed_hashes.add(raw.file_hash)

        # Write per-conversation report immediately
        write_conversation_report_json(report)
        write_conversation_report_md(report)
        write_conversation_report_html(report)

    if not reports:
        logger.warning("No conversations evaluated — nothing to synthesise.")
        _save_hashes(processed_hashes)
        return

    # ── Synthesise ────────────────────────────────────────────────────────────
    logger.info("Synthesising suggestions across %d conversation(s)...", len(reports))
    try:
        suggestions = synthesise_suggestions(reports)
    except Exception as exc:
        logger.error("Synthesis failed: %s", exc, exc_info=True)
        suggestions = []

    logger.info("%d suggestion(s) generated", len(suggestions))

    # Attach suggestions back to each report for full context
    for report in reports:
        report.suggestions = suggestions

    # ── Batch report ──────────────────────────────────────────────────────────
    batch = build_batch_report(reports, suggestions, skipped=skipped)
    write_batch_report_json(batch)
    write_batch_report_md(batch)
    write_batch_report_html(batch)

    _save_hashes(processed_hashes)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Evaluator run complete")
    print(f"  Conversations processed : {len(reports)}")
    print(f"  Conversations skipped   : {skipped}")
    print(f"  Suggestions generated   : {len(suggestions)}")
    print(f"  Reports written to      : {_HERE / 'reports'}/")
    print("=" * 60)

    for s in suggestions:
        print(f"  [{s.category.value.upper():7s}] {s.title}")
        print(f"           priority={s.composite_score:.1f}  freq={s.frequency}")


if __name__ == "__main__":
    main()
