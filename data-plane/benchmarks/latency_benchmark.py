"""
LLM latency benchmark for the voice AI data-plane pipeline.

Measures wall-clock time from utterance receipt to agent speak output across
three core paths:

  - data_collection   — collect first name, last name, email (confirm each)
  - narrative_collection — caller describes their matter across multiple turns
  - scheduling        — present slots, caller picks, confirms (no Calendly call)
  - faq               — single question, multi-question, and legal-deflect turns

Also benchmarks the Planner independently, since it is an LLM call that runs
before every agent invocation in the real pipeline.

Runs each scenario N times (default 10) and reports:
  - Per-turn:       avg, median, p99, max wall time + LLM-only time + overhead
  - Per-LLM tag:   avg, median, p99, max across all runs
  - Scenario total: sum of all turns per run, stats across runs

Note: with N=10 runs, p99 equals max (too few samples for a meaningful
      percentile). Increase --runs to 100+ for accurate p99.

Output:
  benchmarks/reports/latency_<timestamp>.json
  benchmarks/reports/latency_<timestamp>.md

Usage:
  cd data-plane
  PYTHONPATH=. python3 benchmarks/latency_benchmark.py
  PYTHONPATH=. python3 benchmarks/latency_benchmark.py --runs 10 --warmup 3
  PYTHONPATH=. python3 benchmarks/latency_benchmark.py --runs 5 --warmup 0
  PYTHONPATH=. python3 benchmarks/latency_benchmark.py --scenario data_collection
  PYTHONPATH=. python3 benchmarks/latency_benchmark.py --scenario faq
  PYTHONPATH=. python3 benchmarks/latency_benchmark.py --scenario planner

The default --warmup 3 discards the first 3 runs so that the Cerebras KV cache
is populated for the static system prompt before measurements begin. Only the
user-data portion of the context is cold on measured runs.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from app.agents.data_collection import DataCollectionAgent
from app.agents.faq import FAQAgent
from app.agents.narrative_collection import NarrativeCollectionAgent
from app.agents.scheduling import SchedulingAgent
from app.agents.planner import Planner
from app.agents.workflow import WorkflowGraph
from app.agents.graph_config import APPOINTMENT_BOOKING
from app.agents.base import AgentStatus
from app.services.calendar_service import TimeSlot
from app.config import settings

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("benchmark")


# ── Configs ───────────────────────────────────────────────────────────────────

_DC_CONFIG: dict = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [
        {"name": "first_name",    "display_label": "First Name",    "data_type": "name",  "required": True, "collection_order": 1, "extraction_hints": []},
        {"name": "last_name",     "display_label": "Last Name",     "data_type": "name",  "required": True, "collection_order": 2, "extraction_hints": []},
        {"name": "email_address", "display_label": "Email Address", "data_type": "email", "required": True, "collection_order": 3, "extraction_hints": []},
    ],
    "faqs": [], "context_files": [], "practice_areas": [], "global_policy_documents": [],
    "_collected": {}, "_workflow_stages": "Collect caller contact information.",
    "_booking": {}, "_notes": "", "_tool_results": {},
}

_NARR_CONFIG: dict = {
    "assistant": {"persona_name": "Aria", "narrative_topic": "your legal matter"},
    "parameters": [],
    "faqs": [], "context_files": [], "practice_areas": [], "global_policy_documents": [],
    "_collected": {}, "_workflow_stages": "Gather the caller's narrative about their matter.",
    "_booking": {}, "_notes": "", "_tool_results": {},
}

_SCHED_CONFIG: dict = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [],
    "faqs": [], "context_files": [], "practice_areas": [], "global_policy_documents": [],
    "calendly_event_types": [],   # skip event-type matching LLM call
    "_collected": {"first_name": "John", "last_name": "Smith", "email_address": "john@example.com"},
    "_workflow_stages": "Schedule the caller's appointment.",
    "_booking": {}, "_notes": "", "_tool_results": {},
}

_FAQ_CONFIG: dict = {
    "assistant": {"persona_name": "Aria"},
    "parameters": [],
    "context_files": [], "practice_areas": [], "global_policy_documents": [],
    "_collected": {}, "_workflow_stages": "", "_booking": {}, "_notes": "", "_tool_results": {},
    "faqs": [
        {
            "question": "Do you charge for the initial consultation?",
            "answer": "No, the initial consultation is completely free of charge.",
        },
        {
            "question": "What are your office hours?",
            "answer": "We are open Monday through Friday, 9 AM to 6 PM Pacific Time.",
        },
        {
            "question": "What types of cases do you handle?",
            "answer": (
                "We handle personal injury cases including car accidents, slip and fall, "
                "and wrongful death, as well as employment law matters such as wrongful "
                "termination, workplace discrimination, and unpaid wages."
            ),
        },
        {
            "question": "How do your fees work?",
            "answer": (
                "We work on a contingency fee basis — you pay nothing unless we win. "
                "Our fee is typically 33% of the settlement before litigation "
                "and 40% if a lawsuit is filed."
            ),
        },
        {
            "question": "How long does a personal injury case usually take?",
            "answer": (
                "Most personal injury cases settle within 6 to 18 months. "
                "Cases that proceed to trial may take 2 to 3 years."
            ),
        },
    ],
}

_PLANNER_CONFIG: dict = {
    "assistant": {"persona_name": "Aria", "narrative_topic": "your legal matter"},
    "parameters": [
        {"name": "first_name",    "display_label": "First Name",    "data_type": "name",  "required": True, "collection_order": 1, "extraction_hints": []},
        {"name": "last_name",     "display_label": "Last Name",     "data_type": "name",  "required": True, "collection_order": 2, "extraction_hints": []},
        {"name": "email_address", "display_label": "Email Address", "data_type": "email", "required": True, "collection_order": 3, "extraction_hints": []},
    ],
    "faqs": [], "context_files": [], "practice_areas": [], "global_policy_documents": [],
    "_workflow_stages": "Collect contact information, then gather narrative, then schedule appointment.",
}


def _make_fake_slots() -> list[TimeSlot]:
    base = datetime(2026, 4, 7, 9, 0, tzinfo=timezone.utc)
    labels = ["Monday at 9 AM", "Monday at 11 AM", "Tuesday at 2 PM"]
    return [
        TimeSlot(
            slot_id=f"bench-slot-{i:03d}",
            start=base + timedelta(hours=i * 2),
            end=base + timedelta(hours=i * 2 + 1),
            description=label,
            event_type_uri="",
        )
        for i, label in enumerate(labels)
    ]


# ── Scripted scenarios ─────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "data_collection": {
        "description": "Collect first name, last name, and email — confirm each field",
        "agent_class": DataCollectionAgent,
        "config_template": "_DC_CONFIG",
        "turns": [
            "My name is John Smith.",
            "Yes, that's correct.",
            "Yes.",
            "My email is john dot smith at example dot com.",
            "Yes, that's right.",
            "Confirmed.",
        ],
    },
    "narrative_collection": {
        "description": "Caller describes their legal matter across multiple turns",
        "agent_class": NarrativeCollectionAgent,
        "config_template": "_NARR_CONFIG",
        "turns": [
            "I was in a car accident on January 15th. A truck ran a red light and hit my car from the side.",
            "I had whiplash and a broken arm. I was in the hospital for two days.",
            "I've missed three weeks of work and I have medical bills totaling over fifty thousand dollars.",
            "No, I think that covers everything.",
        ],
    },
    "faq": {
        "description": "Single FAQ question, multi-question turn, and a legal-deflect question",
        "agent_class": FAQAgent,
        "config_template": "_FAQ_CONFIG",
        "turns": [
            # T1: single question — matches one FAQ → faq_multi_match + faq_answer
            "Do you charge for consultations?",
            # T2: two questions in one utterance → faq_multi_match + faq_answer (combined)
            "What are your office hours and how do your fees work?",
            # T3: legal question — classified as is_legal → faq_multi_match only (deflect, no faq_answer)
            "Do you think I have a strong case given that the other driver ran the light?",
            # T4: mixed — one answerable FAQ + one legal question → both calls
            "How long do cases typically take, and should I file before the deadline?",
        ],
    },
    "scheduling": {
        "description": "Present available slots, caller picks one, confirms (no Calendly API call)",
        "agent_class": SchedulingAgent,
        "config_template": "_SCHED_CONFIG",
        "turns": [
            "",                      # T1: empty → _present_slots (slot presentation LLM call)
            "Option one please.",    # T2: pick slot 1 (_handle_choice: slot_choice + confirm_speak)
            "Yes, that works.",      # T3: confirm (_handle_confirmation: confirm_intent + booking)
        ],
    },
    "planner": {
        "description": "Planner LLM call latency across representative utterances",
        "agent_class": None,         # handled separately
        "config_template": "_PLANNER_CONFIG",
        "turns": [
            "My name is John Smith.",
            "Yes, that's right.",
            "I was in a car accident.",
            "That covers everything.",
            "Option one please.",
            "Yes, confirmed.",
        ],
    },
}


# ── Metrics capture ────────────────────────────────────────────────────────────

_captured: list[dict] = []


def _patched_log_metrics(tag: str, latency_ms: float, input_tokens: int, output_tokens: int) -> None:
    """Drop-in for llm_utils._log_metrics — captures timing instead of just logging."""
    _captured.append({
        "tag": tag,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })


# ── Result types ───────────────────────────────────────────────────────────────

class TurnResult:
    __slots__ = ("turn_idx", "utterance", "wall_ms", "llm_calls", "speak", "status")

    def __init__(self, turn_idx: int, utterance: str, wall_ms: float,
                 llm_calls: list[dict], speak: str, status: str) -> None:
        self.turn_idx   = turn_idx
        self.utterance  = utterance
        self.wall_ms    = wall_ms
        self.llm_calls  = llm_calls
        self.speak      = speak
        self.status     = status

    @property
    def llm_total_ms(self) -> float:
        return sum(c["latency_ms"] for c in self.llm_calls)

    @property
    def overhead_ms(self) -> float:
        return max(0.0, self.wall_ms - self.llm_total_ms)

    @property
    def llm_call_count(self) -> int:
        return len(self.llm_calls)


class ScenarioRun:
    __slots__ = ("run_idx", "turns")

    def __init__(self, run_idx: int, turns: list[TurnResult]) -> None:
        self.run_idx = run_idx
        self.turns   = turns

    @property
    def total_wall_ms(self) -> float:
        return sum(t.wall_ms for t in self.turns)

    @property
    def total_llm_ms(self) -> float:
        return sum(t.llm_total_ms for t in self.turns)


# ── Runner ─────────────────────────────────────────────────────────────────────

def _run_agent_scenario(scenario_name: str, run_idx: int) -> ScenarioRun:
    """Run a single agent scenario and return timing results."""
    scenario   = SCENARIOS[scenario_name]
    utterances = scenario["turns"]

    # Fresh agent + config per run
    agent  = scenario["agent_class"]()

    if scenario_name == "data_collection":
        config = copy.deepcopy(_DC_CONFIG)
        extra_patches: list = []
    elif scenario_name == "narrative_collection":
        config = copy.deepcopy(_NARR_CONFIG)
        extra_patches = []
    elif scenario_name == "faq":
        config = copy.deepcopy(_FAQ_CONFIG)
        extra_patches = []
    else:  # scheduling
        config = copy.deepcopy(_SCHED_CONFIG)
        config["_tool_results"] = {"prefetched_slots": _make_fake_slots()}
        extra_patches = [
            patch(
                "app.agents.scheduling.book_time_slot",
                return_value={
                    "booking_id": "BK-BENCH-001",
                    "status": "active",
                    "invitee_uri": "https://example.com/bench",
                    "cancel_url": "https://example.com/cancel",
                    "reschedule_url": "https://example.com/reschedule",
                },
            )
        ]

    state:   dict       = {}
    history: list[dict] = []
    results: list[TurnResult] = []

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch("app.agents.llm_utils._log_metrics", side_effect=_patched_log_metrics)
        )
        for p in extra_patches:
            stack.enter_context(p)

        for i, utterance in enumerate(utterances):
            _captured.clear()

            t0 = time.monotonic()
            try:
                resp = agent.process(utterance, state, config, history)
            except Exception as exc:
                logger.warning("[run %d turn %d] raised: %s", run_idx, i + 1, exc)
                resp = None
            wall_ms = (time.monotonic() - t0) * 1000

            llm_calls = list(_captured)

            if resp is not None:
                state = resp.internal_state or {}
                if utterance:
                    history.append({"role": "user", "content": utterance})
                history.append({"role": "assistant", "content": resp.speak or ""})
                speak  = resp.speak or ""
                status = resp.status.value
            else:
                speak  = ""
                status = "error"

            results.append(TurnResult(
                turn_idx  = i + 1,
                utterance = utterance,
                wall_ms   = round(wall_ms, 1),
                llm_calls = llm_calls,
                speak     = speak,
                status    = status,
            ))

            # FAQ is a one-shot handler (always COMPLETED) — keep going through all turns
            if resp is not None and resp.status == AgentStatus.COMPLETED and scenario_name != "faq":
                break   # scenario finished early — that's fine

    return ScenarioRun(run_idx, results)


def _run_planner_scenario(run_idx: int) -> ScenarioRun:
    """Benchmark the planner in isolation — one turn per utterance."""
    scenario   = SCENARIOS["planner"]
    utterances = scenario["turns"]
    graph      = WorkflowGraph(APPOINTMENT_BOOKING)
    planner    = Planner(graph)
    history:   list[dict] = []
    results:   list[TurnResult] = []

    with patch("app.agents.llm_utils._log_metrics", side_effect=_patched_log_metrics):
        for i, utterance in enumerate(utterances):
            _captured.clear()

            t0 = time.monotonic()
            try:
                planner.plan(utterance, history)
            except Exception as exc:
                logger.warning("[planner run %d turn %d] raised: %s", run_idx, i + 1, exc)
            wall_ms = (time.monotonic() - t0) * 1000

            llm_calls = list(_captured)
            history.append({"role": "user",      "content": utterance})
            history.append({"role": "assistant",  "content": "(benchmark)"})

            results.append(TurnResult(
                turn_idx  = i + 1,
                utterance = utterance,
                wall_ms   = round(wall_ms, 1),
                llm_calls = llm_calls,
                speak     = "",
                status    = "completed",
            ))

    return ScenarioRun(run_idx, results)


def _run_scenario(scenario_name: str, run_idx: int) -> ScenarioRun:
    if scenario_name == "planner":
        return _run_planner_scenario(run_idx)
    return _run_agent_scenario(scenario_name, run_idx)


# ── Statistics ─────────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"avg": 0.0, "median": 0.0, "p99": 0.0, "max": 0.0, "min": 0.0, "n": 0}
    n   = len(values)
    s   = sorted(values)
    p99 = s[max(0, int(n * 0.99) - 1)] if n >= 100 else s[-1]
    return {
        "avg":    round(statistics.mean(values),   1),
        "median": round(statistics.median(values), 1),
        "p99":    round(p99,                       1),
        "max":    round(max(values),               1),
        "min":    round(min(values),               1),
        "n":      n,
    }


def aggregate(all_runs: dict[str, list[ScenarioRun]]) -> dict[str, Any]:
    report: dict[str, Any] = {}

    for scenario_name, runs in all_runs.items():
        scenario_desc = SCENARIOS[scenario_name]["description"]
        turn_utterances = SCENARIOS[scenario_name]["turns"]

        turn_wall:     dict[int, list[float]] = {}
        turn_llm:      dict[int, list[float]] = {}
        turn_overhead: dict[int, list[float]] = {}
        turn_n_calls:  dict[int, list[int]]   = {}
        # per-tag: latency, input tokens, output tokens, output throughput (tok/s)
        tag_latency:    dict[str, list[float]] = {}
        tag_in_tokens:  dict[str, list[int]]   = {}
        tag_out_tokens: dict[str, list[int]]   = {}
        tag_throughput: dict[str, list[float]] = {}  # output tok/s
        wall_totals:   list[float] = []
        llm_totals:    list[float] = []

        for run in runs:
            wall_totals.append(run.total_wall_ms)
            llm_totals.append(run.total_llm_ms)
            for t in run.turns:
                turn_wall.setdefault(t.turn_idx, []).append(t.wall_ms)
                turn_llm.setdefault(t.turn_idx, []).append(t.llm_total_ms)
                turn_overhead.setdefault(t.turn_idx, []).append(t.overhead_ms)
                turn_n_calls.setdefault(t.turn_idx, []).append(t.llm_call_count)
                for call in t.llm_calls:
                    tag = call["tag"]
                    lat = call["latency_ms"]
                    out = call["output_tokens"]
                    tag_latency.setdefault(tag, []).append(lat)
                    tag_in_tokens.setdefault(tag, []).append(call["input_tokens"])
                    tag_out_tokens.setdefault(tag, []).append(out)
                    if lat > 0:
                        tag_throughput.setdefault(tag, []).append(out / (lat / 1000))

        report[scenario_name] = {
            "description":             scenario_desc,
            "runs":                    len(runs),
            "scenario_total_wall_ms":  _stats(wall_totals),
            "scenario_total_llm_ms":   _stats(llm_totals),
            "turns": {
                str(idx): {
                    "utterance":  (turn_utterances[idx - 1][:60] if idx <= len(turn_utterances) else ""),
                    "wall_ms":    _stats(turn_wall[idx]),
                    "llm_ms":     _stats(turn_llm[idx]),
                    "overhead_ms": _stats(turn_overhead[idx]),
                    "llm_calls":  _stats([float(x) for x in turn_n_calls[idx]]),
                }
                for idx in sorted(turn_wall.keys())
            },
            "llm_calls_by_tag": {
                tag: {
                    "latency_ms":       _stats(tag_latency[tag]),
                    "input_tokens":     _stats([float(x) for x in tag_in_tokens[tag]]),
                    "output_tokens":    _stats([float(x) for x in tag_out_tokens[tag]]),
                    "throughput_tok_s": _stats(tag_throughput.get(tag, [])),
                }
                for tag in sorted(tag_latency.keys())
            },
        }

    return report


# ── Markdown report ────────────────────────────────────────────────────────────

def _tbl_header_main() -> list[str]:
    return [
        "| Metric                              |  avg (ms) | med (ms) |  p99 (ms) |  max (ms) |  min (ms) |   N |",
        "|-------------------------------------|----------:|----------:|----------:|----------:|----------:|----:|",
    ]


def _tbl_row_main(label: str, s: dict) -> str:
    return (
        f"| {label:<35} | {s['avg']:>9.0f} | {s['median']:>9.0f} | "
        f"{s['p99']:>9.0f} | {s['max']:>9.0f} | {s['min']:>9.0f} | {s['n']:>3} |"
    )


def generate_markdown(report: dict, timestamp: str, n_runs: int, model: str) -> str:
    lines: list[str] = [
        "# LLM Latency Benchmark Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Model:** `{model}`  ",
        f"**Runs per scenario:** {n_runs}  ",
        "",
        "> **What is measured:** wall-clock time from utterance receipt to `agent.process()` return.",
        "> Does **not** include STT audio decoding or TTS synthesis — pure LLM + Python routing.",
        "> **Overhead** = wall time minus sum of individual LLM call durations (JSON parsing, state",
        "> management, Python logic). At N=10 runs, p99 equals max.",
        "",
    ]

    for scenario_name, data in report.items():
        title = scenario_name.replace("_", " ").title()
        w = data["scenario_total_wall_ms"]
        l = data["scenario_total_llm_ms"]

        lines += [
            "---",
            "",
            f"## {title}",
            "",
            f"_{data['description']}_",
            "",
            f"**Runs:** {data['runs']}",
            "",
            "### Scenario Totals  _(sum of all turns per run, stats across runs)_",
            "",
            *_tbl_header_main(),
            _tbl_row_main("Wall time (all turns)", w),
            _tbl_row_main("LLM time (all turns)", l),
            "",
        ]

        # Per-turn
        lines += [
            "### Per-Turn Latency",
            "",
            "| Turn | Utterance (truncated)               | wall avg | wall med | wall p99 | wall max | llm avg | overhead avg | LLM calls |",
            "|-----:|:------------------------------------|--------:|--------:|--------:|--------:|--------:|-------------:|----------:|",
        ]
        for idx_str, td in data["turns"].items():
            utt = (td["utterance"][:36] if td["utterance"] else "_(empty)_")
            wt  = td["wall_ms"]
            lt  = td["llm_ms"]
            ot  = td["overhead_ms"]
            nc  = td["llm_calls"]
            lines.append(
                f"| {idx_str:>4} | {utt:<36} | {wt['avg']:>7.0f}ms | {wt['median']:>7.0f}ms | "
                f"{wt['p99']:>7.0f}ms | {wt['max']:>7.0f}ms | {lt['avg']:>6.0f}ms | "
                f"{ot['avg']:>11.0f}ms | {nc['avg']:>8.1f} |"
            )
        lines.append("")

        # Per-tag
        if data["llm_calls_by_tag"]:
            lines += [
                "### LLM Call Latency by Tag",
                "",
                *_tbl_header_main(),
            ]
            for tag, s in data["llm_calls_by_tag"].items():
                # s may be flat stats dict (old format) or nested with latency_ms key (new format)
                stats = s["latency_ms"] if "latency_ms" in s else s
                lines.append(_tbl_row_main(tag, stats))
            lines.append("")

    return "\n".join(lines)


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(report: dict) -> None:
    print(f"\n{'─' * 72}")
    print(f"{'SCENARIO':<22} {'WALL avg':>10} {'WALL med':>10} {'WALL max':>10} {'LLM avg':>10}")
    print(f"{'─' * 72}")
    for scenario, data in report.items():
        w = data["scenario_total_wall_ms"]
        l = data["scenario_total_llm_ms"]
        print(
            f"{scenario:<22} {w['avg']:>9.0f}ms {w['median']:>9.0f}ms "
            f"{w['max']:>9.0f}ms {l['avg']:>9.0f}ms"
        )
    print(f"{'─' * 72}")

    # Per-tag across all scenarios
    print(f"\n{'LLM CALL TAG':<36} {'avg':>8} {'med':>8} {'max':>8} {'N':>5}")
    print(f"{'─' * 72}")
    all_tags: dict[str, list[float]] = {}
    for data in report.values():
        for tag, s in data["llm_calls_by_tag"].items():
            # Approximate: reconstruct sample from stats (just use avg * n for display)
            all_tags.setdefault(tag, [])
    # Just print from the first scenario that has each tag
    seen: set[str] = set()
    for data in report.values():
        for tag, s in sorted(data["llm_calls_by_tag"].items()):
            if tag not in seen:
                stats = s["latency_ms"] if "latency_ms" in s else s
                print(f"{tag:<36} {stats['avg']:>7.0f}ms {stats['median']:>7.0f}ms {stats['max']:>7.0f}ms {stats['n']:>5}")
                seen.add(tag)
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM latency benchmark for the voice AI pipeline")
    parser.add_argument("--runs",     type=int, default=10,
                        help="Number of measured runs per scenario (default: 10)")
    parser.add_argument("--warmup",   type=int, default=3,
                        help="Warmup runs to discard before measuring (default: 3). "
                             "Allows Cerebras to populate the KV cache for the static "
                             "system prompt so only the user-data portion is cold.")
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()),
                        help="Run only this scenario (default: all)")
    args = parser.parse_args()

    scenarios_to_run = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    n_runs   = args.runs
    n_warmup = args.warmup
    total    = n_warmup + n_runs

    print(f"\nLLM Latency Benchmark")
    print(f"  Model:     {settings.cerebras_model}")
    print(f"  Scenarios: {', '.join(scenarios_to_run)}")
    print(f"  Warmup:    {n_warmup} runs (discarded — KV cache fill)")
    print(f"  Measured:  {n_runs} runs")
    print(f"  Total:     {total} runs per scenario")
    print()

    all_runs: dict[str, list[ScenarioRun]] = {s: [] for s in scenarios_to_run}

    for scenario in scenarios_to_run:
        turns_n = len(SCENARIOS[scenario]["turns"])
        print(f"{'═' * 60}")
        print(f"  {scenario}  ({turns_n} turns × {total} runs, {n_warmup} warmup)")
        print(f"{'─' * 60}")

        for run_idx in range(1, total + 1):
            is_warmup = run_idx <= n_warmup
            label = f"warmup {run_idx:2d}/{n_warmup}" if is_warmup else f"run {run_idx - n_warmup:2d}/{n_runs}"
            print(f"  {label} ...", end="", flush=True)

            run = _run_scenario(scenario, run_idx)

            turn_tags = "  ".join(
                f"T{t.turn_idx}:{t.wall_ms:.0f}ms[{t.llm_call_count}LLM]"
                for t in run.turns
            )
            suffix = "  [discarded]" if is_warmup else ""
            print(f"  {run.total_wall_ms:.0f}ms  ↳ {turn_tags}{suffix}")

            if not is_warmup:
                all_runs[scenario].append(run)

        print()

    report = aggregate(all_runs)

    # Write reports
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"latency_{ts}.json"
    md_path   = reports_dir / f"latency_{ts}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(
        generate_markdown(report, ts, n_runs, settings.cerebras_model)
    )

    print_summary(report)

    print(f"Reports written:")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print()


if __name__ == "__main__":
    main()
