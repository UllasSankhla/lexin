"""Report writers — JSON, Markdown, and HTML outputs for conversation and batch reports."""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path

from evaluator.models import (
    ConversationReport, BatchReport, ImprovementSuggestion, FindingCategory,
)

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path(__file__).parents[1] / "reports"

# ── HTML helpers ──────────────────────────────────────────────────────────────

_HTML_STYLE = """
<style>
  :root { --bg:#f8f9fa; --card:#fff; --border:#dee2e6; --red:#dc3545;
          --orange:#fd7e14; --green:#198754; --blue:#0d6efd; --muted:#6c757d; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:#212529; margin:0; padding:24px; }
  h1 { font-size:1.6rem; margin-bottom:4px; }
  h2 { font-size:1.2rem; border-bottom:2px solid var(--border);
       padding-bottom:6px; margin-top:32px; }
  h3 { font-size:1rem; margin:0 0 6px; }
  .meta { color:var(--muted); font-size:.85rem; margin-bottom:24px; }
  .score-bar { display:flex; align-items:center; gap:10px; margin-bottom:24px; }
  .score-num { font-size:2.5rem; font-weight:700; }
  .score-label { color:var(--muted); font-size:.9rem; }
  .card { background:var(--card); border:1px solid var(--border);
          border-radius:8px; padding:16px; margin-bottom:16px; }
  .card-header { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:12px;
           font-size:.75rem; font-weight:600; }
  .badge-critical { background:#fce8ea; color:var(--red); }
  .badge-moderate { background:#fff3cd; color:#856404; }
  .badge-minor    { background:#d1e7dd; color:var(--green); }
  .badge-none     { background:#e2e3e5; color:var(--muted); }
  .badge-faq      { background:#cfe2ff; color:#084298; }
  .badge-prompt   { background:#e2d9f3; color:#432874; }
  .badge-context  { background:#d1e7dd; color:#0a3622; }
  .score-pill { font-weight:700; padding:2px 10px; border-radius:12px;
                background:#e9ecef; font-size:.85rem; }
  table.exchange { width:100%; border-collapse:collapse; font-size:.9rem; }
  table.exchange td { padding:8px 10px; vertical-align:top;
                      border:1px solid var(--border); }
  table.exchange td:first-child { width:15%; font-weight:600;
                                  color:var(--muted); white-space:nowrap; }
  .gap-block { background:#fff8f0; border-left:3px solid var(--orange);
               padding:8px 12px; font-size:.85rem; margin-top:10px; }
  .suggest-card { background:var(--card); border:1px solid var(--border);
                  border-radius:8px; padding:16px; margin-bottom:16px; }
  .suggest-title { font-size:1rem; font-weight:600; margin:0 0 4px; }
  .suggest-meta { font-size:.8rem; color:var(--muted); margin-bottom:10px; }
  .faq-q { font-weight:600; margin-bottom:2px; }
  .faq-a { margin-top:2px; }
  pre.doc { background:#f1f3f5; border-radius:6px; padding:10px;
            font-size:.82rem; white-space:pre-wrap; margin:6px 0 0; }
  .rule-block { background:#f3f0ff; border-left:3px solid #7048e8;
                padding:8px 12px; font-size:.88rem; border-radius:0 6px 6px 0; }
  .section-label { font-size:.8rem; font-weight:700; letter-spacing:.05em;
                   text-transform:uppercase; color:var(--muted); margin:20px 0 8px; }
  .stat-row { display:flex; gap:24px; margin-bottom:20px; flex-wrap:wrap; }
  .stat { text-align:center; }
  .stat-num { font-size:2rem; font-weight:700; }
  .stat-lbl { font-size:.78rem; color:var(--muted); }
</style>
"""

def _e(s: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(s) if s else "")

def _badge(text: str, cls: str) -> str:
    return f'<span class="badge badge-{cls}">{_e(text)}</span>'

def _score_color(score: int) -> str:
    if score >= 4: return "green"
    if score == 3: return "orange"
    return "red"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _report_to_dict(report: ConversationReport) -> dict:
    return {
        "conversation_id":      report.conversation_id,
        "source_file":          report.source_file,
        "evaluated_at":         report.evaluated_at,
        "overall_quality_score": report.overall_quality_score,
        "findings": [
            {
                "turn_index":           f.turn_index,
                "caller_utterance":     f.caller_utterance,
                "human_agent_response": f.human_agent_response,
                "ai_response":          f.ai_response,
                "agent_id":             f.agent_id,
                "quality_score":        f.quality_score,
                "gap_description":      f.gap_description,
                "category":             f.category.value,
                "severity":             f.severity,
            }
            for f in report.findings
        ],
        "suggestions": [_suggestion_to_dict(s) for s in report.suggestions],
    }


def _suggestion_to_dict(s: ImprovementSuggestion) -> dict:
    return {
        "suggestion_id":            s.suggestion_id,
        "category":                 s.category.value,
        "agent_id":                 s.agent_id,
        "title":                    s.title,
        "faq_question":             s.faq_question,
        "faq_answer":               s.faq_answer,
        "rule_to_add":              s.rule_to_add,
        "document_name":            s.document_name,
        "document_content":         s.document_content,
        "supporting_conversations": s.supporting_conversations,
        "frequency":                s.frequency,
        "composite_score":          s.composite_score,
    }


def _severity_emoji(severity: str) -> str:
    return {"critical": "🔴", "moderate": "🟡", "minor": "🟢", "none": "✅"}.get(severity, "")


# ── Conversation report writers ───────────────────────────────────────────────

def write_conversation_report_json(report: ConversationReport) -> Path:
    _ensure_dir(_REPORTS_DIR)
    out = _REPORTS_DIR / f"{report.conversation_id}.json"
    out.write_text(
        json.dumps(_report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote JSON report: %s", out)
    return out


def write_conversation_report_md(report: ConversationReport) -> Path:
    _ensure_dir(_REPORTS_DIR)
    lines = [
        f"# Evaluation Report: {report.conversation_id}",
        f"",
        f"**Source:** {report.source_file}  ",
        f"**Evaluated:** {report.evaluated_at}  ",
        f"**Overall quality score:** {report.overall_quality_score:.1f} / 5.0",
        f"",
        f"---",
        f"",
        f"## Findings ({len(report.findings)} evaluated turns)",
        f"",
    ]

    for f in report.findings:
        sev = _severity_emoji(f.severity)
        lines += [
            f"### Turn {f.turn_index} — Score {f.quality_score}/5 {sev}",
            f"",
            f"**Caller:** {f.caller_utterance}",
            f"",
            f"**Human agent:** {f.human_agent_response}",
            f"",
            f"**AI agent ({f.agent_id}):** {f.ai_response}",
            f"",
            f"**Gap:** {f.gap_description}  ",
            f"**Category:** `{f.category.value}` | **Severity:** `{f.severity}`",
            f"",
            f"---",
            f"",
        ]

    if report.suggestions:
        lines += [f"## Suggestions", f""]
        for s in report.suggestions:
            lines += [
                f"### {s.title}",
                f"**Category:** `{s.category.value}`",
                f"",
            ]
            if s.faq_question:
                lines += [f"**Q:** {s.faq_question}", f"", f"**A:** {s.faq_answer}", f""]
            if s.rule_to_add:
                lines += [f"**Rule:** {s.rule_to_add}", f""]
            if s.document_name:
                lines += [
                    f"**Document:** `{s.document_name}`",
                    f"```",
                    s.document_content or "",
                    f"```",
                    f"",
                ]

    out = _REPORTS_DIR / f"{report.conversation_id}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote Markdown report: %s", out)
    return out


# ── Batch report writers ──────────────────────────────────────────────────────

def write_batch_report_json(batch: BatchReport) -> Path:
    _ensure_dir(_REPORTS_DIR)
    data = {
        "run_timestamp":            batch.run_timestamp,
        "conversations_processed":  batch.conversations_processed,
        "conversations_skipped":    batch.conversations_skipped,
        "all_suggestions":          [_suggestion_to_dict(s) for s in batch.all_suggestions],
    }
    out = _REPORTS_DIR / "batch_report.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote batch JSON report: %s", out)
    return out


def write_batch_report_md(batch: BatchReport) -> Path:
    _ensure_dir(_REPORTS_DIR)
    lines = [
        f"# Batch Evaluation Report",
        f"",
        f"**Run:** {batch.run_timestamp}  ",
        f"**Conversations processed:** {batch.conversations_processed}  ",
        f"**Conversations skipped:** {batch.conversations_skipped}",
        f"",
        f"---",
        f"",
        f"## Improvement Suggestions ({len(batch.all_suggestions)} total)",
        f"",
    ]

    category_order = [FindingCategory.FAQ, FindingCategory.PROMPT, FindingCategory.CONTEXT]
    by_cat: dict = {c: [] for c in category_order}
    for s in batch.all_suggestions:
        if s.category in by_cat:
            by_cat[s.category].append(s)

    for cat in category_order:
        items = by_cat[cat]
        if not items:
            continue
        lines += [f"### {cat.value.upper()} ({len(items)} suggestions)", f""]
        for s in items:
            lines += [
                f"#### {s.title}",
                f"*Frequency: {s.frequency} | Priority score: {s.composite_score:.1f}*  ",
                f"*Supporting conversations: {', '.join(s.supporting_conversations)}*",
                f"",
            ]
            if s.faq_question:
                lines += [f"**Q:** {s.faq_question}", f"", f"**A:** {s.faq_answer}", f""]
            if s.rule_to_add:
                lines += [f"**Prompt rule:** {s.rule_to_add}", f""]
            if s.document_name:
                lines += [
                    f"**Document:** `{s.document_name}`",
                    f"```",
                    s.document_content or "",
                    f"```",
                    f"",
                ]
            lines.append("---")
            lines.append("")

    out = _REPORTS_DIR / "batch_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote batch Markdown report: %s", out)
    return out


# ── HTML report writers ───────────────────────────────────────────────────────

def _suggestion_html(s: ImprovementSuggestion) -> str:
    cat_cls = s.category.value
    parts = [
        f'<div class="suggest-card">',
        f'  <div class="suggest-title">{_e(s.title)}</div>',
        f'  <div class="suggest-meta">',
        f'    {_badge(s.category.value.upper(), cat_cls)}',
        f'    &nbsp; Frequency: <b>{s.frequency}</b> &nbsp;|&nbsp; Priority: <b>{s.composite_score:.1f}</b>',
        f'    &nbsp;|&nbsp; Conversations: {_e(", ".join(s.supporting_conversations))}',
        f'  </div>',
    ]
    if s.faq_question:
        parts += [
            f'  <div class="faq-q">Q: {_e(s.faq_question)}</div>',
            f'  <div class="faq-a">A: {_e(s.faq_answer or "")}</div>',
        ]
    if s.rule_to_add:
        parts.append(f'  <div class="rule-block">{_e(s.rule_to_add)}</div>')
    if s.document_name:
        parts += [
            f'  <div style="font-size:.85rem;margin-top:8px">Document: <code>{_e(s.document_name)}</code></div>',
            f'  <pre class="doc">{_e(s.document_content or "")}</pre>',
        ]
    parts.append('</div>')
    return "\n".join(parts)


def write_conversation_report_html(report: ConversationReport) -> Path:
    _ensure_dir(_REPORTS_DIR)

    # Build a lookup from turn_index → finding so we can overlay evaluation data
    finding_by_turn: dict[int, "EvaluationFinding"] = {
        f.turn_index: f for f in report.findings
    }

    # Use replay_turns as the source of truth for all turns; fall back to
    # findings alone if replay_turns wasn't populated (older reports).
    source_turns = report.replay_turns if report.replay_turns else [
        type("T", (), {
            "turn_index": f.turn_index,
            "caller_utterance": f.caller_utterance,
            "ai_response": f.ai_response,
            "agent_id": f.agent_id,
            "agent_status": "",
        })()
        for f in report.findings
    ]

    turns_html = []
    for rt in source_turns:
        finding = finding_by_turn.get(rt.turn_index)

        if finding:
            sc = _score_color(finding.quality_score)
            score_pill = f'<span class="score-pill" style="color:var(--{sc})">{finding.quality_score}/5</span>'
            badges = f'{_badge(finding.severity, finding.severity)} {_badge(finding.category.value, finding.category.value)}'
            human_row = f'<tr><td>Human agent</td><td>{_e(finding.human_agent_response)}</td></tr>'
            gap_block = f'<div class="gap-block"><b>Gap:</b> {_e(finding.gap_description)}</div>'
        else:
            score_pill = '<span class="score-pill" style="color:var(--muted)">—</span>'
            badges = '<span style="font-size:.75rem;color:var(--muted)">no human baseline</span>'
            human_row = '<tr><td style="color:var(--muted)">Human agent</td><td style="color:var(--muted);font-style:italic">not available</td></tr>'
            gap_block = ''

        turns_html.append(f"""
<div class="card">
  <div class="card-header">
    {score_pill}
    <b>Turn {rt.turn_index}</b>
    <span style="font-size:.78rem;color:var(--muted)">{_e(rt.agent_id)}</span>
    {badges}
  </div>
  <table class="exchange">
    <tr><td>Caller</td><td>{_e(rt.caller_utterance)}</td></tr>
    {human_row}
    <tr><td>AI agent</td><td>{_e(rt.ai_response)}</td></tr>
  </table>
  {gap_block}
</div>""")

    suggestions_html = ""
    if report.suggestions:
        suggestions_html = "<h2>Improvement Suggestions</h2>\n" + "\n".join(
            _suggestion_html(s) for s in report.suggestions
        )

    score_color = _score_color(int(report.overall_quality_score))
    evaluated_count = len(report.findings)
    total_count = len(source_turns)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Evaluation: {_e(report.conversation_id)}</title>
  {_HTML_STYLE}
</head>
<body>
  <h1>Evaluation Report: {_e(report.conversation_id)}</h1>
  <div class="meta">
    Source: {_e(report.source_file)} &nbsp;|&nbsp; Evaluated: {_e(report.evaluated_at)}
  </div>
  <div class="score-bar">
    <div class="score-num" style="color:var(--{score_color})">{report.overall_quality_score:.1f}</div>
    <div class="score-label">/ 5.0<br>overall quality score</div>
  </div>
  <div class="meta">{total_count} turns replayed &nbsp;|&nbsp; {evaluated_count} evaluated against human baseline</div>
  <h2>All Turns</h2>
  {"".join(turns_html)}
  {suggestions_html}
</body>
</html>"""

    out = _REPORTS_DIR / f"{report.conversation_id}.html"
    out.write_text(page, encoding="utf-8")
    logger.info("Wrote HTML report: %s", out)
    return out


def write_batch_report_html(batch: BatchReport) -> Path:
    _ensure_dir(_REPORTS_DIR)

    category_order = [FindingCategory.FAQ, FindingCategory.PROMPT, FindingCategory.CONTEXT]
    by_cat: dict = {c: [] for c in category_order}
    for s in batch.all_suggestions:
        if s.category in by_cat:
            by_cat[s.category].append(s)

    sections = []
    for cat in category_order:
        items = by_cat[cat]
        if not items:
            continue
        cat_html = "\n".join(_suggestion_html(s) for s in items)
        sections.append(f"""
<div class="section-label">{_e(cat.value.upper())} — {len(items)} suggestion(s)</div>
{cat_html}""")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Batch Evaluation Report</title>
  {_HTML_STYLE}
</head>
<body>
  <h1>Batch Evaluation Report</h1>
  <div class="meta">Run: {_e(batch.run_timestamp)}</div>
  <div class="stat-row">
    <div class="stat"><div class="stat-num">{batch.conversations_processed}</div>
      <div class="stat-lbl">Conversations processed</div></div>
    <div class="stat"><div class="stat-num">{batch.conversations_skipped}</div>
      <div class="stat-lbl">Skipped</div></div>
    <div class="stat"><div class="stat-num">{len(batch.all_suggestions)}</div>
      <div class="stat-lbl">Suggestions</div></div>
  </div>
  <h2>Improvement Suggestions</h2>
  {"".join(sections) if sections else "<p>No suggestions generated.</p>"}
</body>
</html>"""

    out = _REPORTS_DIR / "batch_report.html"
    out.write_text(page, encoding="utf-8")
    logger.info("Wrote HTML batch report: %s", out)
    return out
