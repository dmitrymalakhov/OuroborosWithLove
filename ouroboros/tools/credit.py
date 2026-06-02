"""Corporate credit committee preparation and speaker challenge tools."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ouroboros.tools.documents import (
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_SLIDES,
    MAX_ARCHIVE_FILES,
    _clean_limit,
    _extract_document,
    _resolve_document_path,
    _safe_output_path,
)
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import clip_text, safe_relpath

DEFAULT_OUTPUT_CHARS = 60_000


METRIC_TERMS = [
    "EBITDA", "OCF", "FCF", "DSCR", "ICR", "Net debt", "NetDebt", "LTV",
    "leverage", "ликвид", "денежн", "долг", "ковенант", "залог", "cash flow",
    "working capital", "оборотн", "capex", "процент", "refinancing",
]


def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)]


def _load_playbook(ctx: ToolContext) -> str:
    path = ctx.repo_dir / "prompts" / "CREDIT_COMMITTEE_PLAYBOOK.md"
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return "Credit committee playbook unavailable. Use evidence-based second-line credit challenge."


def _extract_inputs(
    ctx: ToolContext,
    paths: List[str],
    source: str,
    max_chars: int,
    max_pages: int,
    max_slides: int,
    max_archive_files: int,
) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    per_doc_limit = max(3_000, max_chars // max(1, len(paths)))
    for raw_path in paths:
        document_path = _resolve_document_path(ctx, raw_path, source)
        extracted = _extract_document(
            document_path,
            max_pages=max_pages,
            max_slides=max_slides,
            max_archive_files=max_archive_files,
            progress=ctx.emit_progress_fn,
        )
        parts: List[str] = []
        if extracted.warnings:
            parts.append("Warnings:\n" + "\n".join(f"- {warning}" for warning in extracted.warnings))
        for title, text in extracted.sections:
            parts.append(f"### {title}\n{text.strip() or '[Empty]'}")
        docs.append({
            "path": raw_path,
            "name": document_path.name,
            "kind": extracted.kind,
            "content": clip_text("\n\n".join(parts).strip(), per_doc_limit),
        })
    return docs


def _write_optional(ctx: ToolContext, output_path: str, content: str) -> str:
    if not output_path:
        return ""
    target = _safe_output_path(ctx.drive_root, safe_relpath(output_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"\n\nSaved to `{target.relative_to(ctx.drive_root.resolve())}`."


def _combined_text(docs: List[Dict[str, str]], inline_text: str) -> str:
    return "\n\n".join([inline_text] + [doc["content"] for doc in docs]).lower()


def _has_any(text: str, terms: List[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _risk_flags(text: str) -> List[Dict[str, str]]:
    flags: List[Dict[str, str]] = []
    collateral_terms = ["залог", "обеспеч", "ltv", "collateral", "pledge", "haircut"]
    cash_flow_terms = ["ocf", "dscr", "cash flow", "денежн", "операционн", "погашен", "выручк"]
    if _has_any(text, collateral_terms) and not _has_any(text, ["ocf", "dscr", "cash flow", "операционн"]):
        flags.append({
            "type": "collateral",
            "severity": "orange",
            "description": "Collateral appears in the materials, but primary cash-flow repayment evidence is weak or not visible.",
        })
    if _has_any(text, ["parent", "sister", "group support", "групп", "связан", "поручител", "гарант"]) and not _has_any(text, ["connected", "ownership", "оргструкт", "бенефициар"]):
        flags.append({
            "type": "connected_clients",
            "severity": "yellow",
            "description": "Group/support language appears; connected-client and ownership evidence should be explicit.",
        })
    if _has_any(text, ["stress", "downside", "стресс", "сценар"]) is False:
        flags.append({
            "type": "forecast",
            "severity": "yellow",
            "description": "No visible downside/stress scenario. Committee materials should not rely only on base case.",
        })
    if _has_any(text, ["115-фз", "санкц", "compliance", "tax", "налог", "суд", "арбитраж", "лиценз"]) is False:
        flags.append({
            "type": "compliance",
            "severity": "yellow",
            "description": "No visible legal/compliance/tax screening evidence.",
        })
    if _has_any(text, ["концентрац", "n6", "sector concentration", "top-1", "top-5"]) is False:
        flags.append({
            "type": "concentration",
            "severity": "yellow",
            "description": "No visible group/sector/customer concentration analysis.",
        })
    return flags


def _metric_mentions(text: str) -> List[str]:
    mentions: List[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if len(mentions) >= 40:
            break
        if any(term.lower() in line.lower() for term in METRIC_TERMS):
            mentions.append(line[:500])
    return mentions


def _format_credit_brief(
    ctx: ToolContext,
    title: str,
    objective: str,
    framework: List[str],
    requested_output: List[str],
    paths: List[str],
    source: str,
    inline_text: str,
    output_format: str,
    output_path: str,
    max_chars: int,
    max_pages: int,
    max_slides: int,
    max_archive_files: int,
) -> str:
    output_format = (output_format or "markdown").strip().lower()
    if output_format not in {"markdown", "json"}:
        output_format = "markdown"
    max_chars = _clean_limit(max_chars, DEFAULT_OUTPUT_CHARS, 3_000, 120_000)
    max_pages = _clean_limit(max_pages, DEFAULT_MAX_PAGES, 1, 200)
    max_slides = _clean_limit(max_slides, DEFAULT_MAX_SLIDES, 1, 300)
    max_archive_files = _clean_limit(max_archive_files, MAX_ARCHIVE_FILES, 1, 200)
    docs = _extract_inputs(ctx, paths, source, max_chars, max_pages, max_slides, max_archive_files) if paths else []
    playbook = clip_text(_load_playbook(ctx), 14_000)
    combined = _combined_text(docs, inline_text)
    flags = _risk_flags(combined)
    mentions = _metric_mentions(combined)

    if output_format == "json":
        payload = {
            "title": title,
            "objective": objective,
            "framework": framework,
            "requested_output": requested_output,
            "risk_flags": flags,
            "metric_mentions": mentions,
            "playbook": playbook,
            "inline_text": inline_text,
            "documents": docs,
            "output_contract": {
                "recommendation": "approve|approve_with_conditions|rework|escalate|reject",
                "speaker_verdict": "ready|not_ready",
                "material_conclusions_require": "evidence_refs or requires_evidence",
            },
        }
        raw = clip_text(json.dumps(payload, ensure_ascii=False, indent=2), max_chars)
        return raw + _write_optional(ctx, output_path, raw)

    lines = [
        f"# {title}",
        "",
        "## Objective",
        objective,
        "",
        "## Framework",
    ]
    lines.extend(f"- {item}" for item in framework)
    lines.extend(["", "## Requested Output"])
    lines.extend(f"- {item}" for item in requested_output)
    lines.extend([
        "",
        "## Guardrails",
        "- Act as independent second-line challenge, not as a slide polisher.",
        "- Every material conclusion needs an evidence reference or `requires_evidence`.",
        "- Do not recommend auto-approval when red/orange blockers are present.",
        "- This is decision support, not an automatic credit decision.",
        "",
        "## Early Risk Flags",
    ])
    if flags:
        for flag in flags:
            lines.append(f"- {flag['severity'].upper()} / {flag['type']}: {flag['description']}")
    else:
        lines.append("- No automatic early flags from simple keyword checks. Continue full evidence review.")
    if mentions:
        lines.extend(["", "## Metric Mentions To Verify"])
        lines.extend(f"- {mention}" for mention in mentions)
    lines.extend(["", "## Credit Committee Playbook", playbook])
    if inline_text.strip():
        lines.extend(["", "## Inline Input", inline_text.strip()])
    if docs:
        lines.append("")
        lines.append("## Extracted Evidence")
        for doc in docs:
            lines.extend([
                "",
                f"### {doc['path']}",
                f"- file: {doc['name']}",
                f"- type: {doc['kind']}",
                "",
                doc["content"] or "[No extractable content]",
            ])
    raw = clip_text("\n".join(lines).strip() + "\n", max_chars)
    return raw + _write_optional(ctx, output_path, raw)


def _credit_pack_check(
    ctx: ToolContext,
    document_paths: List[str] | str = "",
    deal_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    return _format_credit_brief(
        ctx,
        "Credit Pack Check",
        f"Review corporate credit committee package completeness in {language}.",
        [
            "Check main deck versus evidence pack: request, borrower/group, business model, repayment, financials, deal, collateral, covenants, risks.",
            "Identify missing documents, inconsistent numbers, policy exceptions, and open items.",
            "Classify blockers as yellow, orange, or red.",
        ],
        [
            "Completeness matrix.",
            "Missing evidence and open items.",
            "Risk flags and escalation memo when needed.",
            "Readiness to move to speaker rehearsal.",
        ],
        _as_list(document_paths),
        source,
        deal_text,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _credit_metrics_check(
    ctx: ToolContext,
    document_paths: List[str] | str = "",
    deal_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    return _format_credit_brief(
        ctx,
        "Credit Metrics Check",
        f"Prepare deterministic recalculation checklist for key credit metrics in {language}.",
        [
            "Verify EBITDA/OCF/FCF, NetDebt/EBITDA, ICR, DSCR, liquidity, LTV post-haircut, covenant headroom where data exists.",
            "Check debt roll-forward, interest expense consistency, working-capital growth, capex funding, and covenant consistency.",
            "Mark missing raw data as requires_evidence rather than inventing metrics.",
        ],
        [
            "Metric table with source refs.",
            "Reconciliation issues.",
            "Downside/stress assumptions to run.",
            "Questions for model owner.",
        ],
        _as_list(document_paths),
        source,
        deal_text,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _credit_deck_challenge(
    ctx: ToolContext,
    deck_path: str = "",
    evidence_paths: List[str] | str = "",
    speaker_notes_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    return _format_credit_brief(
        ctx,
        "Credit Deck Challenge",
        f"Challenge speaker deck claims in {language}.",
        [
            "Break the deck into material claims and test each claim for completeness, causality, arithmetic, and stress resilience.",
            "Apply cash flow primacy, connected parties, rating lag, and stress consistency tests.",
            "Generate fair, hard, and escalation questions for every material weak claim.",
        ],
        [
            "Claim-by-claim challenge table.",
            "Supporting and opposing evidence.",
            "PASS / PASS_WITH_RESERVATIONS / REWORK / ESCALATE / REJECT verdicts.",
            "Speaker rehearsal priorities.",
        ],
        _as_list(deck_path) + _as_list(evidence_paths),
        source,
        speaker_notes_text,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _credit_speaker_qna(
    ctx: ToolContext,
    document_paths: List[str] | str = "",
    speaker_claims: str = "",
    question_focus: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    inline = "\n\n".join(part for part in [speaker_claims, question_focus] if part.strip())
    return _format_credit_brief(
        ctx,
        "Credit Speaker Q&A",
        f"Prepare credit committee Q&A in {language}.",
        [
            "Generate fair, hard, and escalation questions across repayment, forecast, EBITDA, working capital, group, collateral, covenants, rate/FX, concentration, ESG, and compliance.",
            "Answers must use: yes/no/partly, one reason, one number, one evidence ref, one residual risk, one mitigation.",
            "Do not answer numeric questions without numbers or evidence refs.",
        ],
        [
            "Question bank grouped by risk block.",
            "Short answer drafts.",
            "Expected evidence for each answer.",
            "Escalation questions that should stop rehearsal until answered.",
        ],
        _as_list(document_paths),
        source,
        inline,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _credit_committee_readiness(
    ctx: ToolContext,
    document_paths: List[str] | str = "",
    speaker_transcript_path: str = "",
    speaker_notes_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    return _format_credit_brief(
        ctx,
        "Credit Committee Readiness",
        f"Score readiness for committee rehearsal in {language}.",
        [
            "Apply speaker scorecard: structure 15, data quality 20, logic 15, risk transparency 20, stress resilience 15, policy fit 10, Q&A 5.",
            "Ready only if request is explicit, numbers reconcile, evidence exists, downside is addressed, residual risk is visible, and answers can be numeric.",
            "Produce blockers and rehearsal plan.",
        ],
        [
            "Speaker scorecard.",
            "ready/not_ready verdict.",
            "Blockers by severity.",
            "Rehearsal plan and must-fix answers.",
        ],
        _as_list(document_paths) + _as_list(speaker_transcript_path),
        source,
        speaker_notes_text,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _credit_memo_draft(
    ctx: ToolContext,
    document_paths: List[str] | str = "",
    deal_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    return _format_credit_brief(
        ctx,
        "Credit Memo Draft",
        f"Draft credit memo and decision request in {language}.",
        [
            "Draft ask, borrower/group, business model, repayment, financials, structure, collateral, covenants, risks, mitigants, CP/CS.",
            "Recommendation options: approve, approve_with_conditions, rework, escalate, reject.",
            "Unsupported sections must be marked requires_evidence.",
        ],
        [
            "Credit memo draft.",
            "Decision request.",
            "Conditions precedent/subsequent.",
            "Residual risks and escalation items.",
        ],
        _as_list(document_paths),
        source,
        deal_text,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _credit_deck_outline(
    ctx: ToolContext,
    document_paths: List[str] | str = "",
    deal_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    return _format_credit_brief(
        ctx,
        "Credit Deck Outline",
        f"Create main deck outline for committee presentation in {language}.",
        [
            "Keep main deck short and evidence-backed.",
            "Use required blocks: decision request, executive summary, borrower/group, business model, repayment, financials, deal structure, collateral, covenants, risks, appendices.",
            "Prepare outline that can be handed to create_presentation.",
        ],
        [
            "Slide-by-slide outline.",
            "Evidence required per slide.",
            "Speaker notes and likely committee questions per slide.",
            "Appendix list.",
        ],
        _as_list(document_paths),
        source,
        deal_text,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


COMMON_PROPS = {
    "source": {"type": "string", "enum": ["drive", "repo"], "default": "drive"},
    "language": {"type": "string", "default": "ru"},
    "output_format": {"type": "string", "enum": ["markdown", "json"], "default": "markdown"},
    "output_path": {"type": "string", "description": "Optional path relative to the user's Drive workspace."},
    "max_chars": {"type": "integer", "default": DEFAULT_OUTPUT_CHARS},
    "max_pages": {"type": "integer", "default": DEFAULT_MAX_PAGES},
    "max_slides": {"type": "integer", "default": DEFAULT_MAX_SLIDES},
    "max_archive_files": {"type": "integer", "default": MAX_ARCHIVE_FILES},
}


def _schema(name: str, description: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {**properties, **COMMON_PROPS}},
    }


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("credit_pack_check", _schema("credit_pack_check", "Check corporate credit committee package completeness and red flags.", {
            "document_paths": {"type": "array", "items": {"type": "string"}, "description": "Credit memo, deck, model, financials, collateral, legal/compliance files."},
            "deal_text": {"type": "string", "description": "Inline deal context."},
        }), _credit_pack_check, timeout_sec=90),
        ToolEntry("credit_metrics_check", _schema("credit_metrics_check", "Prepare recalculation checklist for credit metrics and model consistency.", {
            "document_paths": {"type": "array", "items": {"type": "string"}},
            "deal_text": {"type": "string"},
        }), _credit_metrics_check, timeout_sec=90),
        ToolEntry("credit_deck_challenge", _schema("credit_deck_challenge", "Challenge speaker deck claims against evidence and stress tests.", {
            "deck_path": {"type": "string", "description": "Presentation path in Drive."},
            "evidence_paths": {"type": "array", "items": {"type": "string"}, "description": "Supporting memo/model/evidence files."},
            "speaker_notes_text": {"type": "string"},
        }), _credit_deck_challenge, timeout_sec=90),
        ToolEntry("credit_speaker_qna", _schema("credit_speaker_qna", "Generate committee questions and short numeric evidence-based answers.", {
            "document_paths": {"type": "array", "items": {"type": "string"}},
            "speaker_claims": {"type": "string"},
            "question_focus": {"type": "string"},
        }), _credit_speaker_qna, timeout_sec=90),
        ToolEntry("credit_committee_readiness", _schema("credit_committee_readiness", "Score speaker and package readiness for credit committee.", {
            "document_paths": {"type": "array", "items": {"type": "string"}},
            "speaker_transcript_path": {"type": "string"},
            "speaker_notes_text": {"type": "string"},
        }), _credit_committee_readiness, timeout_sec=90),
        ToolEntry("credit_memo_draft", _schema("credit_memo_draft", "Draft credit memo and recommendation from evidence pack.", {
            "document_paths": {"type": "array", "items": {"type": "string"}},
            "deal_text": {"type": "string"},
        }), _credit_memo_draft, timeout_sec=90),
        ToolEntry("credit_deck_outline", _schema("credit_deck_outline", "Create slide outline for committee presentation.", {
            "document_paths": {"type": "array", "items": {"type": "string"}},
            "deal_text": {"type": "string"},
        }), _credit_deck_outline, timeout_sec=90),
    ]
