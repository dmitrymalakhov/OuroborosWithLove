"""HR hiring tools built on document extraction and a compact hiring playbook."""

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

DEFAULT_OUTPUT_CHARS = 45_000


def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)]


def _load_playbook(ctx: ToolContext) -> str:
    path = ctx.repo_dir / "prompts" / "HR_HIRING_PLAYBOOK.md"
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return "HR playbook unavailable. Use evidence-based hiring, structured scorecards, and fair interview practice."


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
    per_doc_limit = max(2_000, max_chars // max(1, len(paths)))
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


def _format_domain_brief(
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
    max_chars = _clean_limit(max_chars, DEFAULT_OUTPUT_CHARS, 2_000, 100_000)
    max_pages = _clean_limit(max_pages, DEFAULT_MAX_PAGES, 1, 200)
    max_slides = _clean_limit(max_slides, DEFAULT_MAX_SLIDES, 1, 300)
    max_archive_files = _clean_limit(max_archive_files, MAX_ARCHIVE_FILES, 1, 200)
    docs = _extract_inputs(ctx, paths, source, max_chars, max_pages, max_slides, max_archive_files) if paths else []
    playbook = clip_text(_load_playbook(ctx), 12_000)

    if output_format == "json":
        payload = {
            "title": title,
            "objective": objective,
            "framework": framework,
            "requested_output": requested_output,
            "playbook": playbook,
            "inline_text": inline_text,
            "documents": docs,
            "guardrails": [
                "Use evidence from documents and mark unsupported conclusions as requires_evidence.",
                "Do not infer protected attributes or make decisions based on protected attributes.",
                "Treat legal/employment-format guidance as risk prompts, not legal advice.",
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        raw = clip_text(raw, max_chars)
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
        "- Use only provided evidence; mark unsupported conclusions as `requires_evidence`.",
        "- Do not infer or use protected attributes.",
        "- Employment-format and compliance notes are risk prompts, not legal advice.",
        "",
        "## HR Hiring Playbook",
        playbook,
    ])
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


def _hr_vacancy_audit(
    ctx: ToolContext,
    vacancy_path: str = "",
    vacancy_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    return _format_domain_brief(
        ctx,
        "HR Vacancy Audit",
        f"Audit a vacancy/JD for hiring quality and produce recommendations in {language}.",
        [
            "Check honesty, role clarity, business outcome, EVP, compensation realism, channel fit, and red flags.",
            "Separate must-have requirements from inflated or nice-to-have requirements.",
            "Identify whether the vacancy sounds like the real team and attracts the right candidates.",
        ],
        [
            "Vacancy scorecard with strengths, weak spots, and red flags.",
            "Rewritten vacancy outline if enough evidence exists.",
            "Recommended sourcing channels and candidate-screening questions.",
        ],
        _as_list(vacancy_path),
        source,
        vacancy_text,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _hr_role_profile(
    ctx: ToolContext,
    role_path: str = "",
    role_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    return _format_domain_brief(
        ctx,
        "HR Role Profile",
        f"Build a role profile and hiring scorecard in {language}.",
        [
            "Define the business task, outcomes, must-have skills, nice-to-have skills, and sufficient competence level.",
            "Split hard skills, soft skills, motivation, learning ability, values fit, and risk flags.",
            "Design a funnel and interview sequence before candidate screening.",
        ],
        [
            "Role card.",
            "Hiring scorecard with 1-5 rubric.",
            "Interview stages and evidence required at each stage.",
            "Sourcing channel recommendations.",
        ],
        _as_list(role_path),
        source,
        role_text,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _hr_candidate_screen(
    ctx: ToolContext,
    candidate_paths: List[str] | str = "",
    role_path: str = "",
    candidate_text: str = "",
    role_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    paths = _as_list(role_path) + _as_list(candidate_paths)
    inline = "\n\n".join(part for part in [role_text, candidate_text] if part.strip())
    return _format_domain_brief(
        ctx,
        "HR Candidate Screen",
        f"Screen candidate evidence against the role profile in {language}.",
        [
            "Compare candidate evidence to the role scorecard without making protected-attribute inferences.",
            "Distinguish proven fit, gaps, unknowns, and risks.",
            "Generate follow-up questions that verify weak or unsupported claims.",
        ],
        [
            "Evidence matrix by criterion.",
            "Fit/gap/risk summary.",
            "Questions for recruiter and hiring manager.",
            "Verdict options: strong_fit, possible_fit, weak_fit, requires_more_evidence.",
        ],
        paths,
        source,
        inline,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _hr_interview_kit(
    ctx: ToolContext,
    role_path: str = "",
    candidate_path: str = "",
    interview_stage: str = "hiring_manager",
    role_text: str = "",
    candidate_text: str = "",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    inline = "\n\n".join(part for part in [role_text, candidate_text] if part.strip())
    return _format_domain_brief(
        ctx,
        "HR Interview Kit",
        f"Prepare a {interview_stage} interview kit in {language}.",
        [
            "Start with context and a calm setup, then reveal evidence through examples, cases, and motivation questions.",
            "Use stage-specific questions and a score sheet; avoid stress theater and irrelevant puzzles.",
            "Separate hard-skill validation, behavior, motivation, values, compensation expectations, and open risks.",
        ],
        [
            "Interview agenda.",
            "Questions grouped by competency.",
            "Scoring rubric and evidence notes.",
            "Candidate close and next-step checklist.",
        ],
        _as_list(role_path) + _as_list(candidate_path),
        source,
        inline,
        output_format,
        output_path,
        max_chars,
        max_pages,
        max_slides,
        max_archive_files,
    )


def _hr_onboarding_checklist(
    ctx: ToolContext,
    role_path: str = "",
    offer_path: str = "",
    role_text: str = "",
    employment_format: str = "unknown",
    source: str = "drive",
    language: str = "ru",
    output_format: str = "markdown",
    output_path: str = "",
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_slides: int = DEFAULT_MAX_SLIDES,
    max_archive_files: int = MAX_ARCHIVE_FILES,
) -> str:
    return _format_domain_brief(
        ctx,
        "HR Onboarding Checklist",
        f"Prepare onboarding and probation checklist in {language}; employment format: {employment_format}.",
        [
            "Plan documents, first tasks, role expectations, communication cadence, and 30/60/90 checkpoints.",
            "Use employment-format checks as risk prompts only; legal/accounting review is required for formal advice.",
            "Make probation a validation period with clear outcomes, not a discounted labor period.",
        ],
        [
            "Pre-start checklist.",
            "First day / first week / 30-60-90 day plan.",
            "Probation success criteria.",
            "Employment-format risk prompts and escalation items.",
        ],
        _as_list(role_path) + _as_list(offer_path),
        source,
        role_text,
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
        ToolEntry("hr_vacancy_audit", _schema("hr_vacancy_audit", "Audit a vacancy/JD using the HR hiring playbook.", {
            "vacancy_path": {"type": "string", "description": "Vacancy/JD path in Drive."},
            "vacancy_text": {"type": "string", "description": "Inline vacancy/JD text."},
        }), _hr_vacancy_audit, timeout_sec=60),
        ToolEntry("hr_role_profile", _schema("hr_role_profile", "Create a role profile, scorecard, and hiring funnel.", {
            "role_path": {"type": "string", "description": "Role/JD/business request path in Drive."},
            "role_text": {"type": "string", "description": "Inline role description."},
        }), _hr_role_profile, timeout_sec=60),
        ToolEntry("hr_candidate_screen", _schema("hr_candidate_screen", "Screen candidate evidence against a role profile.", {
            "candidate_paths": {"type": "array", "items": {"type": "string"}, "description": "CV, notes, portfolio, or interview files."},
            "role_path": {"type": "string", "description": "Role profile/JD path in Drive."},
            "candidate_text": {"type": "string", "description": "Inline candidate notes."},
            "role_text": {"type": "string", "description": "Inline role profile/JD."},
        }), _hr_candidate_screen, timeout_sec=60),
        ToolEntry("hr_interview_kit", _schema("hr_interview_kit", "Generate structured interview questions, rubric, and score sheet.", {
            "role_path": {"type": "string", "description": "Role/JD path in Drive."},
            "candidate_path": {"type": "string", "description": "Optional candidate material path in Drive."},
            "interview_stage": {"type": "string", "default": "hiring_manager"},
            "role_text": {"type": "string"},
            "candidate_text": {"type": "string"},
        }), _hr_interview_kit, timeout_sec=60),
        ToolEntry("hr_onboarding_checklist", _schema("hr_onboarding_checklist", "Create onboarding, probation, and employment-format risk checklist.", {
            "role_path": {"type": "string", "description": "Role/JD path in Drive."},
            "offer_path": {"type": "string", "description": "Offer or agreement notes path in Drive."},
            "role_text": {"type": "string"},
            "employment_format": {"type": "string", "default": "unknown"},
        }), _hr_onboarding_checklist, timeout_sec=60),
    ]
