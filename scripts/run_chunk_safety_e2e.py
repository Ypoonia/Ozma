#!/usr/bin/env python3
"""Run the real chunk-safety pipeline against the local end-to-end corpus."""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from doc_analyse.detection import DEFAULT_PROMPT_GUARD_MODEL, PromptGuardDetector, YaraDetector
from doc_analyse.detection.detect import CheapRouter
from doc_analyse.ingestion import TextChunker
from doc_analyse.ingestion.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from doc_analyse.orchestration import DocumentAnalysisResult, DocumentOrchestrator
from doc_analyse.workers import build_classifier_worker_pool


DEFAULT_DOCS_DIR = Path("/Users/yash/Desktop/CODEBASE/chunk_safety_test_docs")
DEFAULT_REPORT_PATH = REPO_ROOT / "e2e_reports" / "chunk_safety_e2e_report.json"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_ENV_PREFIX = "DOC_ANALYSE_LLM"

SUPPORTED_ENV_PROVIDERS = frozenset({"anthropic", "claude", "codex", "openai"})
OPENAI_ENV_PROVIDERS = frozenset({"codex", "openai"})
ANTHROPIC_ENV_PROVIDERS = frozenset({"anthropic", "claude"})


LOGGER = logging.getLogger("chunk_safety_e2e")


@dataclass(frozen=True)
class DocumentCase:
    path: Path
    expected: Mapping[str, Any]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real ingestion, YARA, CheapRouter, optional PromptGuard, and the "
            "env-backed Anthropic/OpenAI classifier worker over the chunk-safety corpus."
        )
    )
    parser.add_argument(
        "--docs-dir",
        default=str(DEFAULT_DOCS_DIR),
        help=f"Corpus directory. Default: {DEFAULT_DOCS_DIR}",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help=f"JSON report path. Default: {DEFAULT_REPORT_PATH}",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help=f".env file loaded via python-dotenv. Default: {DEFAULT_ENV_FILE}",
    )
    parser.add_argument(
        "--override-env",
        action="store_true",
        help="Let values from --env-file override already exported environment variables.",
    )
    parser.add_argument(
        "--env-prefix",
        default=DEFAULT_ENV_PREFIX,
        help="Classifier env prefix consumed by doc_analyse workers. Default: DOC_ANALYSE_LLM",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Classifier worker threads. Default: 1 to avoid provider rate spikes.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Text chunk size. Default: {DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Text chunk overlap. Default: {DEFAULT_CHUNK_OVERLAP}",
    )
    parser.add_argument(
        "--disable-prompt-guard",
        action="store_true",
        help="Disable PromptGuard and route with PG score 0.0.",
    )
    parser.add_argument(
        "--allow-prompt-guard-unavailable",
        action="store_true",
        help=(
            "If PromptGuard dependencies or model download are unavailable, log a warning "
            "and continue with PromptGuard disabled."
        ),
    )
    parser.add_argument(
        "--prompt-guard-model",
        default=DEFAULT_PROMPT_GUARD_MODEL,
        help=f"PromptGuard model id. Default: {DEFAULT_PROMPT_GUARD_MODEL}",
    )
    parser.add_argument(
        "--prompt-guard-device",
        type=int,
        default=-1,
        help="Transformers device for PromptGuard. Default: -1 (CPU).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity. Default: WARNING.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional file path for logs. Console output remains concise.",
    )
    return parser.parse_args(argv)


def configure_logging(level_name: str, log_file: Optional[str]) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = _resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, args.log_file)
    report_path = _resolve_path(args.report_path)

    started = _utc_now()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started,
        "finished_at": None,
        "duration_seconds": None,
        "mocks_used": False,
        "configuration": _configuration_report(args, report_path),
        "environment": {},
        "prompt_guard": {},
        "summary": {},
        "documents": [],
    }

    start_time = time.monotonic()
    exit_code = 1
    try:
        exit_code = run(args, report)
    except Exception as exc:
        LOGGER.debug("Set-up failed", exc_info=True)
        report["status"] = "failed"
        report["setup_error"] = _exception_report(exc)
        exit_code = 2
    finally:
        report["finished_at"] = _utc_now()
        report["duration_seconds"] = round(time.monotonic() - start_time, 3)
        if not report.get("summary"):
            report["summary"] = _summarize_documents(report["documents"])
        report["summary"]["setup_failed"] = bool(report.get("setup_error"))
        write_report(report_path, report)
        print_console_summary(report_path, report)

    return exit_code


def run(args: argparse.Namespace, report: dict[str, Any]) -> int:
    env_report = load_environment(args.env_file, args.override_env)
    provider_report = resolve_provider_report(args.env_prefix)
    ensure_supported_classifier_env(provider_report, args.env_prefix)
    report["environment"] = {
        **env_report,
        **provider_report,
    }

    prompt_guard, prompt_guard_report = build_prompt_guard(args)
    report["prompt_guard"] = prompt_guard_report

    docs_dir = _resolve_path(args.docs_dir)
    cases, manifest_report = discover_documents(docs_dir)
    report["corpus"] = manifest_report
    if not cases:
        raise RuntimeError(f"No document cases found in {docs_dir}")

    yara = YaraDetector()
    router = CheapRouter()
    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    with build_classifier_worker_pool(
        prefix=args.env_prefix,
        max_workers=args.max_workers,
    ) as worker_pool:
        orchestrator = DocumentOrchestrator(
            yara=yara,
            pg=prompt_guard,
            router=router,
            worker_pool=worker_pool,
        )
        for case in cases:
            doc_report = analyze_case(orchestrator, chunker, case)
            report["documents"].append(doc_report)

    summary = _summarize_documents(report["documents"])
    report["summary"] = summary
    report["status"] = "passed" if summary["failed_documents"] == 0 else "failed"
    return 0 if report["status"] == "passed" else 1


def load_environment(env_file: str, override: bool) -> dict[str, Any]:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required to load .env for this runner.") from exc

    env_path = _resolve_path(env_file)
    loaded = load_dotenv(dotenv_path=env_path, override=override)
    return {
        "env_file": str(env_path),
        "env_file_exists": env_path.exists(),
        "env_loaded": bool(loaded),
        "env_override": override,
    }


def resolve_provider_report(prefix: str) -> dict[str, Any]:
    provider = os.getenv(f"{prefix}_PROVIDER", "openai").strip().lower()
    model = os.getenv(f"{prefix}_MODEL", "").strip() or None
    provider_api_key_envs = _provider_api_key_env_names(provider)
    key_sources = [f"{prefix}_API_KEY", *provider_api_key_envs]
    return {
        "classifier_env_prefix": prefix,
        "classifier_provider": provider,
        "classifier_model": model,
        "classifier_api_key_present": any(bool(os.getenv(name)) for name in key_sources),
        "classifier_api_key_sources_checked": key_sources,
    }


def ensure_supported_classifier_env(provider_report: Mapping[str, Any], prefix: str) -> None:
    provider = str(provider_report["classifier_provider"])
    if provider not in SUPPORTED_ENV_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_ENV_PROVIDERS))
        raise RuntimeError(
            f"{prefix}_PROVIDER must be Anthropic/OpenAI backed for this runner. "
            f"Got '{provider}'. Supported values: {supported}."
        )

    if not provider_report["classifier_api_key_present"]:
        key_hint = " or ".join(provider_report["classifier_api_key_sources_checked"])
        raise RuntimeError(f"Missing classifier API key. Set {key_hint}.")


def _provider_api_key_env_names(provider: str) -> list[str]:
    if provider in OPENAI_ENV_PROVIDERS:
        return ["OPENAI_API_KEY"]
    if provider in ANTHROPIC_ENV_PROVIDERS:
        return ["ANTHROPIC_API_KEY"]
    return []


def build_prompt_guard(
    args: argparse.Namespace,
) -> tuple[Optional[PromptGuardDetector], dict[str, Any]]:
    if args.disable_prompt_guard:
        return None, {
            "enabled": False,
            "status": "disabled_by_flag",
            "model": args.prompt_guard_model,
            "device": args.prompt_guard_device,
        }

    try:
        detector = PromptGuardDetector(
            model=args.prompt_guard_model,
            device=args.prompt_guard_device,
            eager_load=True,
        )
        return detector, {
            "enabled": True,
            "status": "enabled",
            "model": args.prompt_guard_model,
            "device": args.prompt_guard_device,
        }
    except Exception as exc:
        if not args.allow_prompt_guard_unavailable:
            raise

        LOGGER.info("PromptGuard unavailable; continuing with it disabled: %s", exc)
        return None, {
            "enabled": False,
            "status": "unavailable_disabled_by_flag",
            "model": args.prompt_guard_model,
            "device": args.prompt_guard_device,
            "error": _exception_report(exc, include_traceback=False),
        }


def discover_documents(docs_dir: Path) -> tuple[list[DocumentCase], dict[str, Any]]:
    manifest_path = docs_dir / "manifest_expected_results.json"
    cases: list[DocumentCase] = []
    manifest: dict[str, Any] = {
        "docs_dir": str(docs_dir),
        "manifest_path": str(manifest_path),
        "manifest_found": manifest_path.exists(),
    }

    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in data.get("files", []):
            if not isinstance(item, Mapping):
                continue
            filename = str(item.get("filename", "")).strip()
            if not filename:
                continue
            cases.append(DocumentCase(path=docs_dir / filename, expected=dict(item)))
        manifest["manifest_description"] = data.get("description")
        manifest["recommended_assertions"] = data.get("recommended_assertions", [])
        manifest["case_count"] = len(cases)
        return cases, manifest

    for path in sorted(p for p in docs_dir.rglob("*") if p.is_file()):
        if path.name.startswith(".") or path.suffix.lower() == ".json":
            continue
        cases.append(DocumentCase(path=path, expected={}))

    manifest["case_count"] = len(cases)
    return cases, manifest


def analyze_case(
    orchestrator: DocumentOrchestrator,
    chunker: TextChunker,
    case: DocumentCase,
) -> dict[str, Any]:
    start_time = time.monotonic()
    base_report: dict[str, Any] = {
        "filename": case.path.name,
        "path": str(case.path),
        "expected": dict(case.expected),
        "status": "running",
        "duration_seconds": None,
    }

    if not case.path.exists():
        base_report.update(
            {
                "status": "failed",
                "error": {
                    "type": "FileNotFoundError",
                    "message": f"Document listed in manifest does not exist: {case.path}",
                },
                "matches_expected": False,
                "expectation_checks": [],
                "safe_to_forward": False,
            }
        )
        base_report["duration_seconds"] = round(time.monotonic() - start_time, 3)
        return base_report

    try:
        result = orchestrator.analyze_path(case.path, chunker=chunker)
        doc_report = document_result_report(result, case)
        doc_report["status"] = "completed"
        doc_report["duration_seconds"] = round(time.monotonic() - start_time, 3)
        return doc_report
    except Exception as exc:
        LOGGER.debug("Document failed: %s", case.path, exc_info=True)
        base_report.update(
            {
                "status": "failed",
                "error": _exception_report(exc),
                "matches_expected": False,
                "expectation_checks": [],
                "safe_to_forward": False,
            }
        )
        base_report["duration_seconds"] = round(time.monotonic() - start_time, 3)
        return base_report


def document_result_report(
    result: DocumentAnalysisResult,
    case: DocumentCase,
) -> dict[str, Any]:
    chunks = [chunk_result_report(chunk_result) for chunk_result in result.chunk_results]
    safe_to_forward = result.verdict == "safe" and all(
        chunk["visited"] and chunk["final_verdict"] == "safe" for chunk in chunks
    )
    expectation_checks, matches_expected = compare_expectations(
        expected=case.expected,
        actual_verdict=result.verdict,
        safe_to_forward=safe_to_forward,
    )
    decision_counts = Counter(chunk["cheap_decision"] for chunk in chunks)
    final_verdict_counts = Counter(chunk["final_verdict"] for chunk in chunks)
    llm_verdict_counts = Counter(
        chunk["llm_classification"]["verdict"]
        for chunk in chunks
        if chunk["llm_classification"] is not None
    )
    not_visited = sorted(
        set(range(len(result.ingested_document.chunks)))
        - {chunk["chunk_index"] for chunk in chunks if chunk["visited"]}
    )

    return {
        "filename": case.path.name,
        "path": str(case.path),
        "expected": dict(case.expected),
        "status": "completed",
        "document_verdict": result.verdict,
        "safe_to_forward": safe_to_forward,
        "blocked_from_forwarding": not safe_to_forward,
        "reasons": list(result.reasons),
        "matches_expected": matches_expected,
        "expectation_checks": expectation_checks,
        "ingestion": {
            "source": result.ingested_document.source,
            "text_chars": len(result.ingested_document.text),
            "converter_metadata": dict(result.ingested_document.document.metadata),
            "mime_type": result.ingested_document.document.mime_type,
        },
        "chunks_total": len(chunks),
        "chunks_visited": len(chunks),
        "not_visited_chunk_indices": not_visited,
        "chunks_routed_to_llm": sum(1 for chunk in chunks if chunk["routed_to_llm"]),
        "cheap_decision_counts": dict(sorted(decision_counts.items())),
        "final_verdict_counts": dict(sorted(final_verdict_counts.items())),
        "llm_verdict_counts": dict(sorted(llm_verdict_counts.items())),
        "yara_findings_total": sum(chunk["yara_findings_count"] for chunk in chunks),
        "prompt_guard_findings_total": sum(
            chunk["prompt_guard_findings_count"] for chunk in chunks
        ),
        "max_yara_score": _max_or_zero(chunk["yara_score"] for chunk in chunks),
        "max_prompt_guard_score": _max_or_zero(chunk["prompt_guard_score"] for chunk in chunks),
        "chunks": chunks,
    }


def chunk_result_report(chunk_result: Any) -> dict[str, Any]:
    decision = chunk_result.cheap_decision
    cheap_findings = [
        detection_finding_report(finding) for finding in chunk_result.cheap_findings
    ]
    yara_findings_count = sum(
        1 for finding in cheap_findings if finding["detector"] == "YaraDetector"
    )
    prompt_guard_findings_count = sum(
        1 for finding in cheap_findings if finding["rule_id"] == "prompt_guard"
    )
    return {
        "visited": True,
        "chunk_index": chunk_result.chunk_index,
        "source": chunk_result.chunk.source,
        "start_char": chunk_result.chunk.start_char,
        "end_char": chunk_result.chunk.end_char,
        "char_length": chunk_result.chunk.length,
        "cheap_decision": decision.decision,
        "cheap_reason": decision.reason,
        "risk_score": round(float(decision.risk_score), 4),
        "yara_score": round(float(decision.yara_score), 4),
        "prompt_guard_score": round(float(decision.pg_score), 4),
        "routed_to_llm": bool(chunk_result.routed_to_llm),
        "final_verdict": chunk_result.final_verdict,
        "cheap_findings": cheap_findings,
        "yara_findings_count": yara_findings_count,
        "prompt_guard_findings_count": prompt_guard_findings_count,
        "llm_classification": classification_report(chunk_result.llm_classification),
    }


def detection_finding_report(finding: Any) -> dict[str, Any]:
    metadata = dict(finding.metadata or {})
    return {
        "span": finding.span,
        "category": finding.category,
        "severity": finding.severity,
        "reason": finding.reason,
        "start_char": finding.start_char,
        "end_char": finding.end_char,
        "source": finding.source,
        "rule_id": finding.rule_id,
        "requires_llm_validation": bool(finding.requires_llm_validation),
        "score": finding.score,
        "detector": metadata.get("detector"),
        "metadata": metadata,
    }


def classification_report(classification: Any) -> Optional[dict[str, Any]]:
    if classification is None:
        return None
    return {
        "verdict": classification.verdict,
        "confidence": classification.confidence,
        "reasons": list(classification.reasons),
        "findings": [
            {
                "span": finding.span,
                "attack_type": finding.attack_type,
                "severity": finding.severity,
                "reason": finding.reason,
                "start_char": finding.start_char,
                "end_char": finding.end_char,
            }
            for finding in classification.findings
        ],
        "raw_response_chars": (
            len(classification.raw_response) if classification.raw_response else 0
        ),
    }


def compare_expectations(
    *,
    expected: Mapping[str, Any],
    actual_verdict: str,
    safe_to_forward: bool,
) -> tuple[list[dict[str, Any]], Optional[bool]]:
    checks: list[dict[str, Any]] = []

    if "expected_document_decision" in expected:
        expected_verdict = str(expected["expected_document_decision"]).strip().lower()
        checks.append(
            {
                "field": "document_verdict",
                "expected": expected_verdict,
                "actual": actual_verdict,
                "matches": actual_verdict == expected_verdict,
            }
        )

    if "expected_safe_to_forward" in expected:
        expected_safe = bool(expected["expected_safe_to_forward"])
        checks.append(
            {
                "field": "safe_to_forward",
                "expected": expected_safe,
                "actual": safe_to_forward,
                "matches": safe_to_forward == expected_safe,
            }
        )

    if not checks:
        return checks, None
    return checks, all(check["matches"] for check in checks)


def _summarize_documents(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [doc for doc in documents if doc.get("status") == "completed"]
    failed = [doc for doc in documents if doc.get("status") != "completed"]
    expectation_mismatches = [
        doc for doc in documents if doc.get("matches_expected") is False
    ]
    blocked = [doc for doc in documents if not doc.get("safe_to_forward", False)]
    verdict_counts = Counter(
        str(doc.get("document_verdict", "failed")) for doc in documents
    )

    failed_documents = len(failed) + sum(
        1 for doc in completed if doc.get("matches_expected") is False
    )

    return {
        "setup_failed": False,
        "total_documents": len(documents),
        "completed_documents": len(completed),
        "failed_documents": failed_documents,
        "runtime_failed_documents": len(failed),
        "expectation_mismatches": len(expectation_mismatches),
        "safe_to_forward_documents": sum(
            1 for doc in completed if doc.get("safe_to_forward") is True
        ),
        "blocked_from_forwarding_documents": len(blocked),
        "document_verdict_counts": dict(sorted(verdict_counts.items())),
        "total_chunks": sum(int(doc.get("chunks_total", 0)) for doc in completed),
        "routed_to_llm_chunks": sum(
            int(doc.get("chunks_routed_to_llm", 0)) for doc in completed
        ),
        "yara_findings_total": sum(
            int(doc.get("yara_findings_total", 0)) for doc in completed
        ),
        "prompt_guard_findings_total": sum(
            int(doc.get("prompt_guard_findings_total", 0)) for doc in completed
        ),
    }


def write_report(report_path: Path, report: Mapping[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_console_summary(report_path: Path, report: Mapping[str, Any]) -> None:
    env = report.get("environment", {})
    pg = report.get("prompt_guard", {})
    summary = report.get("summary", {})
    print(
        "chunk-safety e2e: "
        f"status={report.get('status')} "
        f"docs={summary.get('completed_documents', 0)}/{summary.get('total_documents', 0)} "
        f"provider={env.get('classifier_provider', 'unknown')} "
        f"prompt_guard={pg.get('status', 'unknown')}"
    )

    for doc in report.get("documents", []):
        print(_document_console_line(doc))

    if report.get("setup_error"):
        setup_error = report["setup_error"]
        print(f"setup_error={setup_error.get('type')}: {_truncate(setup_error.get('message', ''))}")

    print(f"report={_display_path(report_path)}")


def _document_console_line(doc: Mapping[str, Any]) -> str:
    if doc.get("status") != "completed":
        error = doc.get("error", {})
        return (
            f"FAIL {doc.get('filename')} status={doc.get('status')} "
            f"error={error.get('type')}: {_truncate(error.get('message', ''))}"
        )

    matches_expected = doc.get("matches_expected")
    token = "PASS" if matches_expected is not False else "FAIL"
    expected = doc.get("expected", {}).get("expected_document_decision", "n/a")
    return (
        f"{token} {doc.get('filename')} "
        f"verdict={doc.get('document_verdict')} expected={expected} "
        f"forward={doc.get('safe_to_forward')} "
        f"chunks={doc.get('chunks_total')} routed={doc.get('chunks_routed_to_llm')} "
        f"yara={doc.get('yara_findings_total')} "
        f"pg_max={doc.get('max_prompt_guard_score')}"
    )


def _configuration_report(args: argparse.Namespace, report_path: Path) -> dict[str, Any]:
    return {
        "docs_dir": str(_resolve_path(args.docs_dir)),
        "report_path": str(report_path),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "max_workers": args.max_workers,
        "disable_prompt_guard": args.disable_prompt_guard,
        "allow_prompt_guard_unavailable": args.allow_prompt_guard_unavailable,
        "prompt_guard_model": args.prompt_guard_model,
        "prompt_guard_device": args.prompt_guard_device,
        "log_level": args.log_level,
        "log_file": str(_resolve_path(args.log_file)) if args.log_file else None,
    }


def _exception_report(exc: Exception, *, include_traceback: bool = True) -> dict[str, Any]:
    data = {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }
    if include_traceback:
        data["traceback"] = traceback.format_exc().splitlines()
    return data


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _max_or_zero(values: Sequence[float]) -> float:
    values_tuple = tuple(values)
    if not values_tuple:
        return 0.0
    return round(float(max(values_tuple)), 4)


def _truncate(value: Any, limit: int = 180) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


if __name__ == "__main__":
    raise SystemExit(main())
