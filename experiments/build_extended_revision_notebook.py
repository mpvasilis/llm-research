"""Build and execute the portable higher-budget successor notebook bundle."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = ROOT / "output" / "revision" / "full_study" / "notebook_bundle_manifest_v2.json"
DEFAULT_NOTEBOOK = ROOT / "output" / "jupyter-notebook" / "advice_marker_extended_study.ipynb"
DEFAULT_BUNDLE = ROOT / "output" / "jupyter-notebook" / "advice_marker_extended_study_bundle.zip"
DEFAULT_REPORT_ROOT = ROOT / "output" / "revision" / "extended_notebook"
DEFAULT_COMPLETED_PROBE_BUNDLE = ROOT / "output" / "jupyter-notebook" / "advice_marker_completed_probe_bundle.zip"
DEFAULT_PARENT = ROOT / "output" / "revision" / "full_study" / "protocol_v2.json"
DEFAULT_PROBE_MANIFEST = ROOT / "output" / "revision" / "development_budget_probe" / "development_1024_probe_manifest_v1.json"
DEFAULT_PROBE_RUN_DIR = ROOT / "output" / "revision" / "development_budget_probe" / "runs" / "budget_probe_run_9acdc0e5ce18effd34337b139ff53c24"
DEFAULT_PROBE_EXPORT = ROOT / "output" / "revision" / "development_budget_probe" / "exports" / "development_1024_eight_responses.jsonl"
DEFAULT_PROBE_VERIFICATION = ROOT / "output" / "revision" / "development_budget_probe" / "development_1024_verification_20260915.json"
OLD_NOTEBOOK = ROOT / "output" / "jupyter-notebook" / "advice_marker_revision_study.ipynb"
OLD_BUNDLE = ROOT / "output" / "jupyter-notebook" / "advice_marker_revision_study_bundle.zip"
OLD_NOTEBOOK_SHA256 = "186159433271d0439af706ebe4e43c8b3af722f94b202fefbebd0b7d65730a4b"
OLD_BUNDLE_SHA256 = "aa3f26293432292f1058c71f5b00996e49174bbd93e304e4aa52ec5c7b33524b"
PINNED_EMBEDDING_MODEL_ID = "intfloat/e5-small-v2"
PINNED_EMBEDDING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
PINNED_PREFIX_TOKENIZER_MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
PINNED_PREFIX_TOKENIZER_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
ADDITIONAL_FILES = (
    "config.py",
    "experiments/build_extended_revision_notebook.py",
    "experiments/extended_revision_study.py",
    "experiments/extended_calibration_study.py",
    "experiments/extended_prospective_validation.py",
    "experiments/extended_sensitivity_analysis.py",
    "output/revision/development_budget_probe/development_1024_probe_manifest_v1.json",
    "out/paper/DEVELOPMENT_BUDGET_DECISION_2026-09-15.md",
    "out/paper/EXTENDED_STUDY_PROTOCOL_2026-09-15.md",
    "out/paper/EXTENDED_VALIDATION_PROTOCOL_2026-09-15.md",
    "out/paper/EXTENDED_SENSITIVITY_PROTOCOL_2026-09-15.md",
    "out/paper/EXTENDED_NOTEBOOK_GUIDE_2026-09-15.md",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _verify_preserved_artifacts() -> dict[str, str]:
    actual = {"old_notebook": sha256_file(OLD_NOTEBOOK), "old_bundle": sha256_file(OLD_BUNDLE)}
    expected = {"old_notebook": OLD_NOTEBOOK_SHA256, "old_bundle": OLD_BUNDLE_SHA256}
    if actual != expected:
        raise ValueError(f"Preserved revision notebook artifacts changed: {actual}")
    return actual


def _baseline_files() -> dict[str, dict[str, Any]]:
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if len(files) != 25:
        raise ValueError("The verified baseline bundle must contain exactly 25 files")
    for relative, identity in files.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != identity["sha256"] or path.stat().st_size != identity["size_bytes"]:
            raise ValueError(f"Verified baseline file changed: {relative}")
    return files


def _declared_dependency_check(files: set[str]) -> None:
    from experiments import extended_revision_study as core

    missing = sorted(set(core.SOURCE_PATHS) - files)
    if missing:
        raise ValueError(f"Bundle omits declared successor dependencies: {missing}")


def _optional_probe_members(
    probe_run_dir: Path | None,
    probe_export: Path | None,
    budget_decision: Path | None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    supplied = (probe_run_dir, probe_export, budget_decision)
    if not any(supplied):
        return {}, {
            "status": "pending_complete_probe_run_export_and_budget_decision",
            "included": False,
            "reason": "The fixed manifest and preceding 256/512 evidence are bundled; no complete verified run, export, and decision package was supplied.",
        }
    if not all(supplied):
        raise ValueError("Optional completed-probe inclusion requires run directory, export, and budget decision together")
    from experiments import extended_revision_study as core

    manifest = json.loads(DEFAULT_PROBE_MANIFEST.read_text(encoding="utf-8"))
    prior_paths = core._probe_prior_paths_from_manifest(manifest)
    evidence = core._validate_probe_package(
        DEFAULT_PARENT, DEFAULT_PROBE_MANIFEST, Path(probe_run_dir), Path(probe_export), prior_paths
    )
    decision = json.loads(Path(budget_decision).read_text(encoding="utf-8"))
    parent = core._parent_protocol(DEFAULT_PARENT)
    core._validate_budget_decision(
        Path(budget_decision), parent, evidence, decision.get("primary_cap"), decision.get("sensitivity_cap")
    )
    members = {
        f"inputs/completed_probe/run/{item.relative_to(probe_run_dir).as_posix()}": item
        for item in sorted(Path(probe_run_dir).rglob("*")) if item.is_file()
    }
    members["inputs/completed_probe/export.jsonl"] = Path(probe_export)
    members["inputs/completed_probe/budget_decision.json"] = Path(budget_decision)
    return members, {
        "status": "complete_verified_probe_package_included",
        "included": True,
        "probe_manifest_id": evidence["probe_manifest_id"],
        "run_id": evidence["run_id"],
        "export_sha256": evidence["export_sha256"],
        "budget_decision_id": decision["budget_decision_id"],
    }


def build_bundle(
    bundle_path: Path,
    *,
    probe_run_dir: Path | None = None,
    probe_export: Path | None = None,
    budget_decision: Path | None = None,
) -> dict[str, Any]:
    preserved = _verify_preserved_artifacts()
    baseline = _baseline_files()
    relative_files = set(baseline) | set(ADDITIONAL_FILES)
    missing = sorted(relative for relative in relative_files if not (ROOT / relative).is_file())
    if missing:
        raise FileNotFoundError(f"Cannot build extended notebook bundle; missing files: {missing}")
    _declared_dependency_check(relative_files)
    members = {relative: ROOT / relative for relative in sorted(relative_files)}
    optional_members, probe_status = _optional_probe_members(probe_run_dir, probe_export, budget_decision)
    members.update(optional_members)
    files = {
        name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in sorted(members.items())
    }
    manifest_core = {
        "bundle_schema_version": "advice-marker-extended-study-notebook-bundle-v1",
        "provenance": "Built from hash-verified baseline files and current local successor sources.",
        "verified_baseline": {
            "manifest_sha256": sha256_file(BASE_MANIFEST),
            "files_verified": len(baseline),
            "preserved_artifacts": preserved,
        },
        "probe_input": probe_status,
        "detector_dependencies": {
            "embedding_model": {"model_id": PINNED_EMBEDDING_MODEL_ID, "revision": PINNED_EMBEDDING_REVISION},
            "prefix_window_tokenizer": {"model_id": PINNED_PREFIX_TOKENIZER_MODEL_ID, "revision": PINNED_PREFIX_TOKENIZER_REVISION, "download_scope": "tokenizer files only"},
        },
        "files": files,
    }
    manifest = {**manifest_core, "bundle_content_id": "extended_bundle_" + sha256_bytes(json.dumps(manifest_core, sort_keys=True, separators=(",", ":")).encode())[:32]}
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        def write_member(name: str, payload: bytes) -> None:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)

        write_member("bundle_manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
        for name, path in sorted(members.items()):
            write_member(name, path.read_bytes())
    os.replace(temporary, bundle_path)
    return {**manifest, "bundle_path": str(bundle_path), "bundle_sha256": sha256_file(bundle_path)}


def build_completed_probe_bundle(bundle_path: Path, source_bundle: dict[str, Any]) -> dict[str, Any]:
    """Package only the strictly complete probe artifacts that complement the source bundle."""
    from experiments import extended_revision_study as core

    manifest = json.loads(DEFAULT_PROBE_MANIFEST.read_text(encoding="utf-8"))
    prior_paths = core._probe_prior_paths_from_manifest(manifest)
    evidence = core._validate_probe_package(
        DEFAULT_PARENT, DEFAULT_PROBE_MANIFEST, DEFAULT_PROBE_RUN_DIR, DEFAULT_PROBE_EXPORT, prior_paths
    )
    verification = json.loads(DEFAULT_PROBE_VERIFICATION.read_text(encoding="utf-8"))
    if (verification.get("status") != "passed"
            or verification.get("run_id") != evidence["run_id"]
            or verification.get("probe_manifest_id") != evidence["probe_manifest_id"]
            or verification.get("verification", {}).get("verified_responses") != 8
            or verification.get("verification", {}).get("missing_responses") != 0
            or verification.get("resume", {}).get("new_responses") != 0
            or verification.get("resume_all_run_file_hashes_unchanged") is not True
            or verification.get("export", {}).get("output_sha256") != evidence["export_sha256"]):
        raise ValueError("Final fixed-probe verification does not match the strict complete probe package")
    members = {
        DEFAULT_PROBE_MANIFEST.relative_to(ROOT).as_posix(): DEFAULT_PROBE_MANIFEST,
        DEFAULT_PROBE_EXPORT.relative_to(ROOT).as_posix(): DEFAULT_PROBE_EXPORT,
        DEFAULT_PROBE_VERIFICATION.relative_to(ROOT).as_posix(): DEFAULT_PROBE_VERIFICATION,
    }
    members.update({
        item.relative_to(ROOT).as_posix(): item
        for item in sorted(DEFAULT_PROBE_RUN_DIR.rglob("*")) if item.is_file()
    })
    files = {
        name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in sorted(members.items())
    }
    manifest_core = {
        "manifest_schema_version": "advice-marker-completed-probe-bundle-v1",
        "classification": "completed_exploratory_probe_evidence_not_a_budget_decision_or_successor_freeze",
        "probe_manifest_id": evidence["probe_manifest_id"], "run_id": evidence["run_id"],
        "verified_responses": evidence["verified_responses"], "probe_export_sha256": evidence["export_sha256"],
        "source_bundle_compatibility": {
            "bundle_content_id": source_bundle["bundle_content_id"],
            "bundle_sha256": source_bundle["bundle_sha256"],
        },
        "files": files,
    }
    result = {**manifest_core, "completed_probe_bundle_id": "completed_probe_bundle_" + sha256_bytes(json.dumps(manifest_core, sort_keys=True, separators=(",", ":")).encode())[:32]}
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    os.replace(temporary, bundle_path)
    with zipfile.ZipFile(bundle_path) as archive:
        if sorted(archive.namelist()) != sorted(files):
            raise ValueError("Completed-probe bundle member set mismatch")
        for name, identity in files.items():
            payload = archive.read(name)
            if sha256_bytes(payload) != identity["sha256"] or len(payload) != identity["size_bytes"]:
                raise ValueError(f"Completed-probe bundle member mismatch: {name}")
    return {**result, "bundle_path": str(bundle_path), "bundle_sha256": sha256_file(bundle_path)}


def build_notebook(notebook_path: Path, bundle: dict[str, Any]) -> None:
    bundle_sha = bundle["bundle_sha256"]
    cells = [
        new_markdown_cell("""# Advice-marker extended higher-budget study

This notebook carries the complete successor workflow while keeping every mutation, network action, model load, and human-data step disabled by default. The uploaded bundle is verified member by member before source imports. A candidate cap is never inferred: both caps remain unset until a separate completed probe and budget decision exist."""),
        new_code_cell(f'''from pathlib import Path
import hashlib, json, os, subprocess, sys, zipfile

EXPECTED_BUNDLE_SHA256 = "{bundle_sha}"
PINNED_EMBEDDING_MODEL_ID = "{PINNED_EMBEDDING_MODEL_ID}"
PINNED_EMBEDDING_REVISION = "{PINNED_EMBEDDING_REVISION}"
PINNED_PREFIX_TOKENIZER_MODEL_ID = "{PINNED_PREFIX_TOKENIZER_MODEL_ID}"
PINNED_PREFIX_TOKENIZER_REVISION = "{PINNED_PREFIX_TOKENIZER_REVISION}"
LOCAL_VALIDATION = os.environ.get("EXTENDED_NOTEBOOK_LOCAL_VALIDATION") == "1"

def bootstrap_fs_path(path):
    value = os.fspath(path)
    if os.name != "nt":
        return Path(value)
    if value.startswith("\\\\\\\\?\\\\"):
        return Path(value)
    absolute = os.path.abspath(value)
    return Path("\\\\\\\\?\\\\UNC\\\\" + absolute[2:] if absolute.startswith("\\\\\\\\") else "\\\\\\\\?\\\\" + absolute)

PRIMARY_CAP = None
SENSITIVITY_CAP = None
USE_GOOGLE_DRIVE = False
INSTALL_DEPENDENCIES = False
DOWNLOAD_DETECTORS = False
FREEZE_SUCCESSOR_PROTOCOL = False
CREATE_HUMAN_REVIEW_FORM = False
FREEZE_CALIBRATION_MANIFEST = False
RUN_CALIBRATION = False
EXPORT_CALIBRATION = False
FREEZE_VALIDATION_PLAN = False
RUN_PRIMARY = False
RUN_SENSITIVITY = False
EXPORT_PRIMARY = False
EXPORT_SENSITIVITY = False
RUN_PRIMARY_PREDICTIONS = False
RUN_SENSITIVITY_PREDICTIONS = False
RUN_PRIMARY_ANALYSIS = False
RUN_SENSITIVITY_ANALYSIS = False
PREPARE_VALIDATION_PACKET = False
INIT_HUMAN_PASS_1 = False
INIT_HUMAN_PASS_2 = False
RUN_HUMAN_SCORING = False
DEVICE = "auto"
TORCH_THREADS = 4
ANNOTATOR_1_ID = None
ANNOTATOR_2_ID = None

ACTION_CONTROLS = {{
    "use_google_drive": USE_GOOGLE_DRIVE, "install_dependencies": INSTALL_DEPENDENCIES,
    "download_detectors": DOWNLOAD_DETECTORS, "freeze_successor_protocol": FREEZE_SUCCESSOR_PROTOCOL,
    "create_human_review_form": CREATE_HUMAN_REVIEW_FORM, "freeze_calibration_manifest": FREEZE_CALIBRATION_MANIFEST,
    "run_calibration": RUN_CALIBRATION, "export_calibration": EXPORT_CALIBRATION,
    "freeze_validation_plan": FREEZE_VALIDATION_PLAN, "run_primary": RUN_PRIMARY,
    "run_sensitivity": RUN_SENSITIVITY, "export_primary": EXPORT_PRIMARY,
    "export_sensitivity": EXPORT_SENSITIVITY, "run_primary_predictions": RUN_PRIMARY_PREDICTIONS,
    "run_sensitivity_predictions": RUN_SENSITIVITY_PREDICTIONS, "run_primary_analysis": RUN_PRIMARY_ANALYSIS,
    "run_sensitivity_analysis": RUN_SENSITIVITY_ANALYSIS, "prepare_validation_packet": PREPARE_VALIDATION_PACKET,
    "init_human_pass_1": INIT_HUMAN_PASS_1, "init_human_pass_2": INIT_HUMAN_PASS_2,
    "run_human_scoring": RUN_HUMAN_SCORING,
}}
print({{"local_validation": LOCAL_VALIDATION, "primary_cap": PRIMARY_CAP, "sensitivity_cap": SENSITIVITY_CAP, "all_action_controls_disabled": not any(ACTION_CONTROLS.values()), "action_controls": ACTION_CONTROLS}})'''),
        new_markdown_cell("## Verify and open the source bundle\n\nOutside local validation, upload exactly this bundle. Package installation and Drive mounting require their own switches."),
        new_code_cell('''if INSTALL_DEPENDENCIES:
    if LOCAL_VALIDATION:
        raise ValueError("Dependency installation is disabled during local validation.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.57,<5", "huggingface_hub>=0.34,<2", "sentence-transformers>=5,<6", "accelerate>=1.10,<2", "safetensors>=0.6,<1", "numpy>=1.26,<3", "matplotlib>=3.8,<4"], check=True)
else:
    print("Dependency installation skipped.")'''),
        new_code_cell('''bundle_env = os.environ.get("EXTENDED_NOTEBOOK_BUNDLE")
if bundle_env:
    bundle_path = bootstrap_fs_path(bundle_env)
else:
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError("Set EXTENDED_NOTEBOOK_BUNDLE outside Colab.") from exc
    uploaded = files.upload()
    candidates = [Path(name) for name in uploaded if name.endswith(".zip")]
    if len(candidates) != 1:
        raise ValueError("Upload exactly one extended-study bundle.")
    bundle_path = bootstrap_fs_path(candidates[0])
if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != EXPECTED_BUNDLE_SHA256:
    raise ValueError("Bundle SHA-256 mismatch.")
with zipfile.ZipFile(bundle_path) as archive:
    manifest = json.loads(archive.read("bundle_manifest.json"))
    for info in archive.infolist():
        member = Path(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe archive member: {info.filename}")
    workspace_parent = bootstrap_fs_path(os.environ.get("EXTENDED_NOTEBOOK_WORKSPACE", "/tmp/advice_marker_extended_bundles"))
    workspace = workspace_parent / manifest["bundle_content_id"]
    workspace.mkdir(parents=True, exist_ok=True)
    archive.extractall(workspace)
for relative, expected in manifest["files"].items():
    target = workspace / relative
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected["sha256"] or target.stat().st_size != expected["size_bytes"]:
        raise ValueError(f"Bundled file mismatch: {relative}")
print({"bundle_content_id": manifest["bundle_content_id"], "verified_files": len(manifest["files"]), "baseline_files": manifest["verified_baseline"]["files_verified"], "probe_input_status": manifest["probe_input"]["status"]})'''),
        new_code_cell('''if USE_GOOGLE_DRIVE:
    if LOCAL_VALIDATION:
        raise ValueError("Drive mounting is disabled during local validation.")
    from google.colab import drive
    drive.mount("/content/drive")
    CACHE_ROOT = Path("/content/drive/MyDrive/advice_marker_extended_study")
else:
    CACHE_ROOT = Path(os.environ.get("EXTENDED_NOTEBOOK_CACHE", "/tmp/advice_marker_extended_cache"))
sys.path.insert(0, str(workspace))
from experiments import extended_revision_study as core
from experiments import extended_calibration_study as calibration
from experiments import extended_prospective_validation as prospective
from experiments import extended_sensitivity_analysis as sensitivity_analysis
from experiments import development_budget_probe as budget_probe
from experiments import revision_analysis, validation_v4

CACHE_ROOT = core.fs_path(CACHE_ROOT)
workspace = core.fs_path(workspace)
parent_protocol_path = core.fs_path(workspace / "output/revision/full_study/protocol_v2.json")
fixed_probe_manifest_path = core.fs_path(workspace / "output/revision/development_budget_probe/development_1024_probe_manifest_v1.json")
annotation_guide_path = core.fs_path(workspace / "out/paper/ANNOTATION_GUIDE.md")
parent_protocol = core._parent_protocol(parent_protocol_path)
fixed_probe_manifest = budget_probe.load_manifest(fixed_probe_manifest_path, parent_protocol_path)
print({"imports": [core.__name__, calibration.__name__, prospective.__name__, sensitivity_analysis.__name__], "parent_protocol_id": parent_protocol["protocol_id"], "fixed_probe_manifest_id": fixed_probe_manifest["probe_manifest_id"], "fixed_probe_expected": fixed_probe_manifest["expected_responses"]})'''),
        new_markdown_cell("## Completed probe and explicit budget decision\n\nThe source bundle contains the fixed manifest and preceding 256/512 evidence. The separately verified completed-probe companion supplies the 8/8 run and exact export under their repository-relative paths. A budget decision and both caps remain unset and must be supplied separately before freezing."),
        new_code_cell('''PROBE_RUN_DIR = core.fs_path(os.environ["EXTENDED_PROBE_RUN_DIR"]) if os.environ.get("EXTENDED_PROBE_RUN_DIR") else None
PROBE_EXPORT_PATH = core.fs_path(os.environ["EXTENDED_PROBE_EXPORT"]) if os.environ.get("EXTENDED_PROBE_EXPORT") else None
BUDGET_DECISION_PATH = core.fs_path(os.environ["EXTENDED_BUDGET_DECISION"]) if os.environ.get("EXTENDED_BUDGET_DECISION") else None
SUCCESSOR_PROTOCOL_PATH = core.fs_path(os.environ.get("EXTENDED_PROTOCOL", str(CACHE_ROOT / "full_study/protocol_v1.json")))
protocol = core.load_protocol(SUCCESSOR_PROTOCOL_PATH) if SUCCESSOR_PROTOCOL_PATH.is_file() else None
if PRIMARY_CAP is None or SENSITIVITY_CAP is None:
    print("Pending: set both explicit caps only after the completed probe and decision are available.")
else:
    candidate = core.build_protocol(parent_protocol_path, PRIMARY_CAP, SENSITIVITY_CAP)
    print({"candidate_protocol_id": candidate["candidate_protocol_id"], "primary": len(candidate["primary_grid"]), "sensitivity": len(candidate["longer_budget_sensitivity"]["grid"])})
if FREEZE_SUCCESSOR_PROTOCOL:
    if None in (PRIMARY_CAP, SENSITIVITY_CAP, PROBE_RUN_DIR, PROBE_EXPORT_PATH, BUDGET_DECISION_PATH):
        raise ValueError("Freezing requires explicit caps, complete probe run/export, and budget decision.")
    protocol = core.freeze_protocol(parent_protocol_path, PRIMARY_CAP, SENSITIVITY_CAP, SUCCESSOR_PROTOCOL_PATH, probe_manifest_path=fixed_probe_manifest_path, probe_run_dir=PROBE_RUN_DIR, probe_export_path=PROBE_EXPORT_PATH, budget_decision_path=BUDGET_DECISION_PATH)
    print({"frozen_protocol_id": protocol["protocol_id"]})
else:
    print("Successor protocol freezing disabled.")'''),
        new_markdown_cell("## Genuine prompt review and 96-response calibration"),
        new_code_cell('''HUMAN_REVIEW_PATH = core.fs_path(os.environ.get("EXTENDED_HUMAN_REVIEW", str(CACHE_ROOT / "human_prompt_review.json")))
if CREATE_HUMAN_REVIEW_FORM:
    if protocol is None:
        raise ValueError("Freeze the successor protocol before creating its blank review form.")
    core.create_human_review_manifest(protocol, HUMAN_REVIEW_PATH)
    print({"blank_review_form": str(HUMAN_REVIEW_PATH), "human_fields_completed": 0})
elif protocol is not None and HUMAN_REVIEW_PATH.is_file():
    core.validate_human_review(protocol, HUMAN_REVIEW_PATH)
    print("Genuine completed prompt review validated.")
else:
    print("Prompt review pending; no form or judgments were created.")'''),
        new_code_cell('''CALIBRATION_ROOT = core.fs_path(CACHE_ROOT / "calibration_study")
CALIBRATION_MANIFEST_PATH = core.fs_path(os.environ.get("EXTENDED_CALIBRATION_MANIFEST", str(CALIBRATION_ROOT / "manifest.json")))
CALIBRATION_RUN_DIR = core.fs_path(os.environ["EXTENDED_CALIBRATION_RUN_DIR"]) if os.environ.get("EXTENDED_CALIBRATION_RUN_DIR") else None
CALIBRATION_RESPONSES_PATH = core.fs_path(os.environ.get("EXTENDED_CALIBRATION_RESPONSES", str(CALIBRATION_ROOT / "responses.jsonl")))
calibration_result = None
if FREEZE_CALIBRATION_MANIFEST:
    if protocol is None or PRIMARY_CAP is None:
        raise ValueError("Frozen successor protocol and explicit primary cap required.")
    calibration.freeze_manifest(SUCCESSOR_PROTOCOL_PATH, PRIMARY_CAP, CALIBRATION_MANIFEST_PATH)
if RUN_CALIBRATION:
    calibration_result = calibration.run_generation(CALIBRATION_MANIFEST_PATH, SUCCESSOR_PROTOCOL_PATH, CALIBRATION_ROOT, DEVICE, False, TORCH_THREADS)
    CALIBRATION_RUN_DIR = core.fs_path(calibration_result["run_dir"])
if EXPORT_CALIBRATION:
    calibration.export_run(CALIBRATION_MANIFEST_PATH, SUCCESSOR_PROTOCOL_PATH, CALIBRATION_RUN_DIR, CALIBRATION_RESPONSES_PATH)
if protocol is not None and CALIBRATION_MANIFEST_PATH.is_file() and CALIBRATION_RUN_DIR is not None and CALIBRATION_RESPONSES_PATH.is_file():
    report = calibration.verify_run(CALIBRATION_MANIFEST_PATH, SUCCESSOR_PROTOCOL_PATH, CALIBRATION_RUN_DIR, require_complete=True)
    print({"calibration": report, "responses_export": str(CALIBRATION_RESPONSES_PATH)})
else:
    print("Calibration pending: requires an exact frozen 96-response manifest, complete run, and export.")'''),
        new_markdown_cell("## Human-reviewed development evidence and prospective plan\n\nAt least 60 distinct calibration responses must be genuinely reviewed, with a nonempty timed subset recorded in the evidence file. The validator enforces these requirements; this notebook never fills human fields."),
        new_code_cell('''DEVELOPMENT_EVIDENCE_PATH = core.fs_path(os.environ.get("EXTENDED_DEVELOPMENT_EVIDENCE", str(CALIBRATION_ROOT / "human_development_evidence.json")))
VALIDATION_PLAN_PATH = core.fs_path(os.environ.get("EXTENDED_VALIDATION_PLAN", str(CACHE_ROOT / "prospective_validation/sampling_plan_v1.json")))
if FREEZE_VALIDATION_PLAN:
    prospective.freeze_plan(SUCCESSOR_PROTOCOL_PATH, CALIBRATION_MANIFEST_PATH, CALIBRATION_RESPONSES_PATH, CALIBRATION_RUN_DIR, DEVELOPMENT_EVIDENCE_PATH, VALIDATION_PLAN_PATH, annotation_guide_path)
if protocol is not None and all(path is not None and core.fs_path(path).exists() for path in (CALIBRATION_MANIFEST_PATH, CALIBRATION_RESPONSES_PATH, CALIBRATION_RUN_DIR, DEVELOPMENT_EVIDENCE_PATH)):
    if VALIDATION_PLAN_PATH.is_file():
        print(prospective.validate_plan(VALIDATION_PLAN_PATH, SUCCESSOR_PROTOCOL_PATH, CALIBRATION_MANIFEST_PATH, CALIBRATION_RESPONSES_PATH, CALIBRATION_RUN_DIR, DEVELOPMENT_EVIDENCE_PATH, annotation_guide_path))
    else:
        print(prospective.preview_plan(SUCCESSOR_PROTOCOL_PATH, CALIBRATION_MANIFEST_PATH, CALIBRATION_RESPONSES_PATH, CALIBRATION_RUN_DIR, annotation_guide_path))
else:
    print("Prospective plan pending complete calibration and genuine reviewed/timed development evidence.")

def readiness_mapping():
    return {"plan_path": VALIDATION_PLAN_PATH, "calibration_manifest_path": CALIBRATION_MANIFEST_PATH, "calibration_responses_path": CALIBRATION_RESPONSES_PATH, "calibration_run_dir": CALIBRATION_RUN_DIR, "evidence_path": DEVELOPMENT_EVIDENCE_PATH, "guide_path": annotation_guide_path}'''),
        new_markdown_cell("## Complete primary and longer-budget generation\n\nBoth runs use strict readiness gates and safe per-response resume. Primary is always 2,880 responses; sensitivity is always the pre-specified 192 responses."),
        new_code_cell('''PRIMARY_RUN_DIR = core.fs_path(os.environ["EXTENDED_PRIMARY_RUN_DIR"]) if os.environ.get("EXTENDED_PRIMARY_RUN_DIR") else None
SENSITIVITY_RUN_DIR = core.fs_path(os.environ["EXTENDED_SENSITIVITY_RUN_DIR"]) if os.environ.get("EXTENDED_SENSITIVITY_RUN_DIR") else None
primary_result = sensitivity_result = None
if RUN_PRIMARY:
    primary_result = core.run_generation(SUCCESSOR_PROTOCOL_PATH, "evaluation", CACHE_ROOT, None, HUMAN_REVIEW_PATH, DEVICE, False, TORCH_THREADS, readiness=readiness_mapping())
    PRIMARY_RUN_DIR = core.fs_path(primary_result["run_dir"])
if RUN_SENSITIVITY:
    sensitivity_result = core.run_generation(SUCCESSOR_PROTOCOL_PATH, "sensitivity", CACHE_ROOT, None, HUMAN_REVIEW_PATH, DEVICE, False, TORCH_THREADS, readiness=readiness_mapping())
    SENSITIVITY_RUN_DIR = core.fs_path(sensitivity_result["run_dir"])
print({"primary": primary_result or "disabled", "sensitivity": sensitivity_result or "disabled"})'''),
        new_code_cell('''PRIMARY_RESPONSES_PATH = core.fs_path(os.environ.get("EXTENDED_PRIMARY_RESPONSES", str(CACHE_ROOT / "exports/primary_responses.jsonl")))
SENSITIVITY_RESPONSES_PATH = core.fs_path(os.environ.get("EXTENDED_SENSITIVITY_RESPONSES", str(CACHE_ROOT / "exports/sensitivity_responses.jsonl")))
if EXPORT_PRIMARY:
    core.export_run(SUCCESSOR_PROTOCOL_PATH, PRIMARY_RUN_DIR, PRIMARY_RESPONSES_PATH, require_complete=True)
if EXPORT_SENSITIVITY:
    core.export_run(SUCCESSOR_PROTOCOL_PATH, SENSITIVITY_RUN_DIR, SENSITIVITY_RESPONSES_PATH, require_complete=True)
print({"primary_export_enabled": EXPORT_PRIMARY, "sensitivity_export_enabled": EXPORT_SENSITIVITY})'''),
        new_markdown_cell("## 8,640 primary and 576 sensitivity detector rows"),
        new_code_cell('''if DOWNLOAD_DETECTORS:
    if LOCAL_VALIDATION:
        raise ValueError("Detector downloads are disabled during local validation.")
    from huggingface_hub import snapshot_download
    snapshot_download(PINNED_EMBEDDING_MODEL_ID, revision=PINNED_EMBEDDING_REVISION)
    snapshot_download(PINNED_PREFIX_TOKENIZER_MODEL_ID, revision=PINNED_PREFIX_TOKENIZER_REVISION)
PRIMARY_PREDICTIONS_PATH = core.fs_path(os.environ.get("EXTENDED_PRIMARY_PREDICTIONS", str(CACHE_ROOT / "exports/primary_predictions.jsonl")))
SENSITIVITY_PREDICTIONS_PATH = core.fs_path(os.environ.get("EXTENDED_SENSITIVITY_PREDICTIONS", str(CACHE_ROOT / "exports/sensitivity_predictions.jsonl")))
if RUN_PRIMARY_PREDICTIONS:
    report = core.predict_complete_evaluation(SUCCESSOR_PROTOCOL_PATH, PRIMARY_RUN_DIR, PRIMARY_PREDICTIONS_PATH)
    if report["prediction_rows"] != 8640: raise ValueError("Primary prediction count mismatch")
if RUN_SENSITIVITY_PREDICTIONS:
    report = sensitivity_analysis.predict_complete_sensitivity(SUCCESSOR_PROTOCOL_PATH, SENSITIVITY_RUN_DIR, SENSITIVITY_PREDICTIONS_PATH)
    if report["prediction_rows"] != 576: raise ValueError("Sensitivity prediction count mismatch")
print({"primary_prediction_action": RUN_PRIMARY_PREDICTIONS, "sensitivity_prediction_action": RUN_SENSITIVITY_PREDICTIONS})'''),
        new_markdown_cell("## Primary analysis and matched seed-zero sensitivity"),
        new_code_cell('''PRIMARY_ANALYSIS_DIR = core.fs_path(CACHE_ROOT / "primary_analysis")
SENSITIVITY_ANALYSIS_DIR = core.fs_path(CACHE_ROOT / "sensitivity_analysis")
if RUN_PRIMARY_ANALYSIS:
    subprocess.run([sys.executable, "-m", "experiments.revision_analysis", "--protocol", str(SUCCESSOR_PROTOCOL_PATH), "--responses", str(PRIMARY_RESPONSES_PATH), "--predictions", str(PRIMARY_PREDICTIONS_PATH), "--output-dir", str(PRIMARY_ANALYSIS_DIR), "--bootstrap-replicates", "10000", "--bootstrap-seed", "20260915", "--confirm-real-complete-archive"], check=True, cwd=workspace)
if RUN_SENSITIVITY_ANALYSIS:
    sensitivity_analysis.analyze(SUCCESSOR_PROTOCOL_PATH, PRIMARY_RUN_DIR, PRIMARY_RESPONSES_PATH, PRIMARY_PREDICTIONS_PATH, SENSITIVITY_RUN_DIR, SENSITIVITY_RESPONSES_PATH, SENSITIVITY_PREDICTIONS_PATH, SENSITIVITY_ANALYSIS_DIR, bootstrap_replicates=10000, seed=20260915)
print({"primary_analysis": RUN_PRIMARY_ANALYSIS, "matched_seed0_sensitivity": RUN_SENSITIVITY_ANALYSIS})'''),
        new_markdown_cell("## Prospective packet, two human passes, and scoring"),
        new_code_cell('''VALIDATION_PACKET_DIR = core.fs_path(os.environ.get("EXTENDED_VALIDATION_PACKET", str(CACHE_ROOT / "annotation_v4/evaluation_packet")))
if PREPARE_VALIDATION_PACKET:
    prospective.prepare_packet(VALIDATION_PLAN_PATH, SUCCESSOR_PROTOCOL_PATH, PRIMARY_RESPONSES_PATH, PRIMARY_PREDICTIONS_PATH, CALIBRATION_MANIFEST_PATH, CALIBRATION_RESPONSES_PATH, CALIBRATION_RUN_DIR, DEVELOPMENT_EVIDENCE_PATH, PRIMARY_RUN_DIR, VALIDATION_PACKET_DIR, annotation_guide_path)
if INIT_HUMAN_PASS_1:
    if not ANNOTATOR_1_ID: raise ValueError("A real annotator ID is required for pass 1.")
    validation_v4.init_annotation(1, ANNOTATOR_1_ID, True, VALIDATION_PACKET_DIR)
if INIT_HUMAN_PASS_2:
    if not ANNOTATOR_2_ID: raise ValueError("A real annotator ID is required for pass 2.")
    validation_v4.init_annotation(2, ANNOTATOR_2_ID, True, VALIDATION_PACKET_DIR)
if RUN_HUMAN_SCORING:
    print(validation_v4.score(VALIDATION_PACKET_DIR, 10000, 20260915))
else:
    print({"packet_preparation": PREPARE_VALIDATION_PACKET, "human_pass_1": INIT_HUMAN_PASS_1, "human_pass_2": INIT_HUMAN_PASS_2, "human_scoring": False})'''),
        new_markdown_cell("## Status\n\nThe default execution proves only source/bundle integrity and workflow importability. Scientific completion requires a completed 8/8 probe, explicit budget decision, frozen successor protocol, genuine prompt review, complete 96-response calibration plus reviewed timing evidence, frozen prospective plan, 2,880 primary and 192 sensitivity responses, 8,640 and 576 prediction rows, and two genuine human annotation passes."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "extended_revision_study": {"bundle_content_id": bundle["bundle_content_id"], "bundle_sha256": bundle_sha, "primary_cap_default": None, "sensitivity_cap_default": None, "all_actions_default_false": True},
    })
    nbformat.validate(notebook)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, notebook_path)


def execute_validation(notebook_path: Path, bundle_path: Path, output_path: Path) -> dict[str, Any]:
    if os.name == "nt":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbformat.read(notebook_path, as_version=4)
    temp_root = Path(tempfile.mkdtemp(prefix="extended_notebook_", dir=ROOT / "tmp"))
    prior = {name: os.environ.get(name) for name in ("EXTENDED_NOTEBOOK_LOCAL_VALIDATION", "EXTENDED_NOTEBOOK_BUNDLE", "EXTENDED_NOTEBOOK_WORKSPACE", "EXTENDED_NOTEBOOK_CACHE")}
    os.environ.update({"EXTENDED_NOTEBOOK_LOCAL_VALIDATION": "1", "EXTENDED_NOTEBOOK_BUNDLE": str(bundle_path.resolve()), "EXTENDED_NOTEBOOK_WORKSPACE": str(temp_root / "workspace"), "EXTENDED_NOTEBOOK_CACHE": str(temp_root / "cache")})
    try:
        executed = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    finally:
        for name, value in prior.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value
        shutil.rmtree(temp_root, ignore_errors=True)
    errors = [output for cell in executed.cells if cell.cell_type == "code" for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    if errors:
        raise ValueError(f"Executed notebook contains errors: {errors}")
    outputs = [output for cell in executed.cells if cell.cell_type == "code" for output in cell.get("outputs", [])]
    rendered = json.dumps(outputs, ensure_ascii=False)
    if "Traceback (most recent call last)" in rendered:
        raise ValueError("Executed notebook output contains a traceback")
    control_status = None
    for output in outputs:
        text = output.get("text") if output.get("output_type") == "stream" else None
        if isinstance(text, str) and "all_action_controls_disabled" in text:
            control_status = ast.literal_eval(text.strip())
            break
    if (not isinstance(control_status, dict)
            or control_status.get("primary_cap") is not None
            or control_status.get("sensitivity_cap") is not None
            or control_status.get("all_action_controls_disabled") is not True
            or any(control_status.get("action_controls", {}).values())):
        raise ValueError(f"Notebook safe-default controls were not preserved: {control_status}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, output_path)
    return {"code_cells": sum(cell.cell_type == "code" for cell in executed.cells), "markdown_cells": sum(cell.cell_type == "markdown" for cell in executed.cells), "cell_outputs_inspected": len(outputs), "error_outputs": 0, "safe_defaults_observed": True, "executed_notebook_sha256": sha256_file(output_path)}


def _extended_scientific_files() -> list[str]:
    root = ROOT / "output" / "revision" / "extended_study"
    if not root.exists():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] != "verification"
    )


def verify_bundle(bundle_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    with zipfile.ZipFile(bundle_path) as archive:
        embedded = json.loads(archive.read("bundle_manifest.json"))
        if embedded["bundle_content_id"] != manifest["bundle_content_id"]:
            raise ValueError("Embedded bundle manifest identity mismatch")
        for relative, expected in embedded["files"].items():
            payload = archive.read(relative)
            if sha256_bytes(payload) != expected["sha256"] or len(payload) != expected["size_bytes"]:
                raise ValueError(f"Archive member mismatch: {relative}")
    return {"status": "verified", "files": len(embedded["files"]), "bundle_sha256": sha256_file(bundle_path), "bundle_content_id": embedded["bundle_content_id"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--completed-probe-bundle", type=Path, default=DEFAULT_COMPLETED_PROBE_BUNDLE)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--probe-run-dir", type=Path)
    parser.add_argument("--probe-export", type=Path)
    parser.add_argument("--budget-decision", type=Path)
    args = parser.parse_args(argv)
    scientific_before = _extended_scientific_files()
    bundle = build_bundle(args.bundle, probe_run_dir=args.probe_run_dir, probe_export=args.probe_export, budget_decision=args.budget_decision)
    completed_probe_bundle = build_completed_probe_bundle(args.completed_probe_bundle, bundle)
    build_notebook(args.notebook, bundle)
    bundle_report = verify_bundle(args.bundle, bundle)
    executed_path = args.report_root / "executed_advice_marker_extended_study.ipynb"
    execution = execute_validation(args.notebook, args.bundle, executed_path)
    scientific_after = _extended_scientific_files()
    if scientific_after != scientific_before:
        raise ValueError(f"Notebook validation changed extended-study scientific files: before={scientific_before}, after={scientific_after}")
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "bundle_manifest.json").write_text(json.dumps({key: value for key, value in bundle.items() if key not in {"bundle_path", "bundle_sha256"}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.report_root / "completed_probe_bundle_manifest.json").write_text(json.dumps({key: value for key, value in completed_probe_bundle.items() if key != "bundle_path"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verification = {
        **bundle_report, "status": "complete_local_no_action_validation",
        "notebook": str(args.notebook), "notebook_sha256": sha256_file(args.notebook),
        "bundle": str(args.bundle), **execution,
        "completed_probe_bundle": {
            "path": str(args.completed_probe_bundle), "sha256": completed_probe_bundle["bundle_sha256"],
            "files": len(completed_probe_bundle["files"]), "completed_probe_bundle_id": completed_probe_bundle["completed_probe_bundle_id"],
            "source_bundle_compatibility": completed_probe_bundle["source_bundle_compatibility"],
        },
        "controls": {"primary_cap": None, "sensitivity_cap": None, "all_actions": False},
        "scientific_output_created": False, "extended_study_scientific_files_before": scientific_before,
        "extended_study_scientific_files_after": scientific_after,
        "probe_input": bundle["probe_input"],
    }
    (args.report_root / "candidate_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
