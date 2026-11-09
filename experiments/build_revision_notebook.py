"""Build the portable Colab notebook and its hash-validated local source bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "output" / "jupyter-notebook" / "advice_marker_revision_study.ipynb"
DEFAULT_BUNDLE = ROOT / "output" / "jupyter-notebook" / "advice_marker_revision_study_bundle.zip"
DEFAULT_PROTOCOL = ROOT / "output" / "revision" / "full_study" / "protocol_v2.json"
DEFAULT_REVIEW = ROOT / "output" / "revision" / "full_study" / "human_prompt_review_v2.json"
PINNED_EMBEDDING_MODEL_ID = "intfloat/e5-small-v2"
PINNED_EMBEDDING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
PINNED_PREFIX_TOKENIZER_MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
PINNED_PREFIX_TOKENIZER_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
REQUIRED_BUNDLE_FILES = (
    "experiments/__init__.py",
    "experiments/full_revision_study.py",
    "experiments/calibration_study.py",
    "experiments/development_budget_probe.py",
    "experiments/prospective_validation.py",
    "experiments/validation_v4.py",
    "experiments/revision_analysis.py",
    "experiments/revision_study.py",
    "experiments/detector_spec.py",
    "experiments/prompts.json",
    "experiments/prompts_revision_v1.json",
    "experiments/raw_output_robustness.py",
    "trace/behavior.py",
    "output/revision/full_study/protocol_v2.json",
    "output/revision/full_study/human_prompt_review_v2.json",
    "output/revision/full_study/development_256_verification_20260915.json",
    "output/revision/full_study/exports/development_256_eight_responses.jsonl",
    "output/revision/full_study/development_512_verification_20260915.json",
    "output/revision/full_study/exports/development_512_eight_responses.jsonl",
    "output/revision/study_v1/detector_manifest_v1.json",
    "out/paper/ANNOTATION_GUIDE.md",
    "out/paper/CALIBRATION_PROTOCOL_2026-09-15.md",
    "out/paper/DEVELOPMENT_BUDGET_PROBE_PROTOCOL_2026-09-15.md",
    "out/paper/PROSPECTIVE_VALIDATION_PROTOCOL_2026-09-15.md",
    "out/paper/PROMPT_REVIEW_WORKSHEET_2026-09-15.md",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_bundle(bundle_path: Path) -> dict:
    files = list(REQUIRED_BUNDLE_FILES)
    missing = [relative for relative in files if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot build bundle; missing files: {missing}")
    manifest = {
        "bundle_schema_version": "advice-marker-revision-notebook-bundle-v1",
        "provenance": "Built from the current local workspace, including uncommitted source files.",
        "detector_dependencies": {
            "embedding_model": {
                "model_id": PINNED_EMBEDDING_MODEL_ID,
                "revision": PINNED_EMBEDDING_REVISION,
            },
            "prefix_window_tokenizer": {
                "model_id": PINNED_PREFIX_TOKENIZER_MODEL_ID,
                "revision": PINNED_PREFIX_TOKENIZER_REVISION,
                "download_scope": "tokenizer files only",
            },
        },
        "files": {
            relative: {
                "sha256": sha256_bytes((ROOT / relative).read_bytes()),
                "size_bytes": (ROOT / relative).stat().st_size,
            }
            for relative in sorted(files)
        },
    }
    manifest["bundle_content_id"] = "bundle_" + sha256_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )[:32]
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        def write_member(name: str, payload: bytes) -> None:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)

        write_member(
            "bundle_manifest.json",
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        for relative in sorted(files):
            write_member(relative, (ROOT / relative).read_bytes())
    temporary.replace(bundle_path)
    return {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_bytes(bundle_path.read_bytes()),
    }


def build_notebook(notebook_path: Path, bundle: dict) -> None:
    if notebook_path.exists():
        notebook = nbformat.read(notebook_path, as_version=4)
    else:
        notebook = nbformat.v4.new_notebook()
    bundle_sha = bundle["bundle_sha256"]
    notebook.cells = [
        new_markdown_cell("""# Advice-marker revision study

This notebook runs the frozen prospective factorial study from a **locally uploaded source bundle**. It does not clone a public branch. The bundle contains the current workspace sources, prompt corpus, frozen Hugging Face revisions, blank human-review template, and detector manifest.

The development path is exploratory and may run before prompt review. Evaluation and longer-budget sensitivity paths stay locked until a human completes the corpus-bound review manifest. Prefix detector windows are sensitivity measurements; only full-response predictions may be compared with full-response human labels."""),
        new_code_cell(f'''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile

EXPECTED_BUNDLE_SHA256 = "{bundle_sha}"
PINNED_EMBEDDING_MODEL_ID = "{PINNED_EMBEDDING_MODEL_ID}"
PINNED_EMBEDDING_REVISION = "{PINNED_EMBEDDING_REVISION}"
PINNED_PREFIX_TOKENIZER_MODEL_ID = "{PINNED_PREFIX_TOKENIZER_MODEL_ID}"
PINNED_PREFIX_TOKENIZER_REVISION = "{PINNED_PREFIX_TOKENIZER_REVISION}"
LOCAL_VALIDATION = os.environ.get("REVISION_NOTEBOOK_LOCAL_VALIDATION") == "1"
USE_GOOGLE_DRIVE = False
RUN_DEVELOPMENT = False
FREEZE_BUDGET_PROBE_MANIFEST = False
RUN_BUDGET_PROBE_1024 = False
FREEZE_CALIBRATION_MANIFEST = False
RUN_CALIBRATION = False
FREEZE_VALIDATION_PLAN = False
PREPARE_VALIDATION_PACKET = False
RUN_FULL_EVALUATION = False
RUN_LONGER_SENSITIVITY = False
RUN_PREDICTIONS = False
RUN_ANALYSIS = False
TORCH_THREADS = 4
DEVICE = "auto"
DEV_MAX_NEW_TOKENS = 256  # Use 512 for the matched longer-budget development check.
CALIBRATION_MAX_NEW_TOKENS = None  # Set only after reviewing the development budget evidence.
DEV_FAMILY_IDS = []  # Empty selects one health and one everyday development family below.

print({{"local_validation": LOCAL_VALIDATION, "expected_bundle_sha256": EXPECTED_BUNDLE_SHA256}})'''),
        new_markdown_cell("## Environment and bundle verification\n\nOn Colab, upload `advice_marker_revision_study_bundle.zip` when prompted. Set `USE_GOOGLE_DRIVE=True` only if you want durable run caches in your own Drive."),
        new_code_cell('''if not LOCAL_VALIDATION:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.57,<5", "huggingface_hub>=0.34,<2",
        "sentence-transformers>=5,<6", "accelerate>=1.10,<2", "safetensors>=0.6,<1",
    ], check=True)
else:
    print("Local validation: dependency installation skipped.")'''),
        new_code_cell('''bundle_env = os.environ.get("REVISION_NOTEBOOK_BUNDLE")
if bundle_env:
    bundle_path = Path(bundle_env)
else:
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError("Set REVISION_NOTEBOOK_BUNDLE outside Colab.") from exc
    uploaded = files.upload()
    candidates = [Path(name) for name in uploaded if name.endswith(".zip")]
    if len(candidates) != 1:
        raise ValueError("Upload exactly one revision-study bundle zip.")
    bundle_path = candidates[0]

actual_bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
if actual_bundle_sha != EXPECTED_BUNDLE_SHA256:
    raise ValueError(f"Bundle SHA-256 mismatch: {actual_bundle_sha}")

with zipfile.ZipFile(bundle_path) as archive:
    manifest = json.loads(archive.read("bundle_manifest.json").decode("utf-8"))
    workspace_parent = Path(os.environ.get("REVISION_NOTEBOOK_WORKSPACE", "/tmp/advice_marker_revision_bundles"))
    workspace = workspace_parent / manifest["bundle_content_id"]
    workspace.mkdir(parents=True, exist_ok=True)
    for info in archive.infolist():
        member = Path(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe bundle member: {info.filename}")
    archive.extractall(workspace)

extracted_manifest = json.loads((workspace / "bundle_manifest.json").read_text(encoding="utf-8"))
if extracted_manifest != manifest:
    raise ValueError("Extracted bundle manifest differs from the archive manifest.")
for relative, expected in manifest["files"].items():
    target = workspace / relative
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != expected["sha256"] or target.stat().st_size != expected["size_bytes"]:
        raise ValueError(f"Bundled source mismatch: {relative}")
print({"bundle_content_id": manifest["bundle_content_id"], "verified_files": len(manifest["files"])})'''),
        new_code_cell('''if USE_GOOGLE_DRIVE:
    if LOCAL_VALIDATION:
        raise ValueError("Google Drive mounting is disabled during local validation.")
    from google.colab import drive
    drive.mount("/content/drive")
    CACHE_ROOT = Path("/content/drive/MyDrive/advice_marker_revision_study")
else:
    CACHE_ROOT = Path(os.environ.get("REVISION_NOTEBOOK_CACHE", "/tmp/advice_marker_revision_cache"))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(workspace))
from experiments import full_revision_study as full
from experiments import calibration_study as calibration
from experiments import development_budget_probe as budget_probe
protocol_path = workspace / "output/revision/full_study/protocol_v2.json"
review_template_path = workspace / "output/revision/full_study/human_prompt_review_v2.json"
protocol = full.load_protocol(protocol_path)
print({"cache_root": str(CACHE_ROOT), "protocol_id": protocol["protocol_id"]})'''),
        new_code_cell('''inventory = full.gpu_inventory()
print(json.dumps(inventory, indent=2))
if not LOCAL_VALIDATION and DEVICE in {"auto", "cuda"} and not inventory["cuda_available"]:
    print("No CUDA device is active. Keep full-grid switches off and choose a GPU runtime before generation.")'''),
        new_markdown_cell("## Frozen design and human-review gate"),
        new_code_cell('''overview = {
    "protocol_status": protocol["status"],
    "evaluation_prompts": 160,
    "checkpoints": len(protocol["checkpoints"]),
    "seeds": protocol["primary_seeds"],
    "primary_expected_responses": protocol["primary_expected_responses"],
    "development_256": protocol["development"]["primary_256_expected_responses"],
    "development_512": protocol["development"]["longer_512_expected_responses"],
    "longer_sensitivity": protocol["longer_budget_sensitivity"]["expected_responses"],
    "human_review_template_status": json.loads(review_template_path.read_text())["status"],
}
print(json.dumps(overview, indent=2))'''),
        new_code_cell('''# A human reviewer must edit a copy of this file outside this notebook and set every
# family check after manual review. Code never changes PENDING into COMPLETE.
HUMAN_REVIEW_PATH = Path(os.environ.get("REVISION_HUMAN_REVIEW", str(CACHE_ROOT / "human_prompt_review_completed.json")))
if HUMAN_REVIEW_PATH.exists():
    try:
        completed_review = full.validate_human_review(protocol, HUMAN_REVIEW_PATH)
        review_status = completed_review["status"]
    except Exception as exc:
        review_status = f"INVALID: {exc}"
else:
    review_status = "PENDING: copy and manually complete the bundled blank template"
print({"human_review_path": str(HUMAN_REVIEW_PATH), "status": review_status})'''),
        new_markdown_cell("## Exploratory development run\n\nThe default dry run selects two complete four-variant families (one health and one everyday), yielding eight responses from the pinned 1B Instruct checkpoint. `DEV_MAX_NEW_TOKENS` selects the 256- or 512-token budget. Set `RUN_DEVELOPMENT=True` to execute that budget. Keep both runs exploratory; a cap is not established as adequate for complete answers merely because generation finishes."),
        new_code_cell('''if DEV_MAX_NEW_TOKENS not in (256, 512):
    raise ValueError("The frozen development modes support only 256 or 512 tokens.")
dev_mode = f"development-{DEV_MAX_NEW_TOKENS}"
dev_grid_key = "primary_256_grid" if DEV_MAX_NEW_TOKENS == 256 else "longer_512_grid"
dev_rows = protocol["development"][dev_grid_key]
if not DEV_FAMILY_IDS:
    by_domain = {}
    for row in dev_rows:
        by_domain.setdefault(row["condition"]["domain"], row["scenario_family_id"])
    DEV_FAMILY_IDS = [by_domain["health"], by_domain["everyday"]]
dev_args = [
    "--protocol", str(protocol_path), "--mode", dev_mode,
    "--family-ids", *DEV_FAMILY_IDS, "--output-root", str(CACHE_ROOT),
    "--device", DEVICE, "--torch-threads", str(TORCH_THREADS),
]
if RUN_DEVELOPMENT:
    result = full.run_generation(
        protocol_path, dev_mode, CACHE_ROOT, DEV_FAMILY_IDS, None,
        DEVICE, False, TORCH_THREADS,
    )
else:
    execution = full.execution_profile(DEVICE, TORCH_THREADS)
    _p, grid, run_dir = full.prepare_run(
        protocol_path, dev_mode, CACHE_ROOT, DEV_FAMILY_IDS, None, execution,
    )
    result = {"status": "dry_run", "responses": len(grid), "run_dir": str(run_dir)}
print(json.dumps(result, indent=2))'''),
        new_code_cell('''dev_run_dir = Path(result["run_dir"])
response_files = sorted((dev_run_dir / "responses").glob("*.json")) if (dev_run_dir / "responses").exists() else []
if response_files:
    dev_report = full.verify_run(protocol_path, dev_run_dir, require_complete=False)
    sample = json.loads(response_files[0].read_text())
    print(json.dumps(dev_report, indent=2))
    print({"sample_prompt": sample["generation"]["prompt"], "sample_response": sample["generation"]["response"][:500]})
else:
    print("No development responses yet; this is expected for the default dry run.")'''),
        new_markdown_cell("## Fixed 1,024-token development budget probe\n\nThis optional exploratory probe contains exactly eight fixed development responses: all four variants from each of two fixed families, using the pinned 1B Instruct checkpoint and seed 0. The read-only planning cell validates the completed matched 256- and 512-token evidence. Manifest freezing and generation have separate switches and both default to `False`. A GPU run, or any other runtime-profile change, produces separately profiled evidence and cannot support a claim of matched CPU equivalence. This probe neither changes the primary cap nor replaces the 96-response, six-checkpoint calibration."),
        new_code_cell('''BUDGET_PROBE_ROOT = CACHE_ROOT / "development_budget_probe"
BUDGET_PROBE_MANIFEST_PATH = Path(os.environ.get(
    "REVISION_BUDGET_PROBE_MANIFEST",
    str(BUDGET_PROBE_ROOT / "development_1024_probe_manifest_v1.json"),
))
budget_probe_run_env = os.environ.get("REVISION_BUDGET_PROBE_RUN_DIR")
budget_probe_run_dir = Path(budget_probe_run_env) if budget_probe_run_env else None
budget_probe_plan = budget_probe.plan(protocol_path)
budget_probe_audit = budget_probe.preservation_audit(protocol_path)
if budget_probe_audit["status"] != "pass":
    raise ValueError("Budget-probe preservation audit failed.")
print(json.dumps({
    "plan": budget_probe_plan,
    "preservation_audit": budget_probe_audit,
    "files_written": 0,
    "model_loaded": False,
}, indent=2))'''),
        new_code_cell('''if FREEZE_BUDGET_PROBE_MANIFEST:
    frozen_budget_probe = budget_probe.freeze_manifest(protocol_path, BUDGET_PROBE_MANIFEST_PATH)
    print({
        "frozen_budget_probe_manifest": str(BUDGET_PROBE_MANIFEST_PATH),
        "probe_manifest_id": frozen_budget_probe["probe_manifest_id"],
    })
else:
    print("Budget-probe manifest freezing disabled; the read-only plan remains a candidate.")'''),
        new_code_cell('''budget_probe_result = None
if RUN_BUDGET_PROBE_1024:
    if not BUDGET_PROBE_MANIFEST_PATH.is_file():
        raise ValueError("Freeze the fixed budget-probe manifest before generation.")
    budget_probe_result = budget_probe.run_generation(
        BUDGET_PROBE_MANIFEST_PATH, protocol_path, BUDGET_PROBE_ROOT,
        DEVICE, False, TORCH_THREADS,
    )
    print(json.dumps(budget_probe_result, indent=2))
else:
    print("1,024-token budget-probe generation disabled; no model was loaded by this cell.")'''),
        new_code_cell('''if budget_probe_result and budget_probe_result["status"] == "complete":
    budget_probe_run_dir = Path(budget_probe_result["run_dir"])
if budget_probe_run_dir is not None:
    budget_probe_report = budget_probe.verify_run(
        BUDGET_PROBE_MANIFEST_PATH, protocol_path, budget_probe_run_dir,
        require_complete=True,
    )
    budget_probe_export = BUDGET_PROBE_ROOT / "exports" / f"{budget_probe_run_dir.name}_responses.jsonl"
    budget_probe_export_report = budget_probe.export_run(
        BUDGET_PROBE_MANIFEST_PATH, protocol_path, budget_probe_run_dir,
        budget_probe_export,
    )
    print(json.dumps({
        "verification": budget_probe_report,
        "export": budget_probe_export_report,
    }, indent=2))
else:
    print("Budget-probe export waits for all eight strictly verified responses.")'''),
        new_markdown_cell("## Six-checkpoint development calibration\n\nThe separate 96-response design covers all six retained checkpoints and all ten development families. It stays outside the evaluation. Review the generation-budget evidence before setting `CALIBRATION_MAX_NEW_TOKENS` and explicitly freezing its manifest. Actual humans must use at least 60 distinct development responses to refine the guide and record annotation time before the prospective sampling plan can be frozen. Blank forms or generated responses alone do not establish human calibration."),
        new_code_cell('''CALIBRATION_ROOT = CACHE_ROOT / "calibration_study"
CALIBRATION_MANIFEST_PATH = Path(os.environ.get("REVISION_CALIBRATION_MANIFEST", str(CALIBRATION_ROOT / "calibration_manifest.json")))
calibration_result = None
calibration_exported_responses = None
calibration_run_env = os.environ.get("REVISION_CALIBRATION_RUN_DIR")
calibration_run_dir = Path(calibration_run_env) if calibration_run_env else None
if CALIBRATION_MAX_NEW_TOKENS is None:
    calibration_plans = [calibration.plan(protocol_path, cap) for cap in (256, 512)]
    print(json.dumps({"status": "budget_choice_pending", "candidate_plans": calibration_plans}, indent=2))
else:
    print(json.dumps(calibration.plan(protocol_path, CALIBRATION_MAX_NEW_TOKENS), indent=2))
if FREEZE_CALIBRATION_MANIFEST:
    if CALIBRATION_MAX_NEW_TOKENS is None:
        raise ValueError("Review the budget evidence and choose a calibration cap before freezing.")
    calibration.freeze_manifest(protocol_path, CALIBRATION_MAX_NEW_TOKENS, CALIBRATION_MANIFEST_PATH)
    print({"frozen_calibration_manifest": str(CALIBRATION_MANIFEST_PATH)})
if RUN_CALIBRATION:
    if not CALIBRATION_MANIFEST_PATH.is_file():
        raise ValueError("Freeze the reviewed calibration manifest before generation.")
    calibration_manifest = calibration.load_manifest(CALIBRATION_MANIFEST_PATH, protocol_path)
    if calibration_manifest["max_new_tokens_decision"] != CALIBRATION_MAX_NEW_TOKENS:
        raise ValueError("Selected calibration cap differs from the frozen manifest.")
    calibration_result = calibration.run_generation(
        CALIBRATION_MANIFEST_PATH, protocol_path, CALIBRATION_ROOT, DEVICE, False, TORCH_THREADS,
    )
    print(json.dumps(calibration_result, indent=2))
else:
    print("Calibration generation disabled; no model was loaded by this cell.")'''),
        new_code_cell('''if calibration_result and calibration_result["status"] == "complete":
    calibration_run_dir = Path(calibration_result["run_dir"])
if calibration_run_dir is not None:
    calibration_report = calibration.verify_run(CALIBRATION_MANIFEST_PATH, protocol_path, calibration_run_dir)
    calibration_exported_responses = CALIBRATION_ROOT / "exports" / f"{calibration_run_dir.name}_responses.jsonl"
    calibration_export_report = calibration.export_run(
        CALIBRATION_MANIFEST_PATH, protocol_path, calibration_run_dir, calibration_exported_responses,
    )
    print(json.dumps({"verification": calibration_report, "export": calibration_export_report}, indent=2))
else:
    print("Calibration export waits for all 96 verified development responses.")'''),
        new_markdown_cell("## Freeze human-validation sampling before evaluation\n\nComplete human guide calibration and timing on the development material first. Follow the bundled `PROSPECTIVE_VALIDATION_PROTOCOL_2026-09-15.md` for the human-authored timing record. The frozen plan uses development-derived response-length bins, 13 selections in each of 48 checkpoint/condition/domain cells, and at most 120 separately flagged enriched items. The calibration cap must match the primary evaluation cap. These cells do not create human records or certify any human work."),
        new_code_cell('''VALIDATION_PLAN_PATH = Path(os.environ.get("REVISION_VALIDATION_PLAN", str(CACHE_ROOT / "prospective_validation_plan.json")))
TIMING_EVIDENCE_PATH = Path(os.environ.get("REVISION_CALIBRATION_TIMING", str(CALIBRATION_ROOT / "human_calibration_timing.json")))
calibration_response_env = os.environ.get("REVISION_CALIBRATION_RESPONSES")
if calibration_response_env:
    calibration_exported_responses = Path(calibration_response_env)
def calibration_evidence_args():
    if calibration_exported_responses is None or calibration_run_dir is None:
        raise ValueError("Provide the verified 96-response calibration run directory and matching export.")
    return [
        "--calibration-manifest", str(CALIBRATION_MANIFEST_PATH),
        "--calibration-responses", str(calibration_exported_responses),
        "--calibration-run-dir", str(calibration_run_dir),
        "--timing-evidence", str(TIMING_EVIDENCE_PATH),
    ]
if FREEZE_VALIDATION_PLAN:
    if calibration_exported_responses is None:
        raise ValueError("A complete verified 96-response calibration export is required before freezing sampling.")
    subprocess.run([
        sys.executable, "-m", "experiments.prospective_validation", "freeze-plan",
        "--protocol", str(protocol_path),
        *calibration_evidence_args(),
        "--output", str(VALIDATION_PLAN_PATH),
    ], check=True, cwd=workspace)
    print({"frozen_validation_plan": str(VALIDATION_PLAN_PATH)})
else:
    print("Validation-plan freezing disabled; development human evidence remains a separate requirement.")'''),
        new_markdown_cell("## Reviewed full grid and longer-budget sensitivity\n\nThese cells remain inert until their switches are enabled and `HUMAN_REVIEW_PATH` passes the external manual-review gate. The full grid is always all 2,880 frozen responses; it cannot be subset."),
        new_code_cell('''evaluation_result = None
if RUN_FULL_EVALUATION or RUN_LONGER_SENSITIVITY:
    subprocess.run([
        sys.executable, "-m", "experiments.prospective_validation", "validate-plan",
        "--plan", str(VALIDATION_PLAN_PATH), "--protocol", str(protocol_path),
        *calibration_evidence_args(),
    ], check=True, cwd=workspace)
if RUN_FULL_EVALUATION:
    full.validate_human_review(protocol, HUMAN_REVIEW_PATH)
    evaluation_result = full.run_generation(
        protocol_path, "evaluation", CACHE_ROOT, None, HUMAN_REVIEW_PATH,
        DEVICE, False, TORCH_THREADS,
    )
if RUN_LONGER_SENSITIVITY:
    full.validate_human_review(protocol, HUMAN_REVIEW_PATH)
    sensitivity_result = full.run_generation(
        protocol_path, "sensitivity", CACHE_ROOT, None, HUMAN_REVIEW_PATH,
        DEVICE, False, TORCH_THREADS,
    )
print({"evaluation": evaluation_result, "longer_sensitivity_enabled": RUN_LONGER_SENSITIVITY})'''),
        new_code_cell('''exported_responses = None
if evaluation_result and evaluation_result["status"] == "complete":
    evaluation_run_dir = Path(evaluation_result["run_dir"])
    exported_responses = CACHE_ROOT / "exports" / f"{evaluation_run_dir.name}_responses.jsonl"
    export_report = full.export_run(protocol_path, evaluation_run_dir, exported_responses)
    print(json.dumps(export_report, indent=2))
else:
    print("Full-grid export waits for strict verification of all 2,880 responses.")'''),
        new_markdown_cell("## Detector predictions and validation sampling\n\nThe prediction command emits full-response and 64/128-token rows with pinned dependency identities. Human labels apply only to full responses. The prospective human-validation allocation remains provisional pending development annotation timing."),
        new_code_cell('''prediction_output = None
if RUN_PREDICTIONS:
    if exported_responses is None:
        raise ValueError("Complete and export the full evaluation grid first.")
    from huggingface_hub import snapshot_download
    snapshot_download(
        PINNED_EMBEDDING_MODEL_ID, revision=PINNED_EMBEDDING_REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
    )
    snapshot_download(
        PINNED_PREFIX_TOKENIZER_MODEL_ID, revision=PINNED_PREFIX_TOKENIZER_REVISION,
        allow_patterns=["*.json", "*.txt", "*.model"],
    )
    observed_dependencies = full.shared.prediction_dependencies(include_embedding=True, include_prefix=True)
    expected_dependencies = {
        "embedding_model": (PINNED_EMBEDDING_MODEL_ID, PINNED_EMBEDDING_REVISION),
        "prefix_window_tokenizer": (PINNED_PREFIX_TOKENIZER_MODEL_ID, PINNED_PREFIX_TOKENIZER_REVISION),
    }
    for dependency_name, (model_id, revision) in expected_dependencies.items():
        observed = observed_dependencies[dependency_name]
        if (observed["model_id"], observed["resolved_revision"]) != (model_id, revision):
            raise ValueError(f"Pinned detector dependency mismatch for {dependency_name}: {observed}")
    prediction_output = CACHE_ROOT / "exports" / f"{Path(evaluation_result['run_dir']).name}_union_predictions.jsonl"
    prediction_report = full.predict_complete_evaluation(
        protocol_path, Path(evaluation_result["run_dir"]), prediction_output,
    )
    print(json.dumps(prediction_report, indent=2))
else:
    print("Prediction generation disabled.")'''),
        new_code_cell('''if PREPARE_VALIDATION_PACKET:
    if exported_responses is None or prediction_output is None:
        raise ValueError("Prospective sampling requires the complete evaluation response and prediction exports.")
    validation_packet_dir = CACHE_ROOT / "prospective_validation_packet"
    subprocess.run([
        sys.executable, "-m", "experiments.prospective_validation", "prepare",
        "--plan", str(VALIDATION_PLAN_PATH), "--protocol", str(protocol_path),
        *calibration_evidence_args(),
        "--responses", str(exported_responses), "--predictions", str(prediction_output),
        "--evaluation-run-dir", str(evaluation_run_dir),
        "--out-dir", str(validation_packet_dir),
    ], check=True, cwd=workspace)
    print({"blank_validation_packet": str(validation_packet_dir), "human_passes_completed_by_this_cell": 0})
else:
    print("Prospective packet preparation disabled. It requires a frozen development-derived sampling plan and complete evaluation exports.")'''),
        new_code_cell('''analysis_script = workspace / "experiments/revision_analysis.py"
if RUN_ANALYSIS:
    if not analysis_script.exists() or prediction_output is None or exported_responses is None:
        raise ValueError("Analysis requires bundled analysis source and complete response/prediction exports.")
    analysis_output = CACHE_ROOT / "checkpoint_analysis"
    subprocess.run([
        sys.executable, "-m", "experiments.revision_analysis",
        "--protocol", str(protocol_path),
        "--responses", str(exported_responses),
        "--predictions", str(prediction_output),
        "--output-dir", str(analysis_output),
        "--bootstrap-replicates", "10000",
        "--bootstrap-seed", "20260915",
        "--confirm-real-complete-archive",
    ], check=True, cwd=workspace)
    print({"analysis_output": str(analysis_output)})
else:
    print("Analysis is disabled until a verified full-grid prediction archive exists.")'''),
        new_markdown_cell("## Run status\n\nA complete scientific run requires: a valid manually completed prompt-review manifest; all 2,880 primary responses under one frozen protocol and execution profile; strict export verification; three detector windows with pinned dependencies; and separately collected prospective human annotations. This notebook does not claim any of those stages completed merely because the workflow is runnable."),
    ]
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "revision_study": {
            "bundle_content_id": bundle["bundle_content_id"],
            "bundle_sha256": bundle_sha,
            "protocol_path": "output/revision/full_study/protocol_v2.json",
            "detector_dependencies": bundle["detector_dependencies"],
        },
    })
    nbformat.validate(notebook)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, notebook_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    bundle = build_bundle(args.bundle)
    build_notebook(args.notebook, bundle)
    report_path = ROOT / "output" / "revision" / "full_study" / "notebook_bundle_manifest_v2.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "notebook": str(args.notebook), "bundle": str(args.bundle),
        "bundle_sha256": bundle["bundle_sha256"], "files": len(bundle["files"]),
    }, indent=2))


if __name__ == "__main__":
    main()
