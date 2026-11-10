from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/revision/environment_validation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    tokenizer = json.loads((OUT / "tokenizer_config_compatibility.json").read_text(encoding="utf-8"))
    notebook = json.loads((OUT / "notebook_check/execution_verification.json").read_text(encoding="utf-8"))
    immutability = json.loads((OUT / "notebook_check/protected_artifact_immutability.json").read_text(encoding="utf-8"))
    imports = json.loads((OUT / "import_check.json").read_text(encoding="utf-8"))
    resolved = json.loads((OUT / "resolved_packages.json").read_text(encoding="utf-8"))
    pip_check = (OUT / "pip_check.log").read_text(encoding="utf-8").strip()
    pyvenv = (OUT / "venv/pyvenv.cfg").read_text(encoding="utf-8")
    if "include-system-site-packages = false" not in pyvenv:
        raise AssertionError("The validation environment inherits system packages")
    if pip_check != "No broken requirements found.":
        raise AssertionError(f"pip check failed: {pip_check}")
    if tokenizer["status"] != "passed" or notebook["status"] != "passed" or immutability["status"] != "passed":
        raise AssertionError("A prerequisite verification did not pass")

    expected_sources = {
        "experiments/extended_revision_study.py": "0df8d2538fe519f6374b0d648bbe150c443aefde006c9179d468f779e6f9f61e",
        "experiments/extended_calibration_study.py": "3245adaee72f3c1ce15ef5e6614aa810662c54a19b0fccf601ccd8c7f07d59f9",
        "experiments/extended_prospective_validation.py": "ded635ebd49fdb2612bd78fdf0446479f07a3c74d9391b27286f2712a4e22c4a",
        "experiments/extended_sensitivity_analysis.py": "32531b42abd4d7882ff5d18e8b5179038e0e05fced3c041e6fa714f05b3e7474",
    }
    source_records = []
    for relative, expected in expected_sources.items():
        item = record(relative)
        if item["sha256"] != expected:
            raise AssertionError(f"Scientific source changed: {relative}")
        source_records.append(item)

    verification_files = [
        "output/revision/environment_validation/stated_requirements.txt",
        "output/revision/environment_validation/stated_dependencies_install.log",
        "output/revision/environment_validation/test_only_requirements.txt",
        "output/revision/environment_validation/test_only_install.log",
        "output/revision/environment_validation/resolved_packages.txt",
        "output/revision/environment_validation/resolved_packages.json",
        "output/revision/environment_validation/pip_check.log",
        "output/revision/environment_validation/import_check.json",
        "output/revision/environment_validation/kernels/clean-env/kernel.json",
        "output/revision/environment_validation/tokenizer_config_compatibility.log",
        "output/revision/environment_validation/tokenizer_config_compatibility.json",
        "output/revision/environment_validation/notebook_check/executed_advice_marker_extended_study_clean_env.ipynb",
        "output/revision/environment_validation/notebook_check/execution_console.log",
        "output/revision/environment_validation/notebook_check/execution_verification.json",
        "output/revision/environment_validation/notebook_check/protected_artifact_immutability.json",
        "output/revision/environment_validation/notebook_check/artifact_manifest.json",
    ]
    protected_inputs = [
        "output/jupyter-notebook/advice_marker_extended_study.ipynb",
        "output/jupyter-notebook/advice_marker_extended_study_bundle.zip",
        "output/jupyter-notebook/advice_marker_frozen_preparation_bundle.zip",
        "output/revision/extended_study/development_budget_decision_v1.json",
        "output/revision/extended_study/full_study/protocol_v1.json",
        "output/revision/extended_study/calibration_study/manifest.json",
        "output/revision/extended_study/human_prompt_review_template_v1.json",
        "output/revision/extended_study/prompt_review_worksheet.md",
    ]

    report = {
        "verification_schema_version": "extended-study-clean-environment-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "clean local Windows dependency, safe-default notebook, and offline tokenizer/config compatibility",
        "environment": {
            "host_platform": platform.platform(),
            "selected_interpreter": imports["executable"],
            "python": imports["python"],
            "base_interpreter_prefix": imports["base_prefix"],
            "venv_configuration": "output/revision/environment_validation/venv/pyvenv.cfg",
            "include_system_site_packages": False,
            "venv_retained": True,
            "venv_path": "output/revision/environment_validation/venv",
            "official_index": "https://pypi.org/simple",
            "pip_check": pip_check,
            "imports": imports["imports"],
            "resolved_distribution_count": len(resolved),
        },
        "dependency_sets": {
            "notebook_stated": (OUT / "stated_requirements.txt").read_text(encoding="utf-8").splitlines(),
            "test_only": (OUT / "test_only_requirements.txt").read_text(encoding="utf-8").splitlines(),
            "test_only_classification": "verification tooling only; not claimed as notebook runtime dependencies",
        },
        "tokenizer_and_config": {
            "status": tokenizer["status"],
            "runtime": tokenizer["runtime"],
            "checkpoint_count": len(tokenizer["checkpoints"]),
            "expanded_prompt_count": tokenizer["prompt_corpus"]["expanded_prompt_count"],
            "all_six_per_prompt_counts_identical": tokenizer["prompt_corpus"]["all_six_per_prompt_counts_identical"],
            "input_token_range": [tokenizer["prompt_corpus"]["min_input_tokens"], tokenizer["prompt_corpus"]["max_input_tokens"]],
            "matches_prior_transformers_5_5_4_audit": tokenizer["comparison"],
            "no_model_or_generation": tokenizer["operations"],
        },
        "notebook": {
            "status": notebook["status"],
            "runtime": notebook["runtime"],
            "relocation": notebook["relocation"],
            "source_cells": notebook["source_notebook"]["cells"],
            "code_cells_executed": notebook["code_cells_executed"],
            "cell_type_and_source_sha256": notebook["source_notebook"]["cell_type_and_source_sha256"],
            "source_type_equality": notebook["source_type_equality"],
            "error_outputs": notebook["error_outputs"],
            "traceback_text_found": notebook["traceback_text_found"],
            "safe_default_controls": notebook["safe_default_controls"],
            "network_disabled_by_environment": notebook["network_disabled_by_environment"],
            "protected_inputs_unchanged": notebook["protected_inputs_unchanged"],
            "scientific_interpretation": notebook["scientific_interpretation"],
        },
        "immutability": {
            "status": immutability["status"],
            "original_scientific_artifacts_unchanged": immutability["original_scientific_artifacts_unchanged"],
            "unexpected_original_empirical_output_directories": immutability["unexpected_original_empirical_output_directories"],
            "scientific_sources": source_records,
            "protected_inputs": [record(path) for path in protected_inputs],
        },
        "artifacts": [record(path) for path in verification_files],
        "limits": {
            "validated_host": "local Windows",
            "linux_validated": False,
            "gpu_validated": False,
            "model_weights_loaded": False,
            "model_inference_performed": False,
            "network_used_by_notebook_or_tokenizer_checks": False,
            "network_used_for_authorized_dependency_install_only": True,
            "human_review_created_or_completed": False,
            "study_generation_or_analysis_performed": False,
        },
    }
    destination = OUT / "clean_environment_validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(destination), "sha256": sha256(destination)}))


if __name__ == "__main__":
    main()
