from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from importlib import metadata
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[4]
CHECK = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "output/jupyter-notebook/advice_marker_extended_study.ipynb"
SOURCE_BUNDLE = ROOT / "output/jupyter-notebook/advice_marker_extended_study_bundle.zip"
FROZEN_BUNDLE = ROOT / "output/jupyter-notebook/advice_marker_frozen_preparation_bundle.zip"
EXPECTED = {
    SOURCE_NOTEBOOK: "bf16bb2d7440e091a73034015627431333abb3530bde61a430d38b7ee9c4b7e2",
    SOURCE_BUNDLE: "d148bdb43c6545c0a02c5cb75e6b2683f0b56d3dfa31c2e0c25050b3031ed4cb",
    FROZEN_BUNDLE: "cee63f40d21a44df277f8e4e4b7d581b7463bb3fe370f07f9506481f3fdac796",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fs_path(path: Path) -> Path:
    absolute = str(path.resolve())
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def cell_identity(notebook: nbformat.NotebookNode) -> dict[str, object]:
    types = [cell.cell_type for cell in notebook.cells]
    sources = [cell.source for cell in notebook.cells]
    encoded = json.dumps({"types": types, "sources": sources}, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "cells": len(types),
        "code_cells": types.count("code"),
        "markdown_cells": types.count("markdown"),
        "cell_types": types,
        "cell_type_and_source_sha256": hashlib.sha256(encoded).hexdigest(),
        "sources": sources,
    }


def safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe archive member: {info.filename}")
        archive.extractall(fs_path(destination))


def main() -> None:
    started = time.time()
    before = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED}
    if any(before[str(path.relative_to(ROOT))] != expected for path, expected in EXPECTED.items()):
        raise ValueError(f"Input identity mismatch: {before}")

    input_dir = CHECK / "isolated_workspace/inputs"
    workspace_parent = CHECK / "isolated_workspace/source_workspace"
    cache_dir = CHECK / "isolated_workspace/cache"
    input_dir.mkdir(parents=True, exist_ok=True)
    workspace_parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    relocated_source_bundle = input_dir / SOURCE_BUNDLE.name
    relocated_frozen_bundle = input_dir / FROZEN_BUNDLE.name
    shutil.copy2(SOURCE_BUNDLE, relocated_source_bundle)
    shutil.copy2(FROZEN_BUNDLE, relocated_frozen_bundle)

    with zipfile.ZipFile(relocated_source_bundle) as archive:
        source_manifest = json.loads(archive.read("bundle_manifest.json"))
    extracted_root = fs_path(workspace_parent / source_manifest["bundle_content_id"])
    extracted_root.mkdir(parents=True, exist_ok=True)
    safe_extract(relocated_frozen_bundle, extracted_root)

    protocol = extracted_root / "output/revision/extended_study/full_study/protocol_v1.json"
    calibration = extracted_root / "output/revision/extended_study/calibration_study/manifest.json"
    if sha256(protocol) != "2dbb5e1a6b14eb505456de52487bee0d4faaaeb8c49766aabed7c9c485c92f46":
        raise ValueError("Relocated frozen protocol mismatch")
    if sha256(calibration) != "4775f80d61546e38408fd7549c4113ba3e921c5725677df16b740d9be84ff7b3":
        raise ValueError("Relocated calibration manifest mismatch")

    os.environ.update(
        {
            "EXTENDED_NOTEBOOK_LOCAL_VALIDATION": "1",
            "EXTENDED_NOTEBOOK_BUNDLE": str(relocated_source_bundle),
            "EXTENDED_NOTEBOOK_WORKSPACE": str(workspace_parent),
            "EXTENDED_NOTEBOOK_CACHE": str(cache_dir),
            "EXTENDED_PROTOCOL": str(protocol),
            "EXTENDED_CALIBRATION_MANIFEST": str(calibration),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    source_identity = cell_identity(source)
    if (source_identity["cells"], source_identity["code_cells"], source_identity["markdown_cells"]) != (23, 13, 10):
        raise ValueError(f"Unexpected source cell counts: {source_identity}")

    manager = KernelSpecManager()
    specs = manager.find_kernel_specs()
    if "clean-env" not in specs:
        raise ValueError(f"Scoped clean-env kernel missing: {specs}")
    spec = manager.get_kernel_spec("clean-env")
    expected_python = str((ROOT / "output/revision/environment_validation/venv/Scripts/python.exe").resolve())
    if spec.argv[0] != expected_python:
        raise ValueError(f"Kernel interpreter mismatch: {spec.argv[0]} != {expected_python}")

    executed = NotebookClient(
        source,
        timeout=300,
        kernel_name="clean-env",
        resources={"metadata": {"path": str(CHECK / "isolated_workspace")}},
    ).execute()
    executed_path = CHECK / "executed_advice_marker_extended_study_clean_env.ipynb"
    nbformat.write(executed, executed_path)
    executed_identity = cell_identity(executed)
    source_equal = (
        source_identity["cell_types"] == executed_identity["cell_types"]
        and source_identity["sources"] == executed_identity["sources"]
    )
    errors = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    outputs = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    ]
    rendered = json.dumps(outputs, ensure_ascii=False)
    control = None
    for output in outputs:
        text = output.get("text") if output.get("output_type") == "stream" else None
        if isinstance(text, str) and "all_action_controls_disabled" in text:
            control = ast.literal_eval(text.strip())
            break
    control_ok = (
        isinstance(control, dict)
        and control.get("primary_cap") is None
        and control.get("sensitivity_cap") is None
        and control.get("all_action_controls_disabled") is True
        and len(control.get("action_controls", {})) == 21
        and not any(control.get("action_controls", {}).values())
    )
    after = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED}
    report = {
        "verification_schema_version": "extended-notebook-clean-isolated-environment-v1",
        "status": "passed" if source_equal and not errors and "Traceback (most recent call last)" not in rendered and control_ok and before == after else "failed",
        "scope": "all original notebook cells; safe defaults; no installation, download, model load, generation, freeze, or human records",
        "runtime": {
            "driver_python_executable": sys.executable,
            "python_version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "packages": {name: metadata.version(name) for name in ("nbclient", "nbformat", "ipykernel", "jupyter_client")},
            "kernel_name": "clean-env",
            "kernel_spec_path": specs["clean-env"],
            "kernel_argv": spec.argv,
            "jupyter_path": os.environ.get("JUPYTER_PATH"),
            "duration_seconds": time.time() - started,
        },
        "relocation": {
            "relocated_source_bundle": str(relocated_source_bundle),
            "relocated_frozen_bundle": str(relocated_frozen_bundle),
            "extracted_workspace": str(extracted_root),
            "protocol_path_from_environment": str(protocol),
            "calibration_manifest_path_from_environment": str(calibration),
            "protocol_sha256": sha256(protocol),
            "calibration_manifest_sha256": sha256(calibration),
        },
        "source_notebook": {
            "path": str(SOURCE_NOTEBOOK),
            "sha256": before[str(SOURCE_NOTEBOOK.relative_to(ROOT))],
            **{key: value for key, value in source_identity.items() if key != "sources"},
        },
        "executed_notebook": {
            "path": str(executed_path),
            "sha256": sha256(executed_path),
            **{key: value for key, value in executed_identity.items() if key != "sources"},
        },
        "source_type_equality": source_equal,
        "code_cells_executed": sum(cell.cell_type == "code" and cell.get("execution_count") is not None for cell in executed.cells),
        "output_objects_inspected": len(outputs),
        "error_outputs": len(errors),
        "traceback_text_found": "Traceback (most recent call last)" in rendered,
        "safe_default_controls": control,
        "safe_default_controls_valid": control_ok,
        "network_disabled_by_environment": True,
        "protected_input_sha256_before": before,
        "protected_input_sha256_after": after,
        "protected_inputs_unchanged": before == after,
        "scientific_interpretation": "Environment execution validates safe-default portability/importability only; it creates no empirical or human evidence and does not establish study completion.",
    }
    report_path = CHECK / "execution_verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "report_sha256": sha256(report_path), **{key: report[key] for key in ("status", "source_type_equality", "code_cells_executed", "error_outputs", "safe_default_controls_valid", "protected_inputs_unchanged")}}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
