"""Build and audit an identity-free BlackboxNLP submission package.

The public development repository is not anonymous. This builder creates a
standalone ZIP without git history, public-repository links, local paths, or
camera-ready metadata. It fails closed when an identity marker is detected.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from pypdf import PdfReader

from config import ROOT


PAPER_FILES = (
    "out/paper/acl_latex.pdf",
    "out/paper/acl_latex.tex",
    "out/paper/custom.bib",
    "out/paper/acl.sty",
    "out/paper/acl_natbib.bst",
)
CODE_FILES = (
    "config.py",
    "requirements-paper.txt",
    "experiments/__init__.py",
    "experiments/prompts.json",
    "experiments/summary_adapter.py",
    "experiments/monte_carlo_correction.py",
    "experiments/derive_paper_stats.py",
    "experiments/between_condition_stats.py",
    "experiments/stage_condition_gaps.py",
    "experiments/multiple_testing.py",
    "experiments/threshold_robustness.py",
    "experiments/audit_cluster_bootstrap.py",
    "experiments/dpo_pairwise.py",
    "experiments/recount_sft_olmo2.py",
    "experiments/matched_control_stagewise.py",
    "experiments/render_matched_control_results.py",
)
RESULT_FILES = (
    "out/results/summary_v2.json",
    "out/results/summary_v3.json",
    "out/results/interaction_tests.json",
    "out/results/interaction_tests_v3.json",
    "out/results/between_condition_stats.json",
    "out/results/stage_condition_gaps.json",
    "out/results/multiple_testing.json",
    "out/results/threshold_robustness.json",
    "out/results/dpo_pairwise.json",
    "out/results/sft_recount_olmo2.json",
    "out/results/ai_consensus_validation_v3.json",
    "out/results/ai_consensus_annotations_v3.csv",
    "out/results/ai_codex_annotations_v3.csv",
    "out/results/ai_codex_annotations_2_v3.csv",
    "out/results/ai_consensus_adjudication_details_v3.csv",
    "out/results/ai_claude_opus_adjudication_v3.json",
    "out/results/validation_items_v3.csv",
    "out/results/validation_key_v3.csv",
    "out/results/audit_cluster_bootstrap.json",
)
FORBIDDEN = (
    "vasilis",
    "mpvasilis",
    "innovation bee",
    "github.com/mpvasilis",
    "c:\\users\\",
    "c:/users/",
)
README = """# Anonymous BlackboxNLP supplementary package

This package contains the anonymous review PDF and source, deterministic
analysis scripts, prompt battery, and machine-readable result artifacts.

The 140-prompt evaluation, 90-prompt stagewise subset, and separate
120-sentence AI-only audit are distinct samples. The AI audit is not human
validation. Run the commands below from this directory after installing
`requirements-paper.txt`:

```text
python -m experiments.monte_carlo_correction
python -m experiments.derive_paper_stats
python -m experiments.between_condition_stats
python -m experiments.stage_condition_gaps
python -m experiments.multiple_testing
python -m experiments.threshold_robustness
python -m experiments.audit_cluster_bootstrap
```

The pending GPU matched-control extension is included as executable analysis
code but must not be interpreted as a completed result without its strict
24-test `status=complete` artifact.
"""


def copy_required(relative: str, destination_root: Path) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def audit_text_files(root: Path) -> None:
    failures: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() == ".pdf":
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in FORBIDDEN:
            if marker in text:
                failures.append(f"{path.relative_to(root).as_posix()}: {marker}")
    if failures:
        raise SystemExit("Anonymous-package audit failed:\n" + "\n".join(failures))


def audit_pdf(path: Path) -> None:
    reader = PdfReader(str(path))
    metadata = reader.metadata or {}
    combined = "\n".join((page.extract_text() or "") for page in reader.pages).lower()
    combined += "\n" + "\n".join(str(value) for value in metadata.values()).lower()
    leaks = [marker for marker in FORBIDDEN if marker in combined]
    if leaks:
        raise SystemExit(f"Anonymous PDF audit failed: {leaks}")
    if len(reader.pages) != 11:
        raise SystemExit(f"Unexpected PDF page count: {len(reader.pages)}")


def build(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="blackboxnlp_anonymous_") as temp:
        package = Path(temp) / "blackboxnlp_anonymous_submission"
        for relative in (*PAPER_FILES, *CODE_FILES, *RESULT_FILES):
            copy_required(relative, package)
        (package / "README_ANONYMOUS.md").write_text(README, encoding="utf-8")
        audit_text_files(package)
        audit_pdf(package / "out/paper/acl_latex.pdf")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package).as_posix())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "submission" / "blackboxnlp_anonymous_submission.zip",
    )
    args = parser.parse_args()
    print("Wrote", build(args.output))


if __name__ == "__main__":
    main()
