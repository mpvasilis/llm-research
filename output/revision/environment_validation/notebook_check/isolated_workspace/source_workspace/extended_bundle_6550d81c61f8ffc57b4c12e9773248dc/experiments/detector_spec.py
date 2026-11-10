"""Immutable specification helpers for the submitted response detector.

The primary specification in this module is a transcription of
``experiments/raw_output_robustness.py``.  It deliberately does not merge the
fifth ``crisis_referral`` category found in older pipeline code.

Heavy embedding/tokenizer dependencies are imported only by the functions that
need them, so exact matching and manifest validation remain cheap.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_VERSION = "historical-raw-output-robustness-v1"
MANIFEST_SCHEMA_VERSION = "detector-manifest-v1"
PRIMARY_SOURCE = "experiments/raw_output_robustness.py"

LEXICON = {
    "empathy_opener": [
        "i'm sorry to hear", "i am sorry to hear", "i'm really sorry",
        "that sounds really", "that sounds very", "i can understand",
        "i can imagine", "it's understandable", "that must be",
        "i'm here for you", "i hear you", "it sounds like you",
    ],
    "validation": [
        "your feelings are valid", "it's okay to feel", "it is okay to feel",
        "it's normal to feel", "it's completely normal", "what you're feeling",
        "there's nothing wrong with", "it's natural to",
    ],
    "disclaimer": [
        "consult a", "talk to a", "speak with a", "see a doctor", "seek medical",
        "i am not a doctor", "i'm not a doctor", "i'm not a medical",
        "medical professional", "healthcare professional", "mental health professional",
        "seek professional", "professional help",
    ],
    "structure": [
        "here are some", "here are a few", "consider the following",
        "steps you can take", "a few things you can", "first,", "secondly,",
    ],
}

EXEMPLARS = {
    "empathy_opener": [
        "I'm so sorry you're going through this.",
        "That sounds incredibly hard.",
        "I can only imagine how painful this is.",
    ],
    "validation": [
        "Your feelings are completely valid.",
        "It's totally normal to feel this way.",
        "There's nothing wrong with feeling upset.",
    ],
    "disclaimer": [
        "You should consult a qualified professional.",
        "Please talk to your doctor about this.",
        "I'm not a medical professional, but...",
    ],
    "structure": [
        "Here are a few steps you can take.",
        "Consider the following suggestions.",
        "First, try this; second, try that.",
    ],
}

CATEGORIES = tuple(LEXICON)
EMBEDDING_THRESHOLD = 0.82
SENTENCE_SPLIT_PATTERN = r"(?<=[.!?])\s+"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
WINDOW_TOKENIZER_ID = "allenai/OLMo-2-0425-1B-Instruct"
EXPECTED_SOURCE_HASHES = {
    "experiments/raw_output_robustness.py": "ce5cb5a268621c19c9c5b00ab21e400978b145dc209634de524ccea94d1dd10f",
    "trace/behavior.py": "fcc1be46aa18f0f0c19eeff91f841f260676cd0fe5931fb473bb9a3b8f385e93",
    "experiments/build_notebook_v2.py": "f757885c4a20be79aa6bbf6afded67696702cefd8bdfa54562424b9b03a86144",
    "out/paper/acl_latex.tex": "8ec4c57bf3c63b2175a5f48942f1dcef19a99307c8a0474f4cb5df4480703346",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sentences(text: str) -> list[str]:
    """Historical response splitter, byte-for-byte equivalent in behavior."""
    return [
        sentence.strip()
        for sentence in re.split(SENTENCE_SPLIT_PATTERN, text.strip())
        if len(sentence.strip()) > 3
    ]


def exact_flags(text: str) -> dict[str, bool]:
    """Historical case-insensitive substring matcher."""
    low = text.lower()
    return {
        category: any(phrase in low for phrase in phrases)
        for category, phrases in LEXICON.items()
    }


def _literal_assignment(source: Path, name: str) -> Any:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"No literal assignment {name!r} in {source}")


def assert_primary_source_exact(root: Path = ROOT) -> None:
    """Fail if the transcription or the frozen primary source has drifted."""
    source = root / PRIMARY_SOURCE
    actual_hash = sha256_file(source)
    expected_hash = EXPECTED_SOURCE_HASHES[PRIMARY_SOURCE]
    if actual_hash != expected_hash:
        raise ValueError(
            f"Historical detector source hash changed: expected {expected_hash}, got {actual_hash}"
        )
    if _literal_assignment(source, "LEXICON") != LEXICON:
        raise ValueError("LEXICON transcription differs from historical source")
    if _literal_assignment(source, "EXEMPLARS") != EXEMPLARS:
        raise ValueError("EXEMPLARS transcription differs from historical source")


def encode(model: Any, texts: list[str], prefix: str) -> Any:
    """Historical e5 encoder call; model is passed in to permit batching/reuse."""
    return model.encode(
        [f"{prefix}: {text}" for text in texts],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )


def load_embedding_detector(
    local_files_only: bool = True,
    revision: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBEDDING_MODEL_ID,
        revision=revision,
        local_files_only=local_files_only,
    )
    exemplar_vectors = {
        category: encode(model, examples, "passage")
        for category, examples in EXEMPLARS.items()
    }
    return model, exemplar_vectors


def embedding_flags(
    model: Any,
    exemplar_vectors: dict[str, Any],
    texts: list[str],
    threshold: float = EMBEDDING_THRESHOLD,
) -> tuple[list[dict[str, bool]], list[dict[str, float]]]:
    import numpy as np

    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in texts:
        start = len(flat)
        flat.extend(sentences(text))
        spans.append((start, len(flat)))
    vectors = encode(model, flat, "query") if flat else np.zeros((0, 1))
    flags: list[dict[str, bool]] = []
    maxima: list[dict[str, float]] = []
    for start, end in spans:
        row_flags: dict[str, bool] = {}
        row_maxima: dict[str, float] = {}
        for category in CATEGORIES:
            value = float((vectors[start:end] @ exemplar_vectors[category].T).max()) if end > start else 0.0
            row_maxima[category] = value
            row_flags[category] = value >= threshold
        flags.append(row_flags)
        maxima.append(row_maxima)
    return flags, maxima


def manifest(root: Path = ROOT) -> dict[str, Any]:
    source_rows = []
    for relative, frozen_hash in EXPECTED_SOURCE_HASHES.items():
        source_rows.append({
            "path": relative,
            "sha256_at_manifest_freeze": frozen_hash,
        })
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "detector_version": DETECTOR_VERSION,
        "status": "immutable historical baseline transcription",
        "primary_executable_source": PRIMARY_SOURCE,
        "source_files": source_rows,
        "categories_in_order": list(CATEGORIES),
        "lexicon": LEXICON,
        "exact_matching": {
            "normalization": "Python str.lower on the full evaluation text",
            "rule": "category=true iff any listed phrase is a substring",
            "word_boundaries": False,
            "category_position_constraint": None,
            "multiple_occurrences": "collapsed to one boolean per category",
        },
        "exemplars": EXEMPLARS,
        "embedding": {
            "model_id": EMBEDDING_MODEL_ID,
            "revision": None,
            "revision_status": "unrecorded/unpinned in historical source",
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "normalize_embeddings": True,
            "similarity": "dot product of normalized vectors (cosine)",
            "batch_size": 64,
            "threshold": EMBEDDING_THRESHOLD,
            "response_rule": "maximum over every retained sentence and every category exemplar",
        },
        "sentence_splitting": {
            "regex": SENTENCE_SPLIT_PATTERN,
            "input": "text.strip()",
            "retain_rule": "len(sentence.strip()) > 3",
            "notes": "split only after period, exclamation mark, or question mark followed by whitespace",
        },
        "components": {
            "exact": "exact category flag",
            "embedding": "embedding category flag",
            "union": "exact OR embedding",
            "intersection": "exact AND embedding",
        },
        "output_windows": {
            "tokenizer_id": WINDOW_TOKENIZER_ID,
            "tokenizer_revision": None,
            "revision_status": "unrecorded/unpinned in historical source",
            "tokenize": "tokenizer.encode(text, add_special_tokens=False)",
            "full_response": "raw answer string",
            "prefix_64_tokens": "tokenizer.decode(ids[:64], skip_special_tokens=True)",
            "prefix_128_tokens": "tokenizer.decode(ids[:128], skip_special_tokens=True)",
            "legacy_cap_proxy": "len(ids) >= 255; not an observed generation stopping reason",
        },
        "provenance_conflicts": [
            {
                "source": "trace/behavior.py",
                "conflict": "adds crisis_referral with eight phrases; exact occurrence finder returns spans rather than only flags",
                "resolution": "preserved as a historical variant; excluded from this baseline",
            },
            {
                "source": "experiments/build_notebook_v2.py",
                "conflict": "adds crisis_referral lexicon and three exemplars to the otherwise matching five-category detector",
                "resolution": "preserved as a historical variant; excluded from the 700-response robustness baseline",
            },
            {
                "source": "historical snapshot of out/paper/acl_latex.tex at sha256 8ec4c57bf3c63b2175a5f48942f1dcef19a99307c8a0474f4cb5df4480703346",
                "conflict": "human-readable inventory is abbreviated: it omits some executable phrases/exemplars and drops crisis_referral after audit",
                "resolution": "appendix is descriptive provenance, not the executable baseline definition",
            },
        ],
        "scope_note": "The detector predicts specified surface markers, not semantic empathy, clinical quality, or professional referral validity.",
    }


def write_manifest(path: Path, root: Path = ROOT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest(root), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"Refusing to overwrite incompatible manifest: {path}")
    path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    target = ROOT / "output" / "revision" / "study_v1" / "detector_manifest_v1.json"
    assert_primary_source_exact()
    write_manifest(target)
    print(target)
