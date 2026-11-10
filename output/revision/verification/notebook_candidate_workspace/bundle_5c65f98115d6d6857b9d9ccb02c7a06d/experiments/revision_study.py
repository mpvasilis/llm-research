"""Prospective revision-study setup, audit, prediction, and local generation CLI.

No command downloads a model by default. Generation is a dry run unless
``--execute`` is present, and execution is limited to a small development pilot
unless ``--reviewed-full-study`` is explicitly supplied after pilot review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .detector_spec import (
        CATEGORIES, DETECTOR_VERSION, EMBEDDING_MODEL_ID, EMBEDDING_THRESHOLD, EXPECTED_SOURCE_HASHES,
        LEXICON, ROOT, WINDOW_TOKENIZER_ID, assert_primary_source_exact,
        embedding_flags, exact_flags, load_embedding_detector,
        manifest as detector_manifest, sha256_file, write_manifest,
    )
except ImportError:  # direct script execution
    from detector_spec import (
        CATEGORIES, DETECTOR_VERSION, EMBEDDING_MODEL_ID, EMBEDDING_THRESHOLD, EXPECTED_SOURCE_HASHES,
        LEXICON, ROOT, WINDOW_TOKENIZER_ID, assert_primary_source_exact,
        embedding_flags, exact_flags, load_embedding_detector,
        manifest as detector_manifest, sha256_file, write_manifest,
    )


STUDY_VERSION = "revision-study-v1"
RUNNER_VERSION = "revision-study-local-smoke-runner-v2"
PREDICTION_SCHEMA_VERSION = "historical-detector-prediction-v2"
PROMPT_CORPUS_VERSION = "prompts-revision-v1"
PROMPTS_PATH = ROOT / "experiments" / "prompts_revision_v1.json"
DEFAULT_OUTPUT = ROOT / "output" / "revision" / "study_v1"
HISTORICAL_CACHE = ROOT / "out" / "raw_cache" / "final_7b_answers.jsonl"
EVALUATION_UNITS = ("full_response", "prefix_64_tokens", "prefix_128_tokens")
REQUIRED_SPLITS = {"development", "evaluation", "historical_diagnostic"}
CHECKPOINTS = {
    "1b_sft": "allenai/OLMo-2-0425-1B-SFT",
    "1b_dpo": "allenai/OLMo-2-0425-1B-DPO",
    "1b_instruct": "allenai/OLMo-2-0425-1B-Instruct",
    "7b_sft": "allenai/OLMo-2-1124-7B-SFT",
    "7b_dpo": "allenai/OLMo-2-1124-7B-DPO",
    "7b_instruct": "allenai/OLMo-2-1124-7B-Instruct",
}
DEFAULT_GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 256,
    "seeds": [0, 1, 2],
}
KNOWN_CONSTRUCTION_PROMPTS = {
    prompt
    for values in json.loads((ROOT / "experiments" / "prompts.json").read_text(encoding="utf-8")).values()
    for prompt in values
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any, length: int = 20) -> str:
    return f"{prefix}_{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()[:length]}"


def load_prompt_corpus(path: Path = PROMPTS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expand_prompts(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in corpus["families"]:
        for request_type in ("information", "advice"):
            for emotional_framing in (False, True):
                parts = [family["context"]]
                if emotional_framing:
                    parts.append(family["emotional_clause"])
                parts.append(family["requests"][request_type])
                prompt_text = " ".join(parts)
                condition = {
                    "domain": family["domain"],
                    "emotional_framing": emotional_framing,
                    "request_type": request_type,
                }
                identity = {
                    "corpus_version": corpus["corpus_version"],
                    "scenario_family_id": family["scenario_family_id"],
                    "split": family["split"],
                    "condition": condition,
                    "prompt": prompt_text,
                }
                rows.append({
                    "prompt_id": stable_id("prompt", identity),
                    "scenario_family_id": family["scenario_family_id"],
                    "split": family["split"],
                    "condition": condition,
                    "prompt": prompt_text,
                    "authoring_provenance": corpus["authoring_provenance"],
                    "human_manipulation_check": corpus["human_manipulation_check"],
                })
    return rows


def validate_prompt_corpus(corpus: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if corpus.get("study_version") != STUDY_VERSION:
        errors.append(f"study_version must be {STUDY_VERSION}")
    if corpus.get("corpus_version") != PROMPT_CORPUS_VERSION:
        errors.append(f"corpus_version must be {PROMPT_CORPUS_VERSION}")
    if corpus.get("human_manipulation_check") != "PENDING":
        errors.append("fresh draft must retain human_manipulation_check=PENDING")
    if "human" in corpus.get("authoring_provenance", "").lower() and "AI-assisted" not in corpus.get("authoring_provenance", ""):
        errors.append("authoring provenance must not imply human validation")

    families = corpus.get("families")
    if not isinstance(families, list):
        raise ValueError("families must be a list")
    family_ids = [family.get("scenario_family_id") for family in families]
    if len(family_ids) != len(set(family_ids)) or any(not item for item in family_ids):
        errors.append("scenario_family_id values must be unique and nonblank")
    family_counts = Counter((family.get("split"), family.get("domain")) for family in families)
    expected_family_counts = Counter({
        ("evaluation", "health"): 20,
        ("evaluation", "everyday"): 20,
        ("development", "health"): 5,
        ("development", "everyday"): 5,
    })
    if family_counts != expected_family_counts:
        errors.append(f"family balance differs: {dict(family_counts)}")

    for family in families:
        if family.get("split") not in {"development", "evaluation"}:
            errors.append(f"invalid split for {family.get('scenario_family_id')}")
        context = family.get("context", "").strip()
        emotional_clause = family.get("emotional_clause", "").strip()
        requests = family.get("requests", {})
        if not context or not emotional_clause or set(requests) != {"information", "advice"}:
            errors.append(f"incomplete factorized construction: {family.get('scenario_family_id')}")
        if not all(isinstance(value, str) and value.strip() for value in requests.values()):
            errors.append(f"blank request in {family.get('scenario_family_id')}")

    rows = expand_prompts(corpus)
    prompt_ids = [row["prompt_id"] for row in rows]
    texts = [row["prompt"] for row in rows]
    if len(rows) != 200:
        errors.append(f"expected 200 prompts, found {len(rows)}")
    if len(prompt_ids) != len(set(prompt_ids)):
        errors.append("derived prompt ids are not unique")
    if len(texts) != len(set(texts)):
        errors.append("prompt texts are not unique")
    for row in rows:
        if row["prompt"] in KNOWN_CONSTRUCTION_PROMPTS:
            errors.append(f"known 12-prompt construction overlap: {row['prompt']}")
    cell_counts = Counter(
        (row["split"], row["condition"]["domain"], row["condition"]["emotional_framing"], row["condition"]["request_type"])
        for row in rows
    )
    for split, per_cell in (("evaluation", 20), ("development", 5)):
        for domain in ("health", "everyday"):
            for emotion in (False, True):
                for request in ("information", "advice"):
                    if cell_counts[(split, domain, emotion, request)] != per_cell:
                        errors.append(f"unbalanced cell {(split, domain, emotion, request)}")
    if errors:
        raise ValueError("Prompt corpus validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "valid_draft_pending_human_manipulation_check",
        "families": len(families),
        "prompts": len(rows),
        "family_counts": {"|".join(key): value for key, value in sorted(family_counts.items())},
        "cell_counts": {"|".join(map(str, key)): value for key, value in sorted(cell_counts.items(), key=lambda item: str(item[0]))},
        "known_construction_prompt_overlaps": 0,
    }


def require_keys(record: dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in record or record[key] in (None, "")]
    if missing:
        raise ValueError(f"{label} missing required metadata: {', '.join(missing)}")


def require_present_keys(record: dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"{label} missing required metadata: {', '.join(missing)}")


def validate_response_record(record: dict[str, Any]) -> None:
    historical = record.get("split") == "historical_diagnostic"
    required = (
        "study_version", "response_id", "scenario_family_id", "split", "condition",
        "model_id", "model_revision", "tokenizer_revision", "seed", "serialization_id",
        "model_hash", "generation",
    )
    if not historical:
        required += (
            "prompt_id", "prompt_corpus_sha256", "prompt_text_sha256", "rng_seed",
            "rng_scheme", "response_identity", "runner_version",
        )
    (require_present_keys if historical else require_keys)(record, required, "response")
    if not historical and record["study_version"] != STUDY_VERSION:
        raise ValueError(f"unsupported study_version: {record['study_version']}")
    if record["split"] not in REQUIRED_SPLITS:
        raise ValueError(f"invalid split: {record['split']}")
    generation = record["generation"]
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    generation_required = (
        "prompt", "serialized_prompt", "response", "input_tokens", "output_tokens",
        "max_new_tokens", "stopping", "config",
    )
    if not historical:
        generation_required += ("timing",)
    (require_present_keys if historical else require_keys)(generation, generation_required, "generation")
    if not generation.get("prompt") or not isinstance(generation.get("response"), str):
        raise ValueError("generation prompt/response must be recorded")
    stopping = generation["stopping"]
    if not isinstance(stopping, dict):
        raise ValueError("generation.stopping must be an object")
    stopping_required = ("reason", "eos_reached", "cap_reached")
    (require_present_keys if historical else require_keys)(stopping, stopping_required, "generation.stopping")
    if stopping["eos_reached"] is True and stopping["cap_reached"] is True:
        raise ValueError("eos_reached and cap_reached cannot both be true")
    if not isinstance(record["seed"], int):
        raise ValueError("seed must be an integer")
    if not historical:
        if record["prompt_text_sha256"] != hashlib.sha256(generation["prompt"].encode("utf-8")).hexdigest():
            raise ValueError("prompt_text_sha256 does not match generation.prompt")
        if canonical_json(record["response_identity"]) != canonical_json(_response_identity_from_record(record)):
            raise ValueError("response_identity does not match response metadata")
        if record["response_id"] != stable_id("response", record["response_identity"], 32):
            raise ValueError("response_id does not match response_identity")


def validate_prediction_record(record: dict[str, Any]) -> None:
    require_keys(record, (
        "study_version", "response_id", "detector_version", "prediction_schema_version",
        "prediction_variant", "evaluation_unit", "decision_field", "dependency_identity",
        "predictions",
    ), "prediction")
    if record["evaluation_unit"] not in EVALUATION_UNITS:
        raise ValueError(f"invalid evaluation_unit: {record['evaluation_unit']}")
    if not isinstance(record["predictions"], dict):
        raise ValueError("predictions must be an object")


def _response_identity_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "study_version": record["study_version"],
        "runner_version": record["runner_version"],
        "prompt_corpus_sha256": record["prompt_corpus_sha256"],
        "prompt_id": record["prompt_id"],
        "prompt_text_sha256": record["prompt_text_sha256"],
        "scenario_family_id": record["scenario_family_id"],
        "split": record["split"],
        "condition": record["condition"],
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
        "tokenizer_revision": record["tokenizer_revision"],
        "logical_seed": record["seed"],
        "rng_seed": record["rng_seed"],
        "rng_scheme": record["rng_scheme"],
        "serialization_id": record["serialization_id"],
        "generation_config": {
            key: record["generation"]["config"][key]
            for key in ("do_sample", "temperature", "top_p", "max_new_tokens")
        },
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc


def write_json_compatible(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Refusing to overwrite incompatible artifact: {path}")
    path.write_text(text, encoding="utf-8")


def write_jsonl_compatible(path: Path, rows: list[dict[str, Any]], id_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for row in iter_jsonl(path):
            key = row.get(id_key)
            if not key or key in existing:
                raise ValueError(f"Existing cache has missing/duplicate {id_key}: {path}")
            existing[key] = row
    merged = dict(existing)
    for row in rows:
        key = row[id_key]
        if key in merged and canonical_json(merged[key]) != canonical_json(row):
            raise FileExistsError(f"Refusing incompatible cached record {key} in {path}")
        merged[key] = row
    text = "".join(json.dumps(merged[key], ensure_ascii=False, sort_keys=True) + "\n" for key in sorted(merged))
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def load_unique_jsonl(path: Path, id_key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in iter_jsonl(path):
        key = row.get(id_key)
        if not key:
            raise ValueError(f"Existing cache has missing {id_key}: {path}")
        if key in rows:
            raise ValueError(f"Existing cache has duplicate {id_key}={key}: {path}")
        rows[key] = row
    return rows


def export_run_spec(path: Path) -> dict[str, Any]:
    corpus = load_prompt_corpus()
    validation = validate_prompt_corpus(corpus)
    prompts = [row for row in expand_prompts(corpus) if row["split"] == "evaluation"]
    planned = []
    for checkpoint_name, model_id in CHECKPOINTS.items():
        for prompt in prompts:
            for seed in DEFAULT_GENERATION["seeds"]:
                identity = {
                    "study_version": STUDY_VERSION,
                    "prompt_id": prompt["prompt_id"],
                    "model_id": model_id,
                    "model_revision": "RESOLVE_AT_EXECUTION",
                    "tokenizer_revision": "RESOLVE_AT_EXECUTION",
                    "seed": seed,
                    "serialization_id": "OFFICIAL_CHAT_TEMPLATE_HASH_AT_EXECUTION",
                    "generation_config": {key: value for key, value in DEFAULT_GENERATION.items() if key != "seeds"},
                }
                planned.append({
                    "planned_response_key": stable_id("planned", identity),
                    "prompt_id": prompt["prompt_id"],
                    "scenario_family_id": prompt["scenario_family_id"],
                    "split": prompt["split"],
                    "condition": prompt["condition"],
                    "model_id": model_id,
                    "seed": seed,
                })
    spec = {
        "study_version": STUDY_VERSION,
        "status": "PROSPECTIVE_PENDING_HUMAN_MANIPULATION_CHECK_AND_FULL_RUNNER_IMPLEMENTATION",
        "prompt_corpus_version": PROMPT_CORPUS_VERSION,
        "prompt_validation": validation,
        "evaluation_prompts": len(prompts),
        "scenario_families": len({row["scenario_family_id"] for row in prompts}),
        "checkpoints": CHECKPOINTS,
        "generation": DEFAULT_GENERATION,
        "generation_runner": {
            "implemented_scope": "local development smoke only (1-2 prompts, <=32 new tokens)",
            "full_grid_runner": "NOT_IMPLEMENTED",
            "confirmatory_gate": "pending human manipulation check, infrastructure, and separate runner review",
        },
        "expected_responses": len(planned),
        "join_key": "response_id after model/tokenizer metadata resolution; never row order",
        "planned_rows": planned,
    }
    write_json_compatible(path, spec)
    return spec


def audit_historical(cache_path: Path = HISTORICAL_CACHE) -> dict[str, Any]:
    assert_primary_source_exact()
    source_lexicon = _read_literal_lexicon(ROOT / "experiments" / "raw_output_robustness.py")
    rows = list(iter_jsonl(cache_path))
    required = {"key", "model", "stage", "model_id", "tag", "i", "seed", "is_chat", "question", "answer"}
    missing_rows = [index for index, row in enumerate(rows) if not required.issubset(row)]
    duplicate_keys = len(rows) - len({row.get("key") for row in rows})
    mismatches = []
    for row in rows:
        expected = {
            category: any(phrase in row["answer"].lower() for phrase in phrases)
            for category, phrases in source_lexicon.items()
        }
        actual = exact_flags(row["answer"])
        if expected != actual:
            mismatches.append(row["key"])
    return {
        "audit_version": "historical-cache-audit-v1",
        "status": "pass" if len(rows) == 700 and not missing_rows and not duplicate_keys and not mismatches else "fail",
        "cache_path": str(cache_path.relative_to(ROOT)) if cache_path.is_relative_to(ROOT) else str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "responses": len(rows),
        "unique_response_keys": len({row.get("key") for row in rows}),
        "unique_prompts": len({row.get("question") for row in rows}),
        "condition_counts": dict(sorted(Counter(row.get("tag") for row in rows).items())),
        "missing_required_metadata_rows": missing_rows,
        "duplicate_key_count": duplicate_keys,
        "exact_match_parity": {
            "checked_responses": len(rows),
            "mismatch_count": len(mismatches),
            "mismatch_keys": mismatches,
        },
        "detector_source_sha256": EXPECTED_SOURCE_HASHES["experiments/raw_output_robustness.py"],
        "scope": "Recovered final-7B diagnostic only; not prospective evaluation and not stagewise validation.",
    }


def _read_literal_lexicon(path: Path) -> dict[str, list[str]]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "LEXICON" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"LEXICON not found in {path}")


def _evaluation_texts(
    record: dict[str, Any],
    units: list[str],
    tokenizer_revision: str | None,
) -> dict[str, str]:
    response = record["generation"]["response"]
    texts = {"full_response": response}
    prefix_units = [unit for unit in units if unit != "full_response"]
    if prefix_units:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            WINDOW_TOKENIZER_ID,
            revision=tokenizer_revision,
            local_files_only=True,
        )
        token_ids = tokenizer.encode(response, add_special_tokens=False)
        texts["prefix_64_tokens"] = tokenizer.decode(token_ids[:64], skip_special_tokens=True)
        texts["prefix_128_tokens"] = tokenizer.decode(token_ids[:128], skip_special_tokens=True)
    return {unit: texts[unit] for unit in units}


def predict_records(
    records: list[dict[str, Any]],
    units: list[str],
    include_embedding: bool,
) -> list[dict[str, Any]]:
    seen = set()
    for record in records:
        validate_response_record(record)
        if record["response_id"] in seen:
            raise ValueError(f"duplicate response_id: {record['response_id']}")
        seen.add(record["response_id"])
    dependencies = prediction_dependencies(
        include_embedding=include_embedding,
        include_prefix=any(unit != "full_response" for unit in units),
    )
    model = exemplar_vectors = None
    if include_embedding:
        model, exemplar_vectors = load_embedding_detector(
            local_files_only=True,
            revision=dependencies["embedding_model"]["resolved_revision"],
        )
    prepared = [
        (record, unit, text)
        for record in records
        for unit, text in _evaluation_texts(
            record, units, dependencies["prefix_window_tokenizer"]["resolved_revision"]
        ).items()
    ]
    if include_embedding:
        all_embedding, all_maxima = embedding_flags(
            model, exemplar_vectors, [text for _record, _unit, text in prepared]
        )
    else:
        all_embedding = all_maxima = [None] * len(prepared)
    output = []
    for index, (record, unit, text) in enumerate(prepared):
            exact = exact_flags(text)
            if include_embedding:
                emb, maxima = all_embedding[index], all_maxima[index]
                predictions = {
                    category: {
                        "exact": exact[category],
                        "embedding": emb[category],
                        "union": exact[category] or emb[category],
                        "intersection": exact[category] and emb[category],
                        "embedding_max_similarity": maxima[category],
                    }
                    for category in CATEGORIES
                }
                decision_field = "union"
                variant = "raw_output_robustness_four_category_union_provenance_v2"
            else:
                predictions = {
                    category: {
                        "exact": exact[category],
                        "embedding": "unavailable",
                        "union": "unavailable",
                        "intersection": "unavailable",
                        "embedding_max_similarity": None,
                    }
                    for category in CATEGORIES
                }
                decision_field = "exact"
                variant = "raw_output_robustness_four_category_exact_only_provenance_v2"
            prediction_identity = {
                "response_id": record["response_id"],
                "detector_version": DETECTOR_VERSION,
                "prediction_variant": variant,
                "evaluation_unit": unit,
                "dependency_identity": dependencies,
            }
            output.append({
                "prediction_id": stable_id("prediction", prediction_identity, 32),
                "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
                "study_version": record["study_version"],
                "response_id": record["response_id"],
                "scenario_family_id": record["scenario_family_id"],
                "split": record["split"],
                "detector_version": DETECTOR_VERSION,
                "prediction_variant": variant,
                "evaluation_unit": unit,
                "decision_field": decision_field,
                "dependency_identity": dependencies,
                "threshold": EMBEDDING_THRESHOLD if include_embedding else None,
                "predictions": predictions,
                "unavailable_categories": {
                    "crisis_referral": "not_applicable: absent from raw_output_robustness four-category baseline"
                },
            })
    for row in output:
        validate_prediction_record(row)
    return output


def _resolve_weight_identity(model_id: str, model_revision: str) -> tuple[str, list[str]]:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / model_revision
    blob_ids: list[str] = []
    if snapshot.exists():
        for path in sorted(snapshot.rglob("*.safetensors")):
            try:
                target = path.resolve(strict=True)
                blob_ids.append(target.name)
            except OSError:
                continue
    identity = {"model_id": model_id, "revision": model_revision, "weight_blob_ids": blob_ids}
    return "hf-content-manifest-sha256:" + hashlib.sha256(canonical_json(identity).encode()).hexdigest(), blob_ids


def _cached_hf_revision(repo_id: str) -> str | None:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = cache_root / ("models--" + repo_id.replace("/", "--"))
    main_ref = repo_dir / "refs" / "main"
    if main_ref.exists():
        value = main_ref.read_text(encoding="utf-8").strip()
        return value or None
    snapshots = repo_dir / "snapshots"
    candidates = sorted(path.name for path in snapshots.iterdir()) if snapshots.exists() else []
    return candidates[-1] if len(candidates) == 1 else None


def prediction_dependencies(include_embedding: bool, include_prefix: bool) -> dict[str, Any]:
    embedding_revision = _cached_hf_revision(EMBEDDING_MODEL_ID) if include_embedding else None
    tokenizer_revision = _cached_hf_revision(WINDOW_TOKENIZER_ID) if include_prefix else None
    if include_embedding and not embedding_revision:
        raise ValueError(f"Cannot resolve cached embedding revision for {EMBEDDING_MODEL_ID}")
    if include_prefix and not tokenizer_revision:
        raise ValueError(f"Cannot resolve cached prefix-tokenizer revision for {WINDOW_TOKENIZER_ID}")
    return {
        "embedding_model": {
            "model_id": EMBEDDING_MODEL_ID,
            "resolved_revision": embedding_revision,
            "applied": include_embedding,
        },
        "prefix_window_tokenizer": {
            "model_id": WINDOW_TOKENIZER_ID,
            "resolved_revision": tokenizer_revision,
            "applied": include_prefix,
            "role": "applied only to requested 64/128-token windows; not required for full-response prediction",
        },
        "historical_run_revision_status": "unrecorded; these are current reproduction dependencies",
    }


def _model_metadata(model_id: str) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model_revision = getattr(config, "_commit_hash", None)
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash") or model_revision
    if not model_revision or not tokenizer_revision:
        raise ValueError("Cached model/tokenizer does not expose a resolved Hugging Face revision")
    template = tokenizer.chat_template
    if not template:
        raise ValueError(f"No official chat template for {model_id}; study serialization is undefined")
    serialization_id = "official-chat-template-sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()
    model_hash, blob_ids = _resolve_weight_identity(model_id, model_revision)
    return {
        "config": config,
        "tokenizer": tokenizer,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "serialization_id": serialization_id,
        "model_hash": model_hash,
        "weight_blob_ids": blob_ids,
    }


def generation_plan(model_id: str, pilot_local: bool, max_prompts: int, max_new_tokens: int, seed: int) -> dict[str, Any]:
    corpus = load_prompt_corpus()
    validate_prompt_corpus(corpus)
    if pilot_local and not (1 <= max_prompts <= 2 and 1 <= max_new_tokens <= 32):
        raise ValueError("Local pilot requires 1 <= max_prompts <= 2 and 1 <= max_new_tokens <= 32")
    if not pilot_local:
        raise ValueError("This first-phase runner implements only --pilot-local development smoke generation")
    split = "development"
    prompts = [row for row in expand_prompts(corpus) if row["split"] == split]
    prompts = prompts[:max_prompts] if max_prompts else prompts
    return {
        "study_version": STUDY_VERSION,
        "runner_version": RUNNER_VERSION,
        "response_schema_version": "revision-study-response-v2",
        "mode": "local_development_smoke",
        "model_id": model_id,
        "split": split,
        "prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "local_files_only": True,
        "network_downloads": False,
        "paid_apis": False,
        "prompt_ids": [row["prompt_id"] for row in prompts],
        "prompt_corpus_sha256": sha256_file(PROMPTS_PATH),
        "rng_scheme": "sha256(prompt_id + logical_seed), first 8 bytes modulo 2^63; reset immediately before each generate call",
    }


def execute_generation(
    output_dir: Path,
    model_id: str,
    pilot_local: bool,
    max_prompts: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    execution_started = time.monotonic()
    plan = generation_plan(model_id, pilot_local, max_prompts, max_new_tokens, seed)
    metadata = _model_metadata(model_id)
    tokenizer = metadata.pop("tokenizer")
    metadata.pop("config")
    concrete_config = {**plan, **metadata}
    run_id = stable_id("run", concrete_config)
    run_dir = output_dir / "runs" / run_id
    write_json_compatible(run_dir / "run_config.json", concrete_config)

    import torch
    from transformers import AutoModelForCausalLM

    model_load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=metadata["model_revision"],
        local_files_only=True,
        torch_dtype="auto",
    )
    model.eval()
    model_load_seconds = time.monotonic() - model_load_started
    corpus = load_prompt_corpus()
    split = "development" if pilot_local else "evaluation"
    prompts = [row for row in expand_prompts(corpus) if row["split"] == split]
    prompts = prompts[:max_prompts] if max_prompts else prompts
    responses_path = run_dir / "responses.jsonl"
    existing = load_unique_jsonl(responses_path, "response_id")
    response_rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_text_sha256 = hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest()
        rng_material = {"prompt_id": prompt["prompt_id"], "logical_seed": seed}
        rng_seed = int.from_bytes(hashlib.sha256(canonical_json(rng_material).encode()).digest()[:8], "big") % (2 ** 63)
        generation_config = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": max_new_tokens}
        response_identity = {
            "study_version": STUDY_VERSION,
            "runner_version": RUNNER_VERSION,
            "prompt_corpus_sha256": plan["prompt_corpus_sha256"],
            "prompt_id": prompt["prompt_id"],
            "prompt_text_sha256": prompt_text_sha256,
            "scenario_family_id": prompt["scenario_family_id"],
            "split": prompt["split"],
            "condition": prompt["condition"],
            "model_id": model_id,
            "model_revision": metadata["model_revision"],
            "tokenizer_revision": metadata["tokenizer_revision"],
            "logical_seed": seed,
            "rng_seed": rng_seed,
            "rng_scheme": plan["rng_scheme"],
            "serialization_id": metadata["serialization_id"],
            "generation_config": generation_config,
        }
        response_id = stable_id("response", response_identity, 32)
        if response_id in existing:
            validate_response_record(existing[response_id])
            response_rows.append(existing[response_id])
            continue
        serialized = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["prompt"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = tokenizer(serialized, return_tensors="pt", add_special_tokens=False)
        torch.manual_seed(rng_seed)
        generation_started = time.monotonic()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
        generation_seconds = time.monotonic() - generation_started
        input_count = int(encoded["input_ids"].shape[-1])
        continuation = generated[0, input_count:]
        output_count = int(continuation.shape[-1])
        response_text = tokenizer.decode(continuation, skip_special_tokens=True)
        effective_eos = model.generation_config.eos_token_id
        if effective_eos is None:
            effective_eos = tokenizer.eos_token_id
        eos_ids = [effective_eos] if isinstance(effective_eos, int) else list(effective_eos or [])
        eos_reached = bool(output_count and int(continuation[-1]) in eos_ids)
        cap_reached = bool(output_count >= max_new_tokens and not eos_reached)
        reason = "eos_token" if eos_reached else "max_new_tokens" if cap_reached else "other_model_stop"
        row = {
            "study_version": STUDY_VERSION,
            "runner_version": RUNNER_VERSION,
            "response_id": response_id,
            "response_identity": response_identity,
            "prompt_id": prompt["prompt_id"],
            "scenario_family_id": prompt["scenario_family_id"],
            "split": prompt["split"],
            "condition": prompt["condition"],
            "model_id": model_id,
            "model_revision": metadata["model_revision"],
            "tokenizer_revision": metadata["tokenizer_revision"],
            "model_hash": metadata["model_hash"],
            "seed": seed,
            "rng_seed": rng_seed,
            "rng_scheme": plan["rng_scheme"],
            "prompt_corpus_sha256": plan["prompt_corpus_sha256"],
            "prompt_text_sha256": prompt_text_sha256,
            "serialization_id": metadata["serialization_id"],
            "generation": {
                "prompt": prompt["prompt"],
                "serialized_prompt": serialized,
                "response": response_text,
                "input_tokens": input_count,
                "output_tokens": output_count,
                "max_new_tokens": max_new_tokens,
                "stopping": {"reason": reason, "eos_reached": eos_reached, "cap_reached": cap_reached},
                "config": {**generation_config, "effective_eos_token_ids": eos_ids},
                "timing": {
                    "generation_seconds": generation_seconds,
                    "generated_tokens_per_second": output_count / generation_seconds if generation_seconds > 0 else None,
                    "scope": "model.generate call only; excludes model loading and prompt serialization",
                },
            },
        }
        validate_response_record(row)
        response_rows.append(row)
        write_jsonl_compatible(responses_path, [row], "response_id")
    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "responses": len(response_rows),
        "new_responses": len([row for row in response_rows if row["response_id"] not in existing]),
        "model_load_seconds": model_load_seconds,
        "execution_wall_seconds": time.monotonic() - execution_started,
        "timing_note": "execution wall time includes metadata resolution, model load, serialization, generation, and checkpoint writes",
    }
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = attempts_dir / f"attempt_{time.time_ns()}.json"
    attempt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["attempt_report"] = str(attempt_path)
    return result


def doctor() -> dict[str, Any]:
    package_status = {}
    for package in ("torch", "transformers", "sentence_transformers"):
        try:
            module = __import__(package)
            package_status[package] = getattr(module, "__version__", "installed")
        except Exception as exc:
            package_status[package] = f"unavailable: {exc}"
    cached_models = {}
    for name, model_id in CHECKPOINTS.items():
        try:
            metadata = _model_metadata(model_id)
            cached_models[name] = {
                "model_id": model_id,
                "available_local": True,
                "model_revision": metadata["model_revision"],
                "tokenizer_revision": metadata["tokenizer_revision"],
                "weight_blob_count": len(metadata["weight_blob_ids"]),
            }
        except Exception as exc:
            cached_models[name] = {"model_id": model_id, "available_local": False, "reason": str(exc)}
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    try:
        import psutil
        physical_memory_gib = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        physical_memory_gib = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_gib": physical_memory_gib,
        "cuda_available": cuda,
        "packages": package_status,
        "cached_models": cached_models,
        "historical_detector_dependencies": {
            "embedding_model_id": "intfloat/multilingual-e5-small",
            "embedding_model_cached_revision_current_machine": _cached_hf_revision("intfloat/multilingual-e5-small"),
            "window_tokenizer_id": WINDOW_TOKENIZER_ID,
            "window_tokenizer_cached_revision_current_machine": _cached_hf_revision(WINDOW_TOKENIZER_ID),
            "provenance_warning": "These current cache revisions were not recorded by the historical run and must not be presented as recovered historical revisions.",
        },
        "recommendation": "Use only the two-prompt, 32-token 1B development pilot on CPU; do not run the 2,880-response study locally.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-prompts")
    doctor_parser = sub.add_parser("doctor")
    doctor_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "resource_inventory_v1.json")
    manifest_parser = sub.add_parser("write-detector-manifest")
    manifest_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "detector_manifest_v1.json")
    audit_parser = sub.add_parser("audit-historical")
    audit_parser.add_argument("--cache", type=Path, default=HISTORICAL_CACHE)
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "historical_cache_audit_v1.json")
    spec_parser = sub.add_parser("export-run-spec")
    spec_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective_run_spec_v1.json")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--units", nargs="+", choices=EVALUATION_UNITS, default=["full_response"])
    predict_parser.add_argument("--include-embedding", action="store_true")
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--model-id", default=CHECKPOINTS["1b_instruct"])
    generate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    generate_parser.add_argument("--pilot-local", action="store_true")
    generate_parser.add_argument("--max-prompts", type=int, default=2)
    generate_parser.add_argument("--max-new-tokens", type=int, default=32)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate-prompts":
        result = validate_prompt_corpus(load_prompt_corpus())
    elif args.command == "doctor":
        result = doctor()
        write_json_compatible(args.output, result)
    elif args.command == "write-detector-manifest":
        assert_primary_source_exact()
        write_manifest(args.output)
        result = {"output": str(args.output), "detector_version": DETECTOR_VERSION}
    elif args.command == "audit-historical":
        result = audit_historical(args.cache)
        write_json_compatible(args.output, result)
    elif args.command == "export-run-spec":
        result = export_run_spec(args.output)
    elif args.command == "predict":
        records = list(iter_jsonl(args.input))
        predictions = predict_records(records, args.units, args.include_embedding)
        write_jsonl_compatible(args.output, predictions, "prediction_id")
        result = {"input_records": len(records), "prediction_records": len(predictions), "output": str(args.output)}
    elif args.command == "generate":
        plan = generation_plan(args.model_id, args.pilot_local, args.max_prompts, args.max_new_tokens, args.seed)
        if args.execute:
            result = execute_generation(
                args.output_dir, args.model_id, args.pilot_local,
                args.max_prompts, args.max_new_tokens, args.seed,
            )
        else:
            result = {"status": "dry_run", "execute_required": True, "plan": plan}
    else:
        raise AssertionError(args.command)
    if args.command == "export-run-spec":
        result = {key: result[key] for key in ("study_version", "status", "expected_responses")}
        result["output"] = str(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
