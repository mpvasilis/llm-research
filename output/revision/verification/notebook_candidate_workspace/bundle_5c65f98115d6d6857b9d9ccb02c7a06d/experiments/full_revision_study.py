"""Full prospective generation runner for the advice-marker revision study.

This module is separate from ``revision_study.py``'s local smoke runner.  It
freezes model/tokenizer revisions and serialization before generation, supports
exploratory development runs without human labels, and requires a completed
external human-review manifest for evaluation or sensitivity runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from . import revision_study as shared
except ImportError:
    import revision_study as shared


ROOT = Path(__file__).resolve().parents[1]
RUNNER_VERSION = "revision-study-full-runner-v2"
PROTOCOL_SCHEMA_VERSION = "revision-study-protocol-v1"
REVIEW_SCHEMA_VERSION = "prompt-human-review-v1"
RESPONSE_SCHEMA_VERSION = "revision-study-response-v2"
DEFAULT_OUTPUT = ROOT / "output" / "revision" / "full_study"
PROMPTS_PATH = ROOT / "experiments" / "prompts_revision_v1.json"
SOURCE_PATHS = (
    "experiments/full_revision_study.py",
    "experiments/revision_study.py",
    "experiments/detector_spec.py",
    "experiments/prompts_revision_v1.json",
)
CHECKPOINTS = dict(shared.CHECKPOINTS)
PRIMARY_GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 256,
}
PRIMARY_SEEDS = (0, 1, 2)
LONGER_BUDGET_TOKENS = 512
LONGER_BUDGET_SEEDS = (0,)
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, value: Any, length: int = 24) -> str:
    return f"{prefix}_{sha256_bytes(canonical_json(value).encode('utf-8'))[:length]}"


def atomic_write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite incompatible immutable artifact: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_copy_bytes(source: Path, destination: Path, *, immutable: bool = False) -> None:
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if immutable and destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite incompatible immutable artifact: {destination}")
        return
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_bytes(payload)
    os.replace(temp, destination)


def append_jsonl_durable(path: Path, value: Any) -> None:
    """Append one event and ask the OS to flush it before generation continues."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in SOURCE_PATHS}


def load_corpus(path: Path = PROMPTS_PATH) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    corpus = shared.load_prompt_corpus(path)
    shared.validate_prompt_corpus(corpus)
    return corpus, shared.expand_prompts(corpus), sha256_file(path)


def resolve_revision(model_id: str, local_files_only: bool) -> str:
    if local_files_only:
        revision = shared._cached_hf_revision(model_id)
        if not revision:
            raise FileNotFoundError(f"No single resolved local revision for {model_id}")
        return revision
    from huggingface_hub import HfApi

    revision = HfApi().model_info(model_id, revision="main").sha
    if not revision:
        raise RuntimeError(f"Hugging Face returned no commit SHA for {model_id}")
    return revision


def tokenizer_identity(model_id: str, revision: str, local_files_only: bool) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    resolved = tokenizer.init_kwargs.get("_commit_hash") or revision
    if resolved != revision:
        raise ValueError(f"Tokenizer revision mismatch for {model_id}: requested {revision}, got {resolved}")
    if not tokenizer.chat_template:
        raise ValueError(f"Checkpoint lacks an official chat template: {model_id}@{revision}")
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "SERIALIZATION_PROBE"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    template_hash = sha256_bytes(tokenizer.chat_template.encode("utf-8"))
    rendered_probe_hash = sha256_bytes(probe.encode("utf-8"))
    serialization_id = stable_id("serialization", {
        "template_sha256": template_hash,
        "rendered_probe_sha256": rendered_probe_hash,
    }, 32)
    return {
        "tokenizer_revision": resolved,
        "chat_template_sha256": template_hash,
        "rendered_probe_sha256": rendered_probe_hash,
        "serialization_id": serialization_id,
    }


def select_sensitivity_families(rows: list[dict[str, Any]]) -> list[str]:
    families_by_domain: dict[str, set[str]] = {"health": set(), "everyday": set()}
    for row in rows:
        if row["split"] == "evaluation":
            families_by_domain[row["condition"]["domain"]].add(row["scenario_family_id"])
    selected = []
    for domain in ("health", "everyday"):
        ranked = sorted(
            families_by_domain[domain],
            key=lambda value: sha256_bytes(f"longer-budget-v1|{value}".encode()),
        )
        selected.extend(ranked[:4])
    return selected


def response_identity(
    prompt: dict[str, Any],
    checkpoint: dict[str, Any],
    logical_seed: int,
    generation_config: dict[str, Any],
    corpus_sha256: str,
) -> dict[str, Any]:
    rng_material = {"prompt_id": prompt["prompt_id"], "logical_seed": logical_seed}
    rng_seed = int.from_bytes(hashlib.sha256(canonical_json(rng_material).encode()).digest()[:8], "big") % (2 ** 63)
    rng_scheme = "sha256(canonical_json({prompt_id,logical_seed})) first 8 bytes modulo 2^63; reset before each single-response generate call"
    return {
        "study_version": shared.STUDY_VERSION,
        "runner_version": RUNNER_VERSION,
        "prompt_corpus_sha256": corpus_sha256,
        "prompt_id": prompt["prompt_id"],
        "prompt_text_sha256": sha256_bytes(prompt["prompt"].encode("utf-8")),
        "scenario_family_id": prompt["scenario_family_id"],
        "split": prompt["split"],
        "condition": prompt["condition"],
        "model_id": checkpoint["model_id"],
        "model_revision": checkpoint["model_revision"],
        "tokenizer_revision": checkpoint["tokenizer_revision"],
        "logical_seed": logical_seed,
        "rng_seed": rng_seed,
        "rng_scheme": rng_scheme,
        "serialization_id": checkpoint["serialization_id"],
        "generation_config": generation_config,
    }


def grid_entries(
    prompts: list[dict[str, Any]],
    checkpoint_rows: dict[str, dict[str, Any]],
    aliases: Iterable[str],
    seeds: Iterable[int],
    generation_config: dict[str, Any],
    corpus_sha256: str,
) -> list[dict[str, Any]]:
    entries = []
    for alias in aliases:
        checkpoint = checkpoint_rows[alias]
        for prompt in sorted(prompts, key=lambda row: row["prompt_id"]):
            for seed in seeds:
                identity = response_identity(prompt, checkpoint, seed, generation_config, corpus_sha256)
                entries.append({
                    "response_id": stable_id("response", identity, 32),
                    "checkpoint_alias": alias,
                    "prompt_id": prompt["prompt_id"],
                    "scenario_family_id": prompt["scenario_family_id"],
                    "split": prompt["split"],
                    "condition": prompt["condition"],
                    "prompt": prompt["prompt"],
                    "logical_seed": seed,
                    "response_identity": identity,
                })
    return entries


def freeze_protocol(output: Path, local_files_only: bool = False) -> dict[str, Any]:
    corpus, prompts, corpus_hash = load_corpus()
    checkpoint_rows: dict[str, dict[str, Any]] = {}
    for alias, model_id in CHECKPOINTS.items():
        revision = resolve_revision(model_id, local_files_only)
        tok = tokenizer_identity(model_id, revision, local_files_only)
        checkpoint_rows[alias] = {
            "checkpoint_alias": alias,
            "model_id": model_id,
            "model_revision": revision,
            "model_hash": f"hf-repository-commit:{model_id}@{revision}",
            **tok,
        }
    serialization_validation = {}
    for size in ("1b", "7b"):
        aliases = [f"{size}_sft", f"{size}_dpo", f"{size}_instruct"]
        identities = {checkpoint_rows[alias]["serialization_id"] for alias in aliases}
        if len(identities) != 1:
            raise ValueError(f"Official serialization differs within {size}: {sorted(identities)}")
        serialization_validation[size] = {
            "checkpoint_aliases": aliases,
            "serialization_id": next(iter(identities)),
            "status": "same official interface within model size",
        }
    serialization_validation["cross_size"] = {
        "same": serialization_validation["1b"]["serialization_id"] == serialization_validation["7b"]["serialization_id"],
        "status": "recorded comparison only; cross-size identity is not forced",
    }

    evaluation_prompts = [row for row in prompts if row["split"] == "evaluation"]
    development_prompts = [row for row in prompts if row["split"] == "development"]
    sensitivity_families = select_sensitivity_families(prompts)
    sensitivity_prompts = [row for row in evaluation_prompts if row["scenario_family_id"] in sensitivity_families]
    primary = grid_entries(
        evaluation_prompts, checkpoint_rows, CHECKPOINTS, PRIMARY_SEEDS,
        dict(PRIMARY_GENERATION), corpus_hash,
    )
    longer_config = {**PRIMARY_GENERATION, "max_new_tokens": LONGER_BUDGET_TOKENS}
    sensitivity = grid_entries(
        sensitivity_prompts, checkpoint_rows, CHECKPOINTS, LONGER_BUDGET_SEEDS,
        longer_config, corpus_hash,
    )
    development_primary = grid_entries(
        development_prompts, checkpoint_rows, ("1b_instruct",), (0,),
        dict(PRIMARY_GENERATION), corpus_hash,
    )
    development_longer = grid_entries(
        development_prompts, checkpoint_rows, ("1b_instruct",), (0,),
        longer_config, corpus_hash,
    )
    core = {
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "study_version": shared.STUDY_VERSION,
        "runner_version": RUNNER_VERSION,
        "status": "FROZEN_PROTOCOL_PROMPTS_PENDING_HUMAN_MANIPULATION_REVIEW",
        "prompt_corpus": {
            "path": "experiments/prompts_revision_v1.json",
            "sha256": corpus_hash,
            "corpus_version": corpus["corpus_version"],
            "authoring_provenance": corpus["authoring_provenance"],
            "human_manipulation_check": corpus["human_manipulation_check"],
        },
        "source_sha256": source_hashes(),
        "checkpoints": checkpoint_rows,
        "serialization_validation": serialization_validation,
        "primary_generation": PRIMARY_GENERATION,
        "primary_seeds": list(PRIMARY_SEEDS),
        "primary_grid": primary,
        "primary_expected_responses": len(primary),
        "development": {
            "classification": "exploratory development; independent of human-review completion",
            "primary_256_grid": development_primary,
            "primary_256_expected_responses": len(development_primary),
            "longer_512_grid": development_longer,
            "longer_512_expected_responses": len(development_longer),
        },
        "longer_budget_sensitivity": {
            "classification": "pre-specified sensitivity; requires completed human review before evaluation execution",
            "selection_rule": "four families per domain ranked by sha256('longer-budget-v1|' + scenario_family_id)",
            "scenario_family_ids": sensitivity_families,
            "generation": longer_config,
            "seeds": list(LONGER_BUDGET_SEEDS),
            "grid": sensitivity,
            "expected_responses": len(sensitivity),
        },
        "prospective_human_validation_plan": {
            "status": "PROVISIONAL_PENDING_DEVELOPMENT_ANNOTATION_TIMING",
            "random_sample": {
                "strata_definition": "6 checkpoints x 2 emotional-framing levels x 2 request types x 2 domains",
                "strata": 48,
                "items_per_stratum": 13,
                "planned_total": 624,
            },
            "positive_enrichment": {
                "maximum_total": 120,
                "illustrative_rule": "up to 30 full-response items per original detector category",
            },
            "human_label_evaluation_unit": "full_response",
            "prefix_units": ["prefix_64_tokens", "prefix_128_tokens"],
            "prefix_role": "detector sensitivity only; never joined to full-response human labels",
            "historical_112_item_packet_role": "diagnostic only; not a held-out prospective set",
        },
    }
    if len(primary) != 2880:
        raise AssertionError(f"Primary grid must contain 2,880 responses, found {len(primary)}")
    if len(development_primary) != 40 or len(development_longer) != 40:
        raise AssertionError("Development grids must each contain 40 responses")
    core["protocol_id"] = stable_id("protocol", core, 32)
    atomic_write_json(output, core, immutable=True)
    return core


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Unsupported protocol schema")
    claimed_id = protocol.get("protocol_id")
    core = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if claimed_id != stable_id("protocol", core, 32):
        raise ValueError("Protocol content does not match protocol_id")
    _corpus, _prompts, corpus_hash = load_corpus()
    if protocol["prompt_corpus"]["sha256"] != corpus_hash:
        raise ValueError("Current prompt corpus differs from frozen protocol")
    if protocol["source_sha256"] != source_hashes():
        raise ValueError("Current bundled sources differ from frozen protocol")
    if protocol["primary_expected_responses"] != 2880 or len(protocol["primary_grid"]) != 2880:
        raise ValueError("Frozen protocol primary grid is not complete")
    return protocol


def create_human_review_manifest(protocol: dict[str, Any], output: Path) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for entry in protocol["primary_grid"]:
        family_id = entry["scenario_family_id"]
        family = families.setdefault(family_id, {
            "scenario_family_id": family_id,
            "domain": entry["condition"]["domain"],
            "prompt_ids": [],
            "prompt_text_sha256": [],
            "checks": {
                "topic_consistent_across_variants": None,
                "emotional_framing_manipulation_clear": None,
                "advice_information_manipulation_clear": None,
                "information_variants_do_not_request_advice": None,
                "approved_for_evaluation": None,
            },
            "reviewer_notes": "",
        })
        if entry["prompt_id"] not in family["prompt_ids"]:
            family["prompt_ids"].append(entry["prompt_id"])
            family["prompt_text_sha256"].append(sha256_bytes(entry["prompt"].encode("utf-8")))
    for family in families.values():
        paired = sorted(zip(family["prompt_ids"], family["prompt_text_sha256"]))
        family["prompt_ids"] = [item[0] for item in paired]
        family["prompt_text_sha256"] = [item[1] for item in paired]
    manifest = {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "status": "PENDING_HUMAN_REVIEW",
        "protocol_id": protocol["protocol_id"],
        "prompt_corpus_sha256": protocol["prompt_corpus"]["sha256"],
        "reviewer_id": "",
        "reviewed_at": "",
        "review_method": "",
        "families": [families[key] for key in sorted(families)],
        "provenance_note": "Blank template generated by code. Completion and judgments must be supplied by a human reviewer.",
    }
    atomic_write_json(output, manifest, immutable=True)
    return manifest


def validate_human_review(protocol: dict[str, Any], path: Path) -> dict[str, Any]:
    review = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    for key, expected in (
        ("review_schema_version", REVIEW_SCHEMA_VERSION),
        ("protocol_id", protocol["protocol_id"]),
        ("prompt_corpus_sha256", protocol["prompt_corpus"]["sha256"]),
        ("status", "COMPLETE_HUMAN_REVIEW"),
    ):
        if review.get(key) != expected:
            errors.append(f"{key} must equal {expected!r}")
    for field in ("reviewer_id", "reviewed_at", "review_method"):
        if not str(review.get(field, "")).strip():
            errors.append(f"{field} must be completed by the human reviewer")
    expected_families = {
        entry["scenario_family_id"] for entry in protocol["primary_grid"]
    }
    rows = review.get("families", [])
    if len(rows) != len(expected_families) or {row.get("scenario_family_id") for row in rows} != expected_families:
        errors.append("review family set must equal the 40 evaluation families")
    expected_by_family: dict[str, dict[str, str]] = {}
    for entry in protocol["primary_grid"]:
        expected_by_family.setdefault(entry["scenario_family_id"], {})[entry["prompt_id"]] = sha256_bytes(entry["prompt"].encode("utf-8"))
    for row in rows:
        family_id = row.get("scenario_family_id")
        expected = expected_by_family.get(family_id, {})
        actual = dict(zip(row.get("prompt_ids", []), row.get("prompt_text_sha256", [])))
        if actual != expected:
            errors.append(f"prompt identities differ for {family_id}")
        checks = row.get("checks", {})
        if set(checks) != {
            "topic_consistent_across_variants",
            "emotional_framing_manipulation_clear",
            "advice_information_manipulation_clear",
            "information_variants_do_not_request_advice",
            "approved_for_evaluation",
        } or any(value is not True for value in checks.values()):
            errors.append(f"all review checks must be true for {family_id}")
    if errors:
        raise ValueError("Human review manifest is incomplete or incompatible:\n- " + "\n- ".join(errors))
    return review


def choose_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return requested


def execution_profile(requested_device: str, torch_threads: int | None) -> dict[str, Any]:
    import torch
    import transformers

    if torch_threads is not None:
        if torch_threads < 1:
            raise ValueError("--torch-threads must be at least 1")
        torch.set_num_threads(torch_threads)
    device = choose_device(requested_device)
    if device == "cuda":
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        accelerator = {
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": props.total_memory,
        }
    else:
        dtype = "float32"
        accelerator = None
    core = {
        "profile_schema_version": "revision-study-execution-profile-v1",
        "device_class": device,
        "dtype": dtype,
        "torch_num_threads": torch.get_num_threads(),
        "single_response_generation": True,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "accelerator": accelerator,
    }
    return {**core, "execution_profile_id": stable_id("execution", core, 32)}


def load_checkpoint(checkpoint: dict[str, Any], profile: dict[str, Any], local_files_only: bool) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint["model_id"], revision=checkpoint["tokenizer_revision"],
        local_files_only=local_files_only,
    )
    observed_template = sha256_bytes(tokenizer.chat_template.encode("utf-8")) if tokenizer.chat_template else None
    if observed_template != checkpoint["chat_template_sha256"]:
        raise ValueError(f"Tokenizer serialization differs from protocol for {checkpoint['checkpoint_alias']}")
    device = profile["device_class"]
    dtype_by_name = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_by_name[profile["dtype"]]
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint["model_id"], revision=checkpoint["model_revision"],
            local_files_only=local_files_only, dtype=dtype, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint["model_id"], revision=checkpoint["model_revision"],
            local_files_only=local_files_only, dtype=dtype,
        )
        model.to("cpu")
    model.eval()
    return model, tokenizer, str(dtype)


def entry_to_response(
    entry: dict[str, Any],
    checkpoint: dict[str, Any],
    model: Any,
    tokenizer: Any,
    execution: dict[str, Any],
    dtype: str,
    protocol_id: str,
) -> dict[str, Any]:
    import torch

    identity = entry["response_identity"]
    rng_seed = identity["rng_seed"]
    prompt = entry["prompt"]
    serialized = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(serialized, return_tensors="pt", add_special_tokens=False)
    device = execution["device_class"]
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    started = time.monotonic()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=identity["generation_config"]["do_sample"],
            temperature=identity["generation_config"]["temperature"],
            top_p=identity["generation_config"]["top_p"],
            max_new_tokens=identity["generation_config"]["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    generation_seconds = time.monotonic() - started
    input_tokens = int(encoded["input_ids"].shape[-1])
    continuation = generated[0, input_tokens:].detach().cpu()
    output_tokens = int(continuation.shape[-1])
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    effective_eos = model.generation_config.eos_token_id
    if effective_eos is None:
        effective_eos = tokenizer.eos_token_id
    eos_ids = [effective_eos] if isinstance(effective_eos, int) else list(effective_eos or [])
    eos_reached = bool(output_tokens and int(continuation[-1]) in eos_ids)
    cap = identity["generation_config"]["max_new_tokens"]
    cap_reached = bool(output_tokens >= cap and not eos_reached)
    stop_reason = "eos_token" if eos_reached else "max_new_tokens" if cap_reached else "other_model_stop"
    row = {
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "study_version": shared.STUDY_VERSION,
        "runner_version": RUNNER_VERSION,
        "protocol_id": protocol_id,
        "execution_profile_id": execution["execution_profile_id"],
        "execution_profile": execution,
        "response_id": entry["response_id"],
        "response_identity": identity,
        "prompt_id": entry["prompt_id"],
        "prompt_corpus_sha256": identity["prompt_corpus_sha256"],
        "prompt_text_sha256": identity["prompt_text_sha256"],
        "scenario_family_id": entry["scenario_family_id"],
        "split": entry["split"],
        "condition": entry["condition"],
        "model_id": checkpoint["model_id"],
        "model_revision": checkpoint["model_revision"],
        "tokenizer_revision": checkpoint["tokenizer_revision"],
        "model_hash": checkpoint["model_hash"],
        "serialization_id": checkpoint["serialization_id"],
        "seed": entry["logical_seed"],
        "rng_seed": rng_seed,
        "rng_scheme": identity["rng_scheme"],
        "generation": {
            "prompt": prompt,
            "serialized_prompt": serialized,
            "response": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_new_tokens": cap,
            "stopping": {"reason": stop_reason, "eos_reached": eos_reached, "cap_reached": cap_reached},
            "config": {
                **identity["generation_config"], "effective_eos_token_ids": eos_ids,
                "device": device, "dtype": dtype,
                "execution_profile_id": execution["execution_profile_id"],
            },
            "timing": {
                "generation_seconds": generation_seconds,
                "generated_tokens_per_second": output_tokens / generation_seconds if generation_seconds > 0 else None,
                "scope": "single model.generate call only",
            },
        },
    }
    validate_response(row, entry, protocol_id, execution)
    return row


def validate_response(
    row: dict[str, Any],
    expected: dict[str, Any],
    protocol_id: str,
    expected_execution: dict[str, Any] | None = None,
) -> None:
    shared.validate_response_record(row)
    if row.get("response_schema_version") != RESPONSE_SCHEMA_VERSION:
        raise ValueError("Wrong response schema version")
    if row.get("protocol_id") != protocol_id:
        raise ValueError("Response protocol_id mismatch")
    if row["response_id"] != expected["response_id"] or row["response_identity"] != expected["response_identity"]:
        raise ValueError("Response identity does not match frozen grid entry")
    if row["prompt_id"] != expected["prompt_id"] or row["generation"]["prompt"] != expected["prompt"]:
        raise ValueError("Response prompt does not match frozen grid entry")
    if row["generation"]["serialized_prompt"] == "":
        raise ValueError("Serialized prompt must be retained")
    if expected_execution is not None:
        if row.get("execution_profile") != expected_execution:
            raise ValueError("Response execution profile differs from run config")
        if row.get("execution_profile_id") != expected_execution["execution_profile_id"]:
            raise ValueError("Response execution profile ID differs from run config")
        config = row["generation"]["config"]
        if config.get("device") != expected_execution["device_class"] or config.get("dtype") != f"torch.{expected_execution['dtype']}":
            raise ValueError("Response device or dtype differs from run execution profile")


def _signal_handler(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _mode_grid(
    protocol: dict[str, Any],
    mode: str,
    family_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], str]:
    if mode == "evaluation":
        if family_ids:
            raise ValueError("Evaluation mode cannot subset the frozen 2,880-response grid")
        return protocol["primary_grid"], "confirmatory_evaluation_after_external_human_review"
    if mode == "sensitivity":
        if family_ids:
            raise ValueError("Sensitivity mode cannot subset its pre-specified grid")
        return protocol["longer_budget_sensitivity"]["grid"], "pre_specified_longer_budget_sensitivity"
    if mode in {"development-256", "development-512"}:
        key = "primary_256_grid" if mode.endswith("256") else "longer_512_grid"
        grid = protocol["development"][key]
        if family_ids:
            unknown = set(family_ids) - {entry["scenario_family_id"] for entry in grid}
            if unknown:
                raise ValueError(f"Unknown development family ids: {sorted(unknown)}")
            grid = [entry for entry in grid if entry["scenario_family_id"] in set(family_ids)]
        return grid, "exploratory_development_not_confirmatory"
    raise ValueError(f"Unknown mode: {mode}")


def prepare_run(
    protocol_path: Path,
    mode: str,
    output_root: Path,
    family_ids: list[str] | None,
    human_review_path: Path | None,
    execution: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    protocol = load_protocol(protocol_path)
    grid, classification = _mode_grid(protocol, mode, family_ids)
    review_hash = None
    if mode in {"evaluation", "sensitivity"}:
        if human_review_path is None:
            raise ValueError(f"{mode} requires --human-review with completed external judgments")
        validate_human_review(protocol, human_review_path)
        review_hash = sha256_file(human_review_path)
    config_core = {
        "run_schema_version": "revision-study-run-v1",
        "runner_version": RUNNER_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "mode": mode,
        "classification": classification,
        "human_review_manifest_sha256": review_hash,
        "scenario_family_ids": sorted({entry["scenario_family_id"] for entry in grid}),
        "execution_profile": execution,
        "response_ids": [entry["response_id"] for entry in grid],
        "expected_responses": len(grid),
    }
    config = {**config_core, "run_id": stable_id("run", config_core, 32)}
    run_dir = output_root / "runs" / config["run_id"]
    atomic_write_json(run_dir / "run_config.json", config, immutable=True)
    if human_review_path is not None and mode in {"evaluation", "sensitivity"}:
        atomic_copy_bytes(human_review_path, run_dir / "human_review_manifest.json", immutable=True)
    return protocol, grid, run_dir


def run_generation(
    protocol_path: Path,
    mode: str,
    output_root: Path,
    family_ids: list[str] | None,
    human_review_path: Path | None,
    device_request: str,
    local_files_only: bool,
    torch_threads: int | None,
) -> dict[str, Any]:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    execution = execution_profile(device_request, torch_threads)
    protocol, grid, run_dir = prepare_run(
        protocol_path, mode, output_root, family_ids, human_review_path, execution,
    )
    response_dir = run_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = {entry["response_id"] for entry in grid}
    existing_files = list(response_dir.glob("*.json"))
    if len({path.stem for path in existing_files}) != len(existing_files):
        raise ValueError("Duplicate response filenames in cache")
    unexpected = {path.stem for path in existing_files} - expected_ids
    if unexpected:
        raise ValueError(f"Cache contains unexpected response ids: {sorted(unexpected)[:5]}")
    entry_by_id = {entry["response_id"]: entry for entry in grid}
    completed = set()
    for path in existing_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        validate_response(row, entry_by_id[path.stem], protocol["protocol_id"], execution)
        completed.add(path.stem)
    pending = [entry for entry in grid if entry["response_id"] not in completed]
    device = execution["device_class"]
    print(f"run={run_dir.name} mode={mode} device={device} complete={len(completed)} pending={len(pending)}", flush=True)
    if not pending:
        return {"run_dir": str(run_dir), "completed": len(completed), "new_responses": 0, "remaining": 0, "status": "complete"}

    attempt_id = f"attempt_{time.time_ns()}"
    attempt_started = utc_now()
    attempt_started_monotonic = time.monotonic()
    progress_path = run_dir / "progress.jsonl"
    append_jsonl_durable(progress_path, {
        "event": "attempt_started", "attempt_id": attempt_id, "at": attempt_started,
        "device": device, "torch_num_threads": __import__("torch").get_num_threads(),
        "complete_before_attempt": len(completed), "pending_before_attempt": len(pending),
    })
    prior_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            prior_handlers[signum] = signal.signal(signum, _signal_handler)
        except (ValueError, OSError):
            pass
    new_count = 0
    try:
        for alias in CHECKPOINTS:
            checkpoint_pending = [entry for entry in pending if entry["checkpoint_alias"] == alias]
            if not checkpoint_pending or STOP_REQUESTED:
                continue
            checkpoint = protocol["checkpoints"][alias]
            load_started = time.monotonic()
            model, tokenizer, dtype = load_checkpoint(checkpoint, execution, local_files_only)
            load_seconds = time.monotonic() - load_started
            append_jsonl_durable(progress_path, {
                "event": "checkpoint_loaded", "attempt_id": attempt_id, "at": utc_now(),
                "checkpoint_alias": alias, "model_load_seconds": load_seconds,
                "pending_for_checkpoint": len(checkpoint_pending), "dtype": dtype,
            })
            print(f"loaded={alias} seconds={load_seconds:.2f} pending={len(checkpoint_pending)}", flush=True)
            for index, entry in enumerate(checkpoint_pending, 1):
                row = entry_to_response(entry, checkpoint, model, tokenizer, execution, dtype, protocol["protocol_id"])
                atomic_write_json(response_dir / f"{entry['response_id']}.json", row, immutable=True)
                new_count += 1
                done = len(completed) + new_count
                append_jsonl_durable(progress_path, {
                    "event": "response_checkpointed", "attempt_id": attempt_id, "at": utc_now(),
                    "response_id": entry["response_id"], "checkpoint_alias": alias,
                    "completed": done, "expected": len(grid),
                    "generation_seconds": row["generation"]["timing"]["generation_seconds"],
                    "output_tokens": row["generation"]["output_tokens"],
                })
                if index == 1 or index % 10 == 0 or index == len(checkpoint_pending):
                    print(f"checkpoint={alias} checkpoint_progress={index}/{len(checkpoint_pending)} run_progress={done}/{len(grid)}", flush=True)
                if STOP_REQUESTED:
                    print("stop requested; latest response is checkpointed", flush=True)
                    break
            del model
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
            if STOP_REQUESTED:
                break
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
    complete = len(completed) + new_count
    result = {
        "run_dir": str(run_dir),
        "completed": complete,
        "new_responses": new_count,
        "remaining": len(grid) - complete,
        "status": "complete" if complete == len(grid) else "interrupted_resumable",
        "attempt_id": attempt_id,
        "attempt_started_at": attempt_started,
        "attempt_finished_at": utc_now(),
        "attempt_wall_seconds": time.monotonic() - attempt_started_monotonic,
        "device": device,
        "torch_num_threads": __import__("torch").get_num_threads(),
    }
    atomic_write_json(run_dir / "attempts" / f"{attempt_id}.json", result, immutable=True)
    append_jsonl_durable(progress_path, {"event": "attempt_finished", **result})
    return result


def verify_run(protocol_path: Path, run_dir: Path, require_complete: bool = True) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if config.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Run config protocol_id mismatch")
    if config.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Run config protocol hash mismatch")
    full_grid, classification = _mode_grid(protocol, config["mode"], None)
    if config["classification"] != classification:
        raise ValueError("Run classification mismatch")
    full_by_id = {entry["response_id"]: entry for entry in full_grid}
    config_core = {key: value for key, value in config.items() if key != "run_id"}
    if config.get("run_id") != stable_id("run", config_core, 32) or run_dir.name != config.get("run_id"):
        raise ValueError("Run config content or directory does not match run_id")
    if len(config.get("response_ids", [])) != len(set(config.get("response_ids", []))):
        raise ValueError("Run config contains duplicate response IDs")
    unknown_config_ids = set(config.get("response_ids", [])) - set(full_by_id)
    if unknown_config_ids:
        raise ValueError(f"Run config contains IDs outside the frozen mode grid: {sorted(unknown_config_ids)[:5]}")
    if config["mode"] in {"evaluation", "sensitivity"} and config["response_ids"] != [entry["response_id"] for entry in full_grid]:
        raise ValueError("Evaluation and sensitivity run configs must contain their complete frozen grids")
    configured_families = config.get("scenario_family_ids")
    if configured_families != sorted(set(configured_families or [])):
        raise ValueError("Run scenario_family_ids must be unique and sorted")
    expected_family_set = set(configured_families)
    canonical_selected = [
        entry["response_id"] for entry in full_grid
        if entry["scenario_family_id"] in expected_family_set
    ]
    if config["response_ids"] != canonical_selected:
        raise ValueError("Run response IDs must contain every frozen variant for each selected family")
    if config.get("expected_responses") != len(config["response_ids"]):
        raise ValueError("Run expected_responses does not match its response ID list")
    execution = config.get("execution_profile")
    if not isinstance(execution, dict):
        raise ValueError("Run config lacks an execution profile")
    execution_core = {key: value for key, value in execution.items() if key != "execution_profile_id"}
    if execution.get("execution_profile_id") != stable_id("execution", execution_core, 32):
        raise ValueError("Run execution profile does not match its identity")
    if config["mode"] in {"evaluation", "sensitivity"}:
        review_snapshot = run_dir / "human_review_manifest.json"
        if not review_snapshot.exists():
            raise ValueError("Evaluation run lacks its human-review snapshot")
        if sha256_file(review_snapshot) != config.get("human_review_manifest_sha256"):
            raise ValueError("Human-review snapshot hash mismatch")
        validate_human_review(protocol, review_snapshot)
    elif config.get("human_review_manifest_sha256") is not None:
        raise ValueError("Development run must not claim a human-review gate")
    grid = [full_by_id[response_id] for response_id in config["response_ids"]]
    expected = {entry["response_id"]: entry for entry in grid}
    if config["response_ids"] != [entry["response_id"] for entry in grid]:
        raise ValueError("Run config response order/identity differs from frozen grid")
    paths = list((run_dir / "responses").glob("*.json"))
    rows = []
    seen = set()
    errors = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            response_id = row.get("response_id")
            if response_id in seen:
                errors.append(f"duplicate response_id {response_id}")
                continue
            seen.add(response_id)
            if path.stem != response_id:
                errors.append(f"filename mismatch {path.name}")
                continue
            if response_id not in expected:
                errors.append(f"unexpected response_id {response_id}")
                continue
            validate_response(row, expected[response_id], protocol["protocol_id"], execution)
            rows.append(row)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    missing = sorted(set(expected) - seen)
    if require_complete and missing:
        errors.append(f"missing {len(missing)} expected responses")
    if errors:
        raise ValueError("Run verification failed:\n- " + "\n- ".join(errors[:20]))
    counts = Counter(row["generation"]["stopping"]["reason"] for row in rows)
    return {
        "status": "complete" if len(rows) == len(expected) else "partial_valid",
        "mode": config["mode"],
        "classification": classification,
        "expected_responses": len(expected),
        "verified_responses": len(rows),
        "missing_responses": len(missing),
        "unique_response_ids": len(seen),
        "stopping_reasons": dict(sorted(counts.items())),
        "cap_rate": sum(row["generation"]["stopping"]["cap_reached"] for row in rows) / len(rows) if rows else None,
    }


def export_run(protocol_path: Path, run_dir: Path, output: Path, require_complete: bool = True) -> dict[str, Any]:
    report = verify_run(protocol_path, run_dir, require_complete=require_complete)
    rows = []
    for path in (run_dir / "responses").glob("*.json"):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: row["response_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(temp, output)
    report["output"] = str(output)
    report["output_sha256"] = sha256_file(output)
    return report


def evaluation_windows(
    response: str,
    tokenizer: Any,
    tokenizer_dependency: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    full_tokens = len(token_ids)
    windows = []
    for unit, limit in (("full_response", None), ("prefix_64_tokens", 64), ("prefix_128_tokens", 128)):
        if limit is None:
            text = response
            evaluated_tokens = full_tokens
            shorter = None
            truncated = False
        else:
            evaluated_ids = token_ids[:limit]
            text = tokenizer.decode(evaluated_ids, skip_special_tokens=True)
            evaluated_tokens = len(evaluated_ids)
            shorter = full_tokens < limit
            truncated = full_tokens > limit
        metadata = {
            "tokenizer_model_id": tokenizer_dependency["model_id"],
            "tokenizer_revision": tokenizer_dependency["resolved_revision"],
            "full_response_tokens": full_tokens,
            "window_token_limit": limit,
            "evaluated_tokens": evaluated_tokens,
            "shorter_than_window": shorter,
            "truncated": truncated,
        }
        windows.append((unit, text, metadata))
    return windows


def predict_complete_evaluation(protocol_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    """Score a verified evaluation archive in all three pre-specified windows."""
    report = verify_run(protocol_path, run_dir, require_complete=True)
    if report["mode"] != "evaluation" or report["verified_responses"] != 2880:
        raise ValueError("Prediction export requires the complete 2,880-response evaluation run")
    protocol = load_protocol(protocol_path)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "responses").glob("*.json")
    ]
    rows.sort(key=lambda row: row["response_id"])
    dependencies = shared.prediction_dependencies(include_embedding=True, include_prefix=True)
    tokenizer_dependency = dependencies["prefix_window_tokenizer"]
    from transformers import AutoTokenizer

    window_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dependency["model_id"],
        revision=tokenizer_dependency["resolved_revision"],
        local_files_only=True,
    )
    prepared: list[tuple[dict[str, Any], str, str, dict[str, Any]]] = []
    for row in rows:
        response = row["generation"]["response"]
        for unit, text, metadata in evaluation_windows(response, window_tokenizer, tokenizer_dependency):
            prepared.append((row, unit, text, metadata))

    model, exemplar_vectors = shared.load_embedding_detector(
        local_files_only=True,
        revision=dependencies["embedding_model"]["resolved_revision"],
    )
    embedding, maxima = shared.embedding_flags(
        model, exemplar_vectors, [text for _row, _unit, text, _metadata in prepared],
    )
    grid_by_id = {entry["response_id"]: entry for entry in protocol["primary_grid"]}
    predictions = []
    variant = "raw_output_robustness_four_category_union_provenance_v2"
    for index, (row, unit, text, window_metadata) in enumerate(prepared):
        exact = shared.exact_flags(text)
        component_rows = {
            category: {
                "exact": exact[category],
                "embedding": embedding[index][category],
                "union": exact[category] or embedding[index][category],
                "intersection": exact[category] and embedding[index][category],
                "embedding_max_similarity": maxima[index][category],
            }
            for category in shared.CATEGORIES
        }
        entry = grid_by_id[row["response_id"]]
        alias = entry["checkpoint_alias"]
        size, raw_stage = alias.split("_", 1)
        stage = "final" if raw_stage == "instruct" else raw_stage
        prediction_identity = {
            "response_id": row["response_id"],
            "detector_version": shared.DETECTOR_VERSION,
            "prediction_variant": variant,
            "evaluation_unit": unit,
            "dependency_identity": dependencies,
        }
        prediction = {
            "prediction_id": shared.stable_id("prediction", prediction_identity, 32),
            "prediction_schema_version": shared.PREDICTION_SCHEMA_VERSION,
            "study_version": row["study_version"],
            "protocol_id": protocol["protocol_id"],
            "run_id": config["run_id"],
            "execution_profile_id": config["execution_profile"]["execution_profile_id"],
            "response_id": row["response_id"],
            "scenario_family_id": row["scenario_family_id"],
            "split": row["split"],
            "condition": row["condition"],
            "model_id": row["model_id"],
            "checkpoint_alias": alias,
            "model_size": size,
            "stage": stage,
            "seed": row["seed"],
            "generation_output_tokens": row["generation"]["output_tokens"],
            "generation_stop_reason": row["generation"]["stopping"]["reason"],
            "detector_version": shared.DETECTOR_VERSION,
            "prediction_variant": variant,
            "evaluation_unit": unit,
            "decision_field": "union",
            "dependency_identity": dependencies,
            "window_metadata": window_metadata,
            "threshold": shared.EMBEDDING_THRESHOLD,
            "predictions": component_rows,
            "unavailable_categories": {
                "crisis_referral": "not_applicable: absent from raw_output_robustness four-category baseline"
            },
        }
        shared.validate_prediction_record(prediction)
        predictions.append(prediction)
    if len(predictions) != 8640 or len({row["prediction_id"] for row in predictions}) != 8640:
        raise AssertionError("Complete prospective prediction export must contain 8,640 unique rows")
    shared.write_jsonl_compatible(output, predictions, "prediction_id")
    return {
        "status": "complete",
        "responses": 2880,
        "prediction_rows": len(predictions),
        "evaluation_units": list(shared.EVALUATION_UNITS),
        "dependency_identity": dependencies,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def gpu_inventory() -> dict[str, Any]:
    import torch

    rows = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        rows.append({
            "index": index,
            "name": props.name,
            "total_memory_gib": round(props.total_memory / (1024 ** 3), 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "capability": list(torch.cuda.get_device_capability(index)),
        })
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "devices": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-protocol")
    freeze.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "protocol_v1.json")
    freeze.add_argument("--local-files-only", action="store_true")
    review = sub.add_parser("create-human-review")
    review.add_argument("--protocol", type=Path, required=True)
    review.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "human_prompt_review_v1.json")
    validate_review = sub.add_parser("validate-human-review")
    validate_review.add_argument("--protocol", type=Path, required=True)
    validate_review.add_argument("--human-review", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--protocol", type=Path, required=True)
    plan.add_argument("--mode", choices=("development-256", "development-512", "evaluation", "sensitivity"), required=True)
    plan.add_argument("--family-ids", nargs="*")
    plan.add_argument("--human-review", type=Path)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    plan.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    plan.add_argument("--torch-threads", type=int)
    run = sub.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--mode", choices=("development-256", "development-512", "evaluation", "sensitivity"), required=True)
    run.add_argument("--family-ids", nargs="*")
    run.add_argument("--human-review", type=Path)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--torch-threads", type=int)
    run.add_argument("--execute", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--protocol", type=Path, required=True)
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--allow-partial", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("--protocol", type=Path, required=True)
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--allow-partial", action="store_true")
    predict = sub.add_parser("predict")
    predict.add_argument("--protocol", type=Path, required=True)
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    sub.add_parser("gpu-inventory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-protocol":
        protocol = freeze_protocol(args.output, local_files_only=args.local_files_only)
        result = {"output": str(args.output), "protocol_id": protocol["protocol_id"], "primary_expected_responses": 2880}
    elif args.command == "create-human-review":
        protocol = load_protocol(args.protocol)
        review = create_human_review_manifest(protocol, args.output)
        result = {"output": str(args.output), "status": review["status"], "families": len(review["families"])}
    elif args.command == "validate-human-review":
        protocol = load_protocol(args.protocol)
        review = validate_human_review(protocol, args.human_review)
        result = {"status": "valid_complete_human_review", "reviewer_id": review["reviewer_id"]}
    elif args.command in {"plan", "run"}:
        execution = execution_profile(args.device, args.torch_threads)
        protocol, grid, run_dir = prepare_run(
            args.protocol, args.mode, args.output_root, args.family_ids, args.human_review, execution,
        )
        if args.command == "plan" or not args.execute:
            result = {
                "status": "dry_run", "execute_required": True, "mode": args.mode,
                "classification": json.loads((run_dir / "run_config.json").read_text())["classification"],
                "responses": len(grid), "run_dir": str(run_dir), "protocol_id": protocol["protocol_id"],
            }
        else:
            result = run_generation(
                args.protocol, args.mode, args.output_root, args.family_ids, args.human_review,
                args.device, args.local_files_only, args.torch_threads,
            )
    elif args.command == "verify":
        result = verify_run(args.protocol, args.run_dir, require_complete=not args.allow_partial)
    elif args.command == "export":
        result = export_run(args.protocol, args.run_dir, args.output, require_complete=not args.allow_partial)
    elif args.command == "predict":
        result = predict_complete_evaluation(args.protocol, args.run_dir, args.output)
    elif args.command == "gpu-inventory":
        result = gpu_inventory()
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
