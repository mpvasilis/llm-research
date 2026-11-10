from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from transformers import AutoConfig, AutoTokenizer, __version__ as transformers_version

from experiments import extended_revision_study as core


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    protocol_path = ROOT / "output/revision/extended_study/full_study/protocol_v1.json"
    audit_path = ROOT / "output/revision/verification/checkpoint_context_capacity_audit_20260915.json"
    output_path = ROOT / "output/revision/environment_validation/tokenizer_config_compatibility.json"
    protocol = core.load_protocol(protocol_path)
    prior = json.loads(audit_path.read_text(encoding="utf-8"))
    prior_by_alias = {row["checkpoint_alias"]: row for row in prior["checkpoints"]}
    _, prompts, corpus_hash = core.load_corpus()
    if len(prompts) != 200:
        raise AssertionError(f"Expected 200 expanded prompts, got {len(prompts)}")
    if corpus_hash != protocol["prompt_corpus"]["sha256"]:
        raise AssertionError("Prompt corpus hash differs from frozen successor protocol")

    rows = []
    reference_counts: dict[str, int] | None = None
    for alias, checkpoint in protocol["checkpoints"].items():
        previous = prior_by_alias[alias]
        snapshot = Path(previous["tokenizer_snapshot_path"])
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
        if snapshot.name != checkpoint["tokenizer_revision"]:
            raise AssertionError(f"Snapshot revision leaf mismatch for {alias}")

        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        template = tokenizer.chat_template
        if not template:
            raise AssertionError(f"Missing official chat template for {alias}")
        rendered_probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "SERIALIZATION_PROBE"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        template_hash = sha256_bytes(template.encode("utf-8"))
        rendered_hash = sha256_bytes(rendered_probe.encode("utf-8"))
        serialization_id = core.stable_id(
            "serialization",
            {"template_sha256": template_hash, "rendered_probe_sha256": rendered_hash},
            32,
        )
        expected_identity = {
            "chat_template_sha256": checkpoint["chat_template_sha256"],
            "rendered_probe_sha256": checkpoint["rendered_probe_sha256"],
            "serialization_id": checkpoint["serialization_id"],
        }
        observed_identity = {
            "chat_template_sha256": template_hash,
            "rendered_probe_sha256": rendered_hash,
            "serialization_id": serialization_id,
        }
        if observed_identity != expected_identity:
            raise AssertionError(f"Official serialization identity mismatch for {alias}")

        counts: dict[str, int] = {}
        split_counts: dict[str, list[tuple[str, int]]] = {"development": [], "evaluation": []}
        for prompt in prompts:
            serialized = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            count = len(tokenizer(serialized, add_special_tokens=False)["input_ids"])
            counts[prompt["prompt_id"]] = count
            split_counts[prompt["split"]].append((prompt["prompt_id"], count))
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise AssertionError(f"Per-prompt token counts differ for {alias}")

        split_summary = {}
        for split, values in split_counts.items():
            minimum = min(count for _, count in values)
            maximum = max(count for _, count in values)
            expected = prior["tokenization_result_group"][split]
            summary = {
                "prompt_count": len(values),
                "min_input_tokens": minimum,
                "max_input_tokens": maximum,
                "min_prompt_ids": sorted(pid for pid, count in values if count == minimum),
                "max_prompt_ids": sorted(pid for pid, count in values if count == maximum),
            }
            for key in ("prompt_count", "min_input_tokens", "max_input_tokens", "min_prompt_ids", "max_prompt_ids"):
                if summary[key] != expected[key]:
                    raise AssertionError(f"{alias} {split} {key} differs from prior context audit")
            split_summary[split] = summary

        rows.append(
            {
                "checkpoint_alias": alias,
                "model_id": checkpoint["model_id"],
                "model_revision": checkpoint["model_revision"],
                "tokenizer_revision": checkpoint["tokenizer_revision"],
                "snapshot_path": str(snapshot),
                "snapshot_revision_leaf_matches_pin": True,
                "config_class": type(config).__name__,
                "tokenizer_class": type(tokenizer).__name__,
                "max_position_embeddings": config.max_position_embeddings,
                "expected_max_position_embeddings": previous["config"]["max_position_embeddings"],
                "serialization_identity": observed_identity,
                "serialization_matches_frozen_protocol": True,
                "tokenization": split_summary,
                "per_prompt_counts_match_all_six": True,
            }
        )

    if not all(row["max_position_embeddings"] == 4096 for row in rows):
        raise AssertionError("Unexpected model context capacity")
    all_counts = list((reference_counts or {}).values())
    report = {
        "verification_schema_version": "clean-tokenizer-config-compatibility-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "local_windows_config_and_tokenizer_only_no_model_instantiation",
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "transformers": transformers_version,
            "offline_environment": {
                "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
                "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
                "TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"],
            },
        },
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "protocol_id": protocol["protocol_id"],
            "primary_cap": protocol["primary_cap"],
            "sensitivity_cap": protocol["sensitivity_cap"],
        },
        "prompt_corpus": {
            "sha256": corpus_hash,
            "expanded_prompt_count": len(prompts),
            "min_input_tokens": min(all_counts),
            "max_input_tokens": max(all_counts),
            "all_six_per_prompt_counts_identical": True,
        },
        "checkpoints": rows,
        "operations": {
            "auto_config_loaded": True,
            "auto_tokenizer_loaded": True,
            "local_files_only": True,
            "network_or_download_used": False,
            "auto_model_called": False,
            "model_instantiated": False,
            "model_weights_loaded": False,
            "generation_performed": False,
            "gpu_claimed_or_tested": False,
        },
        "comparison": {
            "prior_context_audit_path": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
            "prior_transformers_version": prior["runtime"]["transformers"],
            "official_template_and_serialization_match_frozen_protocol": True,
            "all_200_prompt_token_counts_match_across_all_six": True,
            "split_extrema_and_prompt_ids_match_prior_audit": True,
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checkpoints": len(rows), "prompts": len(prompts), "output": str(output_path)}))


if __name__ == "__main__":
    main()
