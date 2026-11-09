from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import detector_spec as detector  # noqa: E402
import revision_study as study  # noqa: E402


def response_record() -> dict:
    record = {
        "study_version": study.STUDY_VERSION,
        "runner_version": study.RUNNER_VERSION,
        "response_id": "pending",
        "prompt_id": "prompt_test",
        "prompt_corpus_sha256": "a" * 64,
        "prompt_text_sha256": __import__("hashlib").sha256("Why does this happen?".encode()).hexdigest(),
        "scenario_family_id": "dev_everyday_test",
        "split": "development",
        "condition": {"domain": "everyday", "emotional_framing": False, "request_type": "information"},
        "model_id": "example/model",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "model_hash": "hf-content-manifest-sha256:" + "c" * 64,
        "seed": 0,
        "rng_seed": 12345,
        "rng_scheme": "test deterministic scheme",
        "serialization_id": "official-chat-template-sha256:" + "d" * 64,
        "generation": {
            "prompt": "Why does this happen?",
            "serialized_prompt": "<user>Why does this happen?</user><assistant>",
            "response": "I can understand the process. Here are a few details.",
            "input_tokens": 8,
            "output_tokens": 12,
            "max_new_tokens": 32,
            "stopping": {"reason": "eos_token", "eos_reached": True, "cap_reached": False},
            "config": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 32},
            "timing": {"generation_seconds": 0.5, "generated_tokens_per_second": 24.0},
        },
    }
    record["response_identity"] = study._response_identity_from_record(record)
    record["response_id"] = study.stable_id("response", record["response_identity"], 32)
    return record


def test_prompt_design_has_fixed_families_and_balanced_2x2() -> None:
    corpus = study.load_prompt_corpus()
    report = study.validate_prompt_corpus(corpus)
    rows = study.expand_prompts(corpus)
    assert report["families"] == 50
    assert report["prompts"] == 200
    assert len([family for family in corpus["families"] if family["split"] == "evaluation"]) == 40
    assert len([family for family in corpus["families"] if family["split"] == "development"]) == 10
    assert Counter((family["split"], family["domain"]) for family in corpus["families"]) == Counter({
        ("evaluation", "health"): 20,
        ("evaluation", "everyday"): 20,
        ("development", "health"): 5,
        ("development", "everyday"): 5,
    })
    for family in corpus["families"]:
        family_rows = [row for row in rows if row["scenario_family_id"] == family["scenario_family_id"]]
        assert {family["split"]} == {row["split"] for row in family_rows}
        assert {
            (row["condition"]["emotional_framing"], row["condition"]["request_type"])
            for row in family_rows
        } == {(False, "information"), (True, "information"), (False, "advice"), (True, "advice")}
        for request_type in ("information", "advice"):
            neutral = next(row["prompt"] for row in family_rows if row["condition"] == {
                "domain": family["domain"], "emotional_framing": False, "request_type": request_type,
            })
            emotional = next(row["prompt"] for row in family_rows if row["condition"] == {
                "domain": family["domain"], "emotional_framing": True, "request_type": request_type,
            })
            expected = f"{family['context']} {family['emotional_clause']} {family['requests'][request_type]}"
            assert emotional == expected
            assert neutral == f"{family['context']} {family['requests'][request_type]}"
            assert emotional.replace(f" {family['emotional_clause']}", "", 1) == neutral


def test_prompt_and_response_ids_are_stable_and_unique() -> None:
    corpus = study.load_prompt_corpus()
    first = study.expand_prompts(corpus)
    second = study.expand_prompts(corpus)
    assert [row["prompt_id"] for row in first] == [row["prompt_id"] for row in second]
    assert len({row["prompt_id"] for row in first}) == 200
    identity = {"prompt_id": first[0]["prompt_id"], "seed": 0}
    assert study.stable_id("response", identity) == study.stable_id("response", identity)


def test_no_known_construction_prompt_overlap() -> None:
    texts = {row["prompt"] for row in study.expand_prompts(study.load_prompt_corpus())}
    assert texts.isdisjoint(study.KNOWN_CONSTRUCTION_PROMPTS)


def test_response_missing_metadata_fails() -> None:
    valid = response_record()
    study.validate_response_record(valid)
    for missing in ("model_revision", "tokenizer_revision", "serialization_id", "model_hash"):
        broken = json.loads(json.dumps(valid))
        del broken[missing]
        with pytest.raises(ValueError, match="missing required metadata"):
            study.validate_response_record(broken)
    broken_generation = json.loads(json.dumps(valid))
    del broken_generation["generation"]["stopping"]
    with pytest.raises(ValueError, match="missing required metadata"):
        study.validate_response_record(broken_generation)


def test_incompatible_cached_run_is_not_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "responses.jsonl"
    original = response_record()
    study.write_jsonl_compatible(target, [original], "response_id")
    before = target.read_bytes()
    incompatible = response_record()
    incompatible["generation"]["response"] = "different output"
    with pytest.raises(FileExistsError, match="incompatible cached record"):
        study.write_jsonl_compatible(target, [incompatible], "response_id")
    assert target.read_bytes() == before


def test_duplicate_cached_ids_are_rejected_before_resume(tmp_path: Path) -> None:
    target = tmp_path / "responses.jsonl"
    row = response_record()
    line = json.dumps(row) + "\n"
    target.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate response_id"):
        study.load_unique_jsonl(target, "response_id")


def test_historical_exact_matching_parity_all_700_outputs() -> None:
    report = study.audit_historical()
    assert report["status"] == "pass"
    assert report["responses"] == 700
    assert report["exact_match_parity"] == {
        "checked_responses": 700,
        "mismatch_count": 0,
        "mismatch_keys": [],
    }


def test_detector_manifest_is_complete_and_records_conflicts() -> None:
    detector.assert_primary_source_exact()
    manifest = detector.manifest()
    assert manifest["categories_in_order"] == list(detector.LEXICON)
    assert manifest["embedding"]["threshold"] == 0.82
    assert manifest["embedding"]["revision_status"] == "unrecorded/unpinned in historical source"
    assert set(manifest["output_windows"]) >= {"full_response", "prefix_64_tokens", "prefix_128_tokens"}
    assert len(manifest["provenance_conflicts"]) == 3
    assert all(len(row["sha256_at_manifest_freeze"]) == 64 for row in manifest["source_files"])
    primary = next(row for row in manifest["source_files"] if row["path"] == detector.PRIMARY_SOURCE)
    assert detector.sha256_file(ROOT / detector.PRIMARY_SOURCE) == primary["sha256_at_manifest_freeze"]


def test_exact_prediction_schema_declares_unavailable_crisis_variant() -> None:
    rows = study.predict_records([response_record()], ["full_response"], include_embedding=False)
    assert len(rows) == 1
    prediction = rows[0]
    study.validate_prediction_record(prediction)
    assert prediction["decision_field"] == "exact"
    assert prediction["prediction_variant"] == "raw_output_robustness_four_category_exact_only_provenance_v2"
    assert set(prediction["predictions"]) == set(detector.CATEGORIES)
    assert prediction["unavailable_categories"]["crisis_referral"].startswith("not_applicable")
    assert prediction["predictions"]["empathy_opener"]["exact"] is True


def test_prospective_run_spec_has_2880_stable_join_rows(tmp_path: Path) -> None:
    target = tmp_path / "run_spec.json"
    spec = study.export_run_spec(target)
    assert spec["expected_responses"] == 2880
    assert spec["evaluation_prompts"] == 160
    assert spec["scenario_families"] == 40
    assert len({row["planned_response_key"] for row in spec["planned_rows"]}) == 2880
    assert "never row order" in spec["join_key"]


def test_generation_is_dry_by_default_and_pilot_is_bounded() -> None:
    plan = study.generation_plan(study.CHECKPOINTS["1b_instruct"], True, 2, 32, 0)
    assert plan["mode"] == "local_development_smoke"
    assert plan["prompts"] == 2
    assert plan["local_files_only"] is True
    for prompt_count, token_count in ((0, 32), (-1, 32), (3, 32), (2, 0), (2, 33)):
        with pytest.raises(ValueError, match="Local pilot requires"):
            study.generation_plan(
                study.CHECKPOINTS["1b_instruct"], True, prompt_count, token_count, 0
            )


def test_rng_seed_material_is_prompt_stable_across_checkpoint_order() -> None:
    prompt = study.expand_prompts(study.load_prompt_corpus())[0]
    material = {"prompt_id": prompt["prompt_id"], "logical_seed": 2}
    first = int.from_bytes(__import__("hashlib").sha256(study.canonical_json(material).encode()).digest()[:8], "big") % (2 ** 63)
    second = int.from_bytes(__import__("hashlib").sha256(study.canonical_json(material).encode()).digest()[:8], "big") % (2 ** 63)
    assert first == second
    assert "model_id" not in material


def test_completed_local_smoke_resumes_without_regeneration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    import transformers

    calls = []

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return f"USER:{messages[0]['content']}\nASSISTANT:"

        def __call__(self, text, return_tensors, add_special_tokens):
            return {"input_ids": torch.tensor([[10, 11]])}

        def decode(self, tokens, skip_special_tokens):
            return "mock response"

    class FakeModel:
        generation_config = type("GenerationConfig", (), {"eos_token_id": 2})()

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls.append(1)
            return torch.tensor([[10, 11, 12, 2]])

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs["revision"] == "model-revision"
            return FakeModel()

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(study, "_model_metadata", lambda model_id: {
        "config": object(),
        "tokenizer": FakeTokenizer(),
        "model_revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "serialization_id": "serialization-hash",
        "model_hash": "model-content-hash",
        "weight_blob_ids": ["weight-blob"],
    })
    args = (tmp_path, "example/model", True, 2, 4, 0)
    first = study.execute_generation(*args)
    second = study.execute_generation(*args)
    assert first["new_responses"] == 2
    assert second["new_responses"] == 0
    assert len(calls) == 2
    response_path = Path(first["run_dir"]) / "responses.jsonl"
    rows = list(study.iter_jsonl(response_path))
    assert len(rows) == 2
    assert all(row["prompt_id"] and row["response_identity"] for row in rows)
