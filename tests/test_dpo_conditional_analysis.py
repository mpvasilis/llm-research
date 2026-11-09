from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from experiments import dpo_conditional_analysis as dca


def test_marker_only_fit_is_exact_log_discordant_ratio() -> None:
    chosen_only, rejected_only, ties = 420, 180, 75
    x = np.r_[np.ones(chosen_only), -np.ones(rejected_only), np.zeros(ties)][:, None]
    clusters = np.array([f"p{i}" for i in range(len(x))])
    fit = dca.fit_conditional(x, clusters)
    assert fit["optimizer_success"]
    assert fit["coefficients"][0] == pytest.approx(math.log(chosen_only / rejected_only), abs=2e-6)


def test_paired_reversal_and_row_order_equivariance() -> None:
    rng = np.random.default_rng(9206)
    x = rng.normal(size=(4000, 3))
    x[:, 0] = rng.choice([-1.0, 0.0, 1.0], size=len(x), p=[0.25, 0.45, 0.30])
    clusters = np.array([f"cluster-{i // 2}" for i in range(len(x))])
    forward = dca.fit_conditional(x, clusters)
    reversed_fit = dca.fit_conditional(-x, clusters)
    order = rng.permutation(len(x))
    shuffled = dca.fit_conditional(x[order], clusters[order])
    assert np.asarray(reversed_fit["coefficients"]) == pytest.approx(-np.asarray(forward["coefficients"]), abs=2e-6)
    assert np.asarray(reversed_fit["cluster_standard_errors"]) == pytest.approx(
        np.asarray(forward["cluster_standard_errors"]), abs=2e-6
    )
    assert np.asarray(shuffled["coefficients"]) == pytest.approx(np.asarray(forward["coefficients"]), abs=2e-6)
    assert np.asarray(shuffled["cluster_standard_errors"]) == pytest.approx(
        np.asarray(forward["cluster_standard_errors"]), abs=2e-6
    )


def test_length_adjustment_recovers_known_zero_marker_effect() -> None:
    """Choice depends on log length; marker does not after conditioning on it."""
    rng = np.random.default_rng(1509)
    n = 50000
    log_a = rng.normal(4.5, 0.75, n)
    log_b = rng.normal(4.5, 0.75, n)
    marker_a = rng.binomial(1, expit(-5.0 + 1.15 * log_a))
    marker_b = rng.binomial(1, expit(-5.0 + 1.15 * log_b))
    choose_a = rng.random(n) < expit(1.4 * (log_a - log_b))
    frame = pd.DataFrame(
        {
            "g_chosen": np.where(choose_a, marker_a, marker_b),
            "g_rejected": np.where(choose_a, marker_b, marker_a),
            "chosen_words": np.maximum(1, np.rint(np.expm1(np.where(choose_a, log_a, log_b)))).astype(int),
            "rejected_words": np.maximum(1, np.rint(np.expm1(np.where(choose_a, log_b, log_a)))).astype(int),
            "prompt_cluster_hash": [f"p{i}" for i in range(n)],
        }
    )
    pooled = np.log1p(np.r_[frame["chosen_words"], frame["rejected_words"]])
    knots = np.quantile(pooled, dca.KNOT_PROBABILITIES)
    unadjusted = dca._fit_frame(frame, "g", knots, include_length=False)
    adjusted = dca._fit_frame(frame, "g", knots, include_length=True)
    assert unadjusted["coefficients"][0] > 0.25
    assert abs(adjusted["coefficients"][0]) < 0.10
    assert abs(adjusted["coefficients"][0]) < abs(unadjusted["coefficients"][0]) / 3


def test_legacy_literal_separator_and_group_parsing(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "chosen": [
                    {"role": "user", "content": "prompt one"},
                    {"role": "assistant", "content": "First,"},
                    {"role": "assistant", "content": "consult a clinician"},
                ],
                "rejected": [
                    {"role": "user", "content": "prompt one"},
                    {"role": "assistant", "content": "plain answer"},
                ],
                "chosen_model": "a",
                "rejected_model": "b",
                "id": "pair-1",
                "source": "fixture",
            },
            {
                "chosen": [
                    {"role": "user", "content": "prompt two"},
                    {"role": "assistant", "content": "Talk to your doctor."},
                ],
                "rejected": [
                    {"role": "user", "content": "prompt two"},
                    {"role": "assistant", "content": "See a doctor."},
                ],
                "chosen_model": "a",
                "rejected_model": "b",
                "id": "pair-2",
                "source": "fixture",
            },
        ]
    )
    raw = tmp_path / "fixture.parquet"
    out = tmp_path / "features.parquet"
    rows.to_parquet(raw, index=False)
    connection = duckdb.connect()
    query = dca._feature_query("fixture")
    connection.execute(
        f"COPY ({query}) TO '{out.as_posix()}' (FORMAT PARQUET)", [[raw.as_posix()]]
    )
    features = pd.read_parquet(out)
    first = features.iloc[0]
    # "First,\\nconsult" is one non-whitespace token under the legacy SQL,
    # proving that the separator is the two literal characters rather than LF.
    assert first["chosen_words"] == 3
    assert first["chosen_multiple_assistant"]
    assert first["professional_referral_original_chosen"] == 1
    assert first["professional_referral_high_precision_chosen"] == 0
    assert first["structuring_language_chosen"] == 1
    second = features.iloc[1]
    # The broad historical list has "talk to a", not "talk to your".
    assert second["professional_referral_original_chosen"] == 0
    assert second["professional_referral_high_precision_chosen"] == 1
    assert second["professional_referral_high_precision_rejected"] == 1
    assert features["analysis_valid"].all()


def test_real_preparation_aggregation_parity_when_artifact_exists() -> None:
    manifest_path = dca.DEFAULT_OUTPUT / "feature_manifest.json"
    if not manifest_path.exists():
        pytest.skip("real prepared feature artifact is not present")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not all(model in manifest.get("datasets", {}) for model in dca.DATASETS):
        pytest.skip("real preparation is still in progress")
    for model in dca.DATASETS:
        audit = manifest["datasets"][model].get("audit")
        if audit is None:
            pytest.skip("real preparation is still in progress")
        assert audit["all_parity_checks_pass"], model
        assert all(audit["parity_checks"].values()), model


def test_rcs_basis_is_linear_beyond_boundary_knots() -> None:
    knots = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.array([5.5, 6.0, 6.5])
    basis = dca.rcs_basis(x, knots)
    second_differences = basis[2] - 2 * basis[1] + basis[0]
    assert second_differences == pytest.approx(np.zeros(basis.shape[1]), abs=1e-12)
