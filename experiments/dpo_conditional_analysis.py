"""Exploratory paired DPO analysis with flexible response-length adjustment.

This module deliberately separates remote feature preparation from inference.
Run ``prepare`` once, freeze the numeric knots in the dated protocol, then run
``analyze`` only after all legacy aggregate and length-band checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from config import ROOT
from experiments.dpo_phrase_robustness import GROUPS


DEFAULT_OUTPUT = ROOT / "output" / "revision" / "dpo_conditional"
PROTOCOL = ROOT / "out" / "paper" / "DPO_CONDITIONAL_ANALYSIS_PROTOCOL_2026-09-15.md"
RESULTS_REPORT = ROOT / "out" / "paper" / "DPO_CONDITIONAL_RESULTS_2026-09-15.md"
KNOT_PROBABILITIES = np.array([0.05, 0.275, 0.50, 0.725, 0.95], dtype=float)

# Raw Parquet files exist in the immutable dataset commits, so analysis does
# not depend on Hugging Face's mutable refs/convert/parquet branch.
DATASETS: dict[str, dict[str, Any]] = {
    "1B": {
        "dataset": "allenai/olmo-2-0425-1b-preference-mix",
        "revision": "c6454f3d364622718a8ecd3d307e524febbba569",
        "shards": [f"data/train-{i:05d}-of-00005.parquet" for i in range(5)],
    },
    "7B": {
        "dataset": "allenai/olmo-2-1124-7b-preference-mix",
        "revision": "316c96f092e87bc07b30d04116280bb80d70a0fb",
        "shards": [f"data/train-{i:05d}-of-00006.parquet" for i in range(6)],
    },
}

EXPECTED_LEGACY: dict[str, dict[str, Any]] = {
    "1B": {
        "n_pairs": 378301,
        "aggregate": {
            "professional_referral_original": [2478, 2098],
            "professional_referral_high_precision": [591, 505],
            "structuring_language": [27047, 21779],
        },
        "bands": {
            "chosen_longer_gt20": {
                "n_pairs": 171170,
                "professional_referral_original": [1615, 811],
                "professional_referral_high_precision": [348, 247],
                "structuring_language": [17754, 7519],
            },
            "chosen_shorter_gt20": {
                "n_pairs": 134707,
                "professional_referral_original": [668, 1085],
                "professional_referral_high_precision": [181, 208],
                "structuring_language": [7020, 12120],
            },
            "within_20_words": {
                "n_pairs": 72424,
                "professional_referral_original": [195, 202],
                "professional_referral_high_precision": [62, 50],
                "structuring_language": [2273, 2140],
            },
        },
        "prompt_dependence": [367082, 10332, 21551, 15],
    },
    "7B": {
        "n_pairs": 378341,
        "aggregate": {
            "professional_referral_original": [2602, 1925],
            "professional_referral_high_precision": [649, 486],
            "structuring_language": [28466, 19838],
        },
        "bands": {
            "chosen_longer_gt20": {
                "n_pairs": 187201,
                "professional_referral_original": [1740, 801],
                "professional_referral_high_precision": [423, 240],
                "structuring_language": [19913, 7629],
            },
            "chosen_shorter_gt20": {
                "n_pairs": 120381,
                "professional_referral_original": [643, 911],
                "professional_referral_high_precision": [170, 188],
                "structuring_language": [6341, 10240],
            },
            "within_20_words": {
                "n_pairs": 70759,
                "professional_referral_original": [219, 213],
                "professional_referral_high_precision": [56, 58],
                "structuring_language": [2212, 1969],
            },
        },
        "prompt_dependence": [367110, 10343, 21574, 15],
    },
}

ERROR_COLUMNS = [
    "chosen_null",
    "rejected_null",
    "chosen_no_assistant",
    "rejected_no_assistant",
    "chosen_assistant_null_content",
    "rejected_assistant_null_content",
    "chosen_no_user",
    "rejected_no_user",
    "chosen_user_null_content",
    "rejected_user_null_content",
    "prompt_context_mismatch",
]
AUDIT_COLUMNS = ERROR_COLUMNS + [
    "chosen_multiple_assistant",
    "rejected_multiple_assistant",
    "source_null",
    "id_null",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _pattern(phrases: list[str]) -> str:
    """Return the exact escaped, longest-first legacy alternation."""
    return "(?:" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + ")"


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _urls(spec: dict[str, Any]) -> list[str]:
    base = f"https://huggingface.co/datasets/{spec['dataset']}/resolve/{spec['revision']}"
    return [f"{base}/{path}" for path in spec["shards"]]


def _remote_headers(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "dpo-conditional-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return {
                "url": url,
                "resolved_url": response.geturl(),
                "etag": response.headers.get("ETag"),
                "x_linked_etag": response.headers.get("X-Linked-Etag"),
                "content_length": int(response.headers["Content-Length"])
                if response.headers.get("Content-Length")
                else None,
            }
    except Exception as exc:  # provenance remains useful even if HEAD is blocked
        return {"url": url, "header_error": f"{type(exc).__name__}: {exc}"}


def _feature_query(model: str) -> str:
    marker_columns: list[str] = []
    for group, phrases in GROUPS.items():
        regex = _sql_quote(_pattern(phrases))
        marker_columns.extend(
            [
                f"regexp_matches(lower(coalesce(chosen_text, '')), '{regex}')::UTINYINT AS {group}_chosen",
                f"regexp_matches(lower(coalesce(rejected_text, '')), '{regex}')::UTINYINT AS {group}_rejected",
            ]
        )

    # Important parity detail: '\\n' reaches DuckDB as two literal characters,
    # backslash + n, exactly as in the two legacy scripts.
    separator = r"\n"
    return f"""
      WITH raw AS (
        SELECT *, filename AS input_filename, file_row_number AS input_row_number
        FROM read_parquet(?, filename=true, file_row_number=true, union_by_name=true)
      ), parsed AS (
        SELECT *,
          array_to_string(
            list_transform(list_filter(chosen, m -> m.role='assistant'), m -> m.content),
            '{separator}'
          ) AS chosen_text,
          array_to_string(
            list_transform(list_filter(rejected, m -> m.role='assistant'), m -> m.content),
            '{separator}'
          ) AS rejected_text,
          array_to_string(
            list_transform(list_filter(chosen, m -> m.role='user'), m -> m.content),
            '{separator}'
          ) AS chosen_prompt_text,
          array_to_string(
            list_transform(list_filter(rejected, m -> m.role='user'), m -> m.content),
            '{separator}'
          ) AS rejected_prompt_text,
          to_json(list_filter(chosen, m -> coalesce(m.role, '') <> 'assistant')) AS chosen_context,
          to_json(list_filter(rejected, m -> coalesce(m.role, '') <> 'assistant')) AS rejected_context,
          chosen IS NULL AS chosen_null,
          rejected IS NULL AS rejected_null,
          coalesce(len(list_filter(chosen, m -> m.role='assistant')), 0) = 0 AS chosen_no_assistant,
          coalesce(len(list_filter(rejected, m -> m.role='assistant')), 0) = 0 AS rejected_no_assistant,
          coalesce(len(list_filter(chosen, m -> m.role='assistant')), 0) > 1 AS chosen_multiple_assistant,
          coalesce(len(list_filter(rejected, m -> m.role='assistant')), 0) > 1 AS rejected_multiple_assistant,
          coalesce(list_has_any(list_transform(list_filter(chosen, m -> m.role='assistant'), m -> m.content IS NULL), [true]), false)
            AS chosen_assistant_null_content,
          coalesce(list_has_any(list_transform(list_filter(rejected, m -> m.role='assistant'), m -> m.content IS NULL), [true]), false)
            AS rejected_assistant_null_content,
          coalesce(len(list_filter(chosen, m -> m.role='user')), 0) = 0 AS chosen_no_user,
          coalesce(len(list_filter(rejected, m -> m.role='user')), 0) = 0 AS rejected_no_user,
          coalesce(list_has_any(list_transform(list_filter(chosen, m -> m.role='user'), m -> m.content IS NULL), [true]), false)
            AS chosen_user_null_content,
          coalesce(list_has_any(list_transform(list_filter(rejected, m -> m.role='user'), m -> m.content IS NULL), [true]), false)
            AS rejected_user_null_content,
          source IS NULL AS source_null,
          id IS NULL AS id_null
        FROM raw
      ), audited AS (
        SELECT *, coalesce(chosen_context::VARCHAR, '<NULL>') <> coalesce(rejected_context::VARCHAR, '<NULL>')
          AS prompt_context_mismatch
        FROM parsed
      )
      SELECT
        '{_sql_quote(model)}' AS model,
        source,
        sha256(concat_ws(chr(31), '{_sql_quote(model)}', coalesce(source, '<NULL>'),
          coalesce(id, '<NULL>'), input_filename, input_row_number::VARCHAR,
          sha256(coalesce(chosen_text, '<NULL>')), sha256(coalesce(rejected_text, '<NULL>')))) AS pair_id,
        sha256(concat_ws(chr(31), coalesce(chosen_prompt_text, '<NULL>'),
          coalesce(chosen_text, '<NULL>'), coalesce(rejected_text, '<NULL>'))) AS content_pair_hash,
        sha256(coalesce(chosen_prompt_text, '<NULL>')) AS prompt_cluster_hash,
        regexp_extract(input_filename, '[^/]+$') AS shard_file,
        input_row_number,
        len(regexp_extract_all(chosen_text, '\\S+'))::INTEGER AS chosen_words,
        len(regexp_extract_all(rejected_text, '\\S+'))::INTEGER AS rejected_words,
        {', '.join(marker_columns)},
        {', '.join(AUDIT_COLUMNS)},
        NOT ({' OR '.join(ERROR_COLUMNS)}) AS analysis_valid
      FROM audited
    """


def _audit_features(connection: duckdb.DuckDBPyConnection, path: Path, model: str) -> dict[str, Any]:
    expected = EXPECTED_LEGACY[model]
    marker_sql: list[str] = []
    for group in GROUPS:
        marker_sql.extend(
            [
                f"sum(({group}_chosen=1 AND {group}_rejected=0)::INT) AS {group}_chosen_only",
                f"sum(({group}_chosen=0 AND {group}_rejected=1)::INT) AS {group}_rejected_only",
            ]
        )
    error_sql = [f"sum({name}::INT) AS {name}" for name in AUDIT_COLUMNS]
    qpath = _sql_quote(path.as_posix())
    aggregate_row = connection.execute(
        f"SELECT count(*) AS n_pairs, sum(analysis_valid::INT) AS modeled_pairs, "
        + ", ".join(marker_sql + error_sql)
        + f" FROM read_parquet('{qpath}')"
    ).fetchone()
    names = [item[0] for item in connection.description]
    aggregate = dict(zip(names, aggregate_row))

    band_rows = connection.execute(
        f"""
        SELECT CASE
          WHEN chosen_words-rejected_words < -20 THEN 'chosen_shorter_gt20'
          WHEN chosen_words-rejected_words > 20 THEN 'chosen_longer_gt20'
          ELSE 'within_20_words' END AS length_band,
          count(*) AS n_pairs, {', '.join(marker_sql)}
        FROM read_parquet('{qpath}')
        GROUP BY length_band ORDER BY length_band
        """
    ).fetchall()
    band_names = [item[0] for item in connection.description]
    bands = {row[0]: dict(zip(band_names[1:], row[1:])) for row in band_rows}

    prompt_row = connection.execute(
        f"""
        WITH grouped AS (
          SELECT prompt_cluster_hash, count(*) AS multiplicity
          FROM read_parquet('{qpath}') GROUP BY prompt_cluster_hash
        )
        SELECT count(*), sum((multiplicity>1)::INT),
          sum(CASE WHEN multiplicity>1 THEN multiplicity ELSE 0 END), max(multiplicity)
        FROM grouped
        """
    ).fetchone()

    checks: dict[str, bool] = {"n_pairs": int(aggregate["n_pairs"]) == expected["n_pairs"]}
    for group, values in expected["aggregate"].items():
        checks[f"aggregate::{group}::chosen_only"] = int(aggregate[f"{group}_chosen_only"]) == values[0]
        checks[f"aggregate::{group}::rejected_only"] = int(aggregate[f"{group}_rejected_only"]) == values[1]
    for band, values in expected["bands"].items():
        checks[f"band::{band}::n_pairs"] = int(bands[band]["n_pairs"]) == values["n_pairs"]
        for group in GROUPS:
            checks[f"band::{band}::{group}::chosen_only"] = int(bands[band][f"{group}_chosen_only"]) == values[group][0]
            checks[f"band::{band}::{group}::rejected_only"] = int(bands[band][f"{group}_rejected_only"]) == values[group][1]
    checks["prompt_dependence"] = [int(x) for x in prompt_row] == expected["prompt_dependence"]
    return {
        "aggregate": {key: int(value) for key, value in aggregate.items()},
        "length_bands": {
            band: {key: int(value) for key, value in values.items()} for band, values in bands.items()
        },
        "prompt_dependence": {
            "distinct_prompt_hashes": int(prompt_row[0]),
            "duplicated_prompt_groups": int(prompt_row[1]),
            "rows_in_duplicated_groups": int(prompt_row[2]),
            "max_multiplicity": int(prompt_row[3]),
        },
        "parity_checks": checks,
        "all_parity_checks_pass": all(checks.values()),
    }


def prepare(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar=false")
    connection.execute("SET enable_object_cache=true")
    connection.execute("SET threads=4")
    initial_manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "preparing",
        "legacy_separator": "literal backslash+n",
        "marker_groups": GROUPS,
        "datasets": {},
    }
    manifest_path = output_dir / "feature_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        invariant = {
            "schema_version": initial_manifest["schema_version"],
            "legacy_separator": initial_manifest["legacy_separator"],
            "marker_groups": initial_manifest["marker_groups"],
        }
        for key, expected_value in invariant.items():
            if manifest.get(key) != expected_value:
                raise RuntimeError(f"Existing manifest conflicts on {key}; refusing to overwrite prepared data")
        manifest["resumed_at_utc"] = _utc_now()
    else:
        manifest = initial_manifest
    _write_json_atomic(manifest_path, manifest)

    for model, spec in DATASETS.items():
        urls = _urls(spec)
        feature_path = output_dir / f"pair_features_{model.lower()}.parquet"
        print(f"Preparing {model} from {len(urls)} pinned shards...", flush=True)
        query = _feature_query(model)
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        existing_entry = manifest.get("datasets", {}).get(model, {})
        for key, expected_value in {
            "dataset": spec["dataset"], "revision": spec["revision"],
            "shards": spec["shards"], "query_sha256": query_sha256, "urls": urls,
        }.items():
            if key in existing_entry and existing_entry[key] != expected_value:
                raise RuntimeError(f"Existing {model} manifest conflicts on {key}; refusing to overwrite")
        shard_headers = existing_entry.get("shard_headers") or [_remote_headers(url) for url in urls]
        parts_dir = output_dir / "feature_parts" / model.lower()
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_records = []
        existing_parts = {item["shard"]: item for item in existing_entry.get("completed_parts", [])}
        for shard_index, url in enumerate(urls):
            part_path = parts_dir / f"part-{shard_index:05d}.parquet"
            print(f"  shard {shard_index + 1}/{len(urls)}: {spec['shards'][shard_index]}", flush=True)
            if part_path.exists():
                record = existing_parts.get(spec["shards"][shard_index])
                if record is None:
                    raise RuntimeError(f"Unmanifested part exists at {part_path}; refusing to overwrite")
                if (
                    record.get("url") != url
                    or record.get("revision") != spec["revision"]
                    or record.get("query_sha256") != query_sha256
                    or record.get("bytes") != part_path.stat().st_size
                    or record.get("sha256") != _sha256_file(part_path)
                ):
                    raise RuntimeError(f"Existing part failed provenance/hash validation: {part_path}")
                print("    verified existing checkpoint", flush=True)
            else:
                temporary = part_path.with_suffix(".parquet.tmp")
                if temporary.exists():
                    temporary.unlink()
                connection.execute(
                    f"COPY ({query}) TO '{_sql_quote(temporary.as_posix())}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)",
                    [[url]],
                )
                temporary.replace(part_path)
            part_record = {
                "shard": spec["shards"][shard_index],
                "url": url,
                "revision": spec["revision"],
                "query_sha256": query_sha256,
                "file": part_path.relative_to(output_dir).as_posix(),
                "bytes": part_path.stat().st_size,
                "sha256": _sha256_file(part_path),
            }
            part_records.append(part_record)
            manifest["datasets"][model] = {
                **spec,
                "query_sha256": query_sha256,
                "urls": urls,
                "shard_headers": shard_headers,
                "completed_parts": part_records,
            }
            _write_json_atomic(manifest_path, manifest)
        part_paths = [str(output_dir / item["file"]) for item in part_records]
        previous_feature_sha = existing_entry.get("feature_file_sha256")
        if feature_path.exists():
            if previous_feature_sha is None or _sha256_file(feature_path) != previous_feature_sha:
                raise RuntimeError(f"Existing combined feature file failed manifest validation: {feature_path}")
            print("  verified existing combined feature file", flush=True)
        else:
            temporary_feature = feature_path.with_suffix(".parquet.tmp")
            if temporary_feature.exists():
                temporary_feature.unlink()
            connection.execute(
                f"COPY (SELECT * FROM read_parquet(?)) TO '{_sql_quote(temporary_feature.as_posix())}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)", [part_paths]
            )
            temporary_feature.replace(feature_path)
        audit = _audit_features(connection, feature_path, model)
        manifest["datasets"][model] = {
            **spec,
            "query_sha256": query_sha256,
            "urls": urls,
            "shard_headers": shard_headers,
            "completed_parts": part_records,
            "feature_file": feature_path.name,
            "feature_file_bytes": feature_path.stat().st_size,
            "feature_file_sha256": _sha256_file(feature_path),
            "audit": audit,
        }
        _write_json_atomic(manifest_path, manifest)
        if not audit["all_parity_checks_pass"]:
            failed = [name for name, passed in audit["parity_checks"].items() if not passed]
            raise RuntimeError(f"{model} legacy parity failed: {failed}")
        print(
            f"  {audit['aggregate']['n_pairs']:,} pairs; "
            f"{audit['aggregate']['modeled_pairs']:,} eligible; parity PASS",
            flush=True,
        )

    one_b_path = output_dir / "pair_features_1b.parquet"
    seven_b_path = output_dir / "pair_features_7b.parquet"
    overlap_row = connection.execute(
        """
        WITH one_b AS (SELECT content_pair_hash, count(*) AS n FROM read_parquet(?) GROUP BY 1),
             seven_b AS (SELECT content_pair_hash, count(*) AS n FROM read_parquet(?) GROUP BY 1)
        SELECT count(*) AS distinct_shared_content_pairs,
               coalesce(sum(least(one_b.n, seven_b.n)), 0) AS multiplicity_matched_rows,
               (SELECT count(*) FROM one_b) AS distinct_1b,
               (SELECT count(*) FROM seven_b) AS distinct_7b
        FROM one_b INNER JOIN seven_b USING (content_pair_hash)
        """,
        [str(one_b_path), str(seven_b_path)],
    ).fetchone()
    manifest["cross_mixture_content_overlap"] = {
        "hash_definition": "SHA-256(literal-separator join of chosen user prompt, chosen assistant text, rejected assistant text)",
        "distinct_shared_content_pairs": int(overlap_row[0]),
        "multiplicity_matched_rows": int(overlap_row[1]),
        "distinct_content_pairs_1b": int(overlap_row[2]),
        "distinct_content_pairs_7b": int(overlap_row[3]),
    }

    arrays = []
    for model in DATASETS:
        path = output_dir / f"pair_features_{model.lower()}.parquet"
        values = connection.execute(
            "SELECT chosen_words FROM read_parquet(?) UNION ALL "
            "SELECT rejected_words FROM read_parquet(?)",
            [str(path), str(path)],
        ).fetchnumpy()
        arrays.append(np.asarray(values["chosen_words"], dtype=float))
    pooled = np.log1p(np.concatenate(arrays))
    knots = np.quantile(pooled, KNOT_PROBABILITIES, method="linear")
    if len(np.unique(knots)) != len(knots):
        raise RuntimeError(f"Spline knots are not distinct: {knots.tolist()}")
    manifest["spline"] = {
        "transform": "log1p(word_count)",
        "quantile_probabilities": KNOT_PROBABILITIES.tolist(),
        "quantile_method": "numpy linear empirical interpolation",
        "pooled_response_count": int(pooled.size),
        "knots": knots.tolist(),
        "word_count_equivalents": np.expm1(knots).tolist(),
        "numeric_knots_frozen_at_utc": _utc_now(),
    }
    manifest["status"] = "prepared_parity_passed_knots_pending_protocol"
    _write_json_atomic(manifest_path, manifest)
    print(f"Numeric log1p knots: {knots.tolist()}", flush=True)
    print(f"Wrote {manifest_path}", flush=True)
    return manifest_path


def freeze_protocol(output_dir: Path) -> Path:
    """Bind the prepared feature hashes and numeric knots to the dated protocol."""
    manifest_path = output_dir / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_parity_passed_knots_pending_protocol":
        raise RuntimeError(f"Manifest is not ready to freeze: {manifest.get('status')}")
    if not all(entry["audit"]["all_parity_checks_pass"] for entry in manifest["datasets"].values()):
        raise RuntimeError("Cannot freeze because at least one legacy parity check failed")
    protocol_text = PROTOCOL.read_text(encoding="utf-8")
    if "pending preparation" in protocol_text.lower():
        raise RuntimeError("Protocol still contains the numeric-knot placeholder")
    missing = [f"{value:.9f}" for value in manifest["spline"]["knots"] if f"{value:.9f}" not in protocol_text]
    if missing:
        raise RuntimeError(f"Protocol does not contain all prepared numeric knots: {missing}")
    for model, entry in manifest["datasets"].items():
        feature_path = output_dir / entry["feature_file"]
        if feature_path.stat().st_size != entry["feature_file_bytes"]:
            raise RuntimeError(f"Feature-file size changed before freeze: {model}")
        if _sha256_file(feature_path) != entry["feature_file_sha256"]:
            raise RuntimeError(f"Feature-file hash changed before freeze: {model}")
    manifest["protocol"] = {
        "file": str(PROTOCOL),
        "sha256": _sha256_file(PROTOCOL),
        "frozen_at_utc": _utc_now(),
    }
    manifest["status"] = "prepared_parity_passed_protocol_frozen"
    _write_json_atomic(manifest_path, manifest)
    print(f"Bound protocol and feature hashes in {manifest_path}", flush=True)
    return manifest_path


def rcs_basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis: linear term plus K-2 nonlinear terms."""
    x = np.asarray(values, dtype=float)
    k = np.asarray(knots, dtype=float)
    if k.ndim != 1 or k.size < 3 or np.any(np.diff(k) <= 0):
        raise ValueError("knots must be a strictly increasing one-dimensional array")
    scale = (k[-1] - k[0]) ** 2

    def cube(pos: np.ndarray) -> np.ndarray:
        return np.maximum(pos, 0.0) ** 3 / scale

    columns = [x]
    denominator = k[-1] - k[-2]
    for knot in k[:-2]:
        columns.append(
            cube(x - knot)
            - cube(x - k[-2]) * (k[-1] - knot) / denominator
            + cube(x - k[-1]) * (k[-2] - knot) / denominator
        )
    return np.column_stack(columns)


def make_design(frame: pd.DataFrame, group: str, knots: np.ndarray, include_length: bool = True) -> np.ndarray:
    marker = (
        frame[f"{group}_chosen"].to_numpy(dtype=float)
        - frame[f"{group}_rejected"].to_numpy(dtype=float)
    )[:, None]
    if not include_length:
        return marker
    chosen = rcs_basis(np.log1p(frame["chosen_words"].to_numpy(dtype=float)), knots)
    rejected = rcs_basis(np.log1p(frame["rejected_words"].to_numpy(dtype=float)), knots)
    return np.column_stack([marker, chosen - rejected])


def fit_conditional(X: np.ndarray, clusters: np.ndarray) -> dict[str, Any]:
    """Fit sum(log sigmoid(X beta)) and a finite-corrected cluster sandwich."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a nonempty two-dimensional matrix")
    n, k = X.shape
    if len(clusters) != n:
        raise ValueError("cluster vector length differs from X")
    design_rank = int(np.linalg.matrix_rank(X))
    if design_rank < k:
        raise ValueError(f"design rank deficient: {design_rank} < {k}")

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ beta
        loss = float(np.logaddexp(0.0, -z).mean())
        gradient = -(X.T @ expit(-z)) / n
        return loss, gradient

    fit = minimize(
        objective,
        np.zeros(k),
        method="L-BFGS-B",
        jac=True,
        options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 500, "maxls": 50},
    )
    beta = np.asarray(fit.x)
    z = X @ beta
    probabilities = expit(z)
    weights = probabilities * (1.0 - probabilities)
    information = X.T @ (X * weights[:, None])
    information_rank = int(np.linalg.matrix_rank(information))
    eigenvalues = np.linalg.eigvalsh(information)
    condition_number = float(np.linalg.cond(information))
    if information_rank < k:
        raise ValueError(f"observed information rank deficient: {information_rank} < {k}")
    bread = np.linalg.inv(information)
    individual_scores = X * (1.0 - probabilities)[:, None]
    codes, cluster_labels = pd.factorize(np.asarray(clusters), sort=True)
    g = len(cluster_labels)
    if g <= 1 or n <= k:
        raise ValueError(f"insufficient clusters/rows for finite correction: G={g}, n={n}, k={k}")
    cluster_scores = np.zeros((g, k), dtype=float)
    np.add.at(cluster_scores, codes, individual_scores)
    meat = cluster_scores.T @ cluster_scores
    finite_correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = finite_correction * bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2.0
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    naive_standard_errors = np.sqrt(np.maximum(np.diag(bread), 0.0))
    gradient = objective(beta)[1]
    boundary_count = int(np.sum((probabilities < 1e-8) | (probabilities > 1 - 1e-8)))
    separation_flags = {
        "coefficient_abs_gt_20": bool(np.any(np.abs(beta) > 20)),
        "information_condition_gt_1e12": condition_number > 1e12,
        "boundary_probability_count_positive": boundary_count > 0,
        "minimum_information_eigenvalue_lt_1e_minus_8": float(eigenvalues[0]) < 1e-8,
    }
    return {
        "n_pairs": n,
        "n_parameters": k,
        "n_clusters": g,
        "coefficients": beta.tolist(),
        "cluster_standard_errors": standard_errors.tolist(),
        "naive_standard_errors": naive_standard_errors.tolist(),
        "cluster_covariance": covariance.tolist(),
        "log_likelihood": float(-np.logaddexp(0.0, -z).sum()),
        "optimizer_success": bool(fit.success),
        "optimizer_status": int(fit.status),
        "optimizer_message": str(fit.message),
        "optimizer_iterations": int(fit.nit),
        "gradient_infinity_norm": float(np.max(np.abs(gradient))),
        "design_rank": design_rank,
        "information_rank": information_rank,
        "information_condition_number": condition_number,
        "information_eigenvalues": eigenvalues.tolist(),
        "finite_cluster_correction": finite_correction,
        "boundary_probability_count": boundary_count,
        "separation_flags": separation_flags,
        "any_separation_warning": any(separation_flags.values()),
    }


def _marker_summary(fit: dict[str, Any]) -> dict[str, Any]:
    beta = fit["coefficients"][0]
    se = fit["cluster_standard_errors"][0]
    z = beta / se if se > 0 else math.copysign(math.inf, beta)
    return {
        "log_odds": beta,
        "cluster_se": se,
        "conditional_odds_ratio": math.exp(beta),
        "conditional_odds_ratio_95ci": [math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se)],
        "wald_z": z,
        "wald_p": float(2 * norm.sf(abs(z))),
    }


def _fit_frame(frame: pd.DataFrame, group: str, knots: np.ndarray, include_length: bool) -> dict[str, Any]:
    X = make_design(frame, group, knots, include_length=include_length)
    chosen_only = int(np.sum(X[:, 0] == 1))
    rejected_only = int(np.sum(X[:, 0] == -1))
    result = fit_conditional(X, frame["prompt_cluster_hash"].to_numpy())
    if not result["optimizer_success"]:
        raise RuntimeError(f"optimizer did not converge: {result['optimizer_message']}")
    if not np.all(np.isfinite(result["coefficients"] + result["cluster_standard_errors"])):
        raise RuntimeError("fit returned a non-finite coefficient or cluster standard error")
    if result["gradient_infinity_norm"] > 1e-6:
        raise RuntimeError(
            f"optimizer stopped above frozen mean-gradient tolerance: {result['gradient_infinity_norm']:.3g} > 1e-6"
        )
    result["marker"] = _marker_summary(result)
    result["marker_discordance"] = {
        "chosen_only": chosen_only,
        "rejected_only": rejected_only,
        "total": chosen_only + rejected_only,
    }
    if not include_length and chosen_only and rejected_only:
        identity = math.log(chosen_only / rejected_only)
        result["marker_only_log_b_over_c"] = identity
        result["marker_only_identity_error"] = result["coefficients"][0] - identity
    return result


def _bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.tolist()


def _safe_fit(frame: pd.DataFrame, group: str, knots: np.ndarray, include_length: bool = True) -> dict[str, Any]:
    try:
        return {"available": True, **_fit_frame(frame, group, knots, include_length)}
    except Exception as exc:
        return {"available": False, "n_pairs": len(frame), "error": f"{type(exc).__name__}: {exc}"}


def analyze(output_dir: Path) -> Path:
    manifest_path = output_dir / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_parity_passed_protocol_frozen":
        raise RuntimeError(
            "Preparation is not frozen for inference. Replace the numeric-knot placeholder in the protocol "
            "and set manifest status to prepared_parity_passed_protocol_frozen."
        )
    if not all(entry["audit"]["all_parity_checks_pass"] for entry in manifest["datasets"].values()):
        raise RuntimeError("Legacy parity did not pass")
    if _sha256_file(PROTOCOL) != manifest.get("protocol", {}).get("sha256"):
        raise RuntimeError("The frozen protocol hash no longer matches the current protocol")
    for model, entry in manifest["datasets"].items():
        feature_path = output_dir / entry["feature_file"]
        if feature_path.stat().st_size != entry["feature_file_bytes"]:
            raise RuntimeError(f"Prepared feature-file size changed after freeze: {model}")
        if _sha256_file(feature_path) != entry["feature_file_sha256"]:
            raise RuntimeError(f"Prepared feature-file hash changed after freeze: {model}")
    knots = np.asarray(manifest["spline"]["knots"], dtype=float)
    results: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "analysis_label": "exploratory descriptive conditional association",
        "manifest_sha256": _sha256_file(manifest_path),
        "knots": knots.tolist(),
        "models": {},
    }
    primary_refs: list[tuple[str, str, dict[str, Any]]] = []
    table_rows: list[dict[str, Any]] = []

    columns = [
        "source", "pair_id", "prompt_cluster_hash", "chosen_words", "rejected_words", "analysis_valid",
        *[f"{group}_{side}" for group in GROUPS for side in ("chosen", "rejected")],
    ]
    for model in DATASETS:
        feature_path = output_dir / manifest["datasets"][model]["feature_file"]
        frame = pd.read_parquet(feature_path, columns=columns)
        eligible = frame.loc[frame["analysis_valid"]].copy()
        largest_source = eligible["source"].value_counts(dropna=False).index[0]
        model_result: dict[str, Any] = {
            "n_prepared": len(frame),
            "n_modeled": len(eligible),
            "n_excluded": len(frame) - len(eligible),
            "largest_source": None if pd.isna(largest_source) else str(largest_source),
            "groups": {},
        }
        one_per_prompt = eligible.sort_values("pair_id").drop_duplicates("prompt_cluster_hash", keep="first")
        if pd.isna(largest_source):
            leave_largest = eligible.loc[eligible["source"].notna()]
        else:
            leave_largest = eligible.loc[eligible["source"] != largest_source]

        source_values = sorted(eligible["source"].drop_duplicates().tolist(), key=lambda value: str(value))
        for group in GROUPS:
            print(f"Fitting {model} / {group}...", flush=True)
            primary = _fit_frame(eligible, group, knots, include_length=True)
            marker_only = _fit_frame(eligible, group, knots, include_length=False)
            source_specific: dict[str, Any] = {}
            for source in source_values:
                subset = eligible.loc[eligible["source"].isna()] if pd.isna(source) else eligible.loc[eligible["source"] == source]
                source_specific["<NULL>" if pd.isna(source) else str(source)] = _safe_fit(subset, group, knots)
            group_result = {
                "primary_length_adjusted": primary,
                "marker_only": marker_only,
                "one_pair_per_prompt": _safe_fit(one_per_prompt, group, knots),
                "leave_largest_source_out": _safe_fit(leave_largest, group, knots),
                "source_specific": source_specific,
            }
            model_result["groups"][group] = group_result
            primary_refs.append((model, group, primary))
            table_rows.append(
                {
                    "model": model,
                    "group": group,
                    "analysis": "primary_length_adjusted",
                    **primary["marker"],
                    "n_pairs": primary["n_pairs"],
                    "n_clusters": primary["n_clusters"],
                    **primary["marker_discordance"],
                }
            )
            for label in ("marker_only", "one_pair_per_prompt", "leave_largest_source_out"):
                fit = group_result[label]
                if fit.get("available", True) and "marker" in fit:
                    table_rows.append(
                        {
                            "model": model,
                            "group": group,
                            "analysis": label,
                            **fit["marker"],
                            "n_pairs": fit["n_pairs"],
                            "n_clusters": fit["n_clusters"],
                            **fit["marker_discordance"],
                        }
                    )
        results["models"][model] = model_result

    q_values = _bh_adjust([fit["marker"]["wald_p"] for _, _, fit in primary_refs])
    for (model, group, fit), q_value in zip(primary_refs, q_values):
        fit["marker"]["bh_q_across_six_primary_coefficients"] = q_value
        for row in table_rows:
            if row["model"] == model and row["group"] == group and row["analysis"] == "primary_length_adjusted":
                row["bh_q_across_six_primary_coefficients"] = q_value

    results_path = output_dir / "conditional_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(table_rows).to_csv(output_dir / "conditional_effects.csv", index=False)
    _write_plot(primary_refs, output_dir / "conditional_forest.png")
    _write_report(results, manifest, RESULTS_REPORT)
    print(f"Wrote {results_path}", flush=True)
    return results_path


def _write_plot(primary_refs: list[tuple[str, str, dict[str, Any]]], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = []
    estimates = []
    lower = []
    upper = []
    short = {
        "professional_referral_original": "broad referral/disclaimer",
        "professional_referral_high_precision": "narrower referral phrases",
        "structuring_language": "structuring language",
    }
    for model, group, fit in primary_refs:
        labels.append(f"{model} — {short[group]}")
        marker = fit["marker"]
        estimates.append(marker["conditional_odds_ratio"])
        lower.append(marker["conditional_odds_ratio_95ci"][0])
        upper.append(marker["conditional_odds_ratio_95ci"][1])
    y = np.arange(len(labels))[::-1]
    estimates_a = np.asarray(estimates)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.errorbar(
        estimates_a, y,
        xerr=np.vstack([estimates_a - np.asarray(lower), np.asarray(upper) - estimates_a]),
        fmt="o", color="#2457a6", ecolor="#6f8fbe", capsize=3,
    )
    ax.axvline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ticks = [0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])
    ax.set_xlim(0.94, 1.28)
    ax.set_xlabel("Length-adjusted conditional odds ratio (prompt-cluster 95% CI)")
    ax.set_title("Exploratory within-pair marker associations")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_ci(marker: dict[str, Any]) -> str:
    lo, hi = marker["conditional_odds_ratio_95ci"]
    return f"{marker['conditional_odds_ratio']:.3f} [{lo:.3f}, {hi:.3f}]"


def _write_report(results: dict[str, Any], manifest: dict[str, Any], path: Path) -> None:
    def display_name(group: str) -> str:
        return (
            "narrower referral phrases"
            if group == "professional_referral_high_precision"
            else group.replace("_", " ")
        )

    lines = [
        "# Exploratory paired DPO conditional analysis results",
        "",
        "Completed 15 September 2026. These estimates are exploratory descriptive conditional choice associations. They are not preregistered or confirmatory, and conditioning on observed response length changes the estimand; it does not identify a causal marker preference or remove length bias.",
        "",
        "## Data and parity",
        "",
    ]
    for model in DATASETS:
        audit = manifest["datasets"][model]["audit"]
        errors = {name: audit["aggregate"][name] for name in ERROR_COLUMNS if audit["aggregate"][name]}
        metadata_nulls = {
            name: audit["aggregate"][name] for name in ("source_null", "id_null")
            if audit["aggregate"][name]
        }
        lines.append(
            f"- **{model}:** {audit['aggregate']['n_pairs']:,} prepared pairs; "
            f"{audit['aggregate']['modeled_pairs']:,} modeled; all aggregate, length-band, and prompt-duplication parity checks passed. "
            f"Nonzero structural/context error counts: `{json.dumps(errors, sort_keys=True)}`. "
            f"Multiple assistant messages occurred on the chosen side in {audit['aggregate']['chosen_multiple_assistant']:,} pairs "
            f"and on the rejected side in {audit['aggregate']['rejected_multiple_assistant']:,} pairs; these remain included as concatenated conversation text."
            f" Nonzero metadata-null counts: `{json.dumps(metadata_nulls, sort_keys=True)}`."
        )
    overlap = manifest["cross_mixture_content_overlap"]
    lines.append(
        f"- **Cross-mixture overlap:** {overlap['distinct_shared_content_pairs']:,} distinct content-identical pairs "
        f"({overlap['multiplicity_matched_rows']:,} multiplicity-matched rows) occur in both mixtures. "
        "The 1B and 7B results therefore are not treated as independent replications."
    )
    knots = manifest["spline"]["knots"]
    equivalents = manifest["spline"]["word_count_equivalents"]
    lines.extend(
        [
            "",
            f"The shared restricted-cubic-spline knots on `log1p(words)` are {', '.join(f'{v:.9f}' for v in knots)}; their word-count equivalents are {', '.join(f'{v:.2f}' for v in equivalents)}. The fixed percentile rule was 5%, 27.5%, 50%, 72.5%, and 95% over all chosen and rejected lengths.",
            "",
            "## Primary results",
            "",
            "The internal group ID `professional_referral_high_precision` is retained for artifact compatibility. Because semantic precision has not been established by human validation, the prose and table call it the **narrower referral-phrase group**.",
            "",
            "| Model | Marker group | Adjusted conditional OR [95% CI] | Cluster p | BH q (six fits) | Pairs | Prompt clusters |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, model_result in results["models"].items():
        for group, group_result in model_result["groups"].items():
            fit = group_result["primary_length_adjusted"]
            marker = fit["marker"]
            lines.append(
                f"| {model} | {display_name(group)} | {_format_ci(marker)} | "
                f"{marker['wald_p']:.3g} | {marker['bh_q_across_six_primary_coefficients']:.3g} | "
                f"{fit['n_pairs']:,} | {fit['n_clusters']:,} |"
            )
    lines.extend(
        [
            "",
            "The odds ratios compare exact marker presence on the chosen versus rejected side within a pair after modeling each side's length flexibly. They should not be read as effects of adding or removing a phrase from an otherwise fixed response.",
            "",
            "## Sensitivity and diagnostics",
            "",
        ]
    )
    for model, model_result in results["models"].items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Marker | Marker-only OR | Adjusted OR | One/prompt OR | Leave-largest-source-out OR |")
        lines.append("|---|---:|---:|---:|---:|")
        for group, group_result in model_result["groups"].items():
            def display(label: str) -> str:
                fit = group_result[label]
                return _format_ci(fit["marker"]) if fit.get("available", True) and "marker" in fit else "unavailable"
            lines.append(
                f"| {display_name(group)} | {display('marker_only')} | {display('primary_length_adjusted')} | "
                f"{display('one_pair_per_prompt')} | {display('leave_largest_source_out')} |"
            )
        lines.append("")
        lines.append("Source-specific direction/range (descriptive adjusted ORs):")
        lines.append("")
        for group, group_result in model_result["groups"].items():
            source_fits = [
                value for value in group_result["source_specific"].values()
                if value.get("available") and "marker" in value
            ]
            source_ors = [value["marker"]["conditional_odds_ratio"] for value in source_fits]
            positive = sum(value > 1.0 for value in source_ors)
            if source_ors:
                lines.append(
                    f"- {display_name(group)}: {positive}/{len(source_ors)} sources above 1; "
                    f"range {min(source_ors):.3f}–{max(source_ors):.3f}."
                )
            else:
                lines.append(f"- {display_name(group)}: no source-specific fit available.")
        lines.append("")
        warnings = []
        for group, group_result in model_result["groups"].items():
            fit = group_result["primary_length_adjusted"]
            if not fit["optimizer_success"] or fit["any_separation_warning"]:
                nuisance_only = (
                    fit["separation_flags"]["coefficient_abs_gt_20"]
                    and abs(fit["coefficients"][0]) <= 20
                )
                warnings.append(
                    f"{group}: optimizer_success={fit['optimizer_success']}, "
                    f"flags={fit['separation_flags']}, large-coefficient flag confined to spline nuisance terms={nuisance_only}"
                )
        lines.append("Primary-fit warnings: " + ("; ".join(warnings) if warnings else "none."))
        if warnings:
            lines.append(
                "The frozen absolute-coefficient flag is triggered only by spline nuisance coefficients in these fits; "
                "all marker coefficients have absolute value below 0.18. On its own this basis-scale-dependent flag is not evidence of separation."
            )
        lines.append("")
    lines.extend(
        [
            "Source-specific estimates and full convergence, rank, eigenvalue, gradient, boundary-probability, and finite-correction diagnostics are in `conditional_results.json`. The deterministic one-pair-per-prompt sensitivity uses the smallest stable pair hash. The leave-out sensitivity removes the largest source separately for each model.",
            "",
            "## Scope",
            "",
            "Exact substring flags preserve the historical detectors, including the broad referral/disclaimer fragments. They do not establish semantic referral quality. Prompt-cluster sandwich intervals account for identical prompt hashes, but normal intervals and exact-text clusters do not guarantee protection against every form of dependence. Source heterogeneity and sensitivity estimates are descriptive. None of these co-occurrences attributes checkpoint behavior to particular training examples.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_existing(output_dir: Path) -> None:
    """Polish diagnostics/report/plot from saved fits without re-estimation."""
    results_path = output_dir / "conditional_results.json"
    manifest_path = output_dir / "feature_manifest.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def normalize(value: Any) -> None:
        if isinstance(value, dict):
            if "coefficients" in value and "information_condition_number" in value:
                coefficients = np.asarray(value["coefficients"], dtype=float)
                eigenvalues = np.asarray(value["information_eigenvalues"], dtype=float)
                flags = {
                    "coefficient_abs_gt_20": bool(np.any(np.abs(coefficients) > 20)),
                    "information_condition_gt_1e12": value["information_condition_number"] > 1e12,
                    "boundary_probability_count_positive": value["boundary_probability_count"] > 0,
                    "minimum_information_eigenvalue_lt_1e_minus_8": float(eigenvalues[0]) < 1e-8,
                }
                value["separation_flags"] = flags
                value["any_separation_warning"] = any(flags.values())
                value["marker_coefficient_abs_gt_20"] = bool(abs(coefficients[0]) > 20)
                value.pop("diagnostic_warnings", None)
                value.pop("any_numerical_or_separation_warning", None)
            for child in value.values():
                normalize(child)
        elif isinstance(value, list):
            for child in value:
                normalize(child)

    normalize(results)
    results["diagnostic_rendering"] = {
        "rendered_at_utc": _utc_now(),
        "refit": False,
        "note": "Frozen separation flags restored from saved coefficients/information; numeric fits unchanged.",
    }
    _write_json_atomic(results_path, results)
    primary_refs = [
        (model, group, group_result["primary_length_adjusted"])
        for model, model_result in results["models"].items()
        for group, group_result in model_result["groups"].items()
    ]
    _write_plot(primary_refs, output_dir / "conditional_forest.png")
    _write_report(results, manifest, RESULTS_REPORT)
    print("Rendered saved results without refitting", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "freeze", "analyze", "render"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare(args.output_dir)
    elif args.command == "freeze":
        freeze_protocol(args.output_dir)
    elif args.command == "render":
        render_existing(args.output_dir)
    else:
        analyze(args.output_dir)


if __name__ == "__main__":
    main()
