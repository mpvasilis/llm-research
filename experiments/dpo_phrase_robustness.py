"""Per-phrase, high-precision, source, and duplicate-prompt DPO robustness."""
from __future__ import annotations

import json
import math
import re
import time

import duckdb
from scipy.stats import binomtest

from config import ROOT
from experiments.dpo_pairwise import DATASETS, PHRASES, parquet_urls


HIGH_PRECISION_REFERRAL = [
    "see a doctor",
    "talk to a doctor",
    "talk to your doctor",
    "speak with a doctor",
    "speak with your doctor",
    "consult a doctor",
    "consult your doctor",
    "consult a medical professional",
    "consult a healthcare professional",
    "consult a mental health professional",
    "seek medical advice",
    "seek medical attention",
    "seek professional help",
]

GROUPS = {
    "professional_referral_original": PHRASES["professional_referral"],
    "professional_referral_high_precision": HIGH_PRECISION_REFERRAL,
    "structuring_language": PHRASES["structuring_language"],
}


def regex(phrases: list[str]) -> str:
    return "(?:" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + ")"


def effect(chosen_only: int, rejected_only: int, n_pairs: int) -> dict:
    discordant = chosen_only + rejected_only
    p = binomtest(min(chosen_only, rejected_only), discordant, 0.5).pvalue if discordant else 1.0
    b = chosen_only + 0.5
    c = rejected_only + 0.5
    odds = b / c
    se_log_or = math.sqrt(1 / b + 1 / c)
    odds_ci = [math.exp(math.log(odds) - 1.96 * se_log_or), math.exp(math.log(odds) + 1.96 * se_log_or)]
    risk = (chosen_only - rejected_only) / n_pairs
    variance = ((discordant / n_pairs) - risk**2) / n_pairs if n_pairs else 0.0
    risk_ci = [risk - 1.96 * math.sqrt(max(0.0, variance)), risk + 1.96 * math.sqrt(max(0.0, variance))]
    return {
        "chosen_only": chosen_only,
        "rejected_only": rejected_only,
        "matched_odds_ratio": round(odds, 4),
        "matched_odds_ratio_95ci": [round(value, 4) for value in odds_ci],
        "paired_risk_difference": round(risk, 7),
        "paired_risk_difference_95ci": [round(value, 7) for value in risk_ci],
        "exact_mcnemar_p": p,
    }


def scan(connection: duckdb.DuckDBPyConnection, dataset: str) -> dict:
    urls = parquet_urls(dataset)
    phrase_inventory = sorted(set(PHRASES["professional_referral"] + PHRASES["structuring_language"]), key=str.lower)
    definitions = {f"phrase::{phrase}": [phrase] for phrase in phrase_inventory}
    definitions.update(GROUPS)
    columns = []
    for index, phrases in enumerate(definitions.values()):
        pattern = regex(phrases).replace("'", "''")
        columns.extend([
            f"sum((regexp_matches(lower(chosen_text), '{pattern}') AND NOT regexp_matches(lower(rejected_text), '{pattern}'))::INT) AS c{index}",
            f"sum((NOT regexp_matches(lower(chosen_text), '{pattern}') AND regexp_matches(lower(rejected_text), '{pattern}'))::INT) AS r{index}",
        ])
    base = """
      WITH responses AS (
        SELECT source,
          array_to_string(list_transform(list_filter(chosen, m -> m.role='user'), m -> m.content), '\\n') AS prompt_text,
          array_to_string(list_transform(list_filter(chosen, m -> m.role='assistant'), m -> m.content), '\\n') AS chosen_text,
          array_to_string(list_transform(list_filter(rejected, m -> m.role='assistant'), m -> m.content), '\\n') AS rejected_text
        FROM read_parquet(?)
      )
    """
    row = connection.execute(base + f"SELECT count(*) AS n_pairs, {', '.join(columns)} FROM responses", [urls]).fetchone()
    n_pairs = int(row[0])
    effects = {}
    for index, name in enumerate(definitions):
        effects[name] = effect(int(row[1 + 2 * index]), int(row[2 + 2 * index]), n_pairs)

    source_columns = []
    for index, phrases in enumerate(GROUPS.values()):
        pattern = regex(phrases).replace("'", "''")
        source_columns.extend([
            f"sum((regexp_matches(lower(chosen_text), '{pattern}') AND NOT regexp_matches(lower(rejected_text), '{pattern}'))::INT) AS c{index}",
            f"sum((NOT regexp_matches(lower(chosen_text), '{pattern}') AND regexp_matches(lower(rejected_text), '{pattern}'))::INT) AS r{index}",
        ])
    source_rows = connection.execute(
        base + f"SELECT source, count(*) AS n_pairs, {', '.join(source_columns)} FROM responses GROUP BY source ORDER BY n_pairs DESC",
        [urls],
    ).fetchall()
    by_source = {}
    for source_row in source_rows:
        source = source_row[0]
        source_n = int(source_row[1])
        by_source[source] = {
            name: effect(int(source_row[2 + 2 * index]), int(source_row[3 + 2 * index]), source_n)
            for index, name in enumerate(GROUPS)
        }

    length_rows = connection.execute(
        base
        + f""" , length_rows AS (
               SELECT *,
                 len(regexp_extract_all(chosen_text, '\\S+'))
                 - len(regexp_extract_all(rejected_text, '\\S+')) AS word_delta,
                 CASE
                   WHEN len(regexp_extract_all(chosen_text, '\\S+'))
                        - len(regexp_extract_all(rejected_text, '\\S+')) < -20
                     THEN 'chosen_shorter_gt20'
                   WHEN len(regexp_extract_all(chosen_text, '\\S+'))
                        - len(regexp_extract_all(rejected_text, '\\S+')) > 20
                     THEN 'chosen_longer_gt20'
                   ELSE 'within_20_words'
                 END AS length_band
               FROM responses
             )
             SELECT length_band, count(*) AS n_pairs, avg(word_delta) AS mean_word_delta,
                    {', '.join(source_columns)}
             FROM length_rows
             GROUP BY length_band
             ORDER BY length_band""",
        [urls],
    ).fetchall()
    by_length_delta = {}
    for length_row in length_rows:
        band = length_row[0]
        band_n = int(length_row[1])
        by_length_delta[band] = {
            "n_pairs": band_n,
            "mean_word_delta": round(float(length_row[2]), 1),
            "groups": {
                name: effect(int(length_row[3 + 2 * index]), int(length_row[4 + 2 * index]), band_n)
                for index, name in enumerate(GROUPS)
            },
        }

    duplicate_row = connection.execute(
        base
        + """ , grouped AS (SELECT prompt_text, count(*) AS multiplicity FROM responses GROUP BY prompt_text)
             SELECT count(*) AS distinct_prompts,
                    sum((multiplicity > 1)::INT) AS duplicated_prompt_groups,
                    sum(CASE WHEN multiplicity > 1 THEN multiplicity ELSE 0 END) AS rows_in_duplicated_groups,
                    max(multiplicity) AS max_multiplicity
             FROM grouped""",
        [urls],
    ).fetchone()

    return {
        "dataset": dataset,
        "n_pairs": n_pairs,
        "prompt_dependence": {
            "distinct_prompt_texts": int(duplicate_row[0]),
            "duplicated_prompt_groups": int(duplicate_row[1]),
            "rows_in_duplicated_groups": int(duplicate_row[2]),
            "max_multiplicity": int(duplicate_row[3]),
        },
        "groups": {name: effects[name] for name in GROUPS},
        "per_phrase": {name.removeprefix("phrase::"): value for name, value in effects.items() if name.startswith("phrase::")},
        "by_source": by_source,
        "by_length_delta": by_length_delta,
    }


def main() -> None:
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar=false")
    connection.execute("SET enable_object_cache=true")
    output = {}
    for model, dataset in DATASETS.items():
        for attempt in range(5):
            try:
                print(f"Scanning {model} ({attempt + 1}/5)", flush=True)
                output[model] = scan(connection, dataset)
                break
            except duckdb.HTTPException:
                if attempt == 4:
                    raise
                time.sleep(15 * (attempt + 1))
    destination = ROOT / "out" / "results" / "dpo_phrase_robustness.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
