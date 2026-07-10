"""Paired and length-normalized analysis of the exact OLMo-2 DPO mixes.

The paper originally reported only global chosen/rejected phrase-count ratios.
This script strengthens that analysis in two ways:

1. Pair-level McNemar tests compare chosen-only with rejected-only occurrences.
2. Phrase occurrences are normalized by whitespace-token counts, addressing the
   possibility that chosen responses are simply longer.

The public Hugging Face parquet shards are scanned remotely with DuckDB; they
are not downloaded into the repository.

Run:  python -m experiments.dpo_pairwise
"""
from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request

import duckdb
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom, binomtest

from config import ROOT


DATASETS = {
    "1B": "allenai/olmo-2-0425-1b-preference-mix",
    "7B": "allenai/olmo-2-1124-7b-preference-mix",
}

# These are the exact paper lexicons, grouped under construct-valid names.
PHRASES = {
    "empathy_expression": [
        "i'm sorry to hear",
        "i am sorry to hear",
        "i'm really sorry",
        "that sounds really",
        "that sounds very",
        "i can understand",
        "i can imagine",
        "it's understandable",
        "that must be",
        "i'm here for you",
        "i hear you",
        "it sounds like you",
    ],
    "validation": [
        "your feelings are valid",
        "it's okay to feel",
        "it's normal to feel",
        "it's completely normal",
        "what you're feeling",
        "there's nothing wrong with",
        "it's natural to",
    ],
    "professional_referral": [
        "consult a",
        "talk to a",
        "speak with a",
        "see a doctor",
        "seek medical",
        "i am not a doctor",
        "i'm not a doctor",
        "medical professional",
        "healthcare professional",
        "mental health professional",
        "seek professional",
        "professional help",
    ],
    "structuring_language": [
        "here are some",
        "here are a few",
        "consider the following",
        "steps you can take",
        "a few things you can",
        "first,",
        "secondly,",
    ],
}


def parquet_urls(dataset: str) -> list[str]:
    quoted = urllib.parse.quote(dataset, safe="/")
    api = (
        f"https://huggingface.co/api/datasets/{quoted}/tree/"
        "refs%2Fconvert%2Fparquet/default/train?recursive=true&expand=true&limit=100"
    )
    with urllib.request.urlopen(api, timeout=60) as response:
        entries = json.load(response)
    paths = sorted(entry["path"] for entry in entries if entry["path"].endswith(".parquet"))
    if not paths:
        raise RuntimeError(f"No parquet shards found for {dataset}")
    return [
        f"https://huggingface.co/datasets/{quoted}/resolve/"
        f"refs%2Fconvert%2Fparquet/{path}"
        for path in paths
    ]


def pattern(phrases: list[str]) -> str:
    # Longer alternatives first avoids counting a short alternative inside a
    # longer one when regexp_extract_all is used for token-normalized rates.
    return "(?:" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + ")"


def analyze_dataset(connection: duckdb.DuckDBPyConnection, dataset: str) -> dict:
    urls = parquet_urls(dataset)
    columns = []
    for name, phrases in PHRASES.items():
        regex = pattern(phrases).replace("'", "''")
        columns.extend(
            [
                f"sum(regexp_matches(lower(chosen_text), '{regex}')::INT) AS {name}_chosen_any",
                f"sum(regexp_matches(lower(rejected_text), '{regex}')::INT) AS {name}_rejected_any",
                (
                    f"sum((regexp_matches(lower(chosen_text), '{regex}') AND NOT "
                    f"regexp_matches(lower(rejected_text), '{regex}'))::INT) "
                    f"AS {name}_chosen_only"
                ),
                (
                    f"sum((NOT regexp_matches(lower(chosen_text), '{regex}') AND "
                    f"regexp_matches(lower(rejected_text), '{regex}'))::INT) "
                    f"AS {name}_rejected_only"
                ),
                (
                    f"sum((regexp_matches(lower(chosen_text), '{regex}') AND "
                    f"regexp_matches(lower(rejected_text), '{regex}'))::INT) "
                    f"AS {name}_both"
                ),
                f"sum(len(regexp_extract_all(lower(chosen_text), '{regex}'))) AS {name}_chosen_occ",
                f"sum(len(regexp_extract_all(lower(rejected_text), '{regex}'))) AS {name}_rejected_occ",
            ]
        )

    query = f"""
        WITH responses AS (
          SELECT
            array_to_string(
              list_transform(list_filter(chosen, m -> m.role='assistant'), m -> m.content),
              '\\n'
            ) AS chosen_text,
            array_to_string(
              list_transform(list_filter(rejected, m -> m.role='assistant'), m -> m.content),
              '\\n'
            ) AS rejected_text
          FROM read_parquet(?)
        )
        SELECT
          count(*) AS n_pairs,
          sum(len(regexp_extract_all(chosen_text, '\\S+'))) AS chosen_words,
          sum(len(regexp_extract_all(rejected_text, '\\S+'))) AS rejected_words,
          {', '.join(columns)}
        FROM responses
    """
    row = connection.execute(query, [urls]).fetchone()
    names = [item[0] for item in connection.description]
    aggregate = dict(zip(names, row))
    n_pairs = int(aggregate["n_pairs"])
    chosen_words = int(aggregate["chosen_words"])
    rejected_words = int(aggregate["rejected_words"])

    results = {
        "dataset": dataset,
        "n_pairs": n_pairs,
        "n_shards": len(urls),
        "mean_words": {
            "chosen": round(chosen_words / n_pairs, 2),
            "rejected": round(rejected_words / n_pairs, 2),
            "chosen_over_rejected": round(chosen_words / rejected_words, 3),
        },
        "behaviors": {},
    }
    for name in PHRASES:
        chosen_any = int(aggregate[f"{name}_chosen_any"])
        rejected_any = int(aggregate[f"{name}_rejected_any"])
        chosen_only = int(aggregate[f"{name}_chosen_only"])
        rejected_only = int(aggregate[f"{name}_rejected_only"])
        both = int(aggregate[f"{name}_both"])
        discordant = chosen_only + rejected_only
        p = (
            binomtest(min(chosen_only, rejected_only), discordant, 0.5, alternative="two-sided").pvalue
            if discordant
            else 1.0
        )
        log10_p = (
            (math.log(2) + logsumexp(binom.logpmf(np.arange(min(chosen_only, rejected_only) + 1), discordant, 0.5)))
            / math.log(10)
            if discordant
            else 0.0
        )
        chosen_occ = int(aggregate[f"{name}_chosen_occ"])
        rejected_occ = int(aggregate[f"{name}_rejected_occ"])
        chosen_rate = 1_000_000 * chosen_occ / chosen_words
        rejected_rate = 1_000_000 * rejected_occ / rejected_words
        results["behaviors"][name] = {
            "chosen_any": chosen_any,
            "rejected_any": rejected_any,
            "chosen_only": chosen_only,
            "rejected_only": rejected_only,
            "both": both,
            "paired_risk_difference": round((chosen_any - rejected_any) / n_pairs, 5),
            "matched_odds_ratio_ha": round((chosen_only + 0.5) / (rejected_only + 0.5), 3),
            "mcnemar_exact_p": p,
            "mcnemar_log10_p": log10_p,
            "chosen_occurrences": chosen_occ,
            "rejected_occurrences": rejected_occ,
            "chosen_per_million_words": round(chosen_rate, 2),
            "rejected_per_million_words": round(rejected_rate, 2),
            "token_normalized_ratio": round(chosen_rate / rejected_rate, 3)
            if rejected_rate
            else None,
        }
    return results


def main() -> None:
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar=false")
    connection.execute("SET enable_object_cache=true")
    output = {}
    for model, dataset in DATASETS.items():
        print(f"Scanning {model}: {dataset}", flush=True)
        output[model] = analyze_dataset(connection, dataset)
        print(
            f"  {output[model]['n_pairs']:,} pairs; "
            f"chosen/rejected mean-word ratio="
            f"{output[model]['mean_words']['chosen_over_rejected']:.3f}",
            flush=True,
        )
        for behavior, result in output[model]["behaviors"].items():
            print(
                f"  {behavior:<24} chosen-only={result['chosen_only']:>5} "
                f"rejected-only={result['rejected_only']:>5} "
                f"McNemar p={result['mcnemar_exact_p']:.3g} "
                f"token ratio={result['token_normalized_ratio']}",
                flush=True,
            )

    destination = ROOT / "out" / "results" / "dpo_pairwise.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
