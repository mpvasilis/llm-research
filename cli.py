"""Trace how an LLM answer is shaped by its training data.

Usage:
  python cli.py --question "..." --answer data/answer.txt
  python cli.py --question "..." --answer-text "the model's reply ..."
"""
import argparse
from pathlib import Path
from config import DEFAULT_MODEL, PRETRAIN_INDEXES
from trace.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question", required=True, help="The question you asked the LLM.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--answer", help="Path to a file with the LLM's answer.")
    g.add_argument("--answer-text", help="The LLM's answer inline.")
    g.add_argument("--generate", action="store_true",
                   help="Generate the answer locally with the open-data model "
                        "(downloads ~2.5GB on first run).")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(PRETRAIN_INDEXES),
                    help="Open-data model family whose corpora to search.")
    ap.add_argument("--max-calls", type=int, default=80,
                    help="Cap on infini-gram calls (cost/time bound).")
    ap.add_argument("--no-instruction", action="store_true",
                    help="Skip the (flaky, slow) HF instruction-dataset search; "
                         "pretraining trace only.")
    args = ap.parse_args()

    if args.generate:
        from trace.generate import generate
        print(f"Generating answer locally with {args.model} (first run downloads the model)…")
        answer = generate(args.model, args.question)
        print(f"--- generated answer ---\n{answer}\n------------------------")
        Path("data").mkdir(exist_ok=True)
        Path("data/answer.txt").write_text(answer, encoding="utf-8")
    elif args.answer:
        answer = Path(args.answer).read_text(encoding="utf-8")
    elif args.answer_text:
        answer = args.answer_text
    else:
        raise SystemExit("Provide --answer, --answer-text, or --generate.")
    if not answer.strip():
        raise SystemExit("Answer is empty.")

    print(f"Tracing {args.model} answer ({len(answer)} chars)…")
    t = run(args.model, args.question, answer, max_calls=args.max_calls,
            do_instruction=not args.no_instruction)
    p = t["pretraining"]
    print(f"  pretraining: {p['n_matches']} matched spans, "
          f"longest {p['longest_span_words']} words")
    for d in t["instruction_tuning"]["datasets"]:
        n = d.get("total_matches", "err") if not d.get("error") else "err"
        print(f"  instruction: {d['dataset']} -> {n} matches")
    print(f"Wrote out/report.md, out/trace.json, history record {t.get('record_id')}")


if __name__ == "__main__":
    main()
