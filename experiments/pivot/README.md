# Pivot study runbook

Protocol: `out/paper/PIVOT_PROTOCOL_2026-11-10.md`.

## 1. Check locally (CPU, no GPU needed)

```bash
python experiments/pivot/pivot_runner.py smoke  --out tmp/pivot_smoke
python experiments/pivot/pivot_runner.py dryrun --out tmp/pivot_dry
python -m experiments.pivot.analyze_pivot --results tmp/pivot_dry --out tmp/pivot_dry_analysis.json
python -m experiments.pivot.report_pivot --analysis tmp/pivot_dry_analysis.json --out tmp/pivot_dry.tex
python -m pytest tests/test_pivot.py -q
```

`dryrun` runs the whole schedule (13 atlas tags, 5 arms x 3 seeds) with tiny
random models and synthetic preference pairs, so every file name and every
analysis step is exercised. Its numbers mean nothing.

## 2. Run on Vast.ai

Needs `vastai` logged in and account credit.

```bash
python experiments/pivot/vast_launch.py start --budget 15
```

The remote job first runs `bench` on the rented GPU and writes `bench.json`
with projected hours. `watch` (started by `start`) destroys the instance if the
projection times the hourly price exceeds `--budget`. If that happens, rerun
with a smaller sample for every arm, for example `--pairs 30000`, and report
the value in the paper. Results sync to `output/pivot/results/` every 10
minutes; the instance is destroyed when `ALL_DONE` appears or the budget is
spent. To reattach: `vast_launch.py watch --id <instance> --budget 15`.

## 3. Analyze and update the paper

```bash
python -m experiments.pivot.analyze_pivot --union
python -m experiments.pivot.report_pivot
```

`report_pivot` overwrites `out/paper/pivot_results.tex` with `\pivottrue` and
every number; the paper then builds with RQ4, the atlas appendix, and the
deferred tables. Read the numbers before submitting: the prose around them
states estimates, intervals and p-values but makes no claim about direction.
