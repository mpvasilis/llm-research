# Atlas RQ2: stagewise marker contrasts on the pivot generations

CPU only, offline (uses the locally cached multilingual-e5-small and the OLMo-2 1B Instruct tokenizer).

    python -m experiments.free_analyses.atlas_rq2

Inputs: `output/pivot/results/gen_*.jsonl` (13 tags; bench files untouched). Detector, cleaning and windows are imported from `experiments/detector_spec.py`, `experiments/pivot/analyze_pivot.py` and `experiments/specification_grid.py`.

Outputs: `atlas_rq2.json` (per-tag rates and contrasts, adjacent-stage changes, chat minus qa format effects, grid robustness), `atlas_stage_changes.tex`.

Seed 20261114; 10,000 bootstrap draws over the 40 topic families; TOST margin 5 pp via the 90% interval. Sentence embeddings are cached in `../_cache/`.
