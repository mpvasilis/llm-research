# Specification-curve inference over the 12-cell grid

    python -m experiments.free_analyses.spec_curve

Joint prompt-level permutation test (5,000 permutations, same permutation across all 12 specifications), sums-of-squares decomposition by rule, threshold and window, leave-one-phrase-out and leave-one-exemplar-out, and TOST equivalence per cell (margins 5 and 10 pp, 90% stratified prompt bootstrap, 10,000 draws). Seed 20261114. Observed gaps are checked against `out/results/specification_grid.json` (field `max_abs_dev_from_specification_grid_json`).

Outputs: `spec_curve.json`, `spec_curve_joint.tex`, `tost_cells.tex`.
