# Legacy detector-validation draft — deprecated

The historical 80-row file (`out/results/validation_sheet.csv`) is a
tool-assisted diagnostic first pass. Its human-review provenance is not
established, so it must not be described or cited as human validation.

The current workflow is validation v3, implemented in
`experiments/validation_v3.py` and the complete Colab notebook. It requires:

- blinded items separated from the hidden scoring key;
- two distinct, certified human annotators working in separate files;
- agreement and category/condition/stage error metrics with confidence
  intervals;
- disagreement adjudication with recorded decision rules.

The legacy diagnostic JSON remains available only for provenance. It is never
consumed by validation v3 or treated as publication-grade evidence.
