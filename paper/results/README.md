# Evaluation result snapshot

`all_models_plot_summary.csv` is the compact, three-seed summary used by the
committed figures. It covers the four LigandMPNN noise checkpoints (`005`,
`010`, `020`, and `030`) on the pathway-specific and original LigandMPNN test
sets.

The full per-structure evaluation tables are deliberately not stored in Git.
They can be regenerated with `training/eval_stratified.py` by following
`EXPERIMENT_REPRODUCIBILITY.md` and should be deposited with the accompanying
data archive for the paper.
