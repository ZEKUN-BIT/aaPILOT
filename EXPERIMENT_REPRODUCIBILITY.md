# Experiment reproducibility

## Data contract

Place `train.jsonl`, `val.jsonl`, and `test.jsonl` in one dataset directory.
Every row must contain `name`, `seq`, `target`, and `cluster_id`, plus the
standard LigandMPNN backbone fields. Ligand-bearing rows additionally contain
`ligand_coords` and `ligand_types`. The training command rejects duplicate
sample names and exact sequence overlap between splits.

## Fine-tuning

Run from `training/`:

```bash
python train.py \
  --data-dir /path/to/dataset \
  --checkpoint-dir ../model_params \
  --output-dir training_runs \
  --models ligandmpnn_v_32_005_25 ligandmpnn_v_32_010_25 \
           ligandmpnn_v_32_020_25 ligandmpnn_v_32_030_25 \
  --epochs 10 \
  --batch-size 4 \
  --accumulation-steps 8 \
  --learning-rate 5e-5 \
  --atom-context-num 25 \
  --max-length 2400 \
  --val-max-samples 512 \
  --samples-per-epoch 6000 \
  --use-cluster-balanced \
  --freeze-mode current \
  --use-ema \
  --ema-decay 0.999 \
  --num-workers 4 \
  --device cuda \
  --seed 42
```

The command creates a timestamped run directory containing `config.json` and
one `<model>/best.pt` checkpoint per noise level.

## Paired evaluation

Evaluate each fine-tuned checkpoint against its matching original checkpoint.
Run the command for all four noise levels on both the pathway-specific test set
and the original LigandMPNN test set:

```bash
python eval_stratified.py \
  --finetuned /path/to/run/ligandmpnn_v_32_005_25/best.pt \
  --test-jsonl /path/to/test.jsonl \
  --baseline-dir ../model_params \
  --models ligandmpnn_v_32_005_25 \
  --seeds 42 43 44 \
  --bootstrap-samples 10000 \
  --output-dir evaluation_runs/pathway/ligandmpnn_v_32_005_25
```

Store the general-capability runs under
`evaluation_runs/original_test/<model>/`. The evaluator writes a per-row table
and `stratified_summary.csv`. Use cluster-bootstrap intervals for pathway
conclusions. The original LigandMPNN test export lacks informative homology
cluster labels, so its reported interval uses row bootstrap.

## Publication figures

After the eight evaluations are present:

```bash
python plot_all_model_evaluations.py \
  --root evaluation_runs \
  --output-dir ../paper/figures \
  --summary-csv ../paper/results/all_models_plot_summary.csv \
  --dpi 1600
```

The figure generator produces PNG and vector PDF versions. The committed
`paper/results/all_models_plot_summary.csv` is the compact source table for the
comparison figures; large per-structure tables belong in the archival dataset.

## Tests

```bash
cd training
python -m unittest discover -s tests -v
```
