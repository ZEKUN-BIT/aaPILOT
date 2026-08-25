# Enzyme-family LigandMPNN fine-tuning

This repository contains the training, evaluation, and figure-generation code
for a LigandMPNN fine-tuning experiment on pathway-specific enzyme families.
It is organized as a single reproducible experiment rather than as a sequence
of historical implementations.

## Method

The implementation combines:

- exact sample-name and sequence leakage checks across train/validation/test;
- homology-cluster-aware dataset splits supplied with the experiment dataset;
- target → cluster → member balanced sampling;
- length-sorted mini-batches without changing the sampler's draws;
- checkpoint-matched Gaussian coordinate noise;
- partial fine-tuning with gradient accumulation, bfloat16 AMP, gradient
  clipping, and exponential moving average weights;
- paired baseline/fine-tuned evaluation with identical decoding orders;
- overall recovery, ligand-pocket recovery, perplexity, and paired-bootstrap
  confidence intervals;
- evaluation on the pathway-specific test set and the original LigandMPNN test
  set as a general-capability control.

## Results

The figures below summarize the four fine-tuned LigandMPNN variants. Each model is compared with its matching vanilla checkpoint. The pathway test set measures adaptation to the target enzyme family, while the original LigandMPNN test set checks whether general protein-design performance is retained.

### Overall comparison

<p align="center">
  <img src="paper/figures/all_models_absolute_recovery.png" alt="Absolute sequence and pocket recovery for all four model variants" width="900">
</p>
<p align="center">
  <em>Figure 1. Absolute overall and pocket recovery on the pathway-specific and original LigandMPNN test sets.</em><br>
  <a href="paper/figures/all_models_absolute_recovery.pdf">Vector PDF</a>
</p>

### Fine-tuning effect and uncertainty

<p align="center">
  <img src="paper/figures/all_models_recovery_delta_ci.png" alt="Recovery changes with confidence intervals for all four model variants" width="900">
</p>
<p align="center">
  <em>Figure 2. Paired recovery changes after fine-tuning. Confidence intervals are estimated by cluster bootstrap for the pathway test set and row bootstrap for the original test set.</em><br>
  <a href="paper/figures/all_models_recovery_delta_ci.pdf">Vector PDF</a>
</p>

### Pathway-specific test set

<table>
  <tr>
    <td width="50%" align="center">
      <img src="paper/figures/pathway_overall_recovery_paired_scatter_all_models.png" alt="Paired overall recovery on the pathway-specific test set" width="100%"><br>
      <em>Overall recovery: paired comparison</em><br>
      <a href="paper/figures/pathway_overall_recovery_paired_scatter_all_models.pdf">Vector PDF</a>
    </td>
    <td width="50%" align="center">
      <img src="paper/figures/pathway_pocket_recovery_paired_scatter_all_models.png" alt="Paired pocket recovery on the pathway-specific test set" width="100%"><br>
      <em>Pocket recovery: paired comparison</em><br>
      <a href="paper/figures/pathway_pocket_recovery_paired_scatter_all_models.pdf">Vector PDF</a>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="paper/figures/pathway_overall_recovery_violin_all_models.png" alt="Overall recovery distributions on the pathway-specific test set" width="100%"><br>
      <em>Overall recovery: distribution</em><br>
      <a href="paper/figures/pathway_overall_recovery_violin_all_models.pdf">Vector PDF</a>
    </td>
    <td width="50%" align="center">
      <img src="paper/figures/pathway_pocket_recovery_violin_all_models.png" alt="Pocket recovery distributions on the pathway-specific test set" width="100%"><br>
      <em>Pocket recovery: distribution</em><br>
      <a href="paper/figures/pathway_pocket_recovery_violin_all_models.pdf">Vector PDF</a>
    </td>
  </tr>
</table>

### Original LigandMPNN test set

<table>
  <tr>
    <td width="50%" align="center">
      <img src="paper/figures/original_test_overall_recovery_paired_scatter_all_models.png" alt="Paired overall recovery on the original LigandMPNN test set" width="100%"><br>
      <em>Overall recovery: paired comparison</em><br>
      <a href="paper/figures/original_test_overall_recovery_paired_scatter_all_models.pdf">Vector PDF</a>
    </td>
    <td width="50%" align="center">
      <img src="paper/figures/original_test_pocket_recovery_paired_scatter_all_models.png" alt="Paired pocket recovery on the original LigandMPNN test set" width="100%"><br>
      <em>Pocket recovery: paired comparison</em><br>
      <a href="paper/figures/original_test_pocket_recovery_paired_scatter_all_models.pdf">Vector PDF</a>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="paper/figures/original_test_overall_recovery_violin_all_models.png" alt="Overall recovery distributions on the original LigandMPNN test set" width="100%"><br>
      <em>Overall recovery: distribution</em><br>
      <a href="paper/figures/original_test_overall_recovery_violin_all_models.pdf">Vector PDF</a>
    </td>
    <td width="50%" align="center">
      <img src="paper/figures/original_test_pocket_recovery_violin_all_models.png" alt="Pocket recovery distributions on the original LigandMPNN test set" width="100%"><br>
      <em>Pocket recovery: distribution</em><br>
      <a href="paper/figures/original_test_pocket_recovery_violin_all_models.pdf">Vector PDF</a>
    </td>
  </tr>
</table>

## Repository layout

```text
model_params/                 Original LigandMPNN checkpoints
training/
  train.py                    Fine-tuning entry point
  eval_stratified.py          Paired evaluation and bootstrap summaries
  plot_all_model_evaluations.py  Publication figures
  featurization.py             Shared JSONL batch featurization
  cluster_balanced_sampler.py Balanced sampling implementation
  data_integrity.py           Leakage and input validation
  model_utils.py              LigandMPNN architecture
  finetuned_ligand_models/    Experiment checkpoints
  tests/                      Unit tests
paper/
  figures/                    PNG and vector PDF figures
  results/                    Compact figure source table
```

## Installation

Python 3.10+ is recommended. Install PyTorch for the CUDA version appropriate
for your system, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Reproduction

The complete data contract, commands, hyperparameters, evaluation layout, and
figure commands are documented in
[`EXPERIMENT_REPRODUCIBILITY.md`](EXPERIMENT_REPRODUCIBILITY.md).

Quick validation:

```bash
cd training
python -m unittest discover -s tests -v
```

## Data availability

Large JSONL datasets and per-structure evaluation tables are intentionally not
stored in Git. They should be downloaded from the paper's archival dataset and
placed according to `EXPERIMENT_REPRODUCIBILITY.md`.

## Acknowledgments

This work builds on [LigandMPNN](https://github.com/dauparas/LigandMPNN) and
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN) from the Baker Lab.
