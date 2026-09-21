# ALTAS

**Adversarial Latent-space Task-Aware Selection (ALTAS)** is an embedded,
instance-wise feature-selection framework. It learns a sample-conditioned
binary mask while encouraging masked inputs to preserve both predictive
information and the latent distribution induced by complete inputs.

This repository contains the minimal reference implementation used to train
and evaluate ALTAS on the six synthetic feature-selection mechanisms
Syn1--Syn6.

## Method overview

ALTAS contains four trainable components:

1. a Gumbel--Softmax mask generator that produces an instance-wise feature
   mask;
2. a shared feature extractor for complete and masked inputs;
3. a task predictor trained with complete and masked examples; and
4. a Wasserstein critic with gradient penalty that distinguishes their latent
   distributions.

Each training step performs three updates: the critic is optimized first, the
feature extractor and task predictor are then trained under task supervision,
and the mask generator is finally optimized using adversarial, predictive, and
sparsity losses. During the generator update, the extractor, predictor, and
critic parameters are frozen. This asymmetric optimization keeps the
full-input task representation from being moved by the generator objective.

Removed feature values are replaced with the same coordinates drawn from
shuffled samples in the current batch instead of a fixed mask token.

## Requirements

- Python 3.10 or later
- PyTorch
- NumPy
- pandas
- Matplotlib

A CUDA-capable GPU is optional. The training entry point automatically uses
CUDA when available and otherwise runs on CPU.

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Quick start

The default experiment is configured in [`config.py`](config.py). Adjust the
dataset, architecture, optimization weights, or number of epochs there, then
run:

```bash
python train.py
```

The default configuration trains on a freshly generated 100-dimensional Syn5
dataset. Set `DataConfig.dataset_type` to any value from `"Syn1"` through
`"Syn6"` to use another synthetic mechanism.

Important configuration groups are:

- `DataConfig`: sample count, input dimension, dataset, batch size, and split;
- `ModelConfig`: input dimension, latent width, and number of classes;
- `TrainingConfig`: epochs, learning rates, loss weights, critic steps,
  gradient penalty, and Gumbel--Softmax temperature schedule.

## Outputs

Every execution creates an isolated directory named
`output/run_YYYYMMDD_HHMMSS_microseconds/` containing:

- `config.json`: the complete configuration snapshot;
- `terminal.log`: captured console output;
- `training_history.npz`: epoch-level losses and training metrics;
- `training_losses.png` and `retention_and_accuracy.png`: learning curves;
- `evaluation_masks.npz`: predicted and ground-truth test masks;
- `selected_feature_indices.npy`: features ranked by selection frequency; and
- `feature_mask_generator.pt`: trained mask-generator weights.

The terminal report includes masked-input classification accuracy, mean
per-sample true-positive rate (TPR), false-discovery rate (FDR), and the average
number of retained features.

Generated outputs are intentionally excluded from version control.

## Repository structure

```text
ALTAS/
|-- config.py         # Central experiment configuration
|-- data.py           # Syn1--Syn6 generation and data loaders
|-- models.py         # Generator, extractor, critic, and predictor
|-- masking.py        # Shuffle-replacement masking operation
|-- trainer.py        # Three-stage ALTAS optimization step
|-- training.py       # Epoch loop and metric aggregation
|-- evaluation.py     # Test metrics and feature report
|-- visualization.py  # Training-curve generation
|-- train.py          # End-to-end experiment entry point
`-- requirements.txt
```

## Notes on reproducibility

Synthetic samples are generated at runtime. To reproduce an exact run, set the
random seeds for PyTorch and NumPy before data generation and preserve the
automatically exported `config.json`. The current entry point does not fix a
seed by default.
