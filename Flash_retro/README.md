# FlashRetro: Efficient Retrosynthesis Prediction with Latent-Space MeanFlow Matching and Inversion

**FlashRetro** is a state-of-the-art generative model for retrosynthesis prediction (predicting reactants from products). It builds upon flow-matching and introduces two key mechanisms—**Latent MeanFlow Objective** and **Latent Velocity Inversion**—to achieve superior accuracy and efficiency.



## Project Structure

```
Flash_retro/
├── configs/            # Configuration files (Hydra)
│   └── difm_retro_uspto50k.yaml
├── data/               # Dataset directory
│   └── uspto50k/       # USPTO-50k dataset files
├── flashretro/            # Core source code
│   ├── chem/           # Chemistry utilities (RDKit wrappers, reaction handling)
│   ├── data/           # Data loading (DeltaGraphDataset)
│   ├── models/         # Model definitions (Transformers)
│   └── utils/          # Metrics, evaluation helpers, and training utilities
├── train.py            # Main training entry point
├── test.py             # Main testing entry point
└── requirements.txt    # Project dependencies
```

## Getting Started

### Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- RDKit >= 2023.9.0
- Other dependencies listed in `requirements.txt`

### Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training a Model

Use `train.py` with Hydra configuration. The default configuration file is in `configs/`.

```bash
python train.py --config-name difm_retro_uspto50k
```

You can override parameters directly from the command line:

```bash
python train.py --config-name difm_retro_uspto50k \
    train.batch_size=32 \
    train.devices=1 \
    train.max_iters=100000
```

#### Testing a Model

Use `test.py` to evaluate a trained checkpoint. The model supports fast 1-step sampling.

```bash
python test.py \
    --ckpt_path epoch=108.ckpt \
    --config_path configs/difm_retro_uspto50k.yaml \
    --device cuda \
    --batch_size 4
```

### Key Parameters

Configurations are managed via Hydra (see `configs/difm_retro_uspto50k.yaml`). Key parameters include:

- `train.mode`: Task mode (e.g., 'retro').
- `train.batch_size`: Batch size per GPU.
- `train.lr`: Learning rate.
- `train.max_iters`: Total training iterations.
- `train.devices`: Number of GPUs to use.
- `model.name`: Model architecture (e.g., `vaedifm_uniform`).
- `chem.reaction_data_dir`: Path to the reaction dataset.

## Evaluation Metrics

The model is evaluated using standard retrosynthesis metrics:

1.  **Top-k Accuracy**: Accuracy of the ground truth reactant appearing in the top-k generated candidates.
2.  **Validity**: Percentage of chemically valid generated molecules.


## Logs

Training and testing logs, including checkpoints and configs, will be saved in the `logs/` directory.


