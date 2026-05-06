# FlashRetro: Efficient Retrosynthesis Prediction with Integral Flow Matching and Latent Inversion

**FlashRetro** is a state-of-the-art generative model for retrosynthesis prediction, aiming to predict reactants from a given product molecule. It introduces an **Integral Flow Matching (IFM)** framework with **Latent Inversion** to achieve fast and accurate one-step retrosynthesis generation.

Unlike conventional diffusion or flow-matching methods that rely on iterative ODE/SDE sampling, FlashRetro learns finite-interval integral transport directly in latent space. This training objective is aligned with a principled **1-NFE Latent Inversion** inference rule, which maps a noise latent conditioned on the product latent to the predicted reactant latent using a single learned integral-transport update.

By avoiding test-time numerical integration, FlashRetro significantly improves inference efficiency while maintaining strong retrosynthesis accuracy. Experiments on USPTO-50K show that FlashRetro achieves superior low-latency Top-1/Top-3 performance and provides substantial speedups over iterative diffusion- and flow-based retrosynthesis baselines.



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


