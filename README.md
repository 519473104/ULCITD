# ULCITD

**Cross-Domain Lifelong Learning: Unsupervised Lightweight Class-Incremental Transfer Diagnosis of Machinery Faults in Industrial Data Streams**

## Overview

ULCITD is an unsupervised cross-domain class-incremental learning framework for machinery fault diagnosis. It addresses the challenge of continuously learning new fault classes from unlabeled target-domain data streams while preserving previously learned knowledge. 

## Project Structure

```
ULCITD/
├── main.py                  # Main entry point (multi-run experiments)
├── trainer.py               # Training loop and evaluation
├── IEEE_path.py             # Domain task configs for IEEE dataset
├── Tsinghua_path.py         # Domain task configs for Tsinghua dataset
├── configs/
│   └── foster_uda.json      # Hyperparameter configuration
├── models/
│   ├── foster_uda.py        # Core model (SL+UL + pruning + distillation)
│   ├── base.py              # Base learner class
│   └── pruning_diagnostics.py
├── convs/
│   └── compact_resnet.py    # Backbone network (CompactResNet8)
├── utils/
│   ├── data_manager.py      # Dataset management
│   ├── data_FD.py           # Fault diagnosis data loading
│   ├── inc_net.py           # Incremental network
│   ├── factory.py           # Model factory
│   └── toolkit.py           # Utility functions
├── diagnostics/             # Pruning diagnostics output
└── visualizations/          # t-SNE and confusion matrix output
```

## Quick Start

### 1. Requirements

- Python 3.8+
- PyTorch 1.8+
- numpy, pandas, tqdm, scikit-learn, matplotlib, seaborn

### 2. Dataset Preparation

Organize fault diagnosis data as image folders, with each subfolder representing one fault class:

```
<DATA_ROOT>/
├── Sample_A/          # Working condition A
│   ├── Class_0/       # Fault class 0 (images)
│   ├── Class_1/
│   └── ...
├── Sample_B/          # Working condition B
│   ├── Class_0/
│   └── ...
└── ...
```

### 3. Configure Domain Tasks

Edit `IEEE_path.py` or `Tsinghua_path.py` to define the source/target domain pairs and data paths:

```python
# Set data root directory
DATA_ROOT = r"path/to/your/dataset"

# Define cross-domain task pairs (source -> target)
TASK_CONFIGS = {
    1: {"train": "Sample_A", "test": "Sample_B"},  # A -> B
    2: {"train": "Sample_B", "test": "Sample_A"},  # B -> A
    # Add more tasks as needed
}

# Define fault class subfolder names
SUBFOLDER_NAMES = ['Class_0', 'Class_1', 'Class_2', ...]
```

Then select the desired dataset in `main.py`:

```python
from IEEE_path import setup_task, TASK_CONFIGS
# or
# from Tsinghua_path import setup_task, TASK_CONFIGS
```

### 4. Run

```bash
python main.py --config ./configs/foster_uda.json
```

This runs 5 repeated experiments (different seeds) x 6 cross-domain tasks, and outputs:

- `./diagnostics/results_summary.csv` -- Full results per run
- `./diagnostics/results_key_metrics.csv` -- Key metrics with Mean +/- Std

## Configuration

Key parameters in `configs/foster_uda.json`:

| Parameter | Description |
|-----------|-------------|
| `init_cls` / `increment` | Number of classes in the initial task / per incremental step |
| `init_epochs` / `boosting_epochs` | Training epochs for the initial / incremental tasks |
| `use_uda` | Enable unsupervised domain adaptation |
| `lambda_okd` / `lambda_fkd` | Weights for decision-level / feature-level knowledge distillation |
| `use_pruning` | Enable incremental weight-mask pruning |
| `prune_ratio` | Initial pruning intensity (rho_0) |
| `prune_ratio_decay` | Pruning decay factor (gamma), default 0.9 |
| `min_prune_ratio` | Minimum pruning intensity floor |
| `convnet_type` | Backbone architecture (`compact_resnet8`) |
| `batch_size` | Training batch size |

## Supported Datasets

| Dataset  | Description | Classes | Path Config |
|----------|-------------|---------|-------------|
| IEEE2024 | Bearing fault vibration signals (wavelet transform images) | 10 | `IEEE_path.py` |
| Tsinghua | Gearbox fault vibration signals (wavelet transform images) | 10 | `Tsinghua_path.py` |

## License

See [LICENSE](LICENSE) for details.
