# Source Code (`src/`)

Main implementation of the SINDy (Sparse Identification of Nonlinear Dynamics) framework for modeling screw tip dynamics.

## Files

- **`main.py`**: Entry point for training SINDy models
- **`model.py`**: SINDy model implementation with sparse regression
- **`trainer.py`**: Training loop and optimization logic
- **`dataloader.py`**: Dataset handling and preprocessing

## Quick Usage

```bash
# Basic training
python src/main.py --data_dir ./data/data_1 --epochs 1000

# Advanced configuration
python src/main.py \
    --data_dir ./data/data_1 \
    --poly_order 3 \
    --use_sine true \
    --epochs 1000
```

## Model Features

- **Polynomial library**: Up to 3rd order polynomial features
- **Trigonometric functions**: Optional sine functions
- **Sparse coefficients**: Sequential thresholding for sparsity
- **Multi-dimensional output**: 2D state variable prediction