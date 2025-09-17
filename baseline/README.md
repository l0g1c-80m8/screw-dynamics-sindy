# Baseline Models (`baseline/`)

Baseline model implementations for comparison with the SINDy approach.

## Files

- **`model.py`**: Neural network model implementations (LSTM, MLP)
- **`train.py`**: Training and evaluation scripts for baseline models

## Available Models

- **`LSTMModel`**: LSTM-based sequential model using GRU cells
- **`MLP`**: Multi-Layer Perceptron for direct regression

## Quick Usage

```bash
# Train LSTM model
python baseline/train.py \
    --model_type lstm \
    --data_dir ./data/data_1 \
    --epochs 1000

# Train MLP model
python baseline/train.py \
    --model_type mlp \
    --data_dir ./data/data_1 \
    --epochs 1000
```

## Model Features

- **LSTM**: Sequential modeling with dropout regularization
- **MLP**: Feedforward network with ReLU activations
- **Comparison baseline**: Traditional ML approaches vs SINDy