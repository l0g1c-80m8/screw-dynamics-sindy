# Screw Dynamics SINDy

A computational framework for modeling robotic screw-driving dynamics using **Sparse Identification of Nonlinear Dynamics (SINDy)**.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Access](#data-access)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project develops interpretable dynamical models for robotic screw-driving operations using the SINDy framework. The approach combines computer vision-based feature extraction with sparse regression to identify governing equations that describe screw tip dynamics.

### Key Features

- **SINDy modeling**: Sparse identification of nonlinear dynamics
- **Computer vision**: Feature tracking and pose detection
- **Baseline models**: LSTM and MLP for performance comparison
- **Data processing**: Collection and preprocessing utilities
- **Analysis tools**: Jupyter notebooks for visualization

## Quick Start

```bash
# 1. Setup environment
source activate_env.sh

# 2. Train SINDy model
python src/main.py --data_dir ./data/data_1 --epochs 1000

# 3. Run baseline comparison
python baseline/train.py --model_type lstm --data_dir ./data/data_1

# 4. Open analysis notebooks
jupyter lab notebook/
```

## Repository Structure

```
├── src/                    # SINDy implementation → [README](src/README.md)
├── baseline/               # Baseline models (LSTM, MLP) → [README](baseline/README.md)
├── scripts/                # Data processing utilities → [README](scripts/README.md)
├── notebook/               # Analysis notebooks → [README](notebook/README.md)
├── data/                   # Sample data → [README](data/README.md)
├── archived/               # Previous implementations
├── .github/                # CI/CD workflows
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation
└── LICENSE                # MIT License
```

## Installation

## Installation

### Prerequisites
- Python 3.8+
- PyTorch, OpenCV, scikit-learn, pandas, numpy, matplotlib, jupyter

### Setup

1. Clone the repository:
```bash
git clone https://github.com/l0g1c-80m8/screw-dynamics-sindy.git
cd screw-dynamics-sindy
```

2. Setup environment:
```bash
source activate_env.sh
```

## Usage

### Training Models
```bash
# SINDy model
python src/main.py --data_dir ./data/data_1 --epochs 1000

# Baseline models  
python baseline/train.py --model_type lstm --data_dir ./data/data_1
```

### Data Processing
```bash
# Extract screw tip poses
python scripts/screwtip_pose_detection.py --data_dir ./raw_data
```

### Analysis
```bash
# Open Jupyter notebooks
jupyter lab notebook/
```

## Data Access

Experimental data: [Google Sheets Dataset](https://docs.google.com/spreadsheets/d/14IaxwbMclwKFS25-duvpaQAhQTR5hFq9RrTP6cjfS-Y/edit?usp=sharing)

- **Observation data**: Screw tip pose and position tracking
- **Sensor data**: Force/torque measurements and system states  
- **Image sequences**: Visual data for feature extraction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
  