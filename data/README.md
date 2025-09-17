# Data Directory (`data/`)

Sample experimental data from robotic screw-driving experiments for SINDy modeling.

## Files

```
data/
├── sample_image.png           # Example camera image
└── data_1/                   # Sample experimental dataset
    ├── observation_data.csv   # Computer vision measurements
    └── sensor_data.csv        # Force/torque and robot state data
```

## Data Source

**Primary Dataset**: [Google Sheets](https://docs.google.com/spreadsheets/d/14IaxwbMclwKFS25-duvpaQAhQTR5hFq9RrTP6cjfS-Y/edit?usp=sharing)

Complete experimental dataset with multiple screw types, materials, and operating conditions.

## Data Format

### observation_data.csv
- **Pose data**: x, y, z positions and orientation quaternions
- **Timestamps**: Time-synchronized measurements  
- **Computer vision**: Extracted from camera images

### sensor_data.csv
- **Forces/torques**: fx, fy, fz, mx, my, mz measurements
- **Robot state**: Joint positions and velocities
- **System parameters**: Operating conditions and settings

## Usage

Data files are automatically loaded by the SINDy dataloader for model training and evaluation.