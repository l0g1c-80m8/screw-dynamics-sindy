# Scripts (`scripts/`)

Utility scripts for data collection, image processing, and experimental automation.

## Files

- **`screwtip_pose_detection.py`**: Computer vision-based screw tip pose estimation
- **`screwtip_feature_tracking.py`**: Feature tracking across image sequences  
- **`detect_color_boundaries.py`**: Interactive HSV color range detection
- **`experiment.py`**: ROS-based experimental automation and data recording
- **`camera_recorder.cpp`**: High-performance camera data recording service

## Quick Usage

```bash
# Extract screw tip poses from images
python scripts/screwtip_pose_detection.py \
    --data_dir ./raw_data \
    --out_dir ./processed_data

# Track features across image sequences
python scripts/screwtip_feature_tracking.py \
    --input_dir ./images \
    --output_file ./features.csv

# Run automated experiment
python scripts/experiment.py --config experiment_config.yaml
```

## Script Features

- **Computer vision**: HSV-based color detection and contour analysis
- **ROS integration**: Robot control and data synchronization
- **Real-time processing**: Live camera feed analysis
- **Data export**: CSV format for SINDy model training