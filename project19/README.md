# Barbados Lands and Surveys Plot Automation Challenge

This repository contains a solution for the Barbados Lands and Surveys Plot Automation Challenge hosted on [Zindi Africa](https://zindi.africa/competitions/barbados-lands-and-surveys-plot-automation-challenge).

## Problem Description

The challenge involves automated plot boundary detection and segmentation from aerial/satellite imagery. The goal is to identify and delineate land plot boundaries to assist in land surveying and management.

## Solution Overview

This solution uses a **U-Net deep learning architecture** for semantic segmentation:

- **Input**: RGB aerial/satellite images
- **Output**: Binary masks representing plot boundaries
- **Architecture**: Enhanced U-Net with batch normalization and dropout
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy, Dice coefficient, IoU

## Features

- ✅ **Data Augmentation**: Random flips, rotations, brightness/contrast adjustments
- ✅ **Enhanced U-Net**: Batch normalization, dropout, optimized architecture
- ✅ **Advanced Metrics**: Dice coefficient, IoU tracking during training
- ✅ **Early Stopping**: Prevents overfitting
- ✅ **Threshold Optimization**: Find optimal prediction threshold
- ✅ **Comprehensive Evaluation**: Detailed metrics and visualizations
- ✅ **Modular Code**: Clean, organized, and configurable

## File Structure

```
├── m1.py                    # Main training and inference script
├── config.py               # Configuration parameters
├── utils.py                # Utility functions for evaluation
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── Train.csv              # Training metadata (not included)
├── Test.csv               # Test IDs (not included)
└── data/                  # Directory containing images (not included)
    ├── *.jpg              # Aerial/satellite images
    └── ...
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**:
   - Place `Train.csv` and `Test.csv` in the root directory
   - Create a `data/` directory and place all image files there
   - Ensure image filenames match the IDs in the CSV files

3. **Configure parameters** (optional):
   - Edit `config.py` to adjust hyperparameters, paths, and settings

## Usage

### Training and Prediction

Run the main script:
```bash
python m1.py
```

This will:
1. Load and preprocess the data
2. Train the U-Net model with data augmentation
3. Generate training history plots
4. Evaluate the model on validation data
5. Create a submission file for the test set

### Advanced Usage

#### Find Optimal Threshold
```python
from utils import find_optimal_threshold
optimal_threshold, results = find_optimal_threshold(model, val_ds)
```

#### Detailed Evaluation
```python
from utils import evaluate_model_on_dataset
metrics = evaluate_model_on_dataset(model, val_ds, threshold=0.3)
```

#### Custom Configuration
```python
from config import Config
config = Config()
config.LEARNING_RATE = 5e-5
config.BATCH_SIZE = 16
```

## Model Architecture

The solution uses an enhanced U-Net architecture:

- **Encoder**: 5 levels (64, 128, 256, 512, 1024 filters)
- **Decoder**: 4 levels with skip connections
- **Enhancements**: 
  - Batch normalization after each convolution
  - Dropout for regularization (increasing with depth)
  - Adam optimizer with learning rate scheduling

## Training Strategy

- **Data Split**: 80% training, 20% validation
- **Augmentation**: Flips, rotations, brightness/contrast adjustments
- **Callbacks**: 
  - Model checkpointing (saves best model based on validation Dice score)
  - Learning rate reduction on plateau
  - Early stopping to prevent overfitting
- **Metrics**: Accuracy, Dice coefficient, IoU

## Output Files

After training, the following files are generated:

- `best_model.h5`: Best model weights
- `submission.csv`: Competition submission file
- `training_history.png`: Training curves visualization
- `validation_predictions.png`: Sample predictions visualization
- `threshold_optimization.png`: Threshold analysis (if run)

## Key Improvements Over Baseline

1. **Enhanced Architecture**: Added batch normalization and dropout
2. **Better Training**: Advanced callbacks and metrics tracking
3. **Data Augmentation**: Improves model generalization
4. **Evaluation Tools**: Comprehensive analysis utilities
5. **Modular Design**: Easy to modify and extend
6. **Error Handling**: Robust file checking and validation

## Tips for Better Performance

1. **Increase Image Resolution**: Use larger input sizes if GPU memory allows
2. **Ensemble Methods**: Train multiple models and average predictions
3. **Post-processing**: Apply morphological operations to clean predictions
4. **Advanced Architectures**: Try DeepLab, FPN, or attention mechanisms
5. **Domain-specific Augmentations**: Add geometric transformations relevant to aerial imagery

## Competition Submission

The final submission file (`submission.csv`) contains:
- ID: Plot identifier
- TargetSurvey, Certified date, Total Area, etc.: Metadata fields
- geometry: Predicted polygon coordinates

## License

This project is created for the Zindi Africa competition. Please refer to the competition rules for usage guidelines.

## Acknowledgments

- Zindi Africa for hosting the competition
- The TensorFlow/Keras team for the deep learning framework
- The open-source community for the various Python libraries used
