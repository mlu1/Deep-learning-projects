import os

# Model Configuration
class Config:
    # Data paths
    TRAIN_CSV = "Train.csv"
    TEST_CSV = "Test.csv"
    IMAGE_DIR = "data/"
    
    # Model hyperparameters
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # Training parameters
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
    # Bounds for coordinate system
    BOUNDS = (40600, 42600, 66500, 71000)
    
    # Prediction threshold
    PRED_THRESHOLD = 0.3
    
    # Output paths
    MODEL_CHECKPOINT = "best_model.h5"
    SUBMISSION_FILE = "submission.csv"
    TRAINING_HISTORY_PLOT = "training_history.png"
    VALIDATION_PREDICTIONS_PLOT = "validation_predictions.png"
    
    # Augmentation settings
    USE_AUGMENTATION = True
    
    # Early stopping patience
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 7
    
    def __init__(self):
        # Create necessary directories
        os.makedirs(os.path.dirname(self.MODEL_CHECKPOINT), exist_ok=True)
        
    def print_config(self):
        print("=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        for attr, value in self.__class__.__dict__.items():
            if not attr.startswith('__') and not callable(getattr(self, attr)):
                print(f"{attr}: {value}")
        print("=" * 50)
