import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def calculate_detailed_metrics(y_true, y_pred, threshold=0.5):
    """Calculate detailed metrics for binary segmentation"""
    y_pred_binary = (y_pred > threshold).astype(np.uint8)
    y_true_binary = y_true.astype(np.uint8)
    
    # Flatten arrays
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
    iou = intersection / union if union > 0 else 0
    
    # Dice coefficient
    dice = (2 * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat)) if (np.sum(y_true_flat) + np.sum(y_pred_flat)) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'dice': dice,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(y_true, y_pred, threshold=0.5, save_path=None):
    """Plot confusion matrix"""
    y_pred_binary = (y_pred > threshold).astype(np.uint8)
    y_true_binary = y_true.astype(np.uint8)
    
    cm = confusion_matrix(y_true_binary.flatten(), y_pred_binary.flatten())
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model_on_dataset(model, dataset, threshold=0.5):
    """Evaluate model on entire dataset"""
    all_true = []
    all_pred = []
    
    print("Evaluating model on dataset...")
    for batch_idx, (imgs, masks) in enumerate(dataset):
        preds = model.predict(imgs, verbose=0)
        all_true.append(masks.numpy())
        all_pred.append(preds)
        
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}")
    
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    
    metrics = calculate_detailed_metrics(y_true, y_pred, threshold)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric.upper()}: {value:.4f}")
    print("="*50)
    
    return metrics

def find_optimal_threshold(model, val_dataset, thresholds=None):
    """Find optimal threshold based on validation set"""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    print("Finding optimal threshold...")
    
    # Get predictions once
    all_true = []
    all_pred = []
    
    for imgs, masks in val_dataset:
        preds = model.predict(imgs, verbose=0)
        all_true.append(masks.numpy())
        all_pred.append(preds)
    
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    
    results = []
    for threshold in thresholds:
        metrics = calculate_detailed_metrics(y_true, y_pred, threshold)
        results.append({
            'threshold': threshold,
            'dice': metrics['dice'],
            'iou': metrics['iou'],
            'f1': metrics['f1_score']
        })
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot([r['threshold'] for r in results], [r['dice'] for r in results], 'o-')
    plt.title('Dice Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Dice Score')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot([r['threshold'] for r in results], [r['iou'] for r in results], 'o-')
    plt.title('IoU vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot([r['threshold'] for r in results], [r['f1'] for r in results], 'o-')
    plt.title('F1 Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best threshold based on dice score
    best_result = max(results, key=lambda x: x['dice'])
    print(f"\nOptimal threshold: {best_result['threshold']:.3f}")
    print(f"Best Dice score: {best_result['dice']:.4f}")
    
    return best_result['threshold'], results
