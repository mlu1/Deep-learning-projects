# Barbados Plot Automation - Complete Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import os
import glob
import warnings
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# Optional: raster <-> polygon via scikit-image
try:
    from skimage.draw import polygon as draw_polygon
    from skimage import measure
except ImportError:
    raise ImportError("scikit-image is required: pip install scikit-image")

# Add error handling for required files
def check_required_files():
    required_files = ["Train.csv", "Test.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    if not os.path.exists("data/"):
        raise FileNotFoundError("Missing 'data/' directory with images")
    
    print("✓ All required files found")

check_required_files()

# --- Step 1: Load and preprocess metadata ---
train_csv = "Train.csv"
df = pd.read_csv(train_csv)
df['geometry'] = df['geometry'].apply(wkt.loads)

# Convert 3D polygons to 2D coords
def to_2d(geom):
    if isinstance(geom, Polygon):
        return [(x, y) for x, y, *_ in geom.exterior.coords]
    if isinstance(geom, MultiPolygon):
        pts = []
        for p in geom.geoms:
            pts.extend([(x, y) for x, y, *_ in p.exterior.coords])
        return pts
    return []

df['polygon_coords'] = df['geometry'].apply(to_2d)

# Load blind-test IDs
test_ids = pd.read_csv("Test.csv")
test_set = set(test_ids['ID'].astype(str))

# Exclude test IDs from training pool
train_val = df[~df['ID'].astype(str).isin(test_set)].reset_index(drop=True)
train_df, val_df = train_test_split(train_val, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_ids)}")

# --- Step 2: Constants ---
BOUNDS = (40600, 42600, 66500, 71000)   # Update to match full extent
IMG_SIZE = (512, 512)
BATCH = 8
IMAGE_DIR = "data/"  # Adjust as needed

# --- Step 3: Utilities ---
def rasterize_polygon(coords, size=IMG_SIZE, bounds=BOUNDS):
    mask = np.zeros(size, np.uint8)
    if not coords: return mask
    minx, maxx, miny, maxy = bounds
    xs = [(x-minx)/(maxx-minx)*(size[1]-1) for x,y in coords]
    ys = [(1-(y-miny)/(maxy-miny))*(size[0]-1) for x,y in coords]
    rr, cc = draw_polygon(ys, xs, mask.shape)
    mask[rr, cc] = 1
    return mask


def load_img(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img, np.float32)/255.0

# Resolve image path for a given ID
def resolve_img(pid):
    for pattern in [f"{pid}.jpg", f"anonymised_{pid}.jpg"]:
        p = os.path.join(IMAGE_DIR, pattern)
        if os.path.exists(p): return p
    hits = glob.glob(os.path.join(IMAGE_DIR, f"*{pid}*.jpg"))
    return hits[0] if hits else None

# --- Step 4: tf.data pipelines with augmentation ---
def augment_data(img, mask):
    """Apply random augmentations to improve model generalization"""
    # Random horizontal flip
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    
    # Random vertical flip
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    
    # Random rotation (90, 180, 270 degrees)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    mask = tf.image.rot90(mask, k)
    
    # Random brightness and contrast
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, mask

def dataset_generator(df_split):
    for _, row in df_split.iterrows():
        img_path = resolve_img(row['ID'])
        if not img_path:
            warnings.warn(f"Missing image for {row['ID']}")
            continue
        img = load_img(img_path)
        mask = rasterize_polygon(row['polygon_coords'])[..., None]
        yield img, mask


def make_ds(df_split, augment=True):
    ds = tf.data.Dataset.from_generator(
        lambda: dataset_generator(df_split),
        output_signature=(
            tf.TensorSpec((*IMG_SIZE,3), tf.float32),
            tf.TensorSpec((*IMG_SIZE,1), tf.float32)
        ))
    if augment:
        ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(500).batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_df, augment=True)
val_ds   = make_ds(val_df, augment=False)  # No augmentation for validation

# --- Step 5: Enhanced U-Net model ---
def build_unet():
    inp = layers.Input((*IMG_SIZE,3))
    
    def conv_block(x, f, dropout_rate=0.1):
        x = layers.Conv2D(f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    def enc(x, f, dropout_rate=0.1): 
        c = conv_block(x, f, dropout_rate)
        return c, layers.MaxPooling2D()(c)
    
    def dec(x, s, f, dropout_rate=0.1): 
        x = layers.Conv2DTranspose(f, 2, 2, padding='same')(x)
        x = layers.Concatenate()([x, s])
        return conv_block(x, f, dropout_rate)
    
    # Encoder
    c1, p1 = enc(inp, 64, 0.1)
    c2, p2 = enc(p1, 128, 0.1)
    c3, p3 = enc(p2, 256, 0.2)
    c4, p4 = enc(p3, 512, 0.2)
    
    # Bottleneck
    bn = conv_block(p4, 1024, 0.3)
    
    # Decoder
    u1 = dec(bn, c4, 512, 0.2)
    u2 = dec(u1, c3, 256, 0.2)
    u3 = dec(u2, c2, 128, 0.1)
    u4 = dec(u3, c1, 64, 0.1)
    
    out = layers.Conv2D(1, 1, activation='sigmoid')(u4)
    return models.Model(inp, out)

model = build_unet()

# Custom metrics for better monitoring
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', dice_coefficient, iou_metric]
)
model.summary()

# --- Step 6: Enhanced Training ---
callbacks_list = [
    callbacks.ModelCheckpoint(
        'best_model.h5', 
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=7,
        factor=0.5,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
]

print("Starting training...")
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=50,  # Increased epochs with early stopping
    callbacks=callbacks_list,
    verbose=1
)

# Load best weights
model.load_weights('best_model.h5')
print("✓ Training completed and best weights loaded")

# --- Step 7: Sanity check prediction ---
for imgs, masks in val_ds.take(1):
    preds = model.predict(imgs)
    binp = (preds>0.3).astype(np.uint8)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(imgs[0]); plt.title('Input')
    plt.subplot(1,3,2); plt.imshow(masks[0,...,0],cmap='gray'); plt.title('GT')
    plt.subplot(1,3,3); plt.imshow(binp[0,...,0],cmap='gray'); plt.title('Pred')
    plt.show()
    break

# --- Step 8: Enhanced Inference utils ---
def predict_mask(pid, threshold=0.3):
    """Predict mask for a given plot ID with error handling"""
    try:
        img_path = resolve_img(pid)
        if img_path is None:
            print(f"Warning: No image found for ID {pid}")
            return np.zeros(IMG_SIZE, dtype=np.uint8)
        
        img = load_img(img_path)
        pred = model.predict(img[None], verbose=0)[0, ..., 0]
        return (pred > threshold).astype(np.uint8)
    except Exception as e:
        print(f"Error predicting mask for ID {pid}: {e}")
        return np.zeros(IMG_SIZE, dtype=np.uint8)

def mask_to_polygons(mask):
    """Convert binary mask to polygon coordinates"""
    try:
        contours = measure.find_contours(mask, 0.5)
        polygons = []
        h, w = mask.shape
        minx, maxx, miny, maxy = BOUNDS
        
        for contour in contours:
            # Convert contour coordinates back to original coordinate system
            coords = []
            for r, c in contour:
                x = c / (w - 1) * (maxx - minx) + minx
                y = (1 - r / (h - 1)) * (maxy - miny) + miny
                coords.append([x, y])  # Note: using lists instead of tuples for JSON serialization
            
            # Only keep polygons with enough points and reasonable area
            if len(coords) > 4:  # At least 5 points to form a meaningful polygon
                polygons.append(coords)
        
        return polygons
    except Exception as e:
        print(f"Error converting mask to polygons: {e}")
        return []

def clean_polygon_coords(coords_list):
    """Clean and format polygon coordinates for submission"""
    if not coords_list:
        return []
    
    # Take the largest polygon by number of points (usually the main plot boundary)
    largest_polygon = max(coords_list, key=len)
    
    # Simplify if too many points (optional - helps with file size)
    if len(largest_polygon) > 100:
        # Simple decimation - take every nth point
        step = len(largest_polygon) // 50
        largest_polygon = largest_polygon[::max(1, step)]
    
    return largest_polygon

# --- Step 9: Enhanced Submission Generation ---
def build_test_submission(test_ids_df, output='submission.csv', threshold=0.3):
    """Build submission file with enhanced error handling and validation"""
    print(f"Generating predictions for {len(test_ids_df)} test samples...")
    rows = []
    successful_predictions = 0
    failed_predictions = 0
    
    for idx, pid in enumerate(test_ids_df['ID']):
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(test_ids_df)}...")
        
        try:
            # Convert to string for consistent lookup
            pid_str = str(pid)
            
            # Look up metadata - be more careful with data types
            meta = None
            if pid_str in df['ID'].astype(str).values:
                meta = df[df['ID'].astype(str) == pid_str].iloc[0]
            
            # Extract metadata with defaults
            target = str(meta['Surveyed For']) if meta is not None and pd.notna(meta.get('Surveyed For')) else ''
            cert = str(meta['Certified date']) if meta is not None and pd.notna(meta.get('Certified date')) else ''
            area = str(meta['Total Area']) if meta is not None and pd.notna(meta.get('Total Area')) else ''
            unit = str(meta['Unit of Measurement']) if meta is not None and pd.notna(meta.get('Unit of Measurement')) else ''
            parish = str(meta['Parish']) if meta is not None and pd.notna(meta.get('Parish')) else ''
            lt = str(meta['LT Num']) if meta is not None and pd.notna(meta.get('LT Num')) else ''
            
            # Generate prediction
            mask = predict_mask(pid_str, threshold)
            polygons = mask_to_polygons(mask)
            
            # Clean and select best polygon
            geometry = clean_polygon_coords(polygons)
            
            # Validate prediction
            if len(geometry) == 0:
                print(f"Warning: No valid polygon found for ID {pid_str}")
                failed_predictions += 1
            else:
                successful_predictions += 1
            
            rows.append({
                'ID': pid_str,
                'TargetSurvey': target,
                'Certified date': cert,
                'Total Area': area,
                'Unit of Measurement': unit,
                'Parish': parish,
                'LT Num': lt,
                'geometry': geometry
            })
            
        except Exception as e:
            print(f"Error processing ID {pid}: {e}")
            failed_predictions += 1
            # Add empty row to maintain consistency
            rows.append({
                'ID': str(pid),
                'TargetSurvey': '',
                'Certified date': '',
                'Total Area': '',
                'Unit of Measurement': '',
                'Parish': '',
                'LT Num': '',
                'geometry': []
            })
    
    # Create and save submission DataFrame
    submission_df = pd.DataFrame(rows)
    column_order = ['ID', 'TargetSurvey', 'Certified date', 'Total Area',
                   'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
    submission_df = submission_df[column_order]
    
    # Save to CSV
    submission_df.to_csv(output, index=False)
    
    print(f"\n{'='*50}")
    print(f"SUBMISSION SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {len(test_ids_df)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Success rate: {successful_predictions/len(test_ids_df)*100:.1f}%")
    print(f"Saved to: {output}")
    print(f"{'='*50}")
    
    return submission_df

# Execute submission build with hybrid processing
print("\n" + "="*60)
print("GENERATING TEST PREDICTIONS - HYBRID APPROACH")
print("="*60)

# Try to detect data type and use appropriate processing
try:
    from hybrid_processor import create_adaptive_submission
    print("Using adaptive processing to handle mixed data types...")
    submission_df = create_adaptive_submission(test_ids, image_dir=IMAGE_DIR, model=model)
except ImportError:
    print("Hybrid processor not available, using original approach...")
    submission_df = build_test_submission(test_ids, threshold=0.3)

# Plot training history
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0,0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0,0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0,0].set_title('Model Accuracy')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    
    # Loss
    axes[0,1].plot(history.history['loss'], label='Train Loss')
    axes[0,1].plot(history.history['val_loss'], label='Val Loss')
    axes[0,1].set_title('Model Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    
    # Dice Coefficient
    axes[1,0].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[1,0].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[1,0].set_title('Dice Coefficient')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Dice')
    axes[1,0].legend()
    
    # IoU
    axes[1,1].plot(history.history['iou_metric'], label='Train IoU')
    axes[1,1].plot(history.history['val_iou_metric'], label='Val IoU')
    axes[1,1].set_title('IoU Metric')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('IoU')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history)

# --- Step 7: Enhanced Sanity check prediction ---
for imgs, masks in val_ds.take(1):
    preds = model.predict(imgs)
    
    # Show multiple examples
    n_samples = min(4, imgs.shape[0])
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    for i in range(n_samples):
        if n_samples == 1:
            axes = axes.reshape(1, -1)
            
        # Original image
        axes[i,0].imshow(imgs[i])
        axes[i,0].set_title('Input Image')
        axes[i,0].axis('off')
        
        # Ground truth mask
        axes[i,1].imshow(masks[i,...,0], cmap='gray')
        axes[i,1].set_title('Ground Truth')
        axes[i,1].axis('off')
        
        # Raw prediction
        axes[i,2].imshow(preds[i,...,0], cmap='gray')
        axes[i,2].set_title('Raw Prediction')
        axes[i,2].axis('off')
        
        # Thresholded prediction
        binp = (preds[i,...,0] > 0.3).astype(np.uint8)
        axes[i,3].imshow(binp, cmap='gray')
        axes[i,3].set_title('Thresholded (>0.3)')
        axes[i,3].axis('off')
        
        # Calculate metrics for this sample
        dice = dice_coefficient(masks[i:i+1], preds[i:i+1]).numpy()
        iou = iou_metric(masks[i:i+1], preds[i:i+1]).numpy()
        axes[i,0].text(0.02, 0.98, f'Dice: {dice:.3f}\nIoU: {iou:.3f}', 
                      transform=axes[i,0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('validation_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    break

# --- Step 10: Visualization function for debugging test predictions ---
def visualize_test_predictions(test_ids_df, n_samples=6, threshold=0.3):
    """Visualize test predictions for debugging"""
    print(f"Visualizing {n_samples} test predictions...")
    
    # Select random test samples
    sample_ids = test_ids_df['ID'].sample(n=min(n_samples, len(test_ids_df))).values
    
    fig, axes = plt.subplots(2, n_samples//2, figsize=(4*n_samples//2, 8))
    if n_samples <= 2:
        axes = axes.reshape(-1, 1)
    
    for idx, pid in enumerate(sample_ids):
        row = idx // (n_samples//2)
        col = idx % (n_samples//2)
        
        try:
            # Get image and prediction
            img_path = resolve_img(str(pid))
            if img_path is None:
                axes[row, col].text(0.5, 0.5, f'No image\nfor ID {pid}', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'ID: {pid} (Missing)')
                axes[row, col].axis('off')
                continue
            
            img = load_img(img_path)
            mask = predict_mask(str(pid), threshold)
            polygons = mask_to_polygons(mask)
            
            # Create overlay
            overlay = img.copy()
            if mask.max() > 0:
                # Add red overlay where mask is predicted
                mask_colored = np.zeros_like(img)
                mask_colored[:, :, 0] = mask  # Red channel
                overlay = np.where(mask[..., None] > 0, 
                                 0.7 * img + 0.3 * mask_colored, img)
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'ID: {pid}\n{len(polygons)} polygons found')
            axes[row, col].axis('off')
            
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'ID: {pid} (Error)')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions_debug.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize some test predictions for debugging
visualize_test_predictions(test_ids, n_samples=6)

# --- Step 11: Comprehensive submission validation ---
def validate_submission(submission_df, test_ids_df):
    """Validate submission format and content"""
    print("\n" + "="*50)
    print("SUBMISSION VALIDATION")
    print("="*50)
    
    # Check basic format
    required_columns = ['ID', 'TargetSurvey', 'Certified date', 'Total Area',
                       'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
    
    missing_cols = [col for col in required_columns if col not in submission_df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    else:
        print("✅ All required columns present")
    
    # Check row count
    if len(submission_df) != len(test_ids_df):
        print(f"❌ Row count mismatch: {len(submission_df)} vs {len(test_ids_df)} expected")
        return False
    else:
        print(f"✅ Correct number of rows: {len(submission_df)}")
    
    # Check for missing IDs
    submission_ids = set(submission_df['ID'].astype(str))
    test_ids_set = set(test_ids_df['ID'].astype(str))
    missing_ids = test_ids_set - submission_ids
    if missing_ids:
        print(f"❌ Missing IDs: {list(missing_ids)[:5]}{'...' if len(missing_ids) > 5 else ''}")
        return False
    else:
        print("✅ All test IDs present")
    
    # Check geometry format
    valid_geometries = 0
    empty_geometries = 0
    invalid_geometries = 0
    
    for idx, geom in enumerate(submission_df['geometry']):
        if isinstance(geom, list) and len(geom) > 0:
            # Check if it's a valid list of coordinates
            try:
                if all(isinstance(point, list) and len(point) == 2 
                       and all(isinstance(coord, (int, float)) for coord in point) 
                       for point in geom):
                    valid_geometries += 1
                else:
                    invalid_geometries += 1
            except:
                invalid_geometries += 1
        elif isinstance(geom, list) and len(geom) == 0:
            empty_geometries += 1
        else:
            invalid_geometries += 1
    
    print(f"📊 Geometry statistics:")
    print(f"   Valid geometries: {valid_geometries}")
    print(f"   Empty geometries: {empty_geometries}")
    print(f"   Invalid geometries: {invalid_geometries}")
    
    # Calculate statistics
    total_predictions = len(submission_df)
    success_rate = valid_geometries / total_predictions * 100
    
    print(f"\n📈 Overall statistics:")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Empty predictions: {empty_geometries/total_predictions*100:.1f}%")
    print(f"   Invalid predictions: {invalid_geometries/total_predictions*100:.1f}%")
    
    if success_rate >= 80:
        print("\n✅ Submission looks good!")
        return True
    elif success_rate >= 50:
        print("\n⚠️  Submission has issues but might be acceptable")
        return True
    else:
        print("\n❌ Submission has serious issues")
        return False

# Validate the submission
validation_result = validate_submission(submission_df, test_ids)

# Import debug utilities for additional analysis
try:
    from debug_utils import analyze_submission_geometry, debug_failed_predictions, compare_train_test_distributions
    
    # Debug failed predictions
    debug_failed_predictions(test_ids, IMAGE_DIR)
    
    # Analyze submission geometry
    analyze_submission_geometry(submission_df)
    
    # Compare train/test distributions
    compare_train_test_distributions(df, test_ids)
    
except ImportError:
    print("Debug utilities not available - skipping detailed analysis")

print("="*60)

# --- Quick prediction check function ---
def quick_prediction_check(test_ids_df, n_check=3):
    """Quick manual check of predictions"""
    print(f"\nQuick prediction check for {n_check} samples:")
    print("-" * 40)
    
    sample_ids = test_ids_df['ID'].sample(n=min(n_check, len(test_ids_df)))
    
    for pid in sample_ids:
        try:
            img_path = resolve_img(str(pid))
            if img_path:
                mask = predict_mask(str(pid))
                polygons = mask_to_polygons(mask)
                clean_geom = clean_polygon_coords(polygons)
                print(f"ID {pid}: Found {len(polygons)} polygons, final coords: {len(clean_geom)} points")
            else:
                print(f"ID {pid}: No image found")
        except Exception as e:
            print(f"ID {pid}: Error - {e}")

# Quick manual check
quick_prediction_check(test_ids, n_check=5)

# Final cleanup and summary
print("\n" + "="*60)
print("PROCESSING COMPLETE")
print("="*60)
print("Files generated:")
print("  - submission.csv (or submission_documents.csv)")
print("  - training_history.png")
print("  - validation_predictions.png")
print("  - test_predictions_debug.png")
print("="*60)

