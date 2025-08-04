"""
Debug utilities for Barbados Plot Automation Challenge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import json

def analyze_submission_geometry(submission_df, save_stats=True):
    """Analyze geometry statistics in submission"""
    stats = {
        'total_samples': len(submission_df),
        'valid_geometries': 0,
        'empty_geometries': 0,
        'invalid_geometries': 0,
        'polygon_sizes': [],
        'coordinate_ranges': {'x_min': [], 'x_max': [], 'y_min': [], 'y_max': []}
    }
    
    for idx, geom in enumerate(submission_df['geometry']):
        if isinstance(geom, list) and len(geom) > 0:
            try:
                # Convert to numpy array for analysis
                coords = np.array(geom)
                if coords.shape[1] == 2:  # Valid 2D coordinates
                    stats['valid_geometries'] += 1
                    stats['polygon_sizes'].append(len(geom))
                    
                    # Track coordinate ranges
                    stats['coordinate_ranges']['x_min'].append(coords[:, 0].min())
                    stats['coordinate_ranges']['x_max'].append(coords[:, 0].max())
                    stats['coordinate_ranges']['y_min'].append(coords[:, 1].min())
                    stats['coordinate_ranges']['y_max'].append(coords[:, 1].max())
                else:
                    stats['invalid_geometries'] += 1
            except:
                stats['invalid_geometries'] += 1
        elif isinstance(geom, list) and len(geom) == 0:
            stats['empty_geometries'] += 1
        else:
            stats['invalid_geometries'] += 1
    
    # Print analysis
    print("="*50)
    print("GEOMETRY ANALYSIS")
    print("="*50)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Valid geometries: {stats['valid_geometries']} ({stats['valid_geometries']/stats['total_samples']*100:.1f}%)")
    print(f"Empty geometries: {stats['empty_geometries']} ({stats['empty_geometries']/stats['total_samples']*100:.1f}%)")
    print(f"Invalid geometries: {stats['invalid_geometries']} ({stats['invalid_geometries']/stats['total_samples']*100:.1f}%)")
    
    if stats['polygon_sizes']:
        print(f"\nPolygon size statistics:")
        print(f"  Min points: {min(stats['polygon_sizes'])}")
        print(f"  Max points: {max(stats['polygon_sizes'])}")
        print(f"  Mean points: {np.mean(stats['polygon_sizes']):.1f}")
        print(f"  Median points: {np.median(stats['polygon_sizes']):.1f}")
        
        # Plot polygon size distribution
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(stats['polygon_sizes'], bins=20, alpha=0.7)
        plt.title('Polygon Size Distribution')
        plt.xlabel('Number of Points')
        plt.ylabel('Frequency')
        
        # Plot coordinate ranges
        plt.subplot(1, 2, 2)
        x_range = [min(stats['coordinate_ranges']['x_min']), max(stats['coordinate_ranges']['x_max'])]
        y_range = [min(stats['coordinate_ranges']['y_min']), max(stats['coordinate_ranges']['y_max'])]
        plt.scatter(stats['coordinate_ranges']['x_min'], stats['coordinate_ranges']['y_min'], 
                   alpha=0.5, label='Min coords', s=10)
        plt.scatter(stats['coordinate_ranges']['x_max'], stats['coordinate_ranges']['y_max'], 
                   alpha=0.5, label='Max coords', s=10)
        plt.title('Coordinate Ranges')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('geometry_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nCoordinate ranges:")
        print(f"  X: [{x_range[0]:.1f}, {x_range[1]:.1f}]")
        print(f"  Y: [{y_range[0]:.1f}, {y_range[1]:.1f}]")
    
    if save_stats:
        with open('submission_stats.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            stats_json = stats.copy()
            for key in stats['coordinate_ranges']:
                stats_json['coordinate_ranges'][key] = len(stats['coordinate_ranges'][key])
            json.dump(stats_json, f, indent=2)
    
    return stats

def debug_failed_predictions(test_ids_df, image_dir="data/", max_show=5):
    """Debug samples that have no images or failed predictions"""
    print("="*50)
    print("DEBUGGING FAILED PREDICTIONS")
    print("="*50)
    
    missing_images = []
    found_images = []
    
    for pid in test_ids_df['ID']:
        # Check multiple possible filename patterns
        patterns = [f"{pid}.jpg", f"anonymised_{pid}.jpg"]
        found = False
        
        for pattern in patterns:
            import os
            if os.path.exists(os.path.join(image_dir, pattern)):
                found_images.append((pid, pattern))
                found = True
                break
        
        if not found:
            missing_images.append(pid)
    
    print(f"Images found: {len(found_images)}")
    print(f"Images missing: {len(missing_images)}")
    
    if missing_images:
        print(f"\nFirst {max_show} missing image IDs:")
        for pid in missing_images[:max_show]:
            print(f"  {pid}")
    
    if found_images:
        print(f"\nFirst {max_show} found images:")
        for pid, filename in found_images[:max_show]:
            print(f"  {pid} -> {filename}")
    
    # Try to find pattern mismatches
    import glob
    all_images = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(f"\nTotal images in directory: {len(all_images)}")
    
    if all_images:
        print("Sample filenames:")
        for img_path in all_images[:max_show]:
            print(f"  {os.path.basename(img_path)}")
    
    return missing_images, found_images

def compare_train_test_distributions(train_df, test_ids_df):
    """Compare distributions between training and test data"""
    print("="*50)
    print("TRAIN vs TEST COMPARISON")
    print("="*50)
    
    # Check ID format consistency
    train_ids = train_df['ID'].astype(str)
    test_ids = test_ids_df['ID'].astype(str)
    
    print(f"Train IDs sample: {list(train_ids.head())}")
    print(f"Test IDs sample: {list(test_ids.head())}")
    
    # Check for overlap (should be none)
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        print(f"⚠️  Found overlap between train and test: {len(overlap)} IDs")
        print(f"Sample overlapping IDs: {list(overlap)[:5]}")
    else:
        print("✅ No overlap between train and test sets")
    
    # Check ID ranges/patterns
    try:
        train_ids_numeric = pd.to_numeric(train_ids, errors='coerce')
        test_ids_numeric = pd.to_numeric(test_ids, errors='coerce')
        
        if not train_ids_numeric.isna().all() and not test_ids_numeric.isna().all():
            print(f"\nID ranges:")
            print(f"  Train: {train_ids_numeric.min():.0f} - {train_ids_numeric.max():.0f}")
            print(f"  Test: {test_ids_numeric.min():.0f} - {test_ids_numeric.max():.0f}")
    except:
        print("IDs are not numeric - skipping range analysis")
    
    return overlap
