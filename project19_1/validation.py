"""
Validation and Evaluation Module for Cadastral Plan Extraction

This module provides comprehensive evaluation metrics and validation
tools for assessing the quality of polygon and metadata extraction.
"""

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely import wkt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
import cv2
import os
'''
from scipy.spatial.distance import hausdorff
'''
from skimage.metrics import hausdorff_distance
import warnings
warnings.filterwarnings('ignore')

class CadastralValidator:
    """Comprehensive validation and evaluation for cadastral plan extraction"""
    
    def __init__(self):
        self.metrics = {}
    
    def parse_wkt_polygon(self, wkt_string):
        """Parse WKT polygon string to extract coordinates"""
        try:
            polygon = wkt.loads(wkt_string)
            # Return exterior coordinates as list of (x, y) tuples
            return list(polygon.exterior.coords)
        except:
            return None
        
    def calculate_polygon_metrics(self, predicted_coords, actual_coords):
        """Calculate various metrics for polygon comparison"""
        
        if predicted_coords is None or actual_coords is None:
            return {
                'iou': 0.0,
                'area_difference': float('inf'),
                'hausdorff_distance': float('inf'),
                'centroid_distance': float('inf'),
                'valid_prediction': False
            }
        
        try:
            # Create polygons
            pred_poly = Polygon(predicted_coords)
            actual_poly = Polygon(actual_coords)
            
            # Ensure polygons are valid
            if not pred_poly.is_valid:
                pred_poly = pred_poly.buffer(0)
            if not actual_poly.is_valid:
                actual_poly = actual_poly.buffer(0)
            
            # Calculate IoU (Intersection over Union)
            intersection = pred_poly.intersection(actual_poly).area
            union = pred_poly.union(actual_poly).area
            iou = intersection / union if union > 0 else 0
            
            # Area difference (relative)
            area_diff = abs(pred_poly.area - actual_poly.area) / actual_poly.area
            
            # Hausdorff distance between polygon boundaries
            pred_coords_arr = np.array(predicted_coords)
            actual_coords_arr = np.array(actual_coords)
            hausdorff_dist = hausdorff_distance(pred_coords_arr, actual_coords_arr)
            
            # Centroid distance
            pred_centroid = pred_poly.centroid
            actual_centroid = actual_poly.centroid
            centroid_dist = pred_centroid.distance(actual_centroid)
            
            return {
                'iou': iou,
                'area_difference': area_diff,
                'hausdorff_distance': hausdorff_dist,
                'centroid_distance': centroid_dist,
                'valid_prediction': True
            }
            
        except Exception as e:
            print(f"Error calculating polygon metrics: {e}")
            return {
                'iou': 0.0,
                'area_difference': float('inf'),
                'hausdorff_distance': float('inf'),
                'centroid_distance': float('inf'),
                'valid_prediction': False
            }
    
    def evaluate_metadata_extraction(self, predicted_metadata, actual_metadata):
        """Evaluate metadata extraction accuracy"""
        
        field_accuracies = {}
        
        metadata_fields = ['Land Surveyor', 'Surveyed For', 'Certified date', 
                          'Total Area', 'Unit of Measurement', 'Address', 'Parish', 'LT Num']
        
        for field in metadata_fields:
            pred_val = predicted_metadata.get(field, None)
            actual_val = actual_metadata.get(field, None)
            
            if field == 'Total Area':
                # For numeric fields, use relative error
                if pred_val is not None and actual_val is not None:
                    try:
                        pred_num = float(pred_val)
                        actual_num = float(actual_val)
                        relative_error = abs(pred_num - actual_num) / actual_num
                        field_accuracies[field] = 1.0 - min(1.0, relative_error)
                    except:
                        field_accuracies[field] = 0.0
                else:
                    field_accuracies[field] = 0.0
            else:
                # For text fields, use exact match or fuzzy matching
                if pred_val is not None and actual_val is not None:
                    # Simple exact match (case-insensitive)
                    if str(pred_val).lower().strip() == str(actual_val).lower().strip():
                        field_accuracies[field] = 1.0
                    # Fuzzy matching for names and addresses
                    elif field in ['Land Surveyor', 'Surveyed For', 'Address']:
                        field_accuracies[field] = self.fuzzy_match_score(str(pred_val), str(actual_val))
                    else:
                        field_accuracies[field] = 0.0
                else:
                    field_accuracies[field] = 0.0
        
        return field_accuracies
    
    def fuzzy_match_score(self, pred_text, actual_text):
        """Calculate fuzzy matching score for text fields"""
        
        pred_words = set(pred_text.lower().split())
        actual_words = set(actual_text.lower().split())
        
        if len(actual_words) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = len(pred_words.intersection(actual_words))
        union = len(pred_words.union(actual_words))
        
        return intersection / union if union > 0 else 0.0
    
    def validate_training_predictions(self, extractor, train_df, sample_size=50):
        """Validate extraction on training data subset"""
        
        print(f"Validating on {sample_size} training samples...")
        
        # Ensure polygon_coords column exists
        if 'polygon_coords' not in train_df.columns:
            print("Parsing WKT geometry to polygon coordinates...")
            train_df['polygon_coords'] = train_df['geometry'].apply(self.parse_wkt_polygon)
        
        # Randomly sample training data
        sample_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        
        polygon_metrics = []
        metadata_metrics = []
        
        for idx, row in sample_df.iterrows():
            image_path = f"data/anonymised_{row['ID']}.jpg"
            
            # Skip if no image file exists
            if not os.path.exists(image_path):
                continue
            
            try:
                if hasattr(extractor, 'predict_polygon_and_metadata'):
                    # Standard extractor
                    predicted_polygon, predicted_metadata = extractor.predict_polygon_and_metadata(image_path)
                else:
                    # Advanced extractor
                    predicted_polygon, predicted_metadata = extractor.process_image_complete(image_path)
                
                # Get actual data - ensure polygon_coords exists
                actual_coords = row.get('polygon_coords', None)
                if actual_coords is None and 'geometry' in row:
                    # Try to parse from WKT
                    actual_coords = self.parse_wkt_polygon(row['geometry'])
                
                actual_metadata = row.to_dict()
                
                # Calculate polygon metrics
                poly_metrics = self.calculate_polygon_metrics(predicted_polygon, actual_coords)
                poly_metrics['image_id'] = row['ID']
                polygon_metrics.append(poly_metrics)
                
                # Calculate metadata metrics
                meta_metrics = self.evaluate_metadata_extraction(predicted_metadata, actual_metadata)
                meta_metrics['image_id'] = row['ID']
                metadata_metrics.append(meta_metrics)
                
            except Exception as e:
                print(f"Error processing {row['ID']}: {e}")
                continue
        
        # Convert to DataFrames
        polygon_df = pd.DataFrame(polygon_metrics)
        metadata_df = pd.DataFrame(metadata_metrics)
        
        return polygon_df, metadata_df
    
    def generate_validation_report(self, polygon_df, metadata_df):
        """Generate comprehensive validation report"""
        
        print("\n" + "="*60)
        print("CADASTRAL PLAN EXTRACTION VALIDATION REPORT")
        print("="*60)
        
        # Check if we have any data
        if len(polygon_df) == 0:
            print("\nNo validation data available - all samples failed to process.")
            return {
                'polygon_metrics': polygon_df,
                'metadata_metrics': metadata_df,
                'summary': {
                    'valid_polygon_rate': 0,
                    'mean_iou': 0,
                    'metadata_accuracy': 0
                }
            }
        
        # Polygon Performance
        print("\n1. POLYGON EXTRACTION PERFORMANCE")
        print("-"*40)
        
        # Check if 'valid_prediction' column exists
        if 'valid_prediction' not in polygon_df.columns:
            print("Warning: 'valid_prediction' column missing from polygon data")
            return {
                'polygon_metrics': polygon_df,
                'metadata_metrics': metadata_df,
                'summary': {
                    'valid_polygon_rate': 0,
                    'mean_iou': 0,
                    'metadata_accuracy': 0
                }
            }
        
        valid_predictions = polygon_df['valid_prediction'].sum()
        total_predictions = len(polygon_df)
        
        print(f"Valid Predictions: {valid_predictions}/{total_predictions} ({valid_predictions/total_predictions*100:.1f}%)")
        
        if valid_predictions > 0:
            valid_poly_df = polygon_df[polygon_df['valid_prediction']]
            
            print(f"\nPolygon Metrics (Valid Predictions Only):")
            print(f"  Mean IoU: {valid_poly_df['iou'].mean():.3f} ± {valid_poly_df['iou'].std():.3f}")
            print(f"  Mean Area Difference: {valid_poly_df['area_difference'].mean():.3f} ± {valid_poly_df['area_difference'].std():.3f}")
            print(f"  Mean Hausdorff Distance: {valid_poly_df['hausdorff_distance'].mean():.1f} ± {valid_poly_df['hausdorff_distance'].std():.1f}")
            print(f"  Mean Centroid Distance: {valid_poly_df['centroid_distance'].mean():.1f} ± {valid_poly_df['centroid_distance'].std():.1f}")
            
            # Quality thresholds
            high_quality = (valid_poly_df['iou'] > 0.7).sum()
            medium_quality = ((valid_poly_df['iou'] > 0.5) & (valid_poly_df['iou'] <= 0.7)).sum()
            low_quality = (valid_poly_df['iou'] <= 0.5).sum()
            
            print(f"\nQuality Distribution:")
            print(f"  High Quality (IoU > 0.7): {high_quality} ({high_quality/valid_predictions*100:.1f}%)")
            print(f"  Medium Quality (IoU 0.5-0.7): {medium_quality} ({medium_quality/valid_predictions*100:.1f}%)")
            print(f"  Low Quality (IoU < 0.5): {low_quality} ({low_quality/valid_predictions*100:.1f}%)")
        
        # Metadata Performance
        print("\n2. METADATA EXTRACTION PERFORMANCE")
        print("-"*40)
        
        metadata_fields = ['Land Surveyor', 'Surveyed For', 'Certified date', 
                          'Total Area', 'Unit of Measurement', 'Address', 'Parish', 'LT Num']
        
        for field in metadata_fields:
            if field in metadata_df.columns:
                accuracy = metadata_df[field].mean()
                print(f"  {field}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        overall_metadata_accuracy = metadata_df[metadata_fields].mean().mean()
        print(f"\nOverall Metadata Accuracy: {overall_metadata_accuracy:.3f} ({overall_metadata_accuracy*100:.1f}%)")
        
        return {
            'polygon_metrics': polygon_df,
            'metadata_metrics': metadata_df,
            'summary': {
                'valid_polygon_rate': valid_predictions/total_predictions,
                'mean_iou': polygon_df[polygon_df['valid_prediction']]['iou'].mean() if valid_predictions > 0 else 0,
                'metadata_accuracy': overall_metadata_accuracy
            }
        }
    
    def create_validation_visualizations(self, polygon_df, metadata_df):
        """Create comprehensive validation visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. IoU Distribution
        valid_poly_df = polygon_df[polygon_df['valid_prediction']]
        if len(valid_poly_df) > 0:
            axes[0, 0].hist(valid_poly_df['iou'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(valid_poly_df['iou'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {valid_poly_df["iou"].mean():.3f}')
            axes[0, 0].set_xlabel('IoU Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Polygon IoU Distribution')
            axes[0, 0].legend()
        
        # 2. Area Difference Distribution
        if len(valid_poly_df) > 0:
            axes[0, 1].hist(valid_poly_df['area_difference'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(valid_poly_df['area_difference'].mean(), color='red', linestyle='--',
                              label=f'Mean: {valid_poly_df["area_difference"].mean():.3f}')
            axes[0, 1].set_xlabel('Relative Area Difference')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Area Difference Distribution')
            axes[0, 1].legend()
        
        # 3. Metadata Accuracy by Field
        metadata_fields = ['Land Surveyor', 'Surveyed For', 'Certified date', 
                          'Total Area', 'Unit of Measurement', 'Address', 'Parish', 'LT Num']
        
        field_accuracies = []
        field_names = []
        for field in metadata_fields:
            if field in metadata_df.columns:
                field_accuracies.append(metadata_df[field].mean())
                field_names.append(field)
        
        if field_accuracies:
            bars = axes[0, 2].bar(range(len(field_names)), field_accuracies, alpha=0.7)
            axes[0, 2].set_xlabel('Metadata Field')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].set_title('Metadata Extraction Accuracy')
            axes[0, 2].set_xticks(range(len(field_names)))
            axes[0, 2].set_xticklabels(field_names, rotation=45, ha='right')
            
            # Color code bars
            for i, bar in enumerate(bars):
                if field_accuracies[i] > 0.8:
                    bar.set_color('green')
                elif field_accuracies[i] > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        # 4. IoU vs Area Difference Scatter
        if len(valid_poly_df) > 0:
            scatter = axes[1, 0].scatter(valid_poly_df['iou'], valid_poly_df['area_difference'], 
                                       alpha=0.6, c=valid_poly_df['centroid_distance'], 
                                       cmap='viridis')
            axes[1, 0].set_xlabel('IoU Score')
            axes[1, 0].set_ylabel('Area Difference')
            axes[1, 0].set_title('IoU vs Area Difference')
            plt.colorbar(scatter, ax=axes[1, 0], label='Centroid Distance')
        
        # 5. Performance by Image Complexity (proxy: number of detected polygons)
        if 'num_potential_polygons' in polygon_df.columns:
            complexity_groups = polygon_df.groupby('num_potential_polygons')['iou'].mean()
            axes[1, 1].plot(complexity_groups.index, complexity_groups.values, marker='o')
            axes[1, 1].set_xlabel('Number of Detected Polygons')
            axes[1, 1].set_ylabel('Mean IoU')
            axes[1, 1].set_title('Performance vs Image Complexity')
        
        # 6. Overall Performance Summary
        axes[1, 2].axis('off')
        
        # Create summary text
        summary_text = "PERFORMANCE SUMMARY\n\n"
        
        if len(valid_poly_df) > 0:
            high_quality = (valid_poly_df['iou'] > 0.7).sum()
            total_valid = len(valid_poly_df)
            
            summary_text += f"Polygon Extraction:\n"
            summary_text += f"  • Valid Rate: {len(valid_poly_df)/len(polygon_df)*100:.1f}%\n"
            summary_text += f"  • High Quality: {high_quality/total_valid*100:.1f}%\n"
            summary_text += f"  • Mean IoU: {valid_poly_df['iou'].mean():.3f}\n\n"
        
        if field_accuracies:
            summary_text += f"Metadata Extraction:\n"
            summary_text += f"  • Overall Accuracy: {np.mean(field_accuracies)*100:.1f}%\n"
            summary_text += f"  • Best Field: {field_names[np.argmax(field_accuracies)]}\n"
            summary_text += f"  • Worst Field: {field_names[np.argmin(field_accuracies)]}\n"
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Validation visualizations saved as 'validation_results.png'")
    
    def compare_extractors(self, extractors, train_df, sample_size=20):
        """Compare performance of different extraction methods"""
        
        print(f"Comparing {len(extractors)} extraction methods...")
        
        comparison_results = {}
        
        for name, extractor in extractors.items():
            print(f"\nEvaluating {name}...")
            
            polygon_df, metadata_df = self.validate_training_predictions(
                extractor, train_df, sample_size
            )
            
            # Calculate summary metrics
            valid_rate = polygon_df['valid_prediction'].sum() / len(polygon_df)
            mean_iou = polygon_df[polygon_df['valid_prediction']]['iou'].mean() if valid_rate > 0 else 0
            
            metadata_fields = ['Land Surveyor', 'Total Area', 'Parish', 'Certified date']
            metadata_accuracy = metadata_df[metadata_fields].mean().mean()
            
            comparison_results[name] = {
                'valid_polygon_rate': valid_rate,
                'mean_iou': mean_iou,
                'metadata_accuracy': metadata_accuracy,
                'combined_score': (valid_rate + mean_iou + metadata_accuracy) / 3
            }
        
        # Create comparison visualization
        self.visualize_extractor_comparison(comparison_results)
        
        return comparison_results
    
    def visualize_extractor_comparison(self, comparison_results):
        """Visualize comparison between different extractors"""
        
        extractors = list(comparison_results.keys())
        metrics = ['valid_polygon_rate', 'mean_iou', 'metadata_accuracy', 'combined_score']
        metric_names = ['Valid Polygon Rate', 'Mean IoU', 'Metadata Accuracy', 'Combined Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [comparison_results[extractor][metric] for extractor in extractors]
            
            bars = axes[i].bar(extractors, values, alpha=0.7)
            axes[i].set_ylabel(name)
            axes[i].set_title(f'{name} Comparison')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Color code bars
            for j, bar in enumerate(bars):
                if values[j] > 0.8:
                    bar.set_color('green')
                elif values[j] > 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{values[j]:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('extractor_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Extractor comparison saved as 'extractor_comparison.png'")

# Initialize validator
validator = CadastralValidator()
