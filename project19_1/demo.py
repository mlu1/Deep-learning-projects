#!/usr/bin/env python3
"""
Complete Demo: Cadastral Survey Plan Extraction System

This script demonstrates the complete workflow including:
1. Data loading and analysis
2. Model training
3. Extraction and prediction
4. Validation and evaluation
5. Comparison of different approaches
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from m1 import CadastralPlanExtractor
from advanced_cv import AdvancedCadastralExtractor
from validation import CadastralValidator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_data_availability():
    """Check what data is available for processing"""
    
    print("=== DATA AVAILABILITY CHECK ===")
    
    # Check CSV files
    train_exists = os.path.exists('Train.csv')
    test_exists = os.path.exists('Test.csv')
    
    print(f"Training CSV: {'✓' if train_exists else '✗'}")
    print(f"Test CSV: {'✓' if test_exists else '✗'}")
    
    if not (train_exists and test_exists):
        print("ERROR: Required CSV files not found!")
        return False
    
    # Check image directory
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found!")
        return False
    
    # Count available images
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    print(f"Available images: {len(image_files)}")
    
    # Check training images
    train_df = pd.read_csv('Train.csv')
    train_images = 0
    for idx, row in train_df.iterrows():
        image_path = f"data/anonymised_{row['ID']}.jpg"
        if os.path.exists(image_path):
            train_images += 1
    
    print(f"Training images available: {train_images}/{len(train_df)}")
    
    # Check test images
    test_df = pd.read_csv('Test.csv')
    test_images = 0
    for idx, row in test_df.iterrows():
        image_path = f"data/anonymised_{row['ID']}.jpg"
        if os.path.exists(image_path):
            test_images += 1
    
    print(f"Test images available: {test_images}/{len(test_df)}")
    
    if train_images == 0:
        print("WARNING: No training images found!")
        return False
    
    return True

def demonstrate_basic_extractor():
    """Demonstrate the basic extraction approach"""
    
    print("\n=== BASIC EXTRACTION APPROACH ===")
    
    # Initialize basic extractor
    extractor = CadastralPlanExtractor()
    
    # Load training data
    train_df = extractor.load_training_data('Train.csv')
    
    # Train model on available data
    extractor.train_polygon_prediction_model()
    
    # Test on a sample image
    sample_id = train_df.iloc[0]['ID']
    sample_image_path = f"data/anonymised_{sample_id}.jpg"
    
    if os.path.exists(sample_image_path):
        print(f"\nTesting basic extractor on sample: {sample_id}")
        
        polygon, metadata = extractor.predict_polygon_and_metadata(sample_image_path)
        
        print("Extracted Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"Polygon detected: {'Yes' if polygon and len(polygon) >= 3 else 'No'}")
        if polygon:
            print(f"  Number of vertices: {len(polygon)}")
        
        # Visualize results
        extractor.visualize_results(sample_image_path, polygon, metadata)
        
        return extractor
    else:
        print(f"Sample image not found: {sample_image_path}")
        return extractor

def demonstrate_advanced_extractor():
    """Demonstrate the advanced extraction approach"""
    
    print("\n=== ADVANCED EXTRACTION APPROACH ===")
    
    # Initialize advanced extractor
    advanced_extractor = AdvancedCadastralExtractor()
    
    # Find first available training image
    train_df = pd.read_csv('Train.csv')
    sample_image_path = None
    sample_id = None
    
    for idx, row in train_df.iterrows():
        image_path = f"data/anonymised_{row['ID']}.jpg"
        if os.path.exists(image_path):
            sample_image_path = image_path
            sample_id = row['ID']
            break
    
    if sample_image_path:
        print(f"\nTesting advanced extractor on sample: {sample_id}")
        
        polygon, metadata = advanced_extractor.process_image_complete(sample_image_path)
        
        print("Extracted Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"Polygon detected: {'Yes' if polygon and len(polygon) >= 3 else 'No'}")
        if polygon:
            print(f"  Number of vertices: {len(polygon)}")
        
        # Visualize detection process
        advanced_extractor.visualize_detection_process(sample_image_path)
        
        return advanced_extractor
    else:
        print("No sample images available for advanced testing")
        return advanced_extractor

def run_validation_comparison(basic_extractor, advanced_extractor):
    """Run validation and comparison between extractors"""
    
    print("\n=== VALIDATION AND COMPARISON ===")
    
    # Initialize validator
    validator = CadastralValidator()
    
    # Load training data
    train_df = pd.read_csv('Train.csv')
    
    # Validate basic extractor
    print("\nValidating basic extractor...")
    basic_polygon_df, basic_metadata_df = validator.validate_training_predictions(
        basic_extractor, train_df, sample_size=20
    )
    
    basic_report = validator.generate_validation_report(basic_polygon_df, basic_metadata_df)
    
    # Validate advanced extractor
    print("\nValidating advanced extractor...")
    advanced_polygon_df, advanced_metadata_df = validator.validate_training_predictions(
        advanced_extractor, train_df, sample_size=20
    )
    
    advanced_report = validator.generate_validation_report(advanced_polygon_df, advanced_metadata_df)
    
    # Compare extractors
    extractors = {
        'Basic Extractor': basic_extractor,
        'Advanced Extractor': advanced_extractor
    }
    
    comparison_results = validator.compare_extractors(extractors, train_df, sample_size=15)
    
    # Print comparison summary
    print("\n=== EXTRACTOR COMPARISON SUMMARY ===")
    for name, results in comparison_results.items():
        print(f"\n{name}:")
        print(f"  Valid Polygon Rate: {results['valid_polygon_rate']:.3f}")
        print(f"  Mean IoU: {results['mean_iou']:.3f}")
        print(f"  Metadata Accuracy: {results['metadata_accuracy']:.3f}")
        print(f"  Combined Score: {results['combined_score']:.3f}")
    
    # Create validation visualizations
    print("\nGenerating validation visualizations...")
    validator.create_validation_visualizations(basic_polygon_df, basic_metadata_df)
    
    return comparison_results

def process_test_data(best_extractor):
    """Process test data with the best performing extractor"""
    
    print("\n=== PROCESSING TEST DATA ===")
    
    # Check available test images
    test_df = pd.read_csv('Test.csv')
    available_test_images = 0
    
    for idx, row in test_df.iterrows():
        image_path = f"data/anonymised_{row['ID']}.jpg"
        if os.path.exists(image_path):
            available_test_images += 1
    
    print(f"Processing {available_test_images} test images...")
    
    if available_test_images > 0:
        # Process test images
        if hasattr(best_extractor, 'process_test_images'):
            results_df = best_extractor.process_test_images('Test.csv', 'final_test_predictions.csv')
        else:
            # For advanced extractor, process manually
            results = []
            
            for idx, row in test_df.iterrows():
                image_path = f"data/anonymised_{row['ID']}.jpg"
                
                if os.path.exists(image_path):
                    polygon, metadata = best_extractor.process_image_complete(image_path)
                    
                    result = {'ID': row['ID']}
                    result.update(metadata)
                    
                    # Convert polygon to WKT
                    if polygon and len(polygon) >= 3:
                        try:
                            if polygon[0] != polygon[-1]:
                                polygon.append(polygon[0])
                            
                            coord_str = ', '.join([f"{x} {y} -0.0000234999965869" for x, y in polygon])
                            wkt_string = f"POLYGON Z (({coord_str}))"
                            result['geometry'] = wkt_string
                        except:
                            result['geometry'] = None
                    else:
                        result['geometry'] = None
                    
                    results.append(result)
                
                if idx % 10 == 0:
                    print(f"Processed {idx}/{len(test_df)} test images")
            
            results_df = pd.DataFrame(results)
            
            # Fill missing values
            results_df['Land Surveyor'] = results_df['Land Surveyor'].fillna('Unknown')
            results_df['Surveyed For'] = results_df['Surveyed For'].fillna('Unknown')
            results_df['Certified date'] = results_df['Certified date'].fillna('Unknown')
            results_df['Total Area'] = results_df['Total Area'].fillna(0.0)
            results_df['Unit of Measurement'] = results_df['Unit of Measurement'].fillna('sq m')
            results_df['Address'] = results_df['Address'].fillna('Unknown')
            results_df['Parish'] = results_df['Parish'].fillna('Unknown')
            results_df['LT Num'] = results_df['LT Num'].fillna('Unknown')
            
            results_df.to_csv('final_test_predictions.csv', index=False)
        
        # Generate summary statistics
        print(f"\nTest Results Summary:")
        print(f"- Total test samples processed: {len(results_df)}")
        print(f"- Successful polygon extractions: {results_df['geometry'].notna().sum()}")
        
        metadata_fields = ['Land Surveyor', 'Total Area', 'Parish', 'Certified date']
        for field in metadata_fields:
            if field in results_df.columns:
                if field == 'Total Area':
                    success_rate = (results_df[field] > 0).sum() / len(results_df) * 100
                else:
                    success_rate = (results_df[field] != 'Unknown').sum() / len(results_df) * 100
                print(f"- {field} extraction rate: {success_rate:.1f}%")
        
        print(f"\nResults saved to 'final_test_predictions.csv'")
        
        return results_df
    else:
        print("No test images available for processing")
        return None

def create_final_summary():
    """Create final summary and recommendations"""
    
    print("\n" + "="*60)
    print("FINAL SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    print("\nSYSTEM CAPABILITIES:")
    print("✓ Automated polygon extraction from cadastral plans")
    print("✓ Metadata extraction (surveyor, dates, areas, addresses)")
    print("✓ Multiple extraction approaches (basic ML + advanced CV)")
    print("✓ Comprehensive validation and evaluation")
    print("✓ Performance comparison and optimization")
    
    print("\nRECOMMENDATIONS FOR IMPROVEMENT:")
    print("1. Increase training data size for better ML model performance")
    print("2. Implement specific OCR training for cadastral plan text")
    print("3. Add geometric constraints for polygon validation")
    print("4. Use ensemble methods combining multiple extraction approaches")
    print("5. Implement feedback loop for continuous model improvement")
    
    print("\nDEPLOYMENT CONSIDERATIONS:")
    print("- System can process images in batch or real-time")
    print("- Modular design allows for easy component upgrades")
    print("- Validation framework ensures quality control")
    print("- Results can be exported in various formats (CSV, GeoJSON, etc.)")
    
    print("\nFILES GENERATED:")
    files = [
        'final_test_predictions.csv',
        'validation_results.png',
        'extractor_comparison.png',
        'extraction_analysis.png'
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")

def main():
    """Main demo function"""
    
    print("CADASTRAL SURVEY PLAN EXTRACTION SYSTEM DEMO")
    print("=" * 60)
    
    # Check data availability
    if not check_data_availability():
        print("Demo cannot proceed without required data files.")
        return
    
    # Demonstrate basic extractor
    basic_extractor = demonstrate_basic_extractor()
    
    # Demonstrate advanced extractor
    advanced_extractor = demonstrate_advanced_extractor()
    
    # Run validation and comparison
    comparison_results = run_validation_comparison(basic_extractor, advanced_extractor)
    
    # Determine best extractor
    best_extractor_name = max(comparison_results.keys(), 
                             key=lambda x: comparison_results[x]['combined_score'])
    
    print(f"\nBest performing extractor: {best_extractor_name}")
    
    if best_extractor_name == 'Basic Extractor':
        best_extractor = basic_extractor
    else:
        best_extractor = advanced_extractor
    
    # Process test data
    test_results = process_test_data(best_extractor)
    
    # Create final summary
    create_final_summary()
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE!")
    print("Check the generated files for detailed results and visualizations.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
