#!/usr/bin/env python3
"""
Cadastral Survey Plan Processing Workflow

This script demonstrates the complete workflow for extracting polygon coordinates
and metadata from cadastral survey plan images.
"""

from m1 import CadastralPlanExtractor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    print("=== Cadastral Survey Plan Extraction Workflow ===\n")
    
    # Initialize the extractor
    extractor = CadastralPlanExtractor()
    
    # Step 1: Load and analyze training data
    print("Step 1: Loading training data...")
    train_df = extractor.load_training_data('Train.csv')
    
    # Display basic statistics
    print(f"\nTraining Data Summary:")
    print(f"- Total samples: {len(train_df)}")
    print(f"- Unique surveyors: {train_df['Land Surveyor'].nunique()}")
    print(f"- Unique parishes: {train_df['Parish'].nunique()}")
    print(f"- Date range: {train_df['Certified date'].min()} to {train_df['Certified date'].max()}")
    print(f"- Area range: {train_df['Total Area'].min():.1f} to {train_df['Total Area'].max():.1f} sq m")
    
    # Step 2: Train the polygon prediction model
    print("\nStep 2: Training machine learning models...")
    extractor.train_polygon_prediction_model()
    
    # Step 3: Process a sample training image for visualization
    print("\nStep 3: Analyzing sample training image...")
    sample_id = train_df.iloc[0]['ID']
    sample_image_path = f"data/anonymised_{sample_id}.jpg"
    
    if os.path.exists(sample_image_path):
        print(f"Processing sample image: {sample_id}")
        
        # Extract features and metadata
        features, text_data, polygons = extractor.extract_features_from_image(sample_image_path)
        
        if features:
            print(f"Extracted features:")
            for key, value in features.items():
                print(f"  {key}: {value}")
            
            print(f"\nOCR detected {len(text_data)} text regions")
            print(f"Detected {len(polygons)} potential polygons")
            
            # Get actual metadata for comparison
            actual_row = train_df[train_df['ID'] == sample_id].iloc[0]
            predicted_polygon, predicted_metadata = extractor.predict_polygon_and_metadata(sample_image_path)
            
            print(f"\nActual vs Predicted Metadata:")
            metadata_fields = ['Land Surveyor', 'Surveyed For', 'Total Area', 'Parish']
            for field in metadata_fields:
                actual = actual_row.get(field, 'N/A')
                predicted = predicted_metadata.get(field, 'N/A')
                print(f"  {field}:")
                print(f"    Actual: {actual}")
                print(f"    Predicted: {predicted}")
                print()
            
            # Visualize results
            extractor.visualize_results(sample_image_path, predicted_polygon, predicted_metadata)
        
    else:
        print(f"Sample image not found: {sample_image_path}")
    
    # Step 4: Process all test images
    print("\nStep 4: Processing test images...")
    
    # Check if test images exist
    test_df = pd.read_csv('Test.csv')
    test_images_exist = []
    for test_id in test_df['ID']:
        test_image_path = f"data/anonymised_{test_id}.jpg"
        test_images_exist.append(os.path.exists(test_image_path))
    
    print(f"Test images available: {sum(test_images_exist)}/{len(test_images_exist)}")
    
    if sum(test_images_exist) > 0:
        # Process test images
        results_df = extractor.process_test_images('Test.csv', 'test_predictions.csv')
        
        print(f"\nTest Results Summary:")
        print(f"- Processed {len(results_df)} test images")
        print(f"- Successful polygon extractions: {results_df['geometry'].notna().sum()}")
        print(f"- Metadata extraction rates:")
        
        metadata_fields = ['Land Surveyor', 'Total Area', 'Parish', 'Certified date']
        for field in metadata_fields:
            if field in results_df.columns:
                success_rate = (results_df[field] != 'Unknown').sum() / len(results_df) * 100
                print(f"  {field}: {success_rate:.1f}%")
        
        # Save detailed analysis
        print(f"\nResults saved to 'test_predictions.csv'")
        
        # Create summary visualization
        create_summary_visualization(train_df, results_df)
        
    else:
        print("No test images found in data/ directory")
    
    print("\n=== Workflow Complete ===")

def create_summary_visualization(train_df, results_df):
    """Create visualizations comparing training and test data"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Area distribution comparison
    axes[0, 0].hist(train_df['Total Area'], bins=30, alpha=0.7, label='Training', density=True)
    if 'Total Area' in results_df.columns:
        test_areas = results_df[results_df['Total Area'] > 0]['Total Area']
        if len(test_areas) > 0:
            axes[0, 0].hist(test_areas, bins=30, alpha=0.7, label='Test Predictions', density=True)
    axes[0, 0].set_xlabel('Total Area (sq m)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Area Distribution: Training vs Test')
    axes[0, 0].legend()
    
    # Parish distribution
    train_parishes = train_df['Parish'].value_counts()
    axes[0, 1].bar(range(len(train_parishes)), train_parishes.values)
    axes[0, 1].set_xlabel('Parish')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Parish Distribution (Training)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Date distribution
    train_df['Year'] = pd.to_datetime(train_df['Certified date']).dt.year
    year_counts = train_df['Year'].value_counts().sort_index()
    axes[1, 0].plot(year_counts.index, year_counts.values, marker='o')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Number of Surveys')
    axes[1, 0].set_title('Survey Frequency by Year')
    
    # Metadata extraction success rates
    if len(results_df) > 0:
        metadata_fields = ['Land Surveyor', 'Total Area', 'Parish', 'Certified date']
        success_rates = []
        for field in metadata_fields:
            if field in results_df.columns:
                if field == 'Total Area':
                    rate = (results_df[field] > 0).sum() / len(results_df) * 100
                else:
                    rate = (results_df[field] != 'Unknown').sum() / len(results_df) * 100
                success_rates.append(rate)
            else:
                success_rates.append(0)
        
        axes[1, 1].bar(metadata_fields, success_rates)
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title('Metadata Extraction Success Rates')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('extraction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analysis visualization saved as 'extraction_analysis.png'")

if __name__ == "__main__":
    main()
