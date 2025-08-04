# Barbados Document Analysis - Main Script for Scanned Plans
# This script processes scanned land survey documents to extract polygons and metadata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from document_processor import (
    DocumentProcessor, 
    process_test_documents, 
    visualize_document_processing
)

def main():
    """Main function to process scanned land survey documents"""
    
    print("BARBADOS LAND SURVEY DOCUMENT ANALYSIS")
    print("=" * 60)
    print("Processing scanned plans to extract polygons and metadata")
    print("=" * 60)
    
    # Check required files
    required_files = ["Test.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    if not os.path.exists("data/"):
        raise FileNotFoundError("Missing 'data/' directory with scanned document images")
    
    print("✓ All required files found")
    
    # Load test IDs
    test_ids = pd.read_csv("Test.csv")
    print(f"Found {len(test_ids)} test documents to process")
    
    # Visualize a few samples first for debugging
    print("\nVisualizing sample documents for debugging...")
    visualize_document_processing(test_ids, n_samples=4, image_dir="data/")
    
    # Process all test documents
    print("\nStarting full document processing...")
    submission_df = process_test_documents(
        test_ids_df=test_ids,
        image_dir="data/",
        output_file="submission_documents.csv"
    )
    
    # Analyze results
    analyze_submission_results(submission_df)
    
    print("\nDocument processing completed!")
    return submission_df

def analyze_submission_results(submission_df):
    """Analyze the results of document processing"""
    
    print(f"\n{'='*50}")
    print("SUBMISSION ANALYSIS")
    print("="*50)
    
    total_docs = len(submission_df)
    
    # Analyze geometry extraction
    valid_geometries = 0
    empty_geometries = 0
    
    for geom in submission_df['geometry']:
        if isinstance(geom, list) and len(geom) > 0:
            valid_geometries += 1
        else:
            empty_geometries += 1
    
    print(f"Total documents: {total_docs}")
    print(f"Valid polygon extractions: {valid_geometries} ({valid_geometries/total_docs*100:.1f}%)")
    print(f"Empty polygon extractions: {empty_geometries} ({empty_geometries/total_docs*100:.1f}%)")
    
    # Analyze metadata extraction
    metadata_fields = ['TargetSurvey', 'Certified date', 'Total Area', 
                      'Unit of Measurement', 'Parish', 'LT Num']
    
    print(f"\nMetadata extraction rates:")
    for field in metadata_fields:
        non_empty = sum(1 for val in submission_df[field] if val and str(val).strip())
        rate = non_empty / total_docs * 100
        print(f"  {field}: {non_empty}/{total_docs} ({rate:.1f}%)")
    
    # Create summary visualization
    create_analysis_plots(submission_df)

def create_analysis_plots(submission_df):
    """Create visualization plots for analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Geometry extraction success
    geometry_success = [
        len([g for g in submission_df['geometry'] if isinstance(g, list) and len(g) > 0]),
        len([g for g in submission_df['geometry'] if not isinstance(g, list) or len(g) == 0])
    ]
    
    axes[0, 0].pie(geometry_success, labels=['Success', 'Failed'], autopct='%1.1f%%',
                   colors=['lightgreen', 'lightcoral'])
    axes[0, 0].set_title('Polygon Extraction Success Rate')
    
    # Plot 2: Metadata field completion rates
    metadata_fields = ['TargetSurvey', 'Certified date', 'Total Area', 
                      'Unit of Measurement', 'Parish', 'LT Num']
    completion_rates = []
    
    for field in metadata_fields:
        non_empty = sum(1 for val in submission_df[field] if val and str(val).strip())
        completion_rates.append(non_empty / len(submission_df) * 100)
    
    axes[0, 1].bar(range(len(metadata_fields)), completion_rates, color='skyblue')
    axes[0, 1].set_xlabel('Metadata Fields')
    axes[0, 1].set_ylabel('Completion Rate (%)')
    axes[0, 1].set_title('Metadata Extraction Rates')
    axes[0, 1].set_xticks(range(len(metadata_fields)))
    axes[0, 1].set_xticklabels(metadata_fields, rotation=45, ha='right')
    
    # Plot 3: Polygon size distribution
    polygon_sizes = []
    for geom in submission_df['geometry']:
        if isinstance(geom, list) and len(geom) > 0:
            polygon_sizes.append(len(geom))
    
    if polygon_sizes:
        axes[1, 0].hist(polygon_sizes, bins=20, color='lightblue', alpha=0.7)
        axes[1, 0].set_xlabel('Number of Vertices')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Polygon Complexity Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, 'No valid polygons found', ha='center', va='center')
        axes[1, 0].set_title('Polygon Complexity Distribution')
    
    # Plot 4: Parish distribution (if extracted)
    parish_counts = submission_df['Parish'].value_counts()
    if len(parish_counts) > 0 and parish_counts.iloc[0] != '':
        parish_counts = parish_counts[parish_counts.index != '']  # Remove empty strings
        if len(parish_counts) > 0:
            axes[1, 1].bar(range(len(parish_counts)), parish_counts.values, color='lightcoral')
            axes[1, 1].set_xlabel('Parish')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Parish Distribution')
            axes[1, 1].set_xticks(range(len(parish_counts)))
            axes[1, 1].set_xticklabels(parish_counts.index, rotation=45, ha='right')
        else:
            axes[1, 1].text(0.5, 0.5, 'No parish data extracted', ha='center', va='center')
            axes[1, 1].set_title('Parish Distribution')
    else:
        axes[1, 1].text(0.5, 0.5, 'No parish data extracted', ha='center', va='center')
        axes[1, 1].set_title('Parish Distribution')
    
    plt.tight_layout()
    plt.savefig('document_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_single_document(image_path, plot_id):
    """Test processing of a single document for debugging"""
    
    print(f"Testing single document processing...")
    print(f"Image: {image_path}")
    print(f"Plot ID: {plot_id}")
    print("-" * 40)
    
    processor = DocumentProcessor()
    result = processor.process_document(image_path, plot_id)
    
    print("Results:")
    for key, value in result.items():
        if key == 'geometry':
            print(f"  {key}: {len(value) if isinstance(value, list) else 'N/A'} vertices")
        else:
            print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    # Run the main document processing pipeline
    submission_df = main()
    
    # Optional: Test a single document for detailed debugging
    # Uncomment and modify the path below to test a specific document
    # test_result = test_single_document("data/sample_document.jpg", "12345")
