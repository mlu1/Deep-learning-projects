#!/usr/bin/env python3
"""
Comprehensive test and processing script for Barbados document analysis
Works with actual test data or creates mock data for testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_test_data():
    """Check if test data is available"""
    test_csv_exists = os.path.exists("Test.csv")
    data_dir_exists = os.path.exists("data/")
    
    if test_csv_exists:
        test_df = pd.read_csv("Test.csv")
        print(f"✅ Found Test.csv with {len(test_df)} entries")
        
        if data_dir_exists:
            import glob
            image_files = glob.glob("data/*.jpg") + glob.glob("data/*.png") + glob.glob("data/*.jpeg")
            print(f"✅ Found data/ directory with {len(image_files)} image files")
            
            # Check if any test IDs have corresponding images
            sample_ids = test_df['ID'].head(5).values
            found_images = 0
            for pid in sample_ids:
                for pattern in [f"{pid}.jpg", f"anonymised_{pid}.jpg", f"{pid}.png"]:
                    if os.path.exists(os.path.join("data", pattern)):
                        found_images += 1
                        break
            
            print(f"✅ Found images for {found_images}/{len(sample_ids)} sample IDs")
            return test_df, found_images > 0
        else:
            print("❌ data/ directory not found")
            return test_df, False
    else:
        print("❌ Test.csv not found")
        return None, False

def create_mock_test_data():
    """Create mock test data for demonstration"""
    print("Creating mock test data for demonstration...")
    
    # Create mock Test.csv
    mock_ids = [
        "7703-078", "8606-095", "7703-064", "7703-101", "7707-114",
        "8604-111", "7703-049", "7706-060", "7707-141", "7707-152"
    ]
    
    test_df = pd.DataFrame({'ID': mock_ids})
    test_df.to_csv("Test_mock.csv", index=False)
    
    # Create mock data directory
    os.makedirs("data_mock", exist_ok=True)
    
    # Create mock document images
    created_images = []
    for i, plot_id in enumerate(mock_ids[:3]):  # Create 3 sample images
        img_path = create_mock_document_image(plot_id, f"data_mock/{plot_id}.jpg")
        if img_path:
            created_images.append(img_path)
    
    print(f"✅ Created {len(created_images)} mock document images")
    return test_df, created_images

def create_mock_document_image(plot_id, output_path):
    """Create a mock land survey document image"""
    try:
        # Create a document-like image
        width, height = 800, 600
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_medium = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw title
        draw.text((50, 30), "LAND SURVEY PLAN", fill='black', font=font_large)
        draw.text((50, 60), f"Plot ID: {plot_id}", fill='black', font=font_medium)
        
        # Draw metadata fields
        y_pos = 100
        metadata = [
            f"Surveyed For: John Smith {plot_id[-3:]}",
            f"Parish: St. Michael",
            f"Total Area: {2.5 + int(plot_id[-1])}.{plot_id[-2:]} acres",
            f"Certified Date: 15/06/2023",
            f"LT No: {plot_id.replace('-', '')}"
        ]
        
        for text in metadata:
            draw.text((50, y_pos), text, fill='black', font=font_small)
            y_pos += 25
        
        # Draw a simple polygon (plot boundary)
        polygon_points = [
            (200, 250), (500, 250), (500, 450), (200, 450)
        ]
        draw.polygon(polygon_points, outline='black', width=2)
        draw.text((300, 350), "PLOT BOUNDARY", fill='black', font=font_medium)
        
        # Add some survey details
        draw.text((50, 500), "Scale: 1:500", fill='black', font=font_small)
        draw.text((50, 520), "Surveyed by: ABC Surveyors Ltd", fill='black', font=font_small)
        
        # Save image
        img.save(output_path)
        print(f"Created mock document: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error creating mock image: {e}")
        return None

def test_enhanced_processor(test_df, use_mock=False):
    """Test the enhanced processor on available data"""
    print("\nTesting Enhanced Document Processor...")
    print("=" * 50)
    
    try:
        from enhanced_processor import EnhancedDocumentProcessor, process_test_documents_enhanced
        
        # Test on a small sample first
        sample_size = min(5, len(test_df))
        sample_df = test_df.head(sample_size)
        
        image_dir = "data_mock" if use_mock else "data"
        
        print(f"Testing on {sample_size} samples from {image_dir}/")
        
        # Process sample
        results_df = process_test_documents_enhanced(
            sample_df, 
            image_dir=image_dir, 
            output_file="test_submission.csv"
        )
        
        # Analyze results
        analyze_test_results(results_df)
        
        return results_df
        
    except ImportError as e:
        print(f"Enhanced processor not available: {e}")
        return None

def analyze_test_results(results_df):
    """Analyze the test results"""
    print("\nTEST RESULTS ANALYSIS")
    print("=" * 40)
    
    total = len(results_df)
    
    # Geometry analysis
    valid_geometry = sum(1 for geom in results_df['geometry'] if isinstance(geom, list) and len(geom) > 0)
    print(f"Geometry extraction: {valid_geometry}/{total} ({valid_geometry/total*100:.1f}%)")
    
    # Metadata analysis
    metadata_fields = ['TargetSurvey', 'Certified date', 'Total Area', 
                      'Unit of Measurement', 'Parish', 'LT Num']
    
    print("\nMetadata extraction rates:")
    for field in metadata_fields:
        non_empty = sum(1 for val in results_df[field] if val and str(val).strip())
        rate = non_empty / total * 100
        print(f"  {field}: {non_empty}/{total} ({rate:.1f}%)")
    
    # Overall success rate
    successful_rows = 0
    for _, row in results_df.iterrows():
        has_geometry = isinstance(row['geometry'], list) and len(row['geometry']) > 0
        has_metadata = any(row.get(field, '') for field in metadata_fields)
        if has_geometry or has_metadata:
            successful_rows += 1
    
    print(f"\nOverall success rate: {successful_rows}/{total} ({successful_rows/total*100:.1f}%)")

def main():
    """Main function"""
    print("BARBADOS DOCUMENT ANALYSIS - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Check for real test data
    test_df, has_images = check_test_data()
    
    if test_df is not None and has_images:
        print("Using real test data...")
        results = test_enhanced_processor(test_df, use_mock=False)
    else:
        print("Real test data not available. Creating mock data for demonstration...")
        mock_df, mock_images = create_mock_test_data()
        
        if mock_images:
            print("Testing with mock data...")
            results = test_enhanced_processor(mock_df, use_mock=True)
        
        print("\n" + "="*60)
        print("MOCK DATA DEMONSTRATION COMPLETE")
        print("="*60)
        print("To use with real data:")
        print("1. Place Test.csv in the current directory")
        print("2. Create data/ directory with scanned document images")
        print("3. Run this script again")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
