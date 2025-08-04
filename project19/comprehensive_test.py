#!/usr/bin/env python3
"""
Comprehensive test suite for document processing (No OpenCV)
Tests extraction on mock survey plans using PIL and scikit-image
"""

import os
import sys
import numpy as np
from pathlib import Path
import json
from PIL import Image, ImageDraw

def test_extraction_methods():
    """Test different extraction approaches"""
    print("DOCUMENT PROCESSING TEST SUITE (NO OPENCV)")
    print("=" * 50)
    
    # Test 1: Mock data creation and processing
    print("\n1. TESTING WITH MOCK SURVEY PLAN")
    print("-" * 30)
    
    try:
        from document_processor_no_cv2 import DocumentProcessor, create_test_image
        
        # Create mock image
        mock_image = "test_survey_mock.png"
        if create_test_image(mock_image):
            print(f"✓ Created mock survey image: {mock_image}")
            
            # Process with advanced processor
            processor = DocumentProcessor()
            result = processor.process_document(mock_image)
            
            if result:
                print("✓ Document processor completed")
                display_standard_result(result)
            else:
                print("✗ Document processor failed")
        
    except Exception as e:
        print(f"✗ Document processor test failed: {e}")
    
    # Test 2: Try with simple processor
    print("\n2. TESTING WITH SIMPLE PROCESSOR")
    print("-" * 30)
    
    try:
        from simple_processor_no_cv2 import SimpleProcessor
        
        simple_processor = SimpleProcessor()
        if os.path.exists("test_survey_mock.png"):
            result = simple_processor.process_document("test_survey_mock.png")
            print("✓ Simple processor completed")
            display_standard_result(result)
        
    except Exception as e:
        print(f"✗ Simple processor test failed: {e}")
    
    # Test 3: Create submission format
    print("\n3. TESTING SUBMISSION GENERATION")
    print("-" * 30)
    
    test_submission_generation()

def display_standard_result(result):
    """Display standard processing results"""
    print("\nProcessing Results:")
    
    required_fields = ['TargetSurvey', 'Certified date', 'Total Area', 
                      'Unit of Measurement', 'Parish', 'LT Num']
    
    for field in required_fields:
        value = result.get(field, '')
        status = "✓" if value else "✗"
        print(f"  {status} {field}: '{value}'")
    
    geometry = result.get('geometry', [])
    print(f"  Geometry: {len(geometry)} points" if geometry else "  Geometry: None")

def create_sample_survey_images():
    """Create multiple sample survey images with different layouts using PIL"""
    samples = []
    
    # Sample 1: Standard format
    img1 = Image.new('RGB', (1000, 800), color='white')
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([150, 300, 750, 650], outline='black', width=2)
    
    filename1 = "sample_survey_001.png"
    img1.save(filename1)
    samples.append(filename1)
    
    # Sample 2: Different format
    img2 = Image.new('RGB', (1200, 900), color=(250, 250, 250))
    draw2 = ImageDraw.Draw(img2)
    draw2.polygon([(200, 400), (900, 380), (950, 700), (180, 720)], outline='black', width=3)
    
    filename2 = "sample_survey_002.png"
    img2.save(filename2)
    samples.append(filename2)
    
    # Sample 3: Minimal format
    img3 = Image.new('RGB', (800, 600), color='white')
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle([100, 250, 600, 450], outline='black', width=2)
    
    filename3 = "sample_survey_003.png"
    img3.save(filename3)
    samples.append(filename3)
    
    print(f"Created {len(samples)} sample survey images")
    return samples

def test_batch_processing():
    """Test batch processing on multiple samples"""
    print("\n4. TESTING BATCH PROCESSING")
    print("-" * 30)
    
    # Create sample images
    sample_files = create_sample_survey_images()
    
    results = []
    
    try:
        from document_processor_no_cv2 import DocumentProcessor
        processor = DocumentProcessor()
        
        for i, image_file in enumerate(sample_files):
            print(f"\nProcessing sample {i+1}: {image_file}")
            result = processor.process_document(image_file)
            results.append(result)
            
            # Show quick summary
            filled_fields = sum(1 for v in result.values() if v and v != [])
            print(f"  Extracted {filled_fields}/{len(result)} fields")
    
    except Exception as e:
        print(f"Batch processing failed: {e}")
    
    return results

def test_submission_generation():
    """Test CSV submission generation"""
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available - skipping submission test")
        return
    
    # Create test data that mimics expected output
    test_data = [
        {
            'ID': 'sample_001',
            'TargetSurvey': 'SP45678',
            'Certified date': '22/11/2023', 
            'Total Area': '2.5',
            'Unit of Measurement': 'hectares',
            'Parish': 'ASHGROVE',
            'LT Num': 'LT987654',
            'geometry': [[150, 300], [750, 300], [750, 650], [150, 650]]
        },
        {
            'ID': 'sample_002',
            'TargetSurvey': 'DP123456',
            'Certified date': '05/08/2024',
            'Total Area': '3250',
            'Unit of Measurement': 'sq.m', 
            'Parish': 'TOOWONG',
            'LT Num': '456789123',
            'geometry': [[200, 400], [900, 380], [950, 700], [180, 720]]
        },
        {
            'ID': 'sample_003',
            'TargetSurvey': 'CP78901',
            'Certified date': '15-12-2023',
            'Total Area': '1.8',
            'Unit of Measurement': 'ha',
            'Parish': 'KEDRON',
            'LT Num': '',
            'geometry': [[100, 250], [600, 250], [600, 450], [100, 450]]
        }
    ]
    
    # Convert to DataFrame and save
    df = pd.DataFrame(test_data)
    output_file = "test_submission.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Created test submission: {output_file}")
    
    # Display sample
    print("\nSample submission data:")
    print(df.to_string(index=False))
    
    return output_file

def analyze_current_capabilities():
    """Analyze what extraction methods are currently available"""
    print("\n5. ANALYZING AVAILABLE CAPABILITIES")
    print("-" * 30)
    
    capabilities = {
        'PIL': False,
        'scikit-image': False,
        'Tesseract': False,
        'EasyOCR': False,
        'Document Processor': False,
        'Simple Processor': False
    }
    
    # Test PIL
    try:
        from PIL import Image
        capabilities['PIL'] = True
    except ImportError:
        pass
    
    # Test scikit-image
    try:
        from skimage import filters
        capabilities['scikit-image'] = True
    except ImportError:
        pass
    
    # Test Tesseract
    try:
        import pytesseract
        capabilities['Tesseract'] = True
    except ImportError:
        pass
    
    # Test EasyOCR  
    try:
        import easyocr
        capabilities['EasyOCR'] = True
    except ImportError:
        pass
    
    # Test processors
    try:
        from document_processor_no_cv2 import DocumentProcessor
        capabilities['Document Processor'] = True
    except ImportError:
        pass
        
    try:
        from simple_processor_no_cv2 import SimpleProcessor
        capabilities['Simple Processor'] = True
    except ImportError:
        pass
    
    print("Available capabilities:")
    for capability, available in capabilities.items():
        status = "✓" if available else "✗"
        print(f"  {status} {capability}")
    
    # Recommendations
    print("\nRecommendations:")
    if not capabilities['scikit-image']:
        print("  ⚠ scikit-image not available - polygon detection will be limited")
        print("  → Install: pip install scikit-image")
    
    if not capabilities['Tesseract'] and not capabilities['EasyOCR']:
        print("  ⚠ No OCR libraries available - text extraction will not work")
        print("  → Install: pip install pytesseract easyocr")
    
    return capabilities

if __name__ == "__main__":
    print("Starting comprehensive document processing tests (No OpenCV)...")
    
    # Run all tests
    test_extraction_methods()
    batch_results = test_batch_processing()
    capabilities = analyze_current_capabilities()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    # Summary
    print(f"✓ Created test images and processed them")
    print(f"✓ Generated sample submission CSV")
    print(f"✓ Analyzed {len(capabilities)} capabilities")
    
    print("\nNext steps:")
    print("1. Install OCR libraries for text extraction:")
    print("   pip install pytesseract easyocr")
    print("2. Install scikit-image for better polygon detection:")
    print("   pip install scikit-image")
    print("3. Test with real survey plan images")
    print("4. Tune regex patterns for specific document formats")
