#!/usr/bin/env python3
"""
Comprehensive test suite for document processing capabilities
This script tests extraction on real and mock survey plans
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json

def test_extraction_methods():
    """Test different extraction approaches"""
    print("DOCUMENT PROCESSING TEST SUITE")
    print("=" * 50)
    
    # Test 1: Mock data creation and processing
    print("\n1. TESTING WITH MOCK SURVEY PLAN")
    print("-" * 30)
    
    try:
        from advanced_document_processor import AdvancedDocumentProcessor, create_mock_survey_image
        
        # Create mock image
        mock_image = "test_survey_mock.png"
        if create_mock_survey_image(mock_image):
            print(f"✓ Created mock survey image: {mock_image}")
            
            # Process with advanced processor
            processor = AdvancedDocumentProcessor()
            result = processor.debug_processing(mock_image)
            
            if result:
                print("✓ Advanced processor completed")
                display_extraction_result(result)
            else:
                print("✗ Advanced processor failed")
        
    except Exception as e:
        print(f"✗ Advanced processor test failed: {e}")
    
    # Test 2: Try with enhanced processor if available
    print("\n2. TESTING WITH ENHANCED PROCESSOR")
    print("-" * 30)
    
    try:
        from enhanced_document_processor import EnhancedDocumentProcessor
        
        enhanced_processor = EnhancedDocumentProcessor()
        if os.path.exists("test_survey_mock.png"):
            result = enhanced_processor.process_document("test_survey_mock.png")
            print("✓ Enhanced processor completed")
            display_standard_result(result)
        
    except Exception as e:
        print(f"✗ Enhanced processor test failed: {e}")
    
    # Test 3: Try with simple processor
    print("\n3. TESTING WITH SIMPLE PROCESSOR")
    print("-" * 30)
    
    try:
        from simple_processor import SimpleProcessor
        
        simple_processor = SimpleProcessor()
        if os.path.exists("test_survey_mock.png"):
            result = simple_processor.process_document("test_survey_mock.png")
            print("✓ Simple processor completed")
            display_standard_result(result)
        
    except Exception as e:
        print(f"✗ Simple processor test failed: {e}")
    
    # Test 4: Create submission format
    print("\n4. TESTING SUBMISSION GENERATION")
    print("-" * 30)
    
    test_submission_generation()

def display_extraction_result(result):
    """Display detailed extraction results from debug processing"""
    print("\nExtraction Results:")
    
    # Show metadata
    metadata = result.get('metadata', {})
    for key, value in metadata.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: '{value}'")
    
    # Show text length
    text = result.get('text', '')
    print(f"  Raw text extracted: {len(text)} characters")
    
    # Show polygons
    polygons = result.get('polygons', [])
    print(f"  Polygons detected: {len(polygons)}")
    
    if polygons:
        for i, poly in enumerate(polygons[:2]):  # Show first 2
            print(f"    Polygon {i+1}: {len(poly)} vertices")

def display_standard_result(result):
    """Display standard processing results"""
    print("\nStandard Processing Results:")
    
    required_fields = ['TargetSurvey', 'Certified date', 'Total Area', 
                      'Unit of Measurement', 'Parish', 'LT Num']
    
    for field in required_fields:
        value = result.get(field, '')
        status = "✓" if value else "✗"
        print(f"  {status} {field}: '{value}'")
    
    geometry = result.get('geometry', [])
    print(f"  Geometry: {len(geometry)} points" if geometry else "  Geometry: None")

def create_sample_survey_images():
    """Create multiple sample survey images with different layouts"""
    samples = []
    
    # Sample 1: Standard format
    img1 = np.ones((800, 1000, 3), dtype=np.uint8) * 255
    cv2.putText(img1, "SURVEY PLAN", (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img1, "Survey Number: SP45678", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img1, "Certified: 22/11/2023", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img1, "Area = 2.5 hectares", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img1, "Parish of ASHGROVE", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img1, "Land Title No: LT987654", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add boundary
    points1 = np.array([[150, 300], [750, 300], [750, 650], [150, 650]], np.int32)
    cv2.polylines(img1, [points1], True, (0, 0, 0), 2)
    
    filename1 = "sample_survey_001.png"
    cv2.imwrite(filename1, img1)
    samples.append(filename1)
    
    # Sample 2: Different format
    img2 = np.ones((900, 1200, 3), dtype=np.uint8) * 250  # Light gray background
    cv2.putText(img2, "DEPOSITED PLAN", (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    cv2.putText(img2, "DP 123456", (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img2, "Date: 05/08/2024", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img2, "Total Area: 3250 sq.m", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img2, "P. of TOOWONG", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img2, "L.T. 456789123", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add irregular boundary
    points2 = np.array([[200, 400], [900, 380], [950, 700], [180, 720]], np.int32)
    cv2.polylines(img2, [points2], True, (0, 0, 0), 3)
    
    filename2 = "sample_survey_002.png"
    cv2.imwrite(filename2, img2)
    samples.append(filename2)
    
    # Sample 3: Minimal format
    img3 = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.putText(img3, "PLAN No: CP78901", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img3, "Approved 15-12-2023", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img3, "1.8 ha", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img3, "KEDRON Parish", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Simple rectangle
    points3 = np.array([[100, 250], [600, 250], [600, 450], [100, 450]], np.int32)
    cv2.polylines(img3, [points3], True, (0, 0, 0), 2)
    
    filename3 = "sample_survey_003.png"
    cv2.imwrite(filename3, img3)
    samples.append(filename3)
    
    print(f"Created {len(samples)} sample survey images")
    return samples

def test_batch_processing():
    """Test batch processing on multiple samples"""
    print("\n5. TESTING BATCH PROCESSING")
    print("-" * 30)
    
    # Create sample images
    sample_files = create_sample_survey_images()
    
    results = []
    
    try:
        from advanced_document_processor import AdvancedDocumentProcessor
        processor = AdvancedDocumentProcessor()
        
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
    print("\n6. ANALYZING AVAILABLE CAPABILITIES")
    print("-" * 30)
    
    capabilities = {
        'OpenCV': True,  # Always available
        'Tesseract': False,
        'EasyOCR': False,
        'Advanced Processor': False,
        'Enhanced Processor': False,
        'Simple Processor': False
    }
    
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
        from advanced_document_processor import AdvancedDocumentProcessor
        capabilities['Advanced Processor'] = True
    except ImportError:
        pass
    
    try:
        from enhanced_document_processor import EnhancedDocumentProcessor
        capabilities['Enhanced Processor'] = True
    except ImportError:
        pass
        
    try:
        from simple_processor import SimpleProcessor
        capabilities['Simple Processor'] = True
    except ImportError:
        pass
    
    print("Available capabilities:")
    for capability, available in capabilities.items():
        status = "✓" if available else "✗"
        print(f"  {status} {capability}")
    
    # Recommendations
    print("\nRecommendations:")
    if not capabilities['Tesseract'] and not capabilities['EasyOCR']:
        print("  ⚠ No OCR libraries available - text extraction will be very limited")
        print("  → Install: pip install pytesseract easyocr")
    
    if capabilities['OpenCV']:
        print("  ✓ OpenCV available for basic image processing and polygon detection")
    
    return capabilities

def create_minimal_working_example():
    """Create a minimal processor that works with just OpenCV"""
    print("\n7. CREATING MINIMAL WORKING EXAMPLE")
    print("-" * 30)
    
    minimal_code = '''
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

class MinimalProcessor:
    def process_document(self, image_path):
        """Minimal processing with just OpenCV"""
        result = {
            'ID': Path(image_path).stem,
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': '',
            'geometry': []
        }
        
        try:
            # Load image
            img = cv2.imread(image_path, 0)  # Grayscale
            if img is None:
                return result
            
            # Basic polygon detection
            edges = cv2.Canny(img, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                largest = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)
                
                if len(approx) >= 3:
                    polygon = [[int(point[0][0]), int(point[0][1])] for point in approx]
                    result['geometry'] = polygon
            
        except Exception as e:
            print(f"Error: {e}")
        
        return result

# Usage example:
# processor = MinimalProcessor()
# result = processor.process_document("sample.png")
# print(result)
'''
    
    with open("minimal_processor.py", "w") as f:
        f.write(minimal_code)
    
    print("✓ Created minimal_processor.py")
    print("  This processor only does basic polygon detection with OpenCV")
    print("  No OCR capability, but will extract geometric boundaries")

if __name__ == "__main__":
    print("Starting comprehensive document processing tests...")
    
    # Run all tests
    test_extraction_methods()
    batch_results = test_batch_processing()
    capabilities = analyze_current_capabilities()
    create_minimal_working_example()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    # Summary
    print(f"✓ Created test images and processed them")
    print(f"✓ Generated sample submission CSV")
    print(f"✓ Analyzed {len(capabilities)} capabilities")
    print(f"✓ Created minimal working processor")
    
    print("\nNext steps:")
    print("1. Install OCR libraries for better text extraction:")
    print("   pip install pytesseract easyocr")
    print("2. Test with real survey plan images")
    print("3. Tune regex patterns for specific document formats")
    print("4. Improve polygon detection algorithms")
