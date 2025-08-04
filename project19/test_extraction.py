#!/usr/bin/env python3
"""
Test script to diagnose why metadata extraction is failing
"""

import os
import pandas as pd
from document_processor import DocumentProcessor, MetadataExtractor, DocumentPreprocessor
from PIL import Image
import numpy as np

def test_single_image_extraction(image_path, plot_id):
    """Test extraction on a single image with detailed debugging"""
    
    print(f"\n{'='*60}")
    print(f"TESTING EXTRACTION FOR IMAGE: {os.path.basename(image_path)}")
    print(f"Plot ID: {plot_id}")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        # Test image loading
        img = Image.open(image_path)
        print(f"✓ Image loaded successfully")
        print(f"  Size: {img.size}")
        print(f"  Mode: {img.mode}")
        
        # Test preprocessing
        preprocessor = DocumentPreprocessor()
        enhanced_img = preprocessor.enhance_image(image_path)
        
        if enhanced_img is not None:
            print(f"✓ Image preprocessing successful")
            print(f"  Enhanced shape: {enhanced_img.shape}")
        else:
            print(f"❌ Image preprocessing failed")
            return None
        
        # Test metadata extraction
        extractor = MetadataExtractor()
        
        print(f"\n--- OCR TEXT EXTRACTION ---")
        text = extractor._extract_text_multiple_methods(enhanced_img)
        print(f"Extracted text length: {len(text)} characters")
        print(f"First 500 characters:")
        print(f"'{text[:500]}...'")
        
        print(f"\n--- METADATA PARSING ---")
        metadata = extractor._parse_metadata_from_text(text)
        
        for field, value in metadata.items():
            status = "✓" if value else "❌"
            print(f"{status} {field}: '{value}'")
        
        # Test full document processing
        print(f"\n--- FULL DOCUMENT PROCESSING ---")
        processor = DocumentProcessor()
        result = processor.process_document(image_path, plot_id)
        
        print(f"Final result:")
        for field, value in result.items():
            if field == 'geometry':
                status = "✓" if value else "❌"
                print(f"{status} {field}: {len(value) if isinstance(value, list) else value} coordinates")
            else:
                status = "✓" if value else "❌"
                print(f"{status} {field}: '{value}'")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_multiple_images():
    """Test extraction on multiple images if available"""
    
    print(f"\n{'='*60}")
    print(f"TESTING MULTIPLE IMAGES")
    print(f"{'='*60}")
    
    # Look for test images
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Creating mock test...")
        test_mock_extraction()
        return
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.pdf']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(data_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"❌ No image files found in {data_dir}")
        print("Creating mock test...")
        test_mock_extraction()
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Test first few images
    test_count = min(3, len(image_files))
    results = []
    
    for i in range(test_count):
        image_file = image_files[i]
        image_path = os.path.join(data_dir, image_file)
        plot_id = os.path.splitext(image_file)[0].replace('anonymised_', '')
        
        result = test_single_image_extraction(image_path, plot_id)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    if results:
        # Count successful extractions by field
        field_success = {}
        for result in results:
            for field, value in result.items():
                if field not in field_success:
                    field_success[field] = 0
                if value and value != '' and value != []:
                    field_success[field] += 1
        
        print(f"Extraction success rates ({len(results)} tests):")
        for field, count in field_success.items():
            rate = count / len(results) * 100
            print(f"  {field}: {count}/{len(results)} ({rate:.1f}%)")
    else:
        print("No successful extractions")

def test_mock_extraction():
    """Test with mock data when real images aren't available"""
    
    print(f"\n{'='*60}")
    print(f"TESTING WITH MOCK DATA")
    print(f"{'='*60}")
    
    # Create a simple mock image with text
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a white image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Add some mock survey plan text
    text_lines = [
        "SURVEY PLAN",
        "Plot Number: 12345",
        "Parish: ST. MICHAEL",
        "Total Area: 2.5 acres",
        "Certified Date: 2023-01-15",
        "LT Number: LT-789",
        "Surveyed For: John Smith"
    ]
    
    y_position = 50
    for line in text_lines:
        draw.text((50, y_position), line, fill='black', font=font)
        y_position += 40
    
    # Draw a simple rectangle to represent a plot boundary
    draw.rectangle([200, 300, 600, 500], outline='black', width=2)
    
    # Save mock image
    mock_path = "mock_survey_plan.png"
    img.save(mock_path)
    print(f"Created mock image: {mock_path}")
    
    # Test extraction on mock image
    result = test_single_image_extraction(mock_path, "mock_12345")
    
    # Cleanup
    if os.path.exists(mock_path):
        os.remove(mock_path)
    
    return result

def run_full_diagnostic():
    """Run complete diagnostic of the extraction system"""
    
    print(f"DOCUMENT EXTRACTION DIAGNOSTIC")
    print(f"{'='*60}")
    
    # Check dependencies
    print("Checking dependencies...")
    
    try:
        import pytesseract
        print("✓ Tesseract available")
    except ImportError:
        print("❌ Tesseract not available")
    
    try:
        import easyocr
        print("✓ EasyOCR available")
    except ImportError:
        print("❌ EasyOCR not available")
    
    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        print("❌ OpenCV not available")
    
    # Test pattern matching
    print(f"\n--- TESTING REGEX PATTERNS ---")
    extractor = MetadataExtractor()
    
    sample_texts = [
        "Survey Plan for John Smith Parish: St. Michael Area: 2.5 acres",
        "Plot 12345 LT Number: LT-789 Certified: 2023-01-15",
        "Total area 1.2 hectares surveyed for Jane Doe parish Christ Church"
    ]
    
    for i, text in enumerate(sample_texts):
        print(f"\nSample text {i+1}: '{text}'")
        metadata = extractor._parse_metadata_from_text(text)
        for field, value in metadata.items():
            if value:
                print(f"  ✓ Found {field}: '{value}'")
            else:
                print(f"  ❌ Missing {field}")
    
    # Test on real/mock images
    test_multiple_images()

if __name__ == "__main__":
    run_full_diagnostic()
