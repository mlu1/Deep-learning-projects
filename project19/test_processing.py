#!/usr/bin/env python3
"""
Test script for document processing
This script tests document processing with whatever libraries are available
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def test_basic_opencv():
    """Test if basic OpenCV functionality works"""
    try:
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75] = [255, 255, 255]  # White square
        
        # Test basic OpenCV operations
        gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print("✅ OpenCV basic functionality working")
        print(f"   Found {len(contours)} contours in test image")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_image_loading(image_dir="data/"):
    """Test loading images from the data directory"""
    try:
        if not os.path.exists(image_dir):
            print(f"❌ Directory {image_dir} does not exist")
            return False
        
        # Find image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']:
            import glob
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return False
        
        print(f"✅ Found {len(image_files)} image files")
        
        # Test loading the first image
        test_img_path = image_files[0]
        img = Image.open(test_img_path).convert('RGB')
        img_array = np.array(img)
        
        print(f"✅ Successfully loaded test image: {os.path.basename(test_img_path)}")
        print(f"   Image size: {img.size}")
        print(f"   Array shape: {img_array.shape}")
        
        return True, image_files
        
    except Exception as e:
        print(f"❌ Image loading test failed: {e}")
        return False, []

def create_minimal_submission(test_ids_file="Test.csv", output_file="minimal_submission.csv"):
    """Create a minimal submission with just IDs and empty fields"""
    try:
        # Load test IDs
        if not os.path.exists(test_ids_file):
            print(f"❌ {test_ids_file} not found")
            return False
        
        test_ids = pd.read_csv(test_ids_file)
        print(f"✅ Loaded {len(test_ids)} test IDs")
        
        # Create minimal submission
        results = []
        for plot_id in test_ids['ID']:
            results.append({
                'ID': str(plot_id),
                'TargetSurvey': '',
                'Certified date': '',
                'Total Area': '',
                'Unit of Measurement': '',
                'Parish': '',
                'LT Num': '',
                'geometry': []
            })
        
        # Save submission
        submission_df = pd.DataFrame(results)
        column_order = ['ID', 'TargetSurvey', 'Certified date', 'Total Area',
                       'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
        submission_df = submission_df[column_order]
        submission_df.to_csv(output_file, index=False)
        
        print(f"✅ Minimal submission created: {output_file}")
        print(f"   Sample output:")
        print(submission_df.head().to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal submission creation failed: {e}")
        return False

def test_simple_polygon_extraction(image_path):
    """Test simple polygon extraction on an image"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        h, w = img_array.shape[:2]
        min_area = (h * w) * 0.01
        max_area = (h * w) * 0.5
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) >= 4:
                    valid_contours.append((area, approx))
        
        print(f"✅ Polygon extraction test on {os.path.basename(image_path)}:")
        print(f"   Total contours: {len(contours)}")
        print(f"   Valid polygons: {len(valid_contours)}")
        
        if valid_contours:
            # Get the largest polygon
            valid_contours.sort(key=lambda x: x[0], reverse=True)
            best_contour = valid_contours[0][1]
            coords = [(float(pt[0][0]), float(pt[0][1])) for pt in best_contour]
            
            print(f"   Best polygon: {len(coords)} vertices")
            print(f"   Area: {valid_contours[0][0]:.1f} pixels")
            
            # Visualize
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img_array)
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(edges, cmap='gray')
            plt.title('Edges')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(img_array)
            poly_array = np.array(coords)
            poly_closed = np.vstack([poly_array, poly_array[0]])
            plt.plot(poly_closed[:, 0], poly_closed[:, 1], 'r-', linewidth=2)
            plt.plot(poly_closed[:, 0], poly_closed[:, 1], 'ro', markersize=4)
            plt.title('Detected Polygon')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('polygon_test.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            return coords
        else:
            print("   No valid polygons found")
            return []
            
    except Exception as e:
        print(f"❌ Polygon extraction test failed: {e}")
        return []

def run_comprehensive_test():
    """Run comprehensive test of document processing capabilities"""
    print("="*60)
    print("DOCUMENT PROCESSING COMPREHENSIVE TEST")
    print("="*60)
    
    # Test 1: Basic OpenCV
    print("\n1. TESTING OPENCV FUNCTIONALITY")
    print("-" * 40)
    opencv_ok = test_basic_opencv()
    
    # Test 2: Image loading
    print("\n2. TESTING IMAGE LOADING")
    print("-" * 40)
    images_ok, image_files = test_image_loading()
    
    # Test 3: Minimal submission
    print("\n3. TESTING MINIMAL SUBMISSION CREATION")
    print("-" * 40)
    submission_ok = create_minimal_submission()
    
    # Test 4: Polygon extraction (if images available)
    if images_ok and image_files:
        print("\n4. TESTING POLYGON EXTRACTION")
        print("-" * 40)
        test_image = image_files[0]
        polygon = test_simple_polygon_extraction(test_image)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"OpenCV functionality: {'✅' if opencv_ok else '❌'}")
    print(f"Image loading: {'✅' if images_ok else '❌'}")
    print(f"Minimal submission: {'✅' if submission_ok else '❌'}")
    if images_ok and image_files:
        print(f"Polygon extraction: {'✅' if polygon else '❌'}")
    
    if opencv_ok and images_ok and submission_ok:
        print("\n🎉 Basic document processing capabilities are working!")
        print("You can proceed with more advanced processing.")
    else:
        print("\n⚠️  Some basic functionality is not working.")
        print("Please check the error messages above.")

def create_test_submission_basic():
    """Create a test submission using only basic OpenCV"""
    try:
        # Load test IDs
        test_ids = pd.read_csv("Test.csv")
        print(f"Processing {len(test_ids)} test documents...")
        
        results = []
        successful = 0
        
        for idx, plot_id in enumerate(test_ids['ID']):
            if idx % 20 == 0:
                print(f"Progress: {idx}/{len(test_ids)}")
            
            # Find image
            image_path = None
            for pattern in [f"{plot_id}.jpg", f"anonymised_{plot_id}.jpg", f"{plot_id}.png"]:
                potential_path = os.path.join("data", pattern)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path:
                # Try to extract polygon
                polygon = test_simple_polygon_extraction(image_path)
                if polygon:
                    successful += 1
                    # Convert to geographic coordinates (simplified)
                    geographic_coords = []
                    img = Image.open(image_path)
                    h, w = img.height, img.width
                    minx, maxx, miny, maxy = 40600, 42600, 66500, 71000
                    
                    for x_pixel, y_pixel in polygon:
                        x_norm = x_pixel / (w - 1)
                        y_norm = y_pixel / (h - 1)
                        x_geo = x_norm * (maxx - minx) + minx
                        y_geo = (1 - y_norm) * (maxy - miny) + miny
                        geographic_coords.append([x_geo, y_geo])
                    
                    geometry = geographic_coords
                else:
                    geometry = []
            else:
                geometry = []
            
            results.append({
                'ID': str(plot_id),
                'TargetSurvey': '',
                'Certified date': '',
                'Total Area': '',
                'Unit of Measurement': '',
                'Parish': '',
                'LT Num': '',
                'geometry': geometry
            })
        
        # Save submission
        submission_df = pd.DataFrame(results)
        column_order = ['ID', 'TargetSurvey', 'Certified date', 'Total Area',
                       'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
        submission_df = submission_df[column_order]
        submission_df.to_csv("submission_basic.csv", index=False)
        
        print(f"\nBasic submission created:")
        print(f"  Total documents: {len(test_ids)}")
        print(f"  Successful polygon extractions: {successful}")
        print(f"  Success rate: {successful/len(test_ids)*100:.1f}%")
        print(f"  Saved to: submission_basic.csv")
        
        return submission_df
        
    except Exception as e:
        print(f"Basic submission creation failed: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--submission":
        # Create basic submission
        create_test_submission_basic()
    else:
        # Run comprehensive test
        run_comprehensive_test()
