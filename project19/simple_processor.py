# Simple Document Processor - No external OCR dependencies
# This version uses basic computer vision techniques for testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import re

class SimpleDocumentProcessor:
    """Simple document processor without external OCR dependencies"""
    
    def __init__(self):
        self.coordinate_bounds = (40600, 42600, 66500, 71000)
    
    def process_document_simple(self, image_path, plot_id):
        """Simple document processing for testing"""
        print(f"Simple processing for ID {plot_id}...")
        
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Extract some basic information based on image analysis
            metadata = self._extract_basic_metadata(img_array, image_path)
            
            # Try to extract a simple polygon
            polygon = self._extract_simple_polygon(img_array)
            
            # Convert to geographic coordinates if polygon found
            if polygon:
                geographic_polygon = self._convert_to_geographic(polygon, img_array.shape)
            else:
                geographic_polygon = []
            
            result = {
                'ID': str(plot_id),
                'TargetSurvey': metadata.get('TargetSurvey', ''),
                'Certified date': metadata.get('Certified date', ''),
                'Total Area': metadata.get('Total Area', ''),
                'Unit of Measurement': metadata.get('Unit of Measurement', ''),
                'Parish': metadata.get('Parish', ''),
                'LT Num': metadata.get('LT Num', ''),
                'geometry': geographic_polygon
            }
            
            return result
            
        except Exception as e:
            print(f"Simple processing failed for {plot_id}: {e}")
            return self._empty_result(plot_id)
    
    def _extract_basic_metadata(self, img_array, image_path):
        """Extract basic metadata without OCR"""
        # For now, return empty metadata
        # In a real implementation, you could try:
        # 1. Template matching for common text patterns
        # 2. Color analysis to detect text regions
        # 3. Simple character recognition for numbers
        
        metadata = {
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': ''
        }
        
        # Try to extract LT number from filename if it follows a pattern
        filename = os.path.basename(image_path)
        lt_match = re.search(r'(\d{4}-\d{3})', filename)
        if lt_match:
            metadata['LT Num'] = lt_match.group(1)
        
        return metadata
    
    def _extract_simple_polygon(self, img_array):
        """Extract polygon using basic computer vision"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            h, w = img_array.shape[:2]
            min_area = (h * w) * 0.01  # At least 1% of image
            max_area = (h * w) * 0.5   # At most 50% of image
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:  # At least 4 vertices
                        valid_contours.append((area, approx))
            
            if valid_contours:
                # Select largest valid contour
                valid_contours.sort(key=lambda x: x[0], reverse=True)
                best_contour = valid_contours[0][1]
                
                # Convert to coordinate list
                coords = [(float(pt[0][0]), float(pt[0][1])) for pt in best_contour]
                return coords
            
            return []
            
        except Exception as e:
            print(f"Simple polygon extraction failed: {e}")
            return []
    
    def _convert_to_geographic(self, pixel_coords, image_shape):
        """Convert pixel coordinates to geographic coordinates"""
        if not pixel_coords:
            return []
        
        h, w = image_shape[:2]
        minx, maxx, miny, maxy = self.coordinate_bounds
        
        geographic_coords = []
        for x_pixel, y_pixel in pixel_coords:
            x_norm = x_pixel / (w - 1)
            y_norm = y_pixel / (h - 1)
            
            x_geo = x_norm * (maxx - minx) + minx
            y_geo = (1 - y_norm) * (maxy - miny) + miny
            
            geographic_coords.append([x_geo, y_geo])
        
        return geographic_coords
    
    def _empty_result(self, plot_id):
        """Return empty result structure"""
        return {
            'ID': str(plot_id),
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': '',
            'geometry': []
        }

def create_test_submission(test_ids_df, image_dir="data/", output_file="test_submission.csv"):
    """Create a test submission using simple processing"""
    
    processor = SimpleDocumentProcessor()
    results = []
    
    print(f"Creating test submission for {len(test_ids_df)} documents...")
    
    for idx, plot_id in enumerate(test_ids_df['ID']):
        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(test_ids_df)}")
        
        # Find image
        image_path = None
        for pattern in [f"{plot_id}.jpg", f"anonymised_{plot_id}.jpg", f"{plot_id}.png"]:
            potential_path = os.path.join(image_dir, pattern)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path:
            result = processor.process_document_simple(image_path, plot_id)
        else:
            result = processor._empty_result(plot_id)
        
        results.append(result)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(results)
    column_order = ['ID', 'TargetSurvey', 'Certified date', 'Total Area',
                   'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
    submission_df = submission_df[column_order]
    
    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    
    # Print summary
    total_docs = len(submission_df)
    valid_geometries = sum(1 for geom in submission_df['geometry'] if isinstance(geom, list) and len(geom) > 0)
    
    print(f"\nTest submission created:")
    print(f"  Total documents: {total_docs}")
    print(f"  Valid polygons: {valid_geometries}")
    print(f"  Success rate: {valid_geometries/total_docs*100:.1f}%")
    print(f"  Saved to: {output_file}")
    
    return submission_df

if __name__ == "__main__":
    # Example usage
    print("Simple document processor ready!")
    print("Use create_test_submission(test_ids_df) to generate a basic submission")
