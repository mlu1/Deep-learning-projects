# Barbados Document Analysis - Scanned Plans Processing
# This solution extracts polygon boundaries and metadata from scanned land survey documents

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import os
import glob
import warnings
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import re
import pytesseract
from scipy import ndimage
from skimage import filters, morphology, feature, measure, segmentation
import json

# Optional: Document processing libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. Install with: pip install easyocr")

try:
    from skimage.draw import polygon as draw_polygon
    from skimage import measure
except ImportError:
    raise ImportError("scikit-image is required: pip install scikit-image")

print("Document Analysis Mode - Processing Scanned Plans")
print("=" * 60)

# --- Step 1: Document Image Preprocessing ---
class DocumentPreprocessor:
    """Preprocess scanned document images for better analysis"""
    
    def __init__(self):
        self.target_dpi = 300
        self.target_size = (2048, 2048)  # Larger size for document analysis
    
    def enhance_image(self, image_path):
        """Apply preprocessing to improve document readability"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize if too large
            if max(img.size) > 3000:
                ratio = 3000 / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Apply document enhancement techniques
            enhanced = self._apply_document_enhancement(img_array)
            
            return enhanced
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def _apply_document_enhancement(self, img_array):
        """Apply specific enhancements for scanned documents"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Convert back to RGB
        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        
        return result

# --- Step 2: OCR-based Metadata Extraction ---
class MetadataExtractor:
    """Extract text metadata from scanned documents"""
    
    def __init__(self):
        # Initialize OCR engines
        if EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(['en'])
        
        # Define field patterns for Barbados land survey documents
        self.field_patterns = {
            'Total Area': [
                r'total area[:\s]+([0-9,\.]+)\s*(acres?|sq\.?\s*ft\.?|hectares?)',
                r'area[:\s]+([0-9,\.]+)\s*(acres?|sq\.?\s*ft\.?|hectares?)',
                r'([0-9,\.]+)\s*acres?',
                r'([0-9,\.]+)\s*sq\.?\s*ft\.?'
            ],
            'Unit of Measurement': [
                r'(acres?|sq\.?\s*ft\.?|hectares?|square\s+feet)',
            ],
            'Parish': [
                r'parish[:\s]+([a-zA-Z\s]+)',
                r'(christ church|st\.\s*michael|st\.\s*george|st\.\s*philip|st\.\s*john|st\.\s*james|st\.\s*thomas|st\.\s*joseph|st\.\s*andrew|st\.\s*peter|st\.\s*lucy)',
            ],
            'Surveyed For': [
                r'surveyed for[:\s]+([a-zA-Z\s]+)',
                r'owner[:\s]+([a-zA-Z\s]+)',
                r'client[:\s]+([a-zA-Z\s]+)',
            ],
            'Certified date': [
                r'certified[:\s]+([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})',
                r'date[:\s]+([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})',
                r'([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})',
            ],
            'LT Num': [
                r'lt\.?\s*no\.?\s*([0-9]+)',
                r'lot\s+([0-9]+)',
                r'plot\s+([0-9]+)',
            ]
        }
    
    def extract_metadata(self, image_path):
        """Extract metadata from document image"""
        try:
            # Preprocess image
            preprocessor = DocumentPreprocessor()
            enhanced_img = preprocessor.enhance_image(image_path)
            
            if enhanced_img is None:
                return self._empty_metadata()
            
            # Extract text using multiple OCR methods
            text_data = self._extract_text_multiple_methods(enhanced_img)
            
            # Parse metadata from text
            metadata = self._parse_metadata_from_text(text_data)
            
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            return self._empty_metadata()
    
    def _extract_text_multiple_methods(self, image):
        """Use multiple OCR methods for better accuracy"""
        all_text = []
        
        # Method 1: Tesseract OCR
        try:
            # Convert to PIL Image for tesseract
            pil_img = Image.fromarray(image)
            tesseract_text = pytesseract.image_to_string(pil_img, config='--psm 6')
            all_text.append(tesseract_text)
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")
        
        # Method 2: EasyOCR (if available)
        if EASYOCR_AVAILABLE:
            try:
                easyocr_results = self.reader.readtext(image)
                easyocr_text = ' '.join([result[1] for result in easyocr_results])
                all_text.append(easyocr_text)
            except Exception as e:
                print(f"EasyOCR failed: {e}")
        
        # Combine all text
        combined_text = ' '.join(all_text).lower()
        return combined_text
    
    def _parse_metadata_from_text(self, text):
        """Parse specific metadata fields from extracted text"""
        metadata = self._empty_metadata()
        
        for field, patterns in self.field_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if field == 'Unit of Measurement':
                        metadata[field] = match.group(0).strip()
                    else:
                        metadata[field] = match.group(1).strip()
                    break
        
        return metadata
    
    def _empty_metadata(self):
        """Return empty metadata structure"""
        return {
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': ''
        }

# --- Step 3: Polygon Extraction from Scanned Plans ---
class PolygonExtractor:
    """Extract polygon boundaries from scanned land survey documents"""
    
    def __init__(self):
        self.min_area = 1000  # Minimum area for valid polygons
        self.max_area = 500000  # Maximum area to filter out full page detections
    
    def extract_polygon_from_document(self, image_path):
        """Extract main plot boundary from scanned document"""
        try:
            # Load and preprocess image
            preprocessor = DocumentPreprocessor()
            enhanced_img = preprocessor.enhance_image(image_path)
            
            if enhanced_img is None:
                return []
            
            # Try multiple polygon detection methods
            polygons = []
            
            # Method 1: Contour detection
            contour_polygons = self._extract_via_contours(enhanced_img)
            polygons.extend(contour_polygons)
            
            # Method 2: Line detection and intersection
            line_polygons = self._extract_via_lines(enhanced_img)
            polygons.extend(line_polygons)
            
            # Method 3: Edge detection
            edge_polygons = self._extract_via_edges(enhanced_img)
            polygons.extend(edge_polygons)
            
            # Filter and select best polygon
            best_polygon = self._select_best_polygon(polygons, enhanced_img.shape)
            
            return best_polygon
            
        except Exception as e:
            print(f"Error extracting polygon from {image_path}: {e}")
            return []
    
    def _extract_via_contours(self, image):
        """Extract polygons using contour detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Morphological operations to connect broken lines
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            polygons = []
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Filter by area and number of vertices
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area and len(approx) >= 4:
                    # Convert to coordinate list
                    coords = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                    polygons.append(coords)
            
            return polygons
            
        except Exception as e:
            print(f"Contour extraction failed: {e}")
            return []
    
    def _extract_via_lines(self, image):
        """Extract polygons by detecting lines and finding intersections"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect lines using HoughLines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) < 4:
                return []
            
            # Process lines to find rectangular boundaries
            # This is a simplified approach - in practice, you'd need more sophisticated
            # line intersection and polygon reconstruction algorithms
            
            # For now, return empty list - this method needs more development
            return []
            
        except Exception as e:
            print(f"Line extraction failed: {e}")
            return []
    
    def _extract_via_edges(self, image):
        """Extract polygons using edge detection and morphological operations"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Morphological operations
            kernel = np.ones((5,5), np.uint8)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            polygons = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    # Approximate to polygon
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:
                        coords = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                        polygons.append(coords)
            
            return polygons
            
        except Exception as e:
            print(f"Edge extraction failed: {e}")
            return []
    
    def _select_best_polygon(self, polygons, image_shape):
        """Select the best polygon from candidates"""
        if not polygons:
            return []
        
        # Score polygons based on various criteria
        scored_polygons = []
        
        for poly in polygons:
            score = self._score_polygon(poly, image_shape)
            scored_polygons.append((score, poly))
        
        # Sort by score and return best
        scored_polygons.sort(reverse=True)
        
        if scored_polygons and scored_polygons[0][0] > 0:
            return scored_polygons[0][1]
        
        return []
    
    def _score_polygon(self, polygon, image_shape):
        """Score polygon based on various criteria"""
        if len(polygon) < 4:
            return 0
        
        # Convert to numpy array for calculations
        poly_array = np.array(polygon)
        
        # Calculate area
        area = cv2.contourArea(poly_array.astype(np.int32))
        
        # Normalize area score (prefer medium-sized polygons)
        h, w = image_shape[:2]
        image_area = h * w
        area_ratio = area / image_area
        
        # Prefer polygons that are 5-50% of image area
        if 0.05 < area_ratio < 0.5:
            area_score = 1.0
        else:
            area_score = 0.5
        
        # Prefer polygons closer to center
        center_x, center_y = w // 2, h // 2
        poly_center_x = np.mean(poly_array[:, 0])
        poly_center_y = np.mean(poly_array[:, 1])
        
        center_distance = np.sqrt((poly_center_x - center_x)**2 + (poly_center_y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        center_score = 1.0 - (center_distance / max_distance)
        
        # Prefer rectangularish shapes (4-8 vertices)
        vertex_score = 1.0 if 4 <= len(polygon) <= 8 else 0.5
        
        # Combine scores
        total_score = area_score * center_score * vertex_score
        
        return total_score

# --- Step 4: Coordinate System Conversion ---
class CoordinateConverter:
    """Convert pixel coordinates to geographic coordinates"""
    
    def __init__(self, bounds=(40600, 42600, 66500, 71000)):
        self.bounds = bounds  # (minx, maxx, miny, maxy)
    
    def pixel_to_geographic(self, pixel_coords, image_shape):
        """Convert pixel coordinates to geographic coordinates"""
        if not pixel_coords:
            return []
        
        h, w = image_shape[:2]
        minx, maxx, miny, maxy = self.bounds
        
        geographic_coords = []
        for x_pixel, y_pixel in pixel_coords:
            # Normalize pixel coordinates to [0, 1]
            x_norm = x_pixel / (w - 1)
            y_norm = y_pixel / (h - 1)
            
            # Convert to geographic coordinates
            x_geo = x_norm * (maxx - minx) + minx
            y_geo = (1 - y_norm) * (maxy - miny) + miny  # Flip Y axis
            
            geographic_coords.append([x_geo, y_geo])
        
        return geographic_coords

# --- Step 5: Main Document Processing Pipeline ---
class DocumentProcessor:
    """Main pipeline for processing scanned land survey documents"""
    
    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.polygon_extractor = PolygonExtractor()
        self.coordinate_converter = CoordinateConverter()
        
    def process_document(self, image_path, plot_id):
        """Process a single document and extract all information"""
        try:
            print(f"Processing document for ID {plot_id}...")
            
            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(image_path)
            
            # Extract polygon
            pixel_polygon = self.polygon_extractor.extract_polygon_from_document(image_path)
            
            # Convert coordinates if polygon found
            if pixel_polygon:
                # Get image shape for coordinate conversion
                img = Image.open(image_path)
                image_shape = (img.height, img.width)
                geographic_polygon = self.coordinate_converter.pixel_to_geographic(
                    pixel_polygon, image_shape
                )
            else:
                geographic_polygon = []
            
            # Combine results
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
            print(f"Error processing document {image_path}: {e}")
            return self._empty_result(plot_id)
    
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

# --- Step 6: Batch Processing for Test Set ---
def process_test_documents(test_ids_df, image_dir="data/", output_file="submission.csv"):
    """Process all test documents and create submission"""
    
    processor = DocumentProcessor()
    results = []
    
    print(f"\nProcessing {len(test_ids_df)} test documents...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for idx, plot_id in enumerate(test_ids_df['ID']):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(test_ids_df)} ({idx/len(test_ids_df)*100:.1f}%)")
        
        # Find image file
        image_path = None
        for pattern in [f"{plot_id}.jpg", f"anonymised_{plot_id}.jpg", f"{plot_id}.png", f"{plot_id}.pdf"]:
            potential_path = os.path.join(image_dir, pattern)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"Warning: No image found for ID {plot_id}")
            result = processor._empty_result(plot_id)
            failed += 1
        else:
            result = processor.process_document(image_path, plot_id)
            if result['geometry']:
                successful += 1
            else:
                failed += 1
        
        results.append(result)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(results)
    
    # Ensure column order
    column_order = ['ID', 'TargetSurvey', 'Certified date', 'Total Area',
                   'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
    submission_df = submission_df[column_order]
    
    # Save submission
    submission_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DOCUMENT PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total documents: {len(test_ids_df)}")
    print(f"Successful extractions: {successful}")
    print(f"Failed extractions: {failed}")
    print(f"Success rate: {successful/len(test_ids_df)*100:.1f}%")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")
    
    return submission_df

# --- Step 7: Visualization and Debugging ---
def visualize_document_processing(test_ids_df, n_samples=4, image_dir="data/"):
    """Visualize document processing results for debugging"""
    
    processor = DocumentProcessor()
    sample_ids = test_ids_df['ID'].sample(n=min(n_samples, len(test_ids_df))).values
    
    fig, axes = plt.subplots(2, n_samples//2, figsize=(6*n_samples//2, 12))
    if n_samples <= 2:
        axes = axes.reshape(-1, 1)
    
    for idx, plot_id in enumerate(sample_ids):
        row = idx // (n_samples//2)
        col = idx % (n_samples//2)
        
        # Find image
        image_path = None
        for pattern in [f"{plot_id}.jpg", f"anonymised_{plot_id}.jpg", f"{plot_id}.png"]:
            potential_path = os.path.join(image_dir, pattern)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            axes[row, col].text(0.5, 0.5, f'No image found\nfor ID {plot_id}', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'ID: {plot_id} (Missing)')
            axes[row, col].axis('off')
            continue
        
        try:
            # Load and process image
            img = Image.open(image_path).convert('RGB')
            
            # Extract polygon
            pixel_polygon = processor.polygon_extractor.extract_polygon_from_document(image_path)
            
            # Display image
            axes[row, col].imshow(img)
            
            # Overlay polygon if found
            if pixel_polygon:
                poly_array = np.array(pixel_polygon)
                # Close the polygon
                if len(poly_array) > 0:
                    poly_array = np.vstack([poly_array, poly_array[0]])
                    axes[row, col].plot(poly_array[:, 0], poly_array[:, 1], 'r-', linewidth=2)
                    axes[row, col].plot(poly_array[:, 0], poly_array[:, 1], 'ro', markersize=4)
            
            axes[row, col].set_title(f'ID: {plot_id}\n{len(pixel_polygon)} vertices found')
            axes[row, col].axis('off')
            
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'ID: {plot_id} (Error)')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('document_processing_debug.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # This file should be imported and used with the test data
    print("Document processing pipeline ready!")
    print("Use: process_test_documents(test_ids_df) to process all test documents")
