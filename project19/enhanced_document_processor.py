# Enhanced Document Processor with improved extraction methods
# This version includes fallback methods and better handling of various document types

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import re
import json
from scipy import ndimage
from skimage import filters, morphology, feature, measure, segmentation

# Try to import OCR libraries with fallbacks
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("EasyOCR available")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available")

from skimage.draw import polygon as draw_polygon
from skimage import measure

class EnhancedDocumentProcessor:
    """Enhanced document processor with better extraction methods"""
    
    def __init__(self):
        self.coordinate_bounds = (40600, 42600, 66500, 71000)
        
        # Initialize OCR readers
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                print("EasyOCR reader initialized")
            except Exception as e:
                print(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
        else:
            self.easyocr_reader = None
        
        # Enhanced field patterns
        self.field_patterns = {
            'Total Area': [
                r'(?:total\s+)?area[:\s]+([0-9,\.]+)\s*(acres?|sq\.?\s*ft\.?|hectares?)',
                r'([0-9,\.]+)\s*acres?',
                r'([0-9,\.]+)\s*sq\.?\s*ft\.?',
                r'([0-9,\.]+)\s*hectares?',
                r'area[:\s=]+([0-9,\.]+)',
                r'([0-9,\.]+)\s*ac\.?',
            ],
            'Unit of Measurement': [
                r'(acres?|sq\.?\s*ft\.?|hectares?|square\s+feet|ac\.?)',
            ],
            'Parish': [
                r'parish[:\s]+([a-zA-Z\s]+)',
                r'(christ\s+church|st\.?\s*michael|st\.?\s*george|st\.?\s*philip|st\.?\s*john|st\.?\s*james|st\.?\s*thomas|st\.?\s*joseph|st\.?\s*andrew|st\.?\s*peter|st\.?\s*lucy)',
                r'(christ\s+church|st\s+michael|st\s+george|st\s+philip|st\s+john|st\s+james|st\s+thomas|st\s+joseph|st\s+andrew|st\s+peter|st\s+lucy)',
            ],
            'Surveyed For': [
                r'surveyed\s+for[:\s]+([a-zA-Z\s]+)',
                r'owner[:\s]+([a-zA-Z\s]+)',
                r'client[:\s]+([a-zA-Z\s]+)',
                r'property\s+of[:\s]+([a-zA-Z\s]+)',
            ],
            'Certified date': [
                r'certified[:\s]+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})',
                r'date[:\s]+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})',
                r'([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})',
                r'certified[:\s]+([a-zA-Z]+\s+[0-9]{1,2},?\s+[0-9]{4})',
            ],
            'LT Num': [
                r'lt\.?\s*no\.?\s*([0-9]+)',
                r'lot\s+(?:no\.?\s*)?([0-9]+)',
                r'plot\s+(?:no\.?\s*)?([0-9]+)',
                r'parcel\s+([0-9]+)',
                r'block\s+([0-9]+)',
            ]
        }
    
    def enhanced_preprocess(self, image_path):
        """Enhanced preprocessing for different document types"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Try multiple preprocessing approaches
            processed_images = []
            
            # Method 1: Standard approach
            standard = self._standard_preprocessing(img_array)
            processed_images.append(("standard", standard))
            
            # Method 2: High contrast for text
            high_contrast = self._high_contrast_preprocessing(img_array)
            processed_images.append(("high_contrast", high_contrast))
            
            # Method 3: Edge enhancement
            edge_enhanced = self._edge_enhancement_preprocessing(img_array)
            processed_images.append(("edge_enhanced", edge_enhanced))
            
            return processed_images
            
        except Exception as e:
            print(f"Enhanced preprocessing failed: {e}")
            return [("original", np.array(Image.open(image_path).convert('RGB')))]
    
    def _standard_preprocessing(self, img_array):
        """Standard document preprocessing"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    def _high_contrast_preprocessing(self, img_array):
        """High contrast preprocessing for text extraction"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply strong contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        
        # Apply threshold to create binary image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    def _edge_enhancement_preprocessing(self, img_array):
        """Edge enhancement for polygon detection"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Enhance edges
        edges = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        
        # Combine with original
        combined = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
        
        return cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
    
    def extract_text_robust(self, processed_images):
        """Extract text using multiple methods and preprocessing approaches"""
        all_text = []
        
        for method_name, img in processed_images:
            print(f"  Trying OCR on {method_name} image...")
            
            # Convert to PIL Image for OCR
            pil_img = Image.fromarray(img)
            
            # Method 1: Tesseract with different configs
            if TESSERACT_AVAILABLE:
                try:
                    # Try different PSM modes
                    for psm in [6, 8, 13]:
                        config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:-/() '
                        text = pytesseract.image_to_string(pil_img, config=config)
                        if text.strip():
                            all_text.append(f"tesseract_psm{psm}_{method_name}: {text}")
                except Exception as e:
                    print(f"    Tesseract failed on {method_name}: {e}")
            
            # Method 2: EasyOCR
            if self.easyocr_reader:
                try:
                    results = self.easyocr_reader.readtext(img)
                    easyocr_text = ' '.join([result[1] for result in results if result[2] > 0.5])
                    if easyocr_text.strip():
                        all_text.append(f"easyocr_{method_name}: {easyocr_text}")
                except Exception as e:
                    print(f"    EasyOCR failed on {method_name}: {e}")
        
        # Combine all text
        combined_text = ' '.join(all_text).lower()
        print(f"  Total text extracted: {len(combined_text)} characters")
        return combined_text
    
    def extract_metadata_enhanced(self, image_path):
        """Enhanced metadata extraction with multiple approaches"""
        try:
            # Get multiple preprocessed versions
            processed_images = self.enhanced_preprocess(image_path)
            
            # Extract text from all versions
            all_text = self.extract_text_robust(processed_images)
            
            # Parse metadata
            metadata = self._empty_metadata()
            
            if all_text:
                for field, patterns in self.field_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, all_text, re.IGNORECASE)
                        if matches:
                            # Take the first match
                            if field == 'Unit of Measurement':
                                metadata[field] = matches[0].strip()
                            else:
                                metadata[field] = matches[0].strip() if isinstance(matches[0], str) else matches[0]
                            break
            
            return metadata
            
        except Exception as e:
            print(f"Enhanced metadata extraction failed: {e}")
            return self._empty_metadata()
    
    def extract_polygon_enhanced(self, image_path):
        """Enhanced polygon extraction with multiple methods"""
        try:
            processed_images = self.enhanced_preprocess(image_path)
            
            all_polygons = []
            
            for method_name, img in processed_images:
                print(f"  Trying polygon extraction on {method_name} image...")
                
                # Method 1: Contour detection
                contours = self._extract_contours_enhanced(img)
                all_polygons.extend(contours)
                
                # Method 2: Line detection
                lines = self._extract_via_lines_enhanced(img)
                all_polygons.extend(lines)
                
                # Method 3: Corner detection
                corners = self._extract_via_corners(img)
                all_polygons.extend(corners)
            
            # Filter and select best polygon
            if all_polygons:
                best_polygon = self._select_best_polygon_enhanced(all_polygons, processed_images[0][1].shape)
                return best_polygon
            
            return []
            
        except Exception as e:
            print(f"Enhanced polygon extraction failed: {e}")
            return []
    
    def _extract_contours_enhanced(self, img):
        """Enhanced contour detection"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Multiple edge detection methods
            edges_list = []
            
            # Canny edge detection
            edges1 = cv2.Canny(gray, 30, 100)
            edges_list.append(edges1)
            
            # Sobel edge detection
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges2 = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
            edges_list.append(edges2)
            
            polygons = []
            
            for edges in edges_list:
                # Morphological operations
                kernel = np.ones((2,2), np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 500 < area < 200000:  # Adjusted area thresholds
                        # Approximate contour
                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        if len(approx) >= 3:  # At least triangle
                            coords = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                            polygons.append(coords)
            
            return polygons
            
        except Exception as e:
            print(f"    Enhanced contour extraction failed: {e}")
            return []
    
    def _extract_via_lines_enhanced(self, img):
        """Enhanced line-based polygon extraction"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is None or len(lines) < 3:
                return []
            
            # This is a simplified version - full implementation would need
            # sophisticated line intersection and polygon reconstruction
            return []
            
        except Exception as e:
            print(f"    Enhanced line extraction failed: {e}")
            return []
    
    def _extract_via_corners(self, img):
        """Extract polygons using corner detection"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Harris corner detection
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            corners = cv2.dilate(corners, None)
            
            # Find corner points
            corner_points = np.where(corners > 0.01 * corners.max())
            
            if len(corner_points[0]) < 4:
                return []
            
            # This is a simplified version - full implementation would need
            # clustering and polygon reconstruction from corner points
            return []
            
        except Exception as e:
            print(f"    Corner extraction failed: {e}")
            return []
    
    def _select_best_polygon_enhanced(self, polygons, image_shape):
        """Enhanced polygon selection with better scoring"""
        if not polygons:
            return []
        
        scored_polygons = []
        
        for poly in polygons:
            score = self._score_polygon_enhanced(poly, image_shape)
            if score > 0:
                scored_polygons.append((score, poly))
        
        if not scored_polygons:
            return []
        
        # Sort by score and return best
        scored_polygons.sort(reverse=True)
        return scored_polygons[0][1]
    
    def _score_polygon_enhanced(self, polygon, image_shape):
        """Enhanced polygon scoring"""
        if len(polygon) < 3:
            return 0
        
        try:
            poly_array = np.array(polygon)
            area = cv2.contourArea(poly_array.astype(np.int32))
            
            h, w = image_shape[:2]
            image_area = h * w
            area_ratio = area / image_area
            
            # Score components
            area_score = 1.0 if 0.01 < area_ratio < 0.3 else 0.3
            
            # Prefer polygons with reasonable number of vertices
            vertex_score = 1.0 if 4 <= len(polygon) <= 12 else 0.5
            
            # Prefer polygons not too close to edges
            center_x = np.mean(poly_array[:, 0])
            center_y = np.mean(poly_array[:, 1])
            edge_distance = min(center_x, w - center_x, center_y, h - center_y)
            edge_score = min(1.0, edge_distance / 50)  # Penalize if too close to edge
            
            total_score = area_score * vertex_score * edge_score
            return total_score
            
        except Exception as e:
            print(f"    Polygon scoring failed: {e}")
            return 0
    
    def convert_coordinates(self, pixel_coords, image_shape):
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
    
    def process_document_enhanced(self, image_path, plot_id):
        """Main processing function with enhanced methods"""
        print(f"Enhanced processing for ID {plot_id}...")
        
        try:
            # Extract metadata
            print("Extracting metadata...")
            metadata = self.extract_metadata_enhanced(image_path)
            
            # Extract polygon
            print("Extracting polygon...")
            pixel_polygon = self.extract_polygon_enhanced(image_path)
            
            # Convert coordinates
            if pixel_polygon:
                img = Image.open(image_path)
                image_shape = (img.height, img.width)
                geographic_polygon = self.convert_coordinates(pixel_polygon, image_shape)
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
            
            print(f"Completed processing for ID {plot_id}")
            return result
            
        except Exception as e:
            print(f"Enhanced processing failed for {plot_id}: {e}")
            return self._empty_result(plot_id)
    
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

# Convenience function for batch processing
def process_with_enhanced_methods(test_ids_df, image_dir="data/", output_file="submission_enhanced.csv"):
    """Process documents using enhanced methods"""
    
    processor = EnhancedDocumentProcessor()
    results = []
    
    print(f"Processing {len(test_ids_df)} documents with enhanced methods...")
    
    for idx, plot_id in enumerate(test_ids_df['ID']):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(test_ids_df)} ({idx/len(test_ids_df)*100:.1f}%)")
        
        # Find image
        image_path = None
        for pattern in [f"{plot_id}.jpg", f"anonymised_{plot_id}.jpg", f"{plot_id}.png"]:
            potential_path = os.path.join(image_dir, pattern)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path:
            result = processor.process_document_enhanced(image_path, plot_id)
        else:
            print(f"No image found for ID {plot_id}")
            result = processor._empty_result(plot_id)
        
        results.append(result)
    
    # Create submission
    submission_df = pd.DataFrame(results)
    column_order = ['ID', 'TargetSurvey', 'Certified date', 'Total Area',
                   'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
    submission_df = submission_df[column_order]
    submission_df.to_csv(output_file, index=False)
    
    print(f"Enhanced processing completed. Saved to {output_file}")
    return submission_df
