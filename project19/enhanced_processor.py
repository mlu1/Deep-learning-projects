# Enhanced Document Processor with Better Extraction Methods
# Designed to handle real-world scanned land survey documents

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# Try to import OCR libraries with fallbacks
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract not available - install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available - install with: pip install easyocr")

class EnhancedDocumentProcessor:
    """Enhanced document processor with improved extraction methods"""
    
    def __init__(self):
        self.bounds = (40600, 42600, 66500, 71000)  # Barbados coordinate bounds
        
        # Enhanced metadata patterns for Barbados documents
        self.metadata_patterns = {
            'Total Area': [
                r'area[:\s]+([0-9,\.]+)\s*(acres?|sq\.?\s*ft\.?|hectares?)',
                r'total\s+area[:\s]+([0-9,\.]+)\s*(acres?|sq\.?\s*ft\.?|hectares?)',
                r'([0-9,\.]+)\s*acres?',
                r'([0-9,\.]+)\s*sq\.?\s*ft\.?',
                r'([0-9,\.]+)\s*hectares?',
                r'size[:\s]+([0-9,\.]+)\s*(acres?|sq\.?\s*ft\.?|hectares?)',
            ],
            'Unit of Measurement': [
                r'(acres?|sq\.?\s*ft\.?|hectares?|square\s+feet|sqft)',
            ],
            'Parish': [
                r'parish[:\s]+([a-zA-Z\s]+)',
                r'(christ\s+church|st\.?\s*michael|st\.?\s*george|st\.?\s*philip|st\.?\s*john|st\.?\s*james|st\.?\s*thomas|st\.?\s*joseph|st\.?\s*andrew|st\.?\s*peter|st\.?\s*lucy)',
                r'located\s+in\s+([a-zA-Z\s]+)\s+parish',
            ],
            'Surveyed For': [
                r'surveyed\s+for[:\s]+([a-zA-Z\s\.]+)',
                r'owner[:\s]+([a-zA-Z\s\.]+)',
                r'client[:\s]+([a-zA-Z\s\.]+)',
                r'prepared\s+for[:\s]+([a-zA-Z\s\.]+)',
            ],
            'Certified date': [
                r'certified[:\s]+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})',
                r'date[:\s]+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})',
                r'([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})',
                r'([A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4})',  # "January 15, 2023"
            ],
            'LT Num': [
                r'lt\.?\s*no\.?\s*([0-9]+)',
                r'lot\s+no\.?\s*([0-9]+)',
                r'lot\s+([0-9]+)',
                r'plot\s+([0-9]+)',
                r'parcel\s+([0-9]+)',
                r'block\s+([0-9]+)',
            ]
        }
        
        # Initialize OCR if available
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
            except:
                self.easyocr_reader = None
        else:
            self.easyocr_reader = None
    
    def process_document(self, image_path, plot_id):
        """Process a document and extract all available information"""
        result = {
            'ID': str(plot_id),
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': '',
            'geometry': []
        }
        
        try:
            # Step 1: Enhanced image preprocessing
            processed_img = self._preprocess_image(image_path)
            if processed_img is None:
                return result
            
            # Step 2: Extract text using multiple methods
            extracted_text = self._extract_text_robust(processed_img)
            
            # Step 3: Parse metadata from text
            metadata = self._parse_metadata_enhanced(extracted_text)
            for key, value in metadata.items():
                if key in result and value:
                    result[key] = value
            
            # Step 4: Extract polygon using multiple approaches
            polygon = self._extract_polygon_robust(processed_img)
            if polygon:
                # Convert to geographic coordinates
                img_shape = processed_img.shape[:2]
                geo_coords = self._pixel_to_geo(polygon, img_shape)
                result['geometry'] = geo_coords
            
            return result
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return result
    
    def _preprocess_image(self, image_path):
        """Enhanced image preprocessing for better extraction"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Resize if too large
            max_size = 2048
            if max(img_array.shape[:2]) > max_size:
                scale = max_size / max(img_array.shape[:2])
                new_height = int(img_array.shape[0] * scale)
                new_width = int(img_array.shape[1] * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_array = np.array(img)
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply multiple enhancement techniques
            # 1. Noise reduction
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # 2. Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. Sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Convert back to RGB for consistency
            result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
            
            return result
            
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return None
    
    def _extract_text_robust(self, image):
        """Extract text using multiple OCR methods with fallbacks"""
        all_text = []
        
        # Method 1: Tesseract with different configurations
        if TESSERACT_AVAILABLE:
            try:
                # Convert to PIL Image
                pil_img = Image.fromarray(image)
                
                # Try different PSM modes
                psm_modes = [6, 4, 8, 11]  # Different page segmentation modes
                for psm in psm_modes:
                    try:
                        config = f'--psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;/-() '
                        text = pytesseract.image_to_string(pil_img, config=config)
                        if text.strip():
                            all_text.append(text)
                    except:
                        continue
                        
            except Exception as e:
                print(f"Tesseract extraction failed: {e}")
        
        # Method 2: EasyOCR
        if self.easyocr_reader:
            try:
                results = self.easyocr_reader.readtext(image)
                easyocr_text = ' '.join([result[1] for result in results if result[2] > 0.5])
                if easyocr_text.strip():
                    all_text.append(easyocr_text)
            except Exception as e:
                print(f"EasyOCR extraction failed: {e}")
        
        # Method 3: Simple pattern-based text detection (fallback)
        if not all_text:
            # Use basic image processing to find text-like regions
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Find horizontal text lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # If we detect text-like structures, add a placeholder
            if np.sum(detected_lines) > 1000:
                all_text.append("Text detected but OCR not available")
        
        # Combine all extracted text
        combined_text = ' '.join(all_text).lower()
        return combined_text
    
    def _parse_metadata_enhanced(self, text):
        """Parse metadata with enhanced pattern matching"""
        metadata = {
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': ''
        }
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.lower().strip()
        
        for field, patterns in self.metadata_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if field == 'Unit of Measurement':
                        value = match.group(0).strip()
                    else:
                        value = match.group(1).strip()
                    
                    # Clean extracted value
                    value = re.sub(r'\s+', ' ', value)
                    value = value.strip('.,;:')
                    
                    if value and len(value) > 1:  # Ensure meaningful extraction
                        metadata[field] = value
                        break
        
        return metadata
    
    def _extract_polygon_robust(self, image):
        """Extract polygon using multiple robust methods"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Contour detection with preprocessing
            polygon1 = self._extract_via_contours_enhanced(gray)
            
            # Method 2: Line detection and polygon reconstruction
            polygon2 = self._extract_via_lines_enhanced(gray)
            
            # Method 3: Edge detection with morphological operations
            polygon3 = self._extract_via_edges_enhanced(gray)
            
            # Combine and score polygons
            candidates = [p for p in [polygon1, polygon2, polygon3] if p]
            
            if candidates:
                # Score polygons and return best
                best_polygon = self._select_best_polygon_enhanced(candidates, gray.shape)
                return best_polygon
            
            return []
            
        except Exception as e:
            print(f"Polygon extraction failed: {e}")
            return []
    
    def _extract_via_contours_enhanced(self, gray):
        """Enhanced contour-based polygon extraction"""
        try:
            # Apply multiple edge detection methods
            edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)
            edges2 = cv2.Canny(gray, 30, 100, apertureSize=3)
            
            # Combine edges
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Morphological operations to connect broken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and shape
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 100000:  # Reasonable area range
                    # Approximate contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:  # At least 4 vertices
                        coords = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                        return coords
            
            return []
            
        except Exception as e:
            print(f"Contour extraction failed: {e}")
            return []
    
    def _extract_via_lines_enhanced(self, gray):
        """Enhanced line-based polygon extraction"""
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
            
            if lines is not None and len(lines) >= 4:
                # Simple approach: find bounding rectangle of all lines
                all_points = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    all_points.extend([(x1, y1), (x2, y2)])
                
                if all_points:
                    points = np.array(all_points)
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)
                    
                    # Create rectangular approximation
                    rect_coords = [
                        (float(x_min), float(y_min)),
                        (float(x_max), float(y_min)),
                        (float(x_max), float(y_max)),
                        (float(x_min), float(y_max))
                    ]
                    return rect_coords
            
            return []
            
        except Exception as e:
            print(f"Line extraction failed: {e}")
            return []
    
    def _extract_via_edges_enhanced(self, gray):
        """Enhanced edge-based polygon extraction"""
        try:
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if necessary (text should be dark on light background)
            if np.mean(thresh) > 127:
                thresh = cv2.bitwise_not(thresh)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find largest reasonable contour
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(contour)
                if 2000 < area < 50000:
                    # Approximate to polygon
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:
                        coords = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                        return coords
            
            return []
            
        except Exception as e:
            print(f"Edge extraction failed: {e}")
            return []
    
    def _select_best_polygon_enhanced(self, polygons, image_shape):
        """Select best polygon from candidates using enhanced scoring"""
        if not polygons:
            return []
        
        scores = []
        for poly in polygons:
            score = self._score_polygon_enhanced(poly, image_shape)
            scores.append((score, poly))
        
        # Sort by score and return best
        scores.sort(reverse=True)
        return scores[0][1] if scores[0][0] > 0.3 else []
    
    def _score_polygon_enhanced(self, polygon, image_shape):
        """Enhanced polygon scoring"""
        if len(polygon) < 4:
            return 0.0
        
        h, w = image_shape
        poly_array = np.array(polygon)
        
        # Score components
        scores = []
        
        # 1. Area score (prefer 10-40% of image area)
        area = cv2.contourArea(poly_array.astype(np.int32))
        area_ratio = area / (h * w)
        if 0.1 < area_ratio < 0.4:
            scores.append(1.0)
        elif 0.05 < area_ratio < 0.6:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        # 2. Shape regularity (prefer rectangular shapes)
        if len(polygon) == 4:
            scores.append(1.0)
        elif 4 < len(polygon) <= 8:
            scores.append(0.8)
        else:
            scores.append(0.5)
        
        # 3. Position score (prefer center-ish polygons)
        center_x, center_y = w // 2, h // 2
        poly_center_x = np.mean(poly_array[:, 0])
        poly_center_y = np.mean(poly_array[:, 1])
        
        distance_from_center = np.sqrt((poly_center_x - center_x)**2 + (poly_center_y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        position_score = 1.0 - (distance_from_center / max_distance)
        scores.append(position_score)
        
        # 4. Size reasonableness
        if 2000 < area < 50000:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        return np.mean(scores)
    
    def _pixel_to_geo(self, pixel_coords, image_shape):
        """Convert pixel coordinates to geographic coordinates"""
        if not pixel_coords:
            return []
        
        h, w = image_shape
        minx, maxx, miny, maxy = self.bounds
        
        geo_coords = []
        for x_pixel, y_pixel in pixel_coords:
            # Normalize to [0, 1]
            x_norm = x_pixel / (w - 1) if w > 1 else 0
            y_norm = y_pixel / (h - 1) if h > 1 else 0
            
            # Convert to geographic coordinates
            x_geo = x_norm * (maxx - minx) + minx
            y_geo = (1 - y_norm) * (maxy - miny) + miny
            
            geo_coords.append([x_geo, y_geo])
        
        return geo_coords

def process_test_documents_enhanced(test_ids_df, image_dir="data/", output_file="submission.csv"):
    """Process test documents using enhanced processor"""
    
    print("ENHANCED DOCUMENT PROCESSING")
    print("=" * 60)
    
    processor = EnhancedDocumentProcessor()
    results = []
    
    successful_extractions = {
        'geometry': 0,
        'metadata': 0
    }
    
    for idx, plot_id in enumerate(test_ids_df['ID']):
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(test_ids_df)} ({idx/len(test_ids_df)*100:.1f}%)")
        
        # Find image file
        image_path = None
        for ext in ['jpg', 'jpeg', 'png', 'tiff', 'pdf']:
            for pattern in [f"{plot_id}.{ext}", f"anonymised_{plot_id}.{ext}"]:
                potential_path = os.path.join(image_dir, pattern)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            if image_path:
                break
        
        if image_path:
            result = processor.process_document(image_path, plot_id)
            
            # Count successes
            if result['geometry']:
                successful_extractions['geometry'] += 1
            
            metadata_fields = ['TargetSurvey', 'Certified date', 'Total Area', 
                             'Unit of Measurement', 'Parish', 'LT Num']
            if any(result.get(field, '') for field in metadata_fields):
                successful_extractions['metadata'] += 1
                
        else:
            print(f"Warning: No image found for ID {plot_id}")
            result = {
                'ID': str(plot_id),
                'TargetSurvey': '',
                'Certified date': '',
                'Total Area': '',
                'Unit of Measurement': '',
                'Parish': '',
                'LT Num': '',
                'geometry': []
            }
        
        results.append(result)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(results)
    column_order = ['ID', 'TargetSurvey', 'Certified date', 'Total Area',
                   'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
    submission_df = submission_df[column_order]
    
    # Save submission
    submission_df.to_csv(output_file, index=False)
    
    # Print summary
    total = len(test_ids_df)
    print(f"\n{'='*60}")
    print(f"ENHANCED PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total documents: {total}")
    print(f"Successful geometry extractions: {successful_extractions['geometry']} ({successful_extractions['geometry']/total*100:.1f}%)")
    print(f"Successful metadata extractions: {successful_extractions['metadata']} ({successful_extractions['metadata']/total*100:.1f}%)")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")
    
    return submission_df

if __name__ == "__main__":
    # Test the enhanced processor
    print("Enhanced Document Processor ready!")
    print("Use: process_test_documents_enhanced(test_ids_df) to process documents")
