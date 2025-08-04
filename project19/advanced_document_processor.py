#!/usr/bin/env python3
"""
Advanced Document Processor for Land Survey Plans
Improved extraction with specialized patterns and better polygon detection
"""

import cv2
import numpy as np
import re
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDocumentProcessor:
    """Enhanced processor specifically designed for land survey plans"""
    
    def __init__(self):
        self.setup_patterns()
        
    def setup_patterns(self):
        """Define comprehensive regex patterns for survey plan metadata"""
        
        # Survey number patterns
        self.survey_patterns = [
            r"Survey\s*No[.:]\s*([A-Z0-9\-/]+)",
            r"Survey\s*Number[.:]\s*([A-Z0-9\-/]+)",
            r"Plan\s*No[.:]\s*([A-Z0-9\-/]+)",
            r"S\.?\s*No[.:]\s*([A-Z0-9\-/]+)",
            r"DP\s*([0-9]+)",  # Deposited Plan
            r"CP\s*([0-9]+)",  # Community Plan
            r"SP\s*([0-9]+)",  # Strata Plan
        ]
        
        # Date patterns (more comprehensive)
        self.date_patterns = [
            r"(?:Certified|Approved|Dated?)[:\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"(?:Certified|Approved|Dated?)[:\s]*([0-9]{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+[0-9]{2,4})",
            r"([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",  # Generic date
            r"([0-9]{4}[/-][0-9]{1,2}[/-][0-9]{1,2})",  # ISO format
        ]
        
        # Area patterns
        self.area_patterns = [
            r"(?:Total\s*)?Area[:\s]*([0-9]+\.?[0-9]*)\s*(ha|hectares?|m²|sq\.?\s*m|acres?)",
            r"([0-9]+\.?[0-9]*)\s*(ha|hectares?|m²|sq\.?\s*m|acres?)",
            r"Area\s*=\s*([0-9]+\.?[0-9]*)\s*(ha|hectares?|m²|sq\.?\s*m|acres?)",
        ]
        
        # Parish patterns
        self.parish_patterns = [
            r"Parish\s*of\s*([A-Za-z\s]+?)(?:\n|,|$)",
            r"Parish[:\s]*([A-Za-z\s]+?)(?:\n|,|$)",
            r"P\.?\s*of\s*([A-Za-z\s]+?)(?:\n|,|$)",
        ]
        
        # LT Number patterns
        self.lt_patterns = [
            r"LT\s*(?:No\.?|Number)[:\s]*([A-Z0-9\-/]+)",
            r"L\.T\.\s*([A-Z0-9\-/]+)",
            r"Land\s*Title[:\s]*([A-Z0-9\-/]+)",
        ]

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Enhanced image preprocessing for better OCR"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple preprocessing techniques
            processed_images = []
            
            # 1. Basic denoising and sharpening
            denoised = cv2.fastNlMeansDenoising(gray)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            processed_images.append(sharpened)
            
            # 2. Adaptive thresholding
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            processed_images.append(adaptive)
            
            # 3. Morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            processed_images.append(morph)
            
            # 4. High contrast version
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            contrast = clahe.apply(gray)
            processed_images.append(contrast)
            
            # Return the best preprocessed version (or stack them)
            return processed_images[0]  # Return sharpened version as primary
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return np.zeros((100, 100), dtype=np.uint8)

    def extract_text_opencv(self, img: np.ndarray) -> str:
        """Extract text using OpenCV-based methods (no external OCR)"""
        try:
            # This is a placeholder for OpenCV-only text extraction
            # In practice, this would use template matching or other CV techniques
            # For now, return empty string as OpenCV alone can't do OCR
            return ""
        except Exception as e:
            logger.error(f"OpenCV text extraction failed: {e}")
            return ""

    def extract_text_tesseract(self, img: np.ndarray) -> str:
        """Extract text using Tesseract OCR"""
        try:
            import pytesseract
            # Configure Tesseract for documents
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/:-()[]= '
            text = pytesseract.image_to_string(img, config=config)
            return text
        except ImportError:
            logger.warning("Tesseract not available")
            return ""
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return ""

    def extract_text_easyocr(self, img: np.ndarray) -> str:
        """Extract text using EasyOCR"""
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(img)
            text = ' '.join([result[1] for result in results])
            return text
        except ImportError:
            logger.warning("EasyOCR not available")
            return ""
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""

    def extract_text_multiple_methods(self, img: np.ndarray) -> str:
        """Try multiple OCR methods and combine results"""
        texts = []
        
        # Try different methods
        methods = [
            ("Tesseract", self.extract_text_tesseract),
            ("EasyOCR", self.extract_text_easyocr),
            ("OpenCV", self.extract_text_opencv),
        ]
        
        for method_name, method in methods:
            try:
                text = method(img)
                if text and text.strip():
                    texts.append(text)
                    logger.info(f"{method_name} extracted {len(text)} characters")
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
        
        # Combine texts (prefer longer ones)
        if texts:
            return max(texts, key=len)
        else:
            return ""

    def extract_metadata_from_text(self, text: str) -> Dict[str, str]:
        """Extract metadata using improved regex patterns"""
        metadata = {
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': ''
        }
        
        if not text:
            return metadata
        
        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        # Extract survey number
        for pattern in self.survey_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['TargetSurvey'] = match.group(1).strip()
                break
        
        # Extract date
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['Certified date'] = match.group(1).strip()
                break
        
        # Extract area
        for pattern in self.area_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['Total Area'] = match.group(1).strip()
                if len(match.groups()) > 1:
                    metadata['Unit of Measurement'] = match.group(2).strip()
                break
        
        # Extract parish
        for pattern in self.parish_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['Parish'] = match.group(1).strip()
                break
        
        # Extract LT number
        for pattern in self.lt_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['LT Num'] = match.group(1).strip()
                break
        
        return metadata

    def detect_polygons_advanced(self, img: np.ndarray) -> List[List[List[int]]]:
        """Advanced polygon detection using multiple methods"""
        polygons = []
        
        try:
            # Method 1: Contour detection
            polygons.extend(self._detect_contour_polygons(img))
            
            # Method 2: Line detection and intersection
            polygons.extend(self._detect_line_polygons(img))
            
            # Method 3: Corner detection
            polygons.extend(self._detect_corner_polygons(img))
            
            # Remove duplicates and filter by area
            polygons = self._filter_polygons(polygons, img.shape)
            
        except Exception as e:
            logger.error(f"Polygon detection failed: {e}")
        
        return polygons

    def _detect_contour_polygons(self, img: np.ndarray) -> List[List[List[int]]]:
        """Detect polygons using contour analysis"""
        polygons = []
        
        try:
            # Edge detection
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:  # At least a triangle
                    polygon = [[int(point[0][0]), int(point[0][1])] for point in approx]
                    polygons.append(polygon)
                    
        except Exception as e:
            logger.warning(f"Contour polygon detection failed: {e}")
        
        return polygons

    def _detect_line_polygons(self, img: np.ndarray) -> List[List[List[int]]]:
        """Detect polygons using line detection and intersection"""
        polygons = []
        
        try:
            # Hough line detection
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) >= 3:
                # Find intersections of lines to form polygons
                # This is a simplified approach - in practice, would need more sophisticated geometry
                intersections = []
                for i, line1 in enumerate(lines[:10]):  # Limit to first 10 lines
                    for line2 in lines[i+1:11]:
                        intersection = self._line_intersection(line1[0], line2[0])
                        if intersection:
                            intersections.append(intersection)
                
                if len(intersections) >= 3:
                    # Simple convex hull as polygon
                    points = np.array(intersections, dtype=np.int32)
                    hull = cv2.convexHull(points)
                    polygon = [[int(point[0][0]), int(point[0][1])] for point in hull]
                    polygons.append(polygon)
                    
        except Exception as e:
            logger.warning(f"Line polygon detection failed: {e}")
        
        return polygons

    def _detect_corner_polygons(self, img: np.ndarray) -> List[List[List[int]]]:
        """Detect polygons using corner detection"""
        polygons = []
        
        try:
            # Harris corner detection
            corners = cv2.cornerHarris(img, 2, 3, 0.04)
            corners = cv2.dilate(corners, None)
            
            # Find corner coordinates
            corner_coords = np.where(corners > 0.01 * corners.max())
            
            if len(corner_coords[0]) >= 4:
                points = list(zip(corner_coords[1], corner_coords[0]))
                points = np.array(points, dtype=np.int32)
                
                # Create convex hull
                hull = cv2.convexHull(points)
                polygon = [[int(point[0][0]), int(point[0][1])] for point in hull]
                polygons.append(polygon)
                
        except Exception as e:
            logger.warning(f"Corner polygon detection failed: {e}")
        
        return polygons

    def _line_intersection(self, line1, line2):
        """Find intersection point of two lines"""
        try:
            rho1, theta1 = line1
            rho2, theta2 = line2
            
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            
            if np.abs(np.linalg.det(A)) > 1e-6:  # Lines are not parallel
                intersection = np.linalg.solve(A, b)
                return [int(intersection[0][0]), int(intersection[1][0])]
        except:
            pass
        return None

    def _filter_polygons(self, polygons: List[List[List[int]]], img_shape: Tuple[int, int]) -> List[List[List[int]]]:
        """Filter and clean up detected polygons"""
        filtered = []
        h, w = img_shape[:2]
        
        for polygon in polygons:
            if len(polygon) < 3:
                continue
                
            # Calculate area
            points = np.array(polygon, dtype=np.int32)
            area = cv2.contourArea(points)
            
            # Filter by area (should be significant portion of image)
            min_area = (w * h) * 0.01  # At least 1% of image
            max_area = (w * h) * 0.8   # At most 80% of image
            
            if min_area <= area <= max_area:
                # Ensure points are within image bounds
                valid_polygon = []
                for point in polygon:
                    x, y = point
                    x = max(0, min(w-1, x))
                    y = max(0, min(h-1, y))
                    valid_polygon.append([x, y])
                
                filtered.append(valid_polygon)
        
        return filtered

    def process_document(self, image_path: str) -> Dict[str, Any]:
        """Main processing function"""
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
            # Preprocess image
            img = self.preprocess_image(image_path)
            
            # Extract text
            text = self.extract_text_multiple_methods(img)
            logger.info(f"Extracted text length: {len(text)}")
            
            # Extract metadata
            metadata = self.extract_metadata_from_text(text)
            result.update(metadata)
            
            # Detect polygons
            polygons = self.detect_polygons_advanced(img)
            if polygons:
                result['geometry'] = polygons[0]  # Take the first/best polygon
                logger.info(f"Detected {len(polygons)} polygons")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
        
        return result

    def debug_processing(self, image_path: str) -> Dict[str, Any]:
        """Debug version with detailed logging"""
        print(f"\n=== DEBUGGING: {image_path} ===")
        
        if not os.path.exists(image_path):
            print(f"ERROR: Image file not found: {image_path}")
            return {}
        
        try:
            # Step 1: Load and preprocess
            print("1. Loading and preprocessing image...")
            img = self.preprocess_image(image_path)
            print(f"   Image shape: {img.shape}")
            
            # Step 2: Text extraction
            print("2. Extracting text...")
            text = self.extract_text_multiple_methods(img)
            print(f"   Extracted text ({len(text)} chars):")
            print(f"   '{text[:200]}{'...' if len(text) > 200 else ''}'")
            
            # Step 3: Metadata extraction
            print("3. Extracting metadata...")
            metadata = self.extract_metadata_from_text(text)
            for key, value in metadata.items():
                status = "✓" if value else "✗"
                print(f"   {status} {key}: '{value}'")
            
            # Step 4: Polygon detection
            print("4. Detecting polygons...")
            polygons = self.detect_polygons_advanced(img)
            print(f"   Found {len(polygons)} polygons")
            
            if polygons:
                for i, poly in enumerate(polygons[:3]):  # Show first 3
                    print(f"   Polygon {i+1}: {len(poly)} points")
            
            # Combine results
            result = {
                'ID': Path(image_path).stem,
                'text': text,
                'metadata': metadata,
                'polygons': polygons
            }
            
            return result
            
        except Exception as e:
            print(f"ERROR during processing: {e}")
            import traceback
            traceback.print_exc()
            return {}


def create_mock_survey_image(output_path: str = "test_survey.png"):
    """Create a mock survey plan image for testing"""
    try:
        # Create a simple survey plan image
        img = np.ones((800, 1000, 3), dtype=np.uint8) * 255  # White background
        
        # Add some text content
        cv2.putText(img, "SURVEY PLAN", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(img, "Survey No: DP12345", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "Certified Date: 15/03/2023", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "Total Area: 1.25 hectares", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "Parish of SPRING HILL", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "LT No: 123456789", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add a simple polygon (property boundary)
        points = np.array([[200, 300], [700, 300], [700, 600], [200, 600]], np.int32)
        cv2.polylines(img, [points], True, (0, 0, 0), 3)
        
        # Add some dimension labels
        cv2.putText(img, "100m", (420, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "125m", (180, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.imwrite(output_path, img)
        print(f"Created mock survey image: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error creating mock image: {e}")
        return None


if __name__ == "__main__":
    # Test the advanced processor
    processor = AdvancedDocumentProcessor()
    
    # Create a test image if none exists
    test_image = "test_survey.png"
    if not os.path.exists(test_image):
        create_mock_survey_image(test_image)
    
    if os.path.exists(test_image):
        print("Testing Advanced Document Processor")
        print("=" * 50)
        
        # Run debug processing
        result = processor.debug_processing(test_image)
        
        if result:
            print("\n=== FINAL RESULT ===")
            print(f"ID: {result.get('ID', 'N/A')}")
            
            metadata = result.get('metadata', {})
            for key, value in metadata.items():
                print(f"{key}: '{value}'")
            
            polygons = result.get('polygons', [])
            print(f"Polygons: {len(polygons)} found")
            
            if polygons:
                print("First polygon:", polygons[0][:5] if len(polygons[0]) > 5 else polygons[0])
    else:
        print("No test image available for processing")
