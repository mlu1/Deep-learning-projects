#!/usr/bin/env python3
"""
Document Processor for Land Survey Plans (No OpenCV)
Uses PIL, scikit-image, and numpy for image processing
"""

import numpy as np
import re
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from scipy import ndimage
from skimage import filters, feature, morphology, measure, segmentation
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import find_contours, approximate_polygon
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    """Image preprocessing without OpenCV"""
    
    def enhance_image(self, image_path: str) -> np.ndarray:
        """Enhance image quality using PIL and scikit-image"""
        try:
            # Load image with PIL
            pil_img = Image.open(image_path).convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(pil_img)
            
            # Convert to grayscale using standard weights
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Denoising using gaussian filter
            denoised = gaussian(gray, sigma=1.0)
            
            # Enhance contrast using histogram equalization
            enhanced = self._equalize_histogram(denoised)
            
            # Sharpen the image
            sharpened = self._sharpen_image(enhanced)
            
            # Convert back to 3-channel for consistency
            result = np.stack([sharpened, sharpened, sharpened], axis=-1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing image {image_path}: {e}")
            return None
    
    def _equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """Histogram equalization using numpy"""
        try:
            # Normalize to 0-255 range
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Calculate histogram
            hist, bins = np.histogram(image_uint8.flatten(), 256, [0, 256])
            
            # Calculate cumulative distribution function
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            
            # Use linear interpolation to create the equalized image
            equalized = np.interp(image_uint8.flatten(), bins[:-1], cdf_normalized)
            equalized = equalized.reshape(image_uint8.shape)
            
            return equalized / 255.0
            
        except Exception as e:
            logger.warning(f"Histogram equalization failed: {e}")
            return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image using convolution"""
        try:
            # Sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            
            # Apply convolution
            sharpened = ndimage.convolve(image, kernel)
            
            # Clip values to valid range
            sharpened = np.clip(sharpened, 0, 1)
            
            return sharpened
            
        except Exception as e:
            logger.warning(f"Sharpening failed: {e}")
            return image

class MetadataExtractor:
    """Extract metadata from document images without OCR dependencies"""
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Define regex patterns for survey plan metadata"""
        
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
        
        # Date patterns
        self.date_patterns = [
            r"(?:Certified|Approved|Dated?)[:\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"(?:Certified|Approved|Dated?)[:\s]*([0-9]{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+[0-9]{2,4})",
            r"([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"([0-9]{4}[/-][0-9]{1,2}[/-][0-9]{1,2})",
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
    
    def _extract_text_simple(self, image_path: str) -> str:
        """Simple text extraction placeholder - returns empty string"""
        # Without OCR libraries, we can't extract text from images
        # This would need to be replaced with actual OCR when available
        return ""
    
    def _extract_text_tesseract(self, image_path: str) -> str:
        """Extract text using Tesseract OCR if available"""
        try:
            import pytesseract
            from PIL import Image
            
            img = Image.open(image_path)
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/:-()[]= '
            text = pytesseract.image_to_string(img, config=config)
            return text
        except ImportError:
            logger.warning("Tesseract not available")
            return ""
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return ""
    
    def _extract_text_easyocr(self, image_path: str) -> str:
        """Extract text using EasyOCR if available"""
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(image_path)
            text = ' '.join([result[1] for result in results])
            return text
        except ImportError:
            logger.warning("EasyOCR not available")
            return ""
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""
    
    def _extract_text_multiple_methods(self, image_path: str) -> str:
        """Try multiple OCR methods"""
        methods = [
            ("Tesseract", self._extract_text_tesseract),
            ("EasyOCR", self._extract_text_easyocr),
            ("Simple", self._extract_text_simple),
        ]
        
        for method_name, method in methods:
            try:
                text = method(image_path)
                if text and text.strip():
                    logger.info(f"{method_name} extracted {len(text)} characters")
                    return text
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
        
        return ""
    
    def extract_metadata(self, image_path: str) -> Dict[str, str]:
        """Extract metadata from image"""
        metadata = {
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': ''
        }
        
        # Extract text using available methods
        text = self._extract_text_multiple_methods(image_path)
        
        if not text:
            logger.warning(f"No text extracted from {image_path}")
            return metadata
        
        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        # Extract each field using regex patterns
        metadata = self._extract_fields_from_text(text)
        
        return metadata
    
    def _extract_fields_from_text(self, text: str) -> Dict[str, str]:
        """Extract specific fields from text using regex"""
        metadata = {
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': ''
        }
        
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

class PolygonExtractor:
    """Extract polygon boundaries from documents using scikit-image"""
    
    def extract_polygon_from_document(self, image_path: str) -> List[List[int]]:
        """Extract polygon using edge detection and contour finding"""
        try:
            # Load and preprocess image
            pil_img = Image.open(image_path).convert('RGB')
            img_array = np.array(pil_img)
            
            # Convert to grayscale
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Apply Gaussian filter to reduce noise
            blurred = gaussian(gray, sigma=1.0)
            
            # Edge detection using Canny
            edges = canny(blurred, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
            
            # Morphological operations to close gaps
            closed = binary_closing(edges, disk(2))
            
            # Find contours
            contours = find_contours(closed, 0.5)
            
            if not contours:
                return []
            
            # Find the largest contour (likely the main boundary)
            largest_contour = max(contours, key=len)
            
            # Approximate polygon
            polygon = approximate_polygon(largest_contour, tolerance=2.0)
            
            # Convert to integer coordinates and format as list of [x, y] pairs
            polygon_coords = [[int(point[1]), int(point[0])] for point in polygon]
            
            # Filter out very small polygons
            if len(polygon_coords) < 3:
                return []
            
            return polygon_coords
            
        except Exception as e:
            logger.error(f"Error extracting polygon from {image_path}: {e}")
            return []
    
    def _extract_polygons_alternative(self, image_path: str) -> List[List[int]]:
        """Alternative polygon extraction using threshold and watershed"""
        try:
            # Load image
            pil_img = Image.open(image_path).convert('RGB')
            img_array = np.array(pil_img)
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Apply threshold
            threshold = threshold_otsu(gray)
            binary = gray > threshold
            
            # Remove small objects
            cleaned = morphology.remove_small_objects(binary, min_size=1000)
            
            # Label connected components
            labeled = measure.label(cleaned)
            
            # Find properties of labeled regions
            regions = measure.regionprops(labeled)
            
            if not regions:
                return []
            
            # Get the largest region
            largest_region = max(regions, key=lambda x: x.area)
            
            # Get boundary coordinates
            coords = largest_region.coords
            
            if len(coords) < 3:
                return []
            
            # Create convex hull as simple polygon
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            polygon_coords = [[int(coords[i][1]), int(coords[i][0])] for i in hull.vertices]
            
            return polygon_coords
            
        except Exception as e:
            logger.warning(f"Alternative polygon extraction failed: {e}")
            return []

class DocumentProcessor:
    """Main document processor without OpenCV"""
    
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
        self.metadata_extractor = MetadataExtractor()
        self.polygon_extractor = PolygonExtractor()
    
    def process_document(self, image_path: str, plot_id: str = None) -> Dict[str, Any]:
        """Process a single document image"""
        
        if plot_id is None:
            plot_id = Path(image_path).stem
        
        result = {
            'ID': plot_id,
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': '',
            'geometry': []
        }
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return result
            
            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(image_path)
            result.update(metadata)
            
            # Extract polygon
            polygon = self.polygon_extractor.extract_polygon_from_document(image_path)
            if polygon:
                result['geometry'] = polygon
                logger.info(f"Extracted polygon with {len(polygon)} vertices")
            else:
                logger.warning(f"No polygon extracted from {image_path}")
            
        except Exception as e:
            logger.error(f"Error processing document {image_path}: {e}")
        
        return result
    
    def process_batch(self, image_dir: str, output_path: str = "submission.csv") -> None:
        """Process multiple documents and create submission file"""
        import pandas as pd
        import glob
        
        # Find all images
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
        image_files = []
        
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(image_dir, pattern)))
            image_files.extend(glob.glob(os.path.join(image_dir, pattern.upper())))
        
        if not image_files:
            logger.error(f"No images found in {image_dir}")
            return
        
        logger.info(f"Processing {len(image_files)} images...")
        
        results = []
        for image_path in image_files:
            result = self.process_document(image_path)
            results.append(result)
            logger.info(f"Processed {os.path.basename(image_path)}")
        
        # Create DataFrame and save
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")

def create_test_image(output_path: str = "test_survey.png") -> str:
    """Create a test survey image using PIL"""
    try:
        # Create image using PIL
        img = Image.new('RGB', (1000, 800), color='white')
        
        # We can add text using PIL if we have font support
        # For now, just create a simple image with basic shapes
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a simple rectangle as a property boundary
        draw.rectangle([200, 300, 700, 600], outline='black', width=3)
        
        # Draw some basic text (will only work if fonts are available)
        try:
            draw.text((50, 50), "SURVEY PLAN", fill='black')
            draw.text((50, 100), "Survey No: DP12345", fill='black')
            draw.text((50, 130), "Certified Date: 15/03/2023", fill='black')
            draw.text((50, 160), "Total Area: 1.25 hectares", fill='black')
            draw.text((50, 190), "Parish of SPRING HILL", fill='black')
            draw.text((50, 220), "LT No: 123456789", fill='black')
        except:
            # If font rendering fails, continue without text
            pass
        
        img.save(output_path)
        logger.info(f"Created test image: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating test image: {e}")
        return ""

if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()
    
    # Create test image
    test_image = create_test_image()
    
    if test_image and os.path.exists(test_image):
        print("Testing Document Processor (No OpenCV)")
        print("=" * 50)
        
        result = processor.process_document(test_image)
        
        print("Processing Result:")
        for key, value in result.items():
            if key == 'geometry':
                geom_info = f"{len(value)} vertices" if value else "None"
                print(f"  {key}: {geom_info}")
            else:
                print(f"  {key}: '{value}'")
        
        print("\nNote: Text extraction requires OCR libraries (pytesseract or easyocr)")
        print("Install with: pip install pytesseract easyocr")
    else:
        print("Could not create test image")
