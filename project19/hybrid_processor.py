#!/usr/bin/env python3
"""
Hybrid Processor for Image Analysis (No OpenCV)
Detects image type and applies appropriate processing pipeline
Uses PIL and numpy instead of OpenCV
"""

import os
import numpy as np
from typing import Dict, Any, List
from PIL import Image, ImageFilter, ImageStat
import logging

logger = logging.getLogger(__name__)

def detect_image_type(image_path: str) -> str:
    """
    Detect if image is aerial/satellite or document/scan using PIL
    Returns: 'aerial', 'document', or 'unknown'
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Convert to grayscale for analysis
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        
        # Feature 1: Edge density (documents have more sharp edges)
        edge_density = calculate_edge_density_pil(gray)
        
        # Feature 2: Color variance (aerial images usually more colorful)
        color_variance = calculate_color_variance_pil(img)
        
        # Feature 3: Text-like patterns (high frequency horizontal/vertical lines)
        text_patterns = detect_text_patterns_pil(gray)
        
        # Feature 4: Uniformity (documents often have uniform backgrounds)
        uniformity = calculate_uniformity_pil(gray)
        
        # Simple classification based on features
        score = 0
        
        # High edge density suggests document
        if edge_density > 0.15:
            score += 2
        elif edge_density > 0.08:
            score += 1
        
        # Low color variance suggests document
        if color_variance < 500:
            score += 2
        elif color_variance < 1000:
            score += 1
        
        # Text patterns suggest document
        if text_patterns > 0.3:
            score += 2
        elif text_patterns > 0.15:
            score += 1
        
        # High uniformity suggests document
        if uniformity > 0.7:
            score += 1
        
        # Classification
        if score >= 4:
            return "document"
        elif score <= 1:
            return "aerial"
        else:
            return "document"  # Default to document for land survey plans
            
    except Exception as e:
        logger.error(f"Error detecting image type for {image_path}: {e}")
        return "unknown"

def calculate_edge_density_pil(gray_image: np.ndarray) -> float:
    """Calculate edge density using simple gradients"""
    try:
        # Calculate gradients
        grad_x = np.abs(np.diff(gray_image, axis=1))
        grad_y = np.abs(np.diff(gray_image, axis=0))
        
        # Calculate edge strength
        edges_x = grad_x > (np.mean(grad_x) + 2 * np.std(grad_x))
        edges_y = grad_y > (np.mean(grad_y) + 2 * np.std(grad_y))
        
        # Edge density
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        edge_pixels = np.sum(edges_x) + np.sum(edges_y)
        
        return edge_pixels / total_pixels
        
    except Exception as e:
        logger.warning(f"Edge density calculation failed: {e}")
        return 0.0

def calculate_color_variance_pil(img: Image.Image) -> float:
    """Calculate color variance using PIL ImageStat"""
    try:
        # Calculate statistics for each channel
        stat = ImageStat.Stat(img)
        
        # Calculate variance across RGB channels
        variances = []
        for channel in range(3):
            channel_data = np.array(img.getchannel(channel))
            variances.append(np.var(channel_data))
        
        return np.mean(variances)
        
    except Exception as e:
        logger.warning(f"Color variance calculation failed: {e}")
        return 1000.0  # Default to high variance

def detect_text_patterns_pil(gray_image: np.ndarray) -> float:
    """Detect text-like patterns using line detection"""
    try:
        h, w = gray_image.shape
        
        # Look for horizontal lines (text rows)
        horizontal_score = 0
        for i in range(0, h, 10):  # Sample every 10 rows
            row = gray_image[i, :]
            # Look for alternating patterns (text)
            diff = np.abs(np.diff(row))
            high_freq = np.sum(diff > np.mean(diff))
            horizontal_score += high_freq / len(diff)
        
        # Look for vertical regularity (columns)
        vertical_score = 0
        for j in range(0, w, 10):  # Sample every 10 columns
            col = gray_image[:, j]
            diff = np.abs(np.diff(col))
            high_freq = np.sum(diff > np.mean(diff))
            vertical_score += high_freq / len(diff)
        
        # Normalize by number of samples
        h_samples = len(range(0, h, 10))
        v_samples = len(range(0, w, 10))
        
        total_score = (horizontal_score / h_samples + vertical_score / v_samples) / 2
        
        return min(total_score, 1.0)  # Cap at 1.0
        
    except Exception as e:
        logger.warning(f"Text pattern detection failed: {e}")
        return 0.0

def calculate_uniformity_pil(gray_image: np.ndarray) -> float:
    """Calculate background uniformity"""
    try:
        # Calculate histogram
        hist, _ = np.histogram(gray_image.flatten(), bins=50)
        
        # Find the most common intensity (background)
        max_bin = np.argmax(hist)
        total_pixels = gray_image.size
        
        # Calculate what fraction of pixels are near the background intensity
        uniformity = hist[max_bin] / total_pixels
        
        return uniformity
        
    except Exception as e:
        logger.warning(f"Uniformity calculation failed: {e}")
        return 0.0

class HybridProcessor:
    """Hybrid processor that works without OpenCV"""
    
    def __init__(self):
        self.document_processor = None
        self.aerial_processor = None
        
        # Try to import processors
        try:
            from document_processor_no_cv2 import DocumentProcessor
            self.document_processor = DocumentProcessor()
        except ImportError:
            logger.warning("Document processor not available")
        
        # Aerial processor placeholder (not implemented in this refactor)
        # In the original code, this used U-Net for segmentation
    
    def process_image(self, image_path: str, plot_id: str = None) -> Dict[str, Any]:
        """Process image using appropriate method based on type detection"""
        
        # Default result structure
        result = {
            'ID': plot_id or os.path.splitext(os.path.basename(image_path))[0],
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': '',
            'geometry': []
        }
        
        try:
            # Detect image type
            img_type = detect_image_type(image_path)
            logger.info(f"Detected image type: {img_type} for {image_path}")
            
            if img_type == "document" and self.document_processor:
                # Process as document
                result = self.document_processor.process_document(image_path, plot_id)
                
            elif img_type == "aerial":
                # Process as aerial image (not implemented without deep learning model)
                logger.warning("Aerial image processing not available in this version")
                
            else:
                # Unknown type or no processor available
                logger.warning(f"Cannot process image type: {img_type}")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
        
        return result
    
    def process_batch(self, image_dir: str, test_ids_file: str = None, output_path: str = "submission.csv") -> None:
        """Process batch of images and create submission"""
        import pandas as pd
        import glob
        
        results = []
        
        if test_ids_file and os.path.exists(test_ids_file):
            # Process specific test IDs
            test_ids_df = pd.read_csv(test_ids_file)
            
            for _, row in test_ids_df.iterrows():
                plot_id = str(row['ID'])
                
                # Try different image file patterns
                image_patterns = [
                    f"{plot_id}.jpg",
                    f"{plot_id}.jpeg", 
                    f"{plot_id}.png",
                    f"anonymised_{plot_id}.jpg"
                ]
                
                image_path = None
                for pattern in image_patterns:
                    potential_path = os.path.join(image_dir, pattern)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                if image_path:
                    result = self.process_image(image_path, plot_id)
                else:
                    # Create empty result for missing image
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
                    logger.warning(f"Image not found for ID: {plot_id}")
                
                results.append(result)
                
        else:
            # Process all images in directory
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
            image_files = []
            
            for pattern in image_patterns:
                image_files.extend(glob.glob(os.path.join(image_dir, pattern)))
                image_files.extend(glob.glob(os.path.join(image_dir, pattern.upper())))
            
            for image_path in image_files:
                result = self.process_image(image_path)
                results.append(result)
        
        # Create submission DataFrame
        df = pd.DataFrame(results)
        
        # Ensure required columns are present
        required_columns = ['ID', 'TargetSurvey', 'Certified date', 'Total Area', 
                           'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns
        df = df[required_columns]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results)} results to {output_path}")
        
        # Print summary
        filled_fields = {}
        for col in required_columns:
            if col != 'geometry':
                filled_count = df[col].astype(bool).sum()
                filled_fields[col] = filled_count
        
        geometry_filled = sum(1 for geom in df['geometry'] if geom and len(geom) > 0)
        filled_fields['geometry'] = geometry_filled
        
        print("\nExtraction Summary:")
        for field, count in filled_fields.items():
            percentage = count / len(df) * 100 if len(df) > 0 else 0
            print(f"  {field}: {count}/{len(df)} ({percentage:.1f}%)")

def create_minimal_processor():
    """Create a minimal processor that only extracts IDs"""
    
    class MinimalProcessor:
        def process_image(self, image_path: str, plot_id: str = None) -> Dict[str, Any]:
            return {
                'ID': plot_id or os.path.splitext(os.path.basename(image_path))[0],
                'TargetSurvey': '',
                'Certified date': '',
                'Total Area': '',
                'Unit of Measurement': '',
                'Parish': '',
                'LT Num': '',
                'geometry': []
            }
        
        def process_batch(self, image_dir: str, test_ids_file: str = None, output_path: str = "submission.csv"):
            import pandas as pd
            
            results = []
            
            if test_ids_file and os.path.exists(test_ids_file):
                test_ids_df = pd.read_csv(test_ids_file)
                for _, row in test_ids_df.iterrows():
                    plot_id = str(row['ID'])
                    result = self.process_image("", plot_id)
                    results.append(result)
            
            df = pd.DataFrame(results)
            required_columns = ['ID', 'TargetSurvey', 'Certified date', 'Total Area', 
                               'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
            
            for col in required_columns:
                if col not in df.columns:
                    df[col] = '' if col != 'geometry' else df.apply(lambda x: [], axis=1)
            
            df = df[required_columns]
            df.to_csv(output_path, index=False)
            print(f"Created minimal submission with {len(results)} entries")
    
    return MinimalProcessor()

if __name__ == "__main__":
    # Test the hybrid processor
    processor = HybridProcessor()
    
    # Create test image if document processor is available
    if processor.document_processor:
        from document_processor_no_cv2 import create_test_image
        test_image = create_test_image()
        
        if test_image and os.path.exists(test_image):
            print("Testing Hybrid Processor (No OpenCV)")
            print("=" * 50)
            
            result = processor.process_image(test_image)
            
            print("Processing Result:")
            for key, value in result.items():
                if key == 'geometry':
                    geom_info = f"{len(value)} vertices" if value else "None"
                    print(f"  {key}: {geom_info}")
                else:
                    print(f"  {key}: '{value}'")
    else:
        print("Document processor not available - creating minimal processor")
        minimal = create_minimal_processor()
        minimal.process_batch(".", None, "minimal_submission.csv")
