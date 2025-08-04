# Hybrid Solution - Handles both aerial imagery and scanned documents
# Automatically detects data type and applies appropriate processing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from PIL import Image
import cv2

# Import both processing approaches
try:
    from document_processor import DocumentProcessor
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSOR_AVAILABLE = False
    print("Document processor not available")

def detect_image_type(image_path):
    """Detect whether image is aerial/satellite or scanned document"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Feature 1: Edge density (documents typically have more edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Feature 2: Color variance (aerial images typically more colorful)
        color_variance = np.var(img_array.reshape(-1, 3), axis=0).mean()
        
        # Feature 3: Text detection (documents contain text)
        # Simple approach: look for horizontal/vertical line patterns
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        h_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)
        
        line_density = (np.sum(h_lines > 0) + np.sum(v_lines > 0)) / (gray.shape[0] * gray.shape[1])
        
        # Feature 4: Aspect ratio and size
        aspect_ratio = max(img.size) / min(img.size)
        
        # Decision logic
        document_score = 0
        
        # High edge density suggests document
        if edge_density > 0.1:
            document_score += 2
        
        # Low color variance suggests document (typically grayscale/monochrome)
        if color_variance < 1000:
            document_score += 1
        
        # High line density suggests document
        if line_density > 0.05:
            document_score += 2
        
        # Extreme aspect ratios might suggest scanned documents
        if aspect_ratio > 1.5:
            document_score += 1
        
        # Classify
        image_type = "document" if document_score >= 3 else "aerial"
        
        print(f"Image analysis for {os.path.basename(image_path)}:")
        print(f"  Edge density: {edge_density:.4f}")
        print(f"  Color variance: {color_variance:.1f}")
        print(f"  Line density: {line_density:.4f}")
        print(f"  Aspect ratio: {aspect_ratio:.2f}")
        print(f"  Document score: {document_score}/6")
        print(f"  Classified as: {image_type}")
        
        return image_type
        
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return "unknown"

def process_mixed_dataset(test_ids_df, image_dir="data/", model=None):
    """Process dataset that may contain both aerial and document images"""
    
    print("HYBRID PROCESSING - DETECTING IMAGE TYPES")
    print("=" * 60)
    
    # Analyze a sample of images to determine dominant type
    sample_size = min(10, len(test_ids_df))
    sample_ids = test_ids_df['ID'].sample(n=sample_size).values
    
    type_counts = {"aerial": 0, "document": 0, "unknown": 0}
    
    print(f"Analyzing {sample_size} sample images to determine data type...")
    
    for pid in sample_ids:
        # Find image file
        image_path = None
        for pattern in [f"{pid}.jpg", f"anonymised_{pid}.jpg", f"{pid}.png"]:
            potential_path = os.path.join(image_dir, pattern)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path:
            img_type = detect_image_type(image_path)
            type_counts[img_type] += 1
    
    print(f"\nSample analysis results:")
    for img_type, count in type_counts.items():
        percentage = count / sample_size * 100
        print(f"  {img_type.capitalize()}: {count}/{sample_size} ({percentage:.1f}%)")
    
    # Determine processing approach
    if type_counts["document"] > type_counts["aerial"]:
        print(f"\n🔍 Detected DOCUMENT dataset - using document processing pipeline")
        return process_as_documents(test_ids_df, image_dir)
    else:
        print(f"\n🛰️  Detected AERIAL dataset - using satellite imagery pipeline")
        return process_as_aerial(test_ids_df, image_dir, model)

def process_as_documents(test_ids_df, image_dir):
    """Process as scanned documents"""
    if not DOCUMENT_PROCESSOR_AVAILABLE:
        raise ImportError("Document processor not available. Please check document_processor.py")
    
    from document_processor import process_test_documents
    return process_test_documents(test_ids_df, image_dir, "submission_documents.csv")

def process_as_aerial(test_ids_df, image_dir, model):
    """Process as aerial/satellite imagery"""
    if model is None:
        raise ValueError("Model required for aerial image processing")
    
    # Use the existing aerial processing pipeline from m1.py
    from m1 import build_test_submission
    return build_test_submission(test_ids_df, "submission_aerial.csv")

def create_adaptive_submission(test_ids_df, image_dir="data/", model=None):
    """Create submission with adaptive processing based on image type"""
    
    print("ADAPTIVE SUBMISSION GENERATION")
    print("=" * 60)
    
    # Initialize processors
    if DOCUMENT_PROCESSOR_AVAILABLE:
        doc_processor = DocumentProcessor()
    
    results = []
    
    for idx, pid in enumerate(test_ids_df['ID']):
        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(test_ids_df)} ({idx/len(test_ids_df)*100:.1f}%)")
        
        # Find image
        image_path = None
        for pattern in [f"{pid}.jpg", f"anonymised_{pid}.jpg", f"{pid}.png"]:
            potential_path = os.path.join(image_dir, pattern)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            # No image found - create empty result
            result = {
                'ID': str(pid),
                'TargetSurvey': '',
                'Certified date': '',
                'Total Area': '',
                'Unit of Measurement': '',
                'Parish': '',
                'LT Num': '',
                'geometry': []
            }
        else:
            # Detect image type and process accordingly
            img_type = detect_image_type(image_path)
            
            if img_type == "document" and DOCUMENT_PROCESSOR_AVAILABLE:
                # Process as document
                result = doc_processor.process_document(image_path, pid)
            elif img_type == "aerial" and model is not None:
                # Process as aerial image
                result = process_single_aerial_image(image_path, pid, model)
            else:
                # Fallback to document processing if available
                if DOCUMENT_PROCESSOR_AVAILABLE:
                    result = doc_processor.process_document(image_path, pid)
                else:
                    result = {
                        'ID': str(pid),
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
    submission_df.to_csv("submission_adaptive.csv", index=False)
    
    print(f"\nAdaptive processing completed!")
    print(f"Results saved to: submission_adaptive.csv")
    
    return submission_df

def process_single_aerial_image(image_path, pid, model):
    """Process a single aerial image using the trained model"""
    try:
        # This would use the existing aerial processing functions
        # Simplified version here
        result = {
            'ID': str(pid),
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': '',
            'geometry': []
        }
        return result
    except Exception as e:
        print(f"Error processing aerial image {image_path}: {e}")
        return {
            'ID': str(pid),
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': '',
            'geometry': []
        }

if __name__ == "__main__":
    print("Hybrid processing module loaded!")
    print("Use process_mixed_dataset() or create_adaptive_submission() for processing")
