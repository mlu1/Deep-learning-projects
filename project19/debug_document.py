#!/usr/bin/env python3
"""
Debug script for testing document processing on individual files
Usage: python debug_document.py <image_path> [plot_id]
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from document_processor import DocumentProcessor, DocumentPreprocessor, MetadataExtractor, PolygonExtractor
    from hybrid_processor import detect_image_type
    FULL_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Full processor import error: {e}")
    FULL_PROCESSOR_AVAILABLE = False

try:
    from simple_processor import SimpleDocumentProcessor
    SIMPLE_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Simple processor import error: {e}")
    SIMPLE_PROCESSOR_AVAILABLE = False

if not FULL_PROCESSOR_AVAILABLE and not SIMPLE_PROCESSOR_AVAILABLE:
    print("No document processors available!")
    sys.exit(1)

def debug_single_document(image_path, plot_id="TEST"):
    """Debug processing of a single document with detailed output"""
    
    print("="*60)
    print("DOCUMENT PROCESSING DEBUG")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Plot ID: {plot_id}")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File {image_path} does not exist")
        return
    
    # Determine which processor to use
    if FULL_PROCESSOR_AVAILABLE:
        print("Using full document processor...")
        return debug_with_full_processor(image_path, plot_id)
    elif SIMPLE_PROCESSOR_AVAILABLE:
        print("Using simple document processor...")
        return debug_with_simple_processor(image_path, plot_id)
    else:
        print("❌ No processors available")
        return None

def debug_with_simple_processor(image_path, plot_id):
    """Debug using simple processor"""
    processor = SimpleDocumentProcessor()
    
    print("\n1. SIMPLE PROCESSING")
    print("-" * 30)
    
    result = processor.process_document_simple(image_path, plot_id)
    
    print("Result:")
    for key, value in result.items():
        if key == 'geometry':
            geom_info = f"{len(value)} vertices" if isinstance(value, list) else "None"
            print(f"  {key}: {geom_info}")
        else:
            print(f"  {key}: '{value}'")
    
    # Simple visualization
    try:
        img = Image.open(image_path).convert('RGB')
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        if result['geometry']:
            poly = np.array(result['geometry'])
            poly_closed = np.vstack([poly, poly[0]])
            plt.plot(poly_closed[:, 0], poly_closed[:, 1], 'r-', linewidth=2)
            plt.plot(poly_closed[:, 0], poly_closed[:, 1], 'ro', markersize=4)
        plt.title(f'Detected Polygon ({len(result["geometry"])} vertices)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"debug_simple_{plot_id}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    return result

def debug_with_full_processor(image_path, plot_id):
    """Debug using full processor"""
    
    # Step 1: Image type detection
    print("\n1. IMAGE TYPE DETECTION")
    print("-" * 30)
    img_type = detect_image_type(image_path)
    
    # Step 2: Image preprocessing
    print("\n2. IMAGE PREPROCESSING")
    print("-" * 30)
    preprocessor = DocumentPreprocessor()
    enhanced_img = preprocessor.enhance_image(image_path)
    
    if enhanced_img is None:
        print("❌ Failed to preprocess image")
        return
    else:
        print("✅ Image preprocessing successful")
    
    # Step 3: Metadata extraction
    print("\n3. METADATA EXTRACTION")
    print("-" * 30)
    metadata_extractor = MetadataExtractor()
    
    # Extract raw text first to see what OCR is getting
    try:
        enhanced_img_pil = Image.fromarray(enhanced_img)
        raw_text = metadata_extractor._extract_text_multiple_methods(enhanced_img)
        print(f"Raw OCR text (first 200 chars):")
        print(f"'{raw_text[:200]}...'")
        print()
    except Exception as e:
        print(f"Failed to extract raw text: {e}")
        raw_text = ""
    
    metadata = metadata_extractor.extract_metadata(image_path)
    
    print("Extracted metadata:")
    for key, value in metadata.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: '{value}'")
    
    # Step 4: Polygon extraction
    print("\n4. POLYGON EXTRACTION")
    print("-" * 30)
    polygon_extractor = PolygonExtractor()
    
    # Try different polygon extraction methods separately for debugging
    print("Trying contour detection...")
    contour_polygons = polygon_extractor._extract_via_contours(enhanced_img)
    print(f"  Found {len(contour_polygons)} polygons via contours")
    
    print("Trying edge detection...")
    edge_polygons = polygon_extractor._extract_via_edges(enhanced_img)
    print(f"  Found {len(edge_polygons)} polygons via edges")
    
    polygon = polygon_extractor.extract_polygon_from_document(image_path)
    
    if polygon:
        print(f"✅ Final polygon extracted with {len(polygon)} vertices")
        print(f"   First vertex: {polygon[0]}")
        print(f"   Last vertex: {polygon[-1]}")
        
        # Calculate polygon area for validation
        if len(polygon) >= 3:
            import cv2
            poly_array = np.array(polygon, dtype=np.int32)
            area = cv2.contourArea(poly_array)
            print(f"   Polygon area: {area:.1f} pixels")
    else:
        print("❌ No polygon extracted")
        print("   Debug info:")
        print(f"     Total contour candidates: {len(contour_polygons)}")
        print(f"     Total edge candidates: {len(edge_polygons)}")
    
    # Step 5: Full processing
    print("\n5. FULL PROCESSING")
    print("-" * 30)
    processor = DocumentProcessor()
    result = processor.process_document(image_path, plot_id)
    
    print("Final result:")
    for key, value in result.items():
        if key == 'geometry':
            geom_info = f"{len(value)} vertices" if isinstance(value, list) else "None"
            print(f"  {key}: {geom_info}")
        else:
            print(f"  {key}: '{value}'")
    
    # Step 6: Visualization
    print("\n6. VISUALIZATION")
    print("-" * 30)
    visualize_processing_steps(image_path, enhanced_img, polygon, metadata)
    
    return result

def visualize_processing_steps(image_path, enhanced_img, polygon, metadata):
    """Create visualization of processing steps"""
    
    try:
        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Enhanced image
        axes[0, 1].imshow(enhanced_img)
        axes[0, 1].set_title('Enhanced Image')
        axes[0, 1].axis('off')
        
        # Polygon overlay
        axes[1, 0].imshow(enhanced_img)
        if polygon:
            poly_array = np.array(polygon)
            # Close the polygon for visualization
            poly_closed = np.vstack([poly_array, poly_array[0]])
            axes[1, 0].plot(poly_closed[:, 0], poly_closed[:, 1], 'r-', linewidth=3)
            axes[1, 0].plot(poly_closed[:, 0], poly_closed[:, 1], 'ro', markersize=6)
        axes[1, 0].set_title(f'Extracted Polygon ({len(polygon) if polygon else 0} vertices)')
        axes[1, 0].axis('off')
        
        # Metadata summary
        axes[1, 1].axis('off')
        metadata_text = "Extracted Metadata:\n\n"
        for key, value in metadata.items():
            status = "✅" if value else "❌"
            metadata_text += f"{status} {key}:\n   '{value}'\n\n"
        
        axes[1, 1].text(0.05, 0.95, metadata_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 1].set_title('Metadata Extraction Results')
        
        plt.tight_layout()
        
        # Save debug image
        debug_filename = f"debug_{os.path.basename(image_path)}.png"
        plt.savefig(debug_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Debug visualization saved to: {debug_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def test_batch_detection(image_dir="data/", max_files=5):
    """Test image type detection on multiple files"""
    
    print("="*60)
    print("BATCH IMAGE TYPE DETECTION TEST")
    print("="*60)
    
    if not os.path.exists(image_dir):
        print(f"❌ Directory {image_dir} does not exist")
        return
    
    # Find image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']:
        import glob
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print(f"❌ No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Testing on first {min(max_files, len(image_files))} files...\n")
    
    type_counts = {"aerial": 0, "document": 0, "unknown": 0}
    
    for i, image_path in enumerate(image_files[:max_files]):
        print(f"\n--- File {i+1}: {os.path.basename(image_path)} ---")
        img_type = detect_image_type(image_path)
        type_counts[img_type] += 1
    
    print(f"\n{'='*40}")
    print("SUMMARY")
    print("="*40)
    total = sum(type_counts.values())
    for img_type, count in type_counts.items():
        percentage = count / total * 100 if total > 0 else 0
        print(f"{img_type.capitalize()}: {count}/{total} ({percentage:.1f}%)")

def main():
    """Main function for command line usage"""
    
    if len(sys.argv) < 2:
        print("Usage: python debug_document.py <image_path> [plot_id]")
        print("   or: python debug_document.py --batch [image_dir]")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        image_dir = sys.argv[2] if len(sys.argv) > 2 else "data/"
        test_batch_detection(image_dir)
    else:
        image_path = sys.argv[1]
        plot_id = sys.argv[2] if len(sys.argv) > 2 else "TEST"
        debug_single_document(image_path, plot_id)

if __name__ == "__main__":
    main()
