#!/usr/bin/env python3
"""
Main script for land survey plan processing (No OpenCV)
Refactored to remove cv2 dependency and use PIL + scikit-image instead
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check which processing libraries are available"""
    available = {
        'PIL': False,
        'numpy': False,
        'pandas': False,
        'scikit-image': False,
        'scipy': False,
        'tesseract': False,
        'easyocr': False
    }
    
    try:
        from PIL import Image
        available['PIL'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        available['numpy'] = True
    except ImportError:
        pass
    
    try:
        import pandas
        available['pandas'] = True
    except ImportError:
        pass
    
    try:
        from skimage import filters
        available['scikit-image'] = True
    except ImportError:
        pass
    
    try:
        from scipy import ndimage
        available['scipy'] = True
    except ImportError:
        pass
    
    try:
        import pytesseract
        available['tesseract'] = True
    except ImportError:
        pass
    
    try:
        import easyocr
        available['easyocr'] = True
    except ImportError:
        pass
    
    return available

def get_best_processor(dependencies):
    """Get the best available processor based on dependencies"""
    
    if dependencies['PIL'] and dependencies['scikit-image'] and dependencies['scipy']:
        try:
            from document_processor_no_cv2 import DocumentProcessor
            from hybrid_processor_no_cv2 import HybridProcessor
            logger.info("Using advanced document processor (PIL + scikit-image)")
            return HybridProcessor()
        except ImportError:
            pass
    
    if dependencies['PIL']:
        try:
            from simple_processor_no_cv2 import SimpleProcessor
            logger.info("Using simple processor (PIL only)")
            return SimpleProcessor()
        except ImportError:
            pass
    
    # Fallback to minimal processor
    logger.warning("Using minimal processor (no image processing)")
    return MinimalProcessor()

class MinimalProcessor:
    """Minimal processor that creates empty submissions"""
    
    def process_batch_from_ids(self, test_ids_file: str, image_dir: str = "data/", 
                              output_path: str = "submission.csv"):
        """Create minimal submission with empty fields"""
        try:
            import pandas as pd
            
            if not os.path.exists(test_ids_file):
                logger.error(f"Test IDs file not found: {test_ids_file}")
                return
            
            # Load test IDs
            test_ids_df = pd.read_csv(test_ids_file)
            logger.info(f"Creating minimal submission for {len(test_ids_df)} samples")
            
            # Create empty results
            results = []
            for _, row in test_ids_df.iterrows():
                plot_id = str(row['ID'])
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
                results.append(result)
            
            # Save to CSV
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved minimal submission to {output_path}")
            
        except ImportError:
            logger.error("pandas not available - cannot create submission")
        except Exception as e:
            logger.error(f"Error creating minimal submission: {e}")

def main():
    """Main execution function"""
    print("Land Survey Plan Processing (No OpenCV)")
    print("=" * 50)
    
    # Check dependencies
    dependencies = check_dependencies()
    
    print("Available dependencies:")
    for lib, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"  {status} {lib}")
    
    # Check for required files
    if not os.path.exists("data/"):
        logger.warning("No 'data/' directory found")
        print("\n⚠️  No 'data/' directory found")
        print("Expected structure:")
        print("  data/")
        print("    ├── image1.jpg")
        print("    ├── image2.jpg") 
        print("    └── ...")
    
    # Look for test IDs file
    test_ids_files = ["test_ids.csv", "test.csv", "sample_submission.csv"]
    test_ids_file = None
    
    for filename in test_ids_files:
        if os.path.exists(filename):
            test_ids_file = filename
            break
    
    if not test_ids_file:
        logger.warning(f"No test IDs file found. Looking for: {test_ids_files}")
        
        # Create a sample for demonstration
        if dependencies['pandas']:
            print("\nCreating sample test IDs for demonstration...")
            sample_ids = [
                "7703-078", "8606-095", "7703-064", "7703-101", "7707-114",
                "8604-111", "7703-049", "7706-060", "7707-141", "7707-152",
                "7706-062", "7703-066", "8606-091", "7703-071", "7703-072",
                "8604-128", "8605-014", "8606-092", "7703-087", "7702-112",
                "7702-115", "7704-135", "7704-137", "8603-125", "7707-192",
                "5601-027", "7711-039", "7714-088"
            ]
            
            sample_df = pd.DataFrame({'ID': sample_ids})
            test_ids_file = "sample_test_ids.csv"
            sample_df.to_csv(test_ids_file, index=False)
            print(f"Created {test_ids_file}")
        else:
            print("Cannot create sample - pandas not available")
            return
    
    print(f"\nUsing test IDs file: {test_ids_file}")
    
    # Get processor
    processor = get_best_processor(dependencies)
    
    # Process documents
    output_file = "submission.csv"
    
    try:
        if hasattr(processor, 'process_batch_from_ids'):
            processor.process_batch_from_ids(test_ids_file, "data/", output_file)
        elif hasattr(processor, 'process_batch'):
            processor.process_batch("data/", test_ids_file, output_file)
        else:
            logger.error("Processor doesn't have expected methods")
            return
        
        # Check if file was created
        if os.path.exists(output_file):
            print(f"\n✓ Submission created: {output_file}")
            
            # Show sample of results
            if dependencies['pandas']:
                df = pd.read_csv(output_file)
                print(f"\nSubmission contains {len(df)} rows")
                print("\nFirst 5 rows:")
                print(df.head().to_string(index=False))
                
                # Count non-empty fields
                print(f"\nField extraction summary:")
                for col in df.columns:
                    if col == 'geometry':
                        filled = sum(1 for geom in df['geometry'] if geom and str(geom) != '[]')
                    else:
                        filled = df[col].astype(bool).sum()
                    
                    percentage = filled / len(df) * 100 if len(df) > 0 else 0
                    print(f"  {col}: {filled}/{len(df)} ({percentage:.1f}%)")
        else:
            print(f"\n✗ Failed to create submission file")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"\n✗ Processing failed: {e}")
    
    print(f"\n" + "=" * 50)
    print("Processing complete!")
    
    if not dependencies['scikit-image'] or not dependencies['scipy']:
        print("\n💡 To improve extraction capabilities, install:")
        print("   pip install scikit-image scipy")
    
    if not dependencies['tesseract'] and not dependencies['easyocr']:
        print("\n💡 For text extraction (OCR), install:")
        print("   pip install pytesseract easyocr")

if __name__ == "__main__":
    main()
