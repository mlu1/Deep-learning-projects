#!/usr/bin/env python3
"""
Simple Document Processor (Pure Python + PIL)
Minimal dependencies - only uses PIL for image loading and basic Python libraries
"""

import os
import re
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class SimpleProcessor:
    """Simple processor using only basic libraries"""
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Define regex patterns for survey plan metadata"""
        
        # Survey number patterns (simplified)
        self.survey_patterns = [
            r"(?:Survey|Plan|DP|CP|SP)\s*(?:No\.?|Number)?\s*[:\s]*([A-Z0-9\-/]+)",
            r"([0-9]{4}-[0-9]{3})",  # Format like 7703-078
            r"([0-9]{4}-[0-9]{2,3})", # Alternative format
        ]
        
        # Date patterns
        self.date_patterns = [
            r"([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"([0-9]{4}[/-][0-9]{1,2}[/-][0-9]{1,2})",
        ]
        
        # Area patterns
        self.area_patterns = [
            r"([0-9]+\.?[0-9]*)\s*(ha|hectares?|m²|sq\.?\s*m|acres?)",
        ]
        
        # Parish patterns
        self.parish_patterns = [
            r"Parish\s*(?:of\s*)?([A-Z][A-Za-z\s]+)",
        ]
        
        # LT Number patterns
        self.lt_patterns = [
            r"(?:LT|L\.T\.)\s*(?:No\.?)?\s*([A-Z0-9\-/]+)",
        ]
    
    def extract_basic_polygon(self, image_path: str) -> List[List[int]]:
        """Extract a basic rectangular polygon based on image dimensions"""
        try:
            # Load image to get dimensions
            img = Image.open(image_path)
            width, height = img.size
            
            # Create a simple rectangular boundary (placeholder)
            # In a real implementation, this would need actual computer vision
            margin = min(width, height) // 10
            
            polygon = [
                [margin, margin],
                [width - margin, margin],
                [width - margin, height - margin],
                [margin, height - margin]
            ]
            
            return polygon
            
        except Exception as e:
            logger.error(f"Error extracting polygon from {image_path}: {e}")
            return []
    
    def extract_metadata_mock(self, image_path: str) -> Dict[str, str]:
        """Mock metadata extraction - returns empty values"""
        # Without OCR, we can't extract text from images
        # This is a placeholder that could be enhanced with actual OCR
        
        return {
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': ''
        }
    
    def extract_metadata_from_filename(self, image_path: str) -> Dict[str, str]:
        """Try to extract metadata from filename patterns"""
        metadata = {
            'TargetSurvey': '',
            'Certified date': '',
            'Total Area': '',
            'Unit of Measurement': '',
            'Parish': '',
            'LT Num': ''
        }
        
        # Get filename without extension
        filename = Path(image_path).stem
        
        # Try to extract survey number from filename
        for pattern in self.survey_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                metadata['TargetSurvey'] = match.group(1)
                break
        
        # If no survey number found, use the filename as ID
        if not metadata['TargetSurvey']:
            # Clean up filename
            clean_filename = re.sub(r'[^A-Za-z0-9\-]', '', filename)
            if clean_filename:
                metadata['TargetSurvey'] = clean_filename
        
        return metadata
    
    def process_document(self, image_path: str, plot_id: str = None) -> Dict[str, Any]:
        """Process a document image with basic methods"""
        
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
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return result
            
            # Try to extract metadata from filename
            metadata = self.extract_metadata_from_filename(image_path)
            result.update(metadata)
            
            # Extract basic polygon
            polygon = self.extract_basic_polygon(image_path)
            if polygon:
                result['geometry'] = polygon
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
        
        return result
    
    def process_batch_from_ids(self, test_ids_file: str, image_dir: str = "data/", 
                              output_path: str = "submission.csv") -> None:
        """Process batch based on test IDs file"""
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not available - cannot create CSV")
            return
        
        if not os.path.exists(test_ids_file):
            logger.error(f"Test IDs file not found: {test_ids_file}")
            return
        
        # Load test IDs
        test_ids_df = pd.read_csv(test_ids_file)
        logger.info(f"Processing {len(test_ids_df)} test samples")
        
        results = []
        
        for _, row in test_ids_df.iterrows():
            plot_id = str(row['ID'])
            
            # Try different image filename patterns
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
                result = self.process_document(image_path, plot_id)
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
                logger.debug(f"Image not found for ID: {plot_id}")
            
            results.append(result)
        
        # Create DataFrame and save
        df = pd.DataFrame(results)
        
        # Ensure proper column order
        required_columns = ['ID', 'TargetSurvey', 'Certified date', 'Total Area',
                           'Unit of Measurement', 'Parish', 'LT Num', 'geometry']
        
        df = df[required_columns]
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
        
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total samples: {len(results)}")
        
        # Count filled fields
        for col in required_columns:
            if col == 'geometry':
                filled = sum(1 for geom in df['geometry'] if geom and len(geom) > 0)
            else:
                filled = df[col].astype(bool).sum()
            
            percentage = filled / len(df) * 100 if len(df) > 0 else 0
            print(f"{col}: {filled}/{len(df)} ({percentage:.1f}%)")

def create_minimal_submission(test_ids_file: str, output_path: str = "minimal_submission.csv"):
    """Create minimal submission with just IDs and empty fields"""
    try:
        import pandas as pd
        
        if not os.path.exists(test_ids_file):
            print(f"Test IDs file not found: {test_ids_file}")
            return
        
        # Load test IDs
        test_ids_df = pd.read_csv(test_ids_file)
        
        # Create empty submission
        submission_data = []
        for _, row in test_ids_df.iterrows():
            plot_id = str(row['ID'])
            submission_data.append({
                'ID': plot_id,
                'TargetSurvey': '',
                'Certified date': '',
                'Total Area': '',
                'Unit of Measurement': '',
                'Parish': '',
                'LT Num': '',
                'geometry': []
            })
        
        df = pd.DataFrame(submission_data)
        df.to_csv(output_path, index=False)
        
        print(f"Created minimal submission with {len(submission_data)} entries: {output_path}")
        
        # Show sample
        print(f"\nFirst 5 rows:")
        print(df.head().to_string(index=False))
        
    except ImportError:
        print("pandas not available - cannot create CSV")
    except Exception as e:
        print(f"Error creating minimal submission: {e}")

def test_simple_processor():
    """Test the simple processor"""
    processor = SimpleProcessor()
    
    # Create a dummy image for testing
    try:
        img = Image.new('RGB', (800, 600), color='white')
        test_path = "test_simple.png"
        img.save(test_path)
        
        print("Testing Simple Processor")
        print("=" * 30)
        
        result = processor.process_document(test_path, "TEST-001")
        
        print("Result:")
        for key, value in result.items():
            if key == 'geometry':
                geom_info = f"{len(value)} points" if value else "None"
                print(f"  {key}: {geom_info}")
            else:
                print(f"  {key}: '{value}'")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    print("Simple Document Processor (Minimal Dependencies)")
    print("=" * 50)
    
    # Test the processor
    test_simple_processor()
    
    # Check for test IDs file
    test_ids_files = ["test_ids.csv", "test.csv", "sample_submission.csv"]
    test_ids_file = None
    
    for filename in test_ids_files:
        if os.path.exists(filename):
            test_ids_file = filename
            break
    
    if test_ids_file:
        print(f"\nFound test IDs file: {test_ids_file}")
        print("Creating minimal submission...")
        create_minimal_submission(test_ids_file, "simple_submission.csv")
    else:
        print(f"\nNo test IDs file found. Looking for: {test_ids_files}")
        print("You can create a minimal submission by providing a CSV with 'ID' column")
        
        # Create a sample test IDs file for demonstration
        try:
            import pandas as pd
            sample_ids = [
                "7703-078", "8606-095", "7703-064", "7703-101", "7707-114",
                "8604-111", "7703-049", "7706-060", "7707-141", "7707-152"
            ]
            
            sample_df = pd.DataFrame({'ID': sample_ids})
            sample_df.to_csv("sample_test_ids.csv", index=False)
            
            print("Created sample_test_ids.csv for demonstration")
            create_minimal_submission("sample_test_ids.csv", "demo_submission.csv")
            
        except ImportError:
            print("pandas not available - cannot create sample files")
