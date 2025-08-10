# Barbados Document Analysis - Scanned Plans Processing

## Overview

This solution is designed to handle **scanned land survey documents** rather than aerial/satellite imagery. It extracts both polygon boundaries and metadata from scanned plan documents using computer vision and OCR techniques.

## Problem Understanding

The test data contains **300 scanned land survey plans** (documents/blueprints) that need to be processed to extract:

1. **Polygon coordinates** - Plot boundaries drawn on the plans
2. **Metadata fields** - Text information like area, parish, survey date, etc.

This is fundamentally different from the satellite imagery segmentation approach and requires document analysis techniques.

## Solution Components

### 1. Document Image Preprocessing (`DocumentPreprocessor`)
- Noise reduction and contrast enhancement
- Image sharpening for better text/line visibility
- Adaptive sizing and format handling

### 2. OCR-based Metadata Extraction (`MetadataExtractor`)
- **Multiple OCR engines**: Tesseract + EasyOCR for better accuracy
- **Pattern matching**: Regular expressions for field extraction
- **Field-specific parsing**: Custom patterns for Barbados survey documents

### 3. Polygon Detection (`PolygonExtractor`)
- **Contour detection**: Find closed boundaries in the document
- **Line detection**: Identify straight edges and intersections
- **Edge-based extraction**: Morphological operations for shape detection
- **Multi-method approach**: Combines different techniques for robustness

### 4. Coordinate Conversion (`CoordinateConverter`)
- Converts pixel coordinates to geographic coordinate system
- Handles coordinate system transformation and scaling

## File Structure

```
├── document_processor.py     # Main document processing classes
├── process_documents.py      # Main script for document processing
├── hybrid_processor.py       # Adaptive processing (auto-detects data type)
├── m1.py                    # Original aerial imagery approach
├── requirements.txt         # Updated dependencies
└── README_DOCUMENTS.md      # This file
```

## Installation

Install additional dependencies for document processing:

```bash
pip install -r requirements.txt

# Install Tesseract OCR (system dependency)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from GitHub releases
```

## Usage

### Option 1: Pure Document Processing

```python
# For scanned documents only
python process_documents.py
```

### Option 2: Hybrid Approach (Recommended)

```python
# Automatically detects data type and uses appropriate processing
python m1.py  # Now includes hybrid processing
```

### Option 3: Manual Processing

```python
from document_processor import DocumentProcessor, process_test_documents

# Process all test documents
submission_df = process_test_documents(
    test_ids_df=test_ids,
    image_dir="data/",
    output_file="submission.csv"
)
```

## Key Features

### 🔍 **Advanced Document Analysis**
- Multi-method polygon detection
- Robust OCR with multiple engines
- Document-specific preprocessing

### 📝 **Metadata Extraction**
- Pattern-based field detection
- Barbados-specific parsing rules
- Handles various document formats

### 🎯 **Polygon Detection Methods**
1. **Contour Detection**: Finds closed boundaries
2. **Line Detection**: Identifies straight edges
3. **Edge-based**: Uses morphological operations
4. **Scoring System**: Selects best polygon candidates

### 🔄 **Hybrid Processing**
- Automatically detects image type (aerial vs document)
- Applies appropriate processing pipeline
- Fallback mechanisms for robustness

## Expected Metadata Fields

The system looks for these fields in scanned documents:

- **Total Area**: `"area: 1.23 acres"`, `"1.5 sq ft"`
- **Unit of Measurement**: `"acres"`, `"sq ft"`, `"hectares"`
- **Parish**: `"Christ Church"`, `"St. Michael"`, etc.
- **Surveyed For**: `"surveyed for: John Smith"`
- **Certified Date**: `"certified: 01/02/2023"`
- **LT Num**: `"LT No. 12345"`, `"Lot 67"`

## Polygon Detection Process

1. **Image Enhancement**: Improve contrast and reduce noise
2. **Multiple Detection Methods**:
   - Edge detection + contour finding
   - Line detection + intersection calculation
   - Morphological operations
3. **Polygon Filtering**: Remove invalid/noisy detections
4. **Scoring and Selection**: Choose best polygon candidate
5. **Coordinate Conversion**: Transform to geographic coordinates

## Output Format

Generates a CSV file with columns:
- `ID`: Plot identifier
- `TargetSurvey`: Surveyed for field
- `Certified date`: Survey certification date
- `Total Area`: Plot area value
- `Unit of Measurement`: Area units
- `Parish`: Parish name
- `LT Num`: Lot/plot number
- `geometry`: List of polygon coordinates

## Performance Optimization

### For Better Results:
1. **High-quality scans**: Use 300+ DPI for input documents
2. **Preprocessing**: Ensure documents are properly oriented
3. **OCR tuning**: Adjust OCR parameters for specific document types
4. **Pattern refinement**: Customize regex patterns for your document format

### Troubleshooting:
- **No polygons detected**: Check if document contains clear boundaries
- **Poor OCR results**: Verify image quality and contrast
- **Wrong coordinates**: Adjust coordinate system bounds
- **Missing metadata**: Update regex patterns for your document format

## Advanced Configuration

### Custom OCR Patterns
```python
# Add custom patterns for specific fields
metadata_extractor.field_patterns['Custom Field'] = [
    r'custom field[:\s]+([a-zA-Z0-9\s]+)'
]
```

### Polygon Detection Tuning
```python
# Adjust polygon detection parameters
polygon_extractor.min_area = 2000  # Minimum polygon area
polygon_extractor.max_area = 300000  # Maximum polygon area
```

### Coordinate System
```python
# Update coordinate bounds for your region
converter = CoordinateConverter(bounds=(minx, maxx, miny, maxy))
```

## Comparison: Aerial vs Document Processing

| Aspect | Aerial Imagery | Scanned Documents |
|--------|---------------|-------------------|
| **Input** | Satellite/aerial photos | Scanned paper plans |
| **Approach** | Deep learning segmentation | Computer vision + OCR |
| **Challenges** | Cloud cover, shadows | Scan quality, text recognition |
| **Polygon Source** | Land boundaries from above | Drawn boundaries on plans |
| **Metadata** | External databases | Embedded in document |

## License

Created for the Zindi Africa Barbados competition. Please refer to competition rules for usage guidelines.

## Acknowledgments

- OpenCV team for computer vision tools
- Tesseract/EasyOCR teams for OCR capabilities
- Scikit-image for advanced image processing
