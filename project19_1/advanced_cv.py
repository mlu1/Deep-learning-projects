"""
Advanced Computer Vision Pipeline for Cadastral Plan Analysis

This module implements deep learning approaches for more accurate polygon detection
and metadata extraction from cadastral survey plans.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
from PIL import Image
import easyocr
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import re
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import albumentations as A

class PolygonDetectionCNN(nn.Module):
    """Custom CNN for polygon coordinate prediction"""
    
    def __init__(self, num_coordinates=20):
        super(PolygonDetectionCNN, self).__init__()
        
        # Use pretrained ResNet as backbone
        self.backbone = resnet50(pretrained=True)
        
        # Remove final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add custom regression head for polygon coordinates
        self.regression_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_coordinates)  # x,y coordinates for polygon vertices
        )
        
    def forward(self, x):
        features = self.backbone(x)
        coordinates = self.regression_head(features)
        return coordinates

class AdvancedCadastralExtractor:
    """Advanced extractor using deep learning and sophisticated CV techniques"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.ocr_reader = easyocr.Reader(['en'])
        self.polygon_model = None
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation pipeline for training
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.3),
        ])
    
    def preprocess_image_advanced(self, image_path):
        """Advanced image preprocessing with multiple enhancement techniques"""
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, None, None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Noise reduction using Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 2. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # 4. Adaptive thresholding for text/line detection
        binary = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        return rgb_image, enhanced, binary
    
    def detect_lines_advanced(self, binary_image):
        """Advanced line detection using probabilistic Hough transform"""
        
        # Edge detection with hysteresis
        edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        
        # Morphological closing to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=20)
        
        return lines, edges
    
    def extract_text_advanced(self, image):
        """Advanced text extraction with region filtering"""
        
        # Use EasyOCR for text detection
        results = self.ocr_reader.readtext(image, detail=1)
        
        # Filter and categorize text by region and confidence
        text_regions = {
            'title': [],
            'surveyor': [],
            'dates': [],
            'measurements': [],
            'addresses': [],
            'other': []
        }
        
        for (bbox, text, confidence) in results:
            if confidence < 0.3:
                continue
            
            text = text.strip()
            
            # Calculate text position (top, middle, bottom third)
            y_center = np.mean([point[1] for point in bbox])
            height = image.shape[0]
            
            # Categorize text based on content and position
            text_lower = text.lower()
            
            if any(keyword in text_lower for keyword in ['surveyor', 'prepared by', 'drawn by']):
                text_regions['surveyor'].append({
                    'text': text, 'bbox': bbox, 'confidence': confidence
                })
            elif re.search(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', text):
                text_regions['dates'].append({
                    'text': text, 'bbox': bbox, 'confidence': confidence
                })
            elif re.search(r'\d+\.?\d*\s*(sq\s*m|m²|square|area)', text_lower):
                text_regions['measurements'].append({
                    'text': text, 'bbox': bbox, 'confidence': confidence
                })
            elif any(keyword in text_lower for keyword in ['lot', 'block', 'street', 'road', 'avenue']):
                text_regions['addresses'].append({
                    'text': text, 'bbox': bbox, 'confidence': confidence
                })
            elif y_center < height * 0.2:  # Top 20% likely title
                text_regions['title'].append({
                    'text': text, 'bbox': bbox, 'confidence': confidence
                })
            else:
                text_regions['other'].append({
                    'text': text, 'bbox': bbox, 'confidence': confidence
                })
        
        return text_regions
    
    def detect_polygon_ml(self, image):
        """Machine learning-based polygon detection"""
        
        if self.polygon_model is None:
            return self.detect_polygon_traditional(image)
        
        # Prepare image for ML model
        pil_image = Image.fromarray(image)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict coordinates
        with torch.no_grad():
            predicted_coords = self.polygon_model(input_tensor)
            predicted_coords = predicted_coords.cpu().numpy().flatten()
        
        # Convert to polygon format
        polygon_points = []
        for i in range(0, len(predicted_coords), 2):
            if i+1 < len(predicted_coords):
                x = predicted_coords[i] * image.shape[1]  # Scale to image size
                y = predicted_coords[i+1] * image.shape[0]
                if x > 0 and y > 0:  # Filter out padding
                    polygon_points.append((x, y))
        
        return polygon_points if len(polygon_points) >= 3 else None
    
    def detect_polygon_traditional(self, image):
        """Traditional computer vision polygon detection with improvements (accepts image array)"""
        # If image is not grayscale, convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Enhance and binarize
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Multi-scale contour detection
        polygons = []
        for scale in [1.0, 0.8, 0.6]:
            if scale != 1.0:
                h, w = binary.shape
                resized = cv2.resize(binary, (int(w*scale), int(h*scale)))
            else:
                resized = binary

            contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000 or area > image.shape[0] * image.shape[1] * 0.8:
                    continue
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) >= 3:
                    if scale != 1.0:
                        approx = approx / scale
                    coords = [(point[0][0], point[0][1]) for point in approx]
                    if self.validate_polygon_shape(coords):
                        polygons.append({
                            'coords': coords,
                            'area': area,
                            'confidence': self.calculate_polygon_confidence(coords, area)
                        })
        if polygons:
            best_polygon = max(polygons, key=lambda x: x['confidence'])
            return best_polygon['coords']
        return None
    
    def validate_polygon_shape(self, coords):
        """Validate if detected shape is a reasonable polygon"""
        
        if len(coords) < 3:
            return False
        
        try:
            polygon = Polygon(coords)
            
            # Check if polygon is valid
            if not polygon.is_valid:
                return False
            
            # Check area ratio (not too elongated)
            bbox = polygon.bounds
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox_area > 0:
                ratio = polygon.area / bbox_area
                if ratio < 0.1:  # Too elongated
                    return False
            
            # Check for reasonable number of vertices
            if len(coords) > 20:  # Too many vertices
                return False
            
            return True
            
        except:
            return False
    
    def calculate_polygon_confidence(self, coords, area):
        """Calculate confidence score for detected polygon"""
        
        try:
            polygon = Polygon(coords)
            
            # Factors that increase confidence:
            # 1. Reasonable area
            area_score = min(1.0, area / 10000)  # Normalize to typical lot size
            
            # 2. Regular shape (convex hull ratio)
            convex_hull = polygon.convex_hull
            convexity = polygon.area / convex_hull.area if convex_hull.area > 0 else 0
            
            # 3. Number of vertices (4-8 is typical for lots)
            vertex_score = 1.0 - abs(len(coords) - 6) / 10
            vertex_score = max(0, vertex_score)
            
            # Combine scores
            confidence = (area_score * 0.4 + convexity * 0.4 + vertex_score * 0.2)
            return confidence
            
        except:
            return 0.0
    
    def extract_metadata_advanced(self, text_regions):
        """Advanced metadata extraction using categorized text regions"""
        
        metadata = {
            'Land Surveyor': None,
            'Surveyed For': None,
            'Certified date': None,
            'Total Area': None,
            'Unit of Measurement': None,
            'Address': None,
            'Parish': None,
            'LT Num': None
        }
        
        # Extract surveyor from surveyor regions
        for item in text_regions['surveyor']:
            text = item['text']
            
            # Common patterns for surveyor names
            patterns = [
                r'(?:surveyor|prepared by|drawn by)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+(?:P\.?L\.?S\.?|R\.?L\.?S\.?))?'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metadata['Land Surveyor'] = match.group(1).strip()
                    break
        
        # Extract dates
        for item in text_regions['dates']:
            text = item['text']
            
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{2}/\d{2}/\d{4})',
                r'(\d{2}-\d{2}-\d{4})',
                r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metadata['Certified date'] = match.group(1)
                    break
        
        # Extract measurements
        for item in text_regions['measurements']:
            text = item['text']
            
            area_patterns = [
                r'(\d+\.?\d*)\s*(?:sq\s*m|square\s*meter|m²)',
                r'area[:\s]*(\d+\.?\d*)\s*(?:sq\s*m|square\s*meter|m²)'
            ]
            
            for pattern in area_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metadata['Total Area'] = float(match.group(1))
                    metadata['Unit of Measurement'] = 'sq m'
                    break
        
        # Extract address information
        address_parts = []
        for item in text_regions['addresses']:
            address_parts.append(item['text'])
        
        if address_parts:
            metadata['Address'] = ' '.join(address_parts)
        
        # Extract lot numbers from all text
        all_text = ' '.join([item['text'] for region in text_regions.values() 
                            for item in region])
        
        lot_patterns = [
            r'Lot\s+(\d+)',
            r'LT\s+([\d.]+)',
            r'Plot\s+(\d+)'
        ]
        
        for pattern in lot_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                metadata['LT Num'] = match.group(1)
                break
        
        # Extract parish
        parishes = ['St. Philip', 'St. Michael', 'St. George', 'St. John', 'St. Peter', 
                   'St. Andrew', 'St. James', 'St. Thomas', 'St. Joseph', 'St. Lucy', 'Christ Church']
        
        for parish in parishes:
            if parish.lower() in all_text.lower():
                metadata['Parish'] = parish
                break
        
        return metadata
    
    def process_image_complete(self, image_path):
        """Complete processing pipeline for a single image"""
        
        # Load and preprocess image
        rgb_image, enhanced, binary = self.preprocess_image_advanced(image_path)
        
        if rgb_image is None:
            return None, None
        
        # Extract text regions
        text_regions = self.extract_text_advanced(rgb_image)
        
        # Extract metadata
        metadata = self.extract_metadata_advanced(text_regions)
        
        # Detect polygon
        polygon = self.detect_polygon_ml(rgb_image)
        if polygon is None:
            polygon = self.detect_polygon_traditional(rgb_image)
        
        return polygon, metadata
    
    def visualize_detection_process(self, image_path):
        """Visualize the complete detection process"""
        
        rgb_image, enhanced, binary = self.preprocess_image_advanced(image_path)
        text_regions = self.extract_text_advanced(rgb_image)
        polygon = self.detect_polygon_traditional(rgb_image)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Enhanced image
        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title('Enhanced Image')
        axes[0, 1].axis('off')
        
        # Binary image
        axes[0, 2].imshow(binary, cmap='gray')
        axes[0, 2].set_title('Binary Image')
        axes[0, 2].axis('off')
        
        # Text regions
        axes[1, 0].imshow(rgb_image)
        for region_type, items in text_regions.items():
            if region_type == 'surveyor':
                color = 'red'
            elif region_type == 'dates':
                color = 'blue'
            elif region_type == 'measurements':
                color = 'green'
            else:
                color = 'yellow'
            
            for item in items:
                bbox = np.array(item['bbox'])
                rect = plt.Polygon(bbox, fill=False, edgecolor=color, linewidth=2)
                axes[1, 0].add_patch(rect)
        
        axes[1, 0].set_title('Text Regions')
        axes[1, 0].axis('off')
        
        # Detected polygon
        axes[1, 1].imshow(rgb_image)
        if polygon:
            poly_x = [p[0] for p in polygon] + [polygon[0][0]]
            poly_y = [p[1] for p in polygon] + [polygon[0][1]]
            axes[1, 1].plot(poly_x, poly_y, 'r-', linewidth=3)
            axes[1, 1].scatter(poly_x[:-1], poly_y[:-1], c='red', s=50, zorder=5)
        axes[1, 1].set_title('Detected Polygon')
        axes[1, 1].axis('off')
        
        # Combined result
        axes[1, 2].imshow(rgb_image)
        # Add text annotations
        y_pos = 50
        for region_type, items in text_regions.items():
            if items:
                text = f"{region_type}: {len(items)} regions"
                axes[1, 2].text(10, y_pos, text, fontsize=10, 
                               bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                y_pos += 30
        
        axes[1, 2].set_title('Analysis Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

# Initialize advanced extractor
advanced_extractor = AdvancedCadastralExtractor()
