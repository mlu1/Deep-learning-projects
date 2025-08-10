import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely import wkt
import easyocr
import re
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CadastralPlanExtractor:
    """
    A comprehensive system for extracting polygon coordinates and metadata 
    from cadastral survey plan images.
    """
    
    def __init__(self):
        self.ocr_reader = easyocr.Reader(['en'])
        self.scaler = StandardScaler()
        self.polygon_model = None
        self.metadata_extractors = {}
        
    def load_training_data(self, train_csv_path):
        """Load and parse training data"""
        print("Loading training data...")
        self.train_df = pd.read_csv(train_csv_path)
        
        # Parse polygon coordinates from WKT format
        self.train_df['polygon_coords'] = self.train_df['geometry'].apply(self.parse_wkt_polygon)
        self.train_df['polygon_area_calculated'] = self.train_df['polygon_coords'].apply(self.calculate_polygon_area)
        
        print(f"Loaded {len(self.train_df)} training samples")
        return self.train_df
    
    def parse_wkt_polygon(self, wkt_string):
        """Parse WKT polygon string to extract coordinates"""
        try:
            polygon = wkt.loads(wkt_string)
            # Return exterior coordinates as list of (x, y) tuples
            return list(polygon.exterior.coords)
        except:
            return None
    
    def calculate_polygon_area(self, coords):
        """Calculate area of polygon from coordinates"""
        if coords is None or len(coords) < 3:
            return 0
        try:
            polygon = Polygon(coords)
            return polygon.area
        except:
            return 0
    
    def preprocess_image(self, image_path):
        """Preprocess cadastral plan image for analysis"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, None
            
        # Convert to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to enhance text
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Denoise
        denoised = cv2.medianBlur(binary, 3)
        
        return rgb_image, denoised
    
    def extract_text_regions(self, image):
        """Extract text from image using OCR"""
        try:
            results = self.ocr_reader.readtext(image)
            
            text_data = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence detections
                    text_data.append({
                        'text': text.strip(),
                        'bbox': bbox,
                        'confidence': confidence
                    })
            
            return text_data
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return []
    
    def extract_metadata_from_text(self, text_data):
        """Extract metadata fields from OCR text"""
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
        
        # Combine all text
        all_text = ' '.join([item['text'] for item in text_data])
        
        # Extract surveyor names (look for common patterns)
        surveyor_patterns = [
            r'Surveyor[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'By[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Prepared by[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        
        for pattern in surveyor_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                metadata['Land Surveyor'] = match.group(1)
                break
        
        # Extract dates
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{2}/\d{2}/\d{4})',
            r'(\d{2}-\d{2}-\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, all_text)
            if match:
                metadata['Certified date'] = match.group(1)
                break
        
        # Extract area measurements
        area_patterns = [
            r'(\d+\.?\d*)\s*(sq\s*m|square\s*meter|m²)',
            r'Area[:\s]+(\d+\.?\d*)\s*(sq\s*m|square\s*meter|m²)'
        ]
        
        for pattern in area_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                metadata['Total Area'] = float(match.group(1))
                metadata['Unit of Measurement'] = 'sq m'
                break
        
        # Extract lot numbers
        lot_patterns = [
            r'Lot\s+(\d+)',
            r'LT\s+(\d+\.?\d*\.?\d*\.?\d*)'
        ]
        
        for pattern in lot_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                metadata['LT Num'] = match.group(1)
                break
        
        # Extract parish information
        parishes = ['St. Philip', 'St. Michael', 'St. George', 'St. John', 'St. Peter', 
                   'St. Andrew', 'St. James', 'St. Thomas', 'St. Joseph', 'St. Lucy', 'Christ Church']
        
        for parish in parishes:
            if parish.lower() in all_text.lower():
                metadata['Parish'] = parish
                break
        
        return metadata
    
    def detect_polygon_boundaries(self, image):
        """Detect polygon boundaries using edge detection and contour analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to connect edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        potential_polygons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:  # At least triangle
                    # Convert to coordinate list
                    coords = [(point[0][0], point[0][1]) for point in approx]
                    potential_polygons.append({
                        'coords': coords,
                        'area': area,
                        'contour': contour
                    })
        
        # Sort by area (largest first, likely to be the main parcel)
        potential_polygons.sort(key=lambda x: x['area'], reverse=True)
        
        return potential_polygons
    
    def extract_features_from_image(self, image_path):
        """Extract comprehensive features from an image for ML model"""
        rgb_image, processed = self.preprocess_image(image_path)
        if rgb_image is None:
            return None
        
        # Image properties
        height, width = rgb_image.shape[:2]
        
        # Text extraction
        text_data = self.extract_text_regions(rgb_image)
        
        # Polygon detection
        polygons = self.detect_polygon_boundaries(rgb_image)
        
        # Extract features
        features = {
            'image_width': width,
            'image_height': height,
            'image_aspect_ratio': width / height,
            'num_text_regions': len(text_data),
            'avg_text_confidence': np.mean([t['confidence'] for t in text_data]) if text_data else 0,
            'num_potential_polygons': len(polygons),
            'largest_polygon_area': polygons[0]['area'] if polygons else 0,
            'total_polygon_area': sum([p['area'] for p in polygons]) if polygons else 0
        }
        
        # Add edge density features
        edges = cv2.Canny(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (width * height)
        
        return features, text_data, polygons
    
    def train_polygon_prediction_model(self):
        """Train ML model to predict polygon coordinates from image features"""
        print("Training polygon prediction model...")
        
        # Extract features for all training images
        training_features = []
        training_targets = []
        
        for idx, row in self.train_df.iterrows():
            image_path = f"data/anonymised_{row['ID']}.jpg"
            
            if os.path.exists(image_path):
                features, _, _ = self.extract_features_from_image(image_path)
                coords = row['polygon_coords']
                
                # Only include samples that have both valid features and valid coordinates
                if features is not None and coords is not None and len(coords) >= 3:
                    training_features.append(list(features.values()))
                    
                    # Target: normalized polygon coordinates
                    # Flatten coordinates and normalize
                    flat_coords = []
                    for coord in coords[:-1]:  # Exclude last point (duplicate of first)
                        # Handle both 2D (x, y) and 3D (x, y, z) coordinates
                        if len(coord) >= 2:
                            x, y = coord[0], coord[1]  # Take only x, y coordinates
                            flat_coords.extend([x, y])
                    
                    # Pad/truncate to fixed length (e.g., 20 coordinates = 10 points)
                    target_length = 20
                    if len(flat_coords) > target_length:
                        flat_coords = flat_coords[:target_length]
                    else:
                        flat_coords.extend([0] * (target_length - len(flat_coords)))
                    
                    training_targets.append(flat_coords)
            
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(self.train_df)} training images")
        
        if len(training_features) > 0:
            # Verify that features and targets have the same length
            if len(training_features) != len(training_targets):
                print(f"Warning: Mismatch in training data - features: {len(training_features)}, targets: {len(training_targets)}")
                # Truncate to the minimum length
                min_length = min(len(training_features), len(training_targets))
                training_features = training_features[:min_length]
                training_targets = training_targets[:min_length]
                print(f"Truncated both to length: {min_length}")
            
            X = np.array(training_features)
            y = np.array(training_targets)
            
            print(f"Training data shape: X={X.shape}, y={y.shape}")
            
            # Need at least 10 samples for training
            if len(X) < 10:
                print(f"Insufficient training data: only {len(X)} valid samples found")
                print("Need at least 10 samples with both valid images and polygon coordinates")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest model
            self.polygon_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.polygon_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.polygon_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model trained. MSE: {mse:.2f}, R²: {r2:.3f}")
            print(f"Trained on {len(training_features)} samples")
        else:
            print("No valid training data found!")
    
    def predict_polygon_and_metadata(self, image_path):
        """Predict polygon coordinates and extract metadata for a test image"""
        features, text_data, detected_polygons = self.extract_features_from_image(image_path)
        
        if features is None:
            return None, None
        
        # Extract metadata
        metadata = self.extract_metadata_from_text(text_data)
        
        # Predict polygon using ML model if available
        predicted_polygon = None
        if self.polygon_model is not None:
            feature_vector = np.array([list(features.values())]).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            pred_coords = self.polygon_model.predict(feature_vector_scaled)[0]
            
            # Reshape predicted coordinates
            predicted_polygon = []
            for i in range(0, len(pred_coords), 2):
                if pred_coords[i] != 0 or pred_coords[i+1] != 0:
                    predicted_polygon.append((pred_coords[i], pred_coords[i+1]))
        
        # Fallback to detected polygons if ML prediction not available
        if predicted_polygon is None or len(predicted_polygon) < 3:
            if detected_polygons:
                predicted_polygon = detected_polygons[0]['coords']
        
        return predicted_polygon, metadata
    
    def process_test_images(self, test_csv_path, output_path):
        """Process all test images and generate predictions"""
        test_df = pd.read_csv(test_csv_path)
        results = []
        
        print(f"Processing {len(test_df)} test images...")
        
        for idx, row in test_df.iterrows():
            image_id = row['ID']
            image_path = f"data/anonymised_{image_id}.jpg"
            
            if os.path.exists(image_path):
                polygon, metadata = self.predict_polygon_and_metadata(image_path)
                
                # Create result entry
                result = {'ID': image_id}
                result.update(metadata)
                
                # Store polygon coordinates as a list of tuples
                if polygon and len(polygon) >= 3:
                    try:
                        # Ensure polygon is closed
                        if polygon[0] != polygon[-1]:
                            polygon.append(polygon[0])
                        
                        # Format as list of tuples string
                        coord_str = str([f"({x}, {y})" for x, y in polygon]).replace("'", "")
                        result['geometry'] = coord_str
                    except:
                        result['geometry'] = None
                else:
                    result['geometry'] = None
                
                results.append(result)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(test_df)} test images")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Fill missing values with reasonable defaults
        results_df['Land Surveyor'] = results_df['Land Surveyor'].fillna('Unknown')
        results_df['Surveyed For'] = results_df['Surveyed For'].fillna('Unknown')
        results_df['Certified date'] = results_df['Certified date'].fillna('Unknown')
        results_df['Total Area'] = results_df['Total Area'].fillna(0.0)
        results_df['Unit of Measurement'] = results_df['Unit of Measurement'].fillna('sq m')
        results_df['Address'] = results_df['Address'].fillna('Unknown')
        results_df['Parish'] = results_df['Parish'].fillna('Unknown')
        results_df['LT Num'] = results_df['LT Num'].fillna('Unknown')
        
        # Create TargetSurvey column by concatenating and cleaning fields
        results_df['TargetSurvey'] = (
            results_df['Land Surveyor'].astype(str).str.strip() + " " +
            results_df['Surveyed For'].astype(str).str.strip() + " " +
            results_df['Address'].astype(str).str.strip()
        )
        
        # Clean TargetSurvey: lowercase, remove punctuation, normalize spaces
        results_df['TargetSurvey'] = results_df['TargetSurvey'].apply(
            lambda x: re.sub(r"\s+", " ", re.sub(r"[.,]", " ", x.lower())).strip()
        )
        
        # Final columns to keep in the correct order
        required_columns = [
            'ID', 'TargetSurvey', 'Certified date', 'Total Area',
            'Unit of Measurement', 'Parish', 'LT Num', 'geometry'
        ]
        
        # Add any missing columns with default values
        for col in required_columns:
            if col not in results_df.columns:
                results_df[col] = 'Unknown'
        
        # Reorder columns to match requirements
        results_df = results_df[required_columns]
        
        # Save results without quoting the TargetSurvey field
        results_df.to_csv(output_path, index=False, quoting=1, quotechar='"', 
                         columns=required_columns,
                         escapechar='\\',
                         doublequote=False)
        print(f"Results saved to {output_path}")
        
        return results_df
    
    def visualize_results(self, image_path, polygon, metadata):
        """Visualize extraction results for a single image"""
        rgb_image, _ = self.preprocess_image(image_path)
        
        if rgb_image is None:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(rgb_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Image with detected polygon
        plt.subplot(2, 2, 2)
        plt.imshow(rgb_image)
        if polygon and len(polygon) >= 3:
            poly_x = [p[0] for p in polygon] + [polygon[0][0]]
            poly_y = [p[1] for p in polygon] + [polygon[0][1]]
            plt.plot(poly_x, poly_y, 'r-', linewidth=2, label='Detected Polygon')
            plt.scatter(poly_x[:-1], poly_y[:-1], c='red', s=50, zorder=5)
        plt.title('Detected Polygon')
        plt.legend()
        plt.axis('off')
        
        # Metadata
        plt.subplot(2, 1, 2)
        plt.axis('off')
        metadata_text = "Extracted Metadata:\n"
        for key, value in metadata.items():
            metadata_text += f"{key}: {value}\n"
        plt.text(0.1, 0.5, metadata_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.show()

# Initialize the extractor
extractor = CadastralPlanExtractor()
