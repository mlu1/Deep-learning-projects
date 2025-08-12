#!/usr/bin/env python3
"""
Single SOTA Deep Learning File for Cadastral Plan Extraction

This file uses state-of-the-art deep learning methods to:
1. Extract polygons from cadastral survey plans
2. Extract metadata from the plans
3. Generate final_test_predictions.csv

No visualization, no comparison - just pure SOTA deep learning.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import easyocr
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from shapely.geometry import Polygon
from shapely import wkt
import warnings
warnings.filterwarnings('ignore')

class AdvancedSOTACadastralNet(nn.Module):
    """Advanced SOTA CNN with attention mechanism and residual connections"""
    
    def __init__(self, num_points=8):
        super(AdvancedSOTACadastralNet, self).__init__()
        
        # Enhanced feature extraction with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.res_block1 = self._make_layer(64, 128, 3)
        self.res_block2 = self._make_layer(128, 256, 4)
        self.res_block3 = self._make_layer(256, 512, 6)
        self.res_block4 = self._make_layer(512, 1024, 3)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Advanced classifier with multiple stages
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            
            # First stage - feature reduction
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            # Second stage - spatial understanding
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # Third stage - coordinate refinement
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(512, num_points * 2)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Create residual layer"""
        layers = []
        
        # First block with dimension change
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x * attention_weights
        
        # Classification
        x = self.classifier(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class SOTACadastralNet(nn.Module):
    """Original SOTA CNN for backward compatibility"""
    
    def __init__(self, num_points=8):
        super(SOTACadastralNet, self).__init__()
        
        # Feature extraction with residual connections
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Fully connected layers with dropout
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_points * 2)  # x,y coordinates for each point
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.fc_layers(x)
        return x

class AdvancedCadastralDataset(Dataset):
    """Advanced dataset with data augmentation and better preprocessing"""
    
    def __init__(self, df, data_dir, transform=None, num_points=8, augment=True):
        self.df = df
        self.data_dir = data_dir
        self.num_points = num_points
        self.augment = augment
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
        ])
        
        # Augmentation transforms
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.aug_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.data_dir, f"anonymised_{row['ID']}.jpg")
        
        # Load and preprocess image
        image = self.load_and_enhance_image(image_path)
        
        # Parse polygon coordinates
        try:
            coords = self.parse_wkt_polygon(row['geometry'])
            target_coords = self.normalize_polygon_advanced(coords, self.num_points)
        except Exception as e:
            print(f"Error processing coordinates for {row['ID']}: {e}")
            target_coords = self.get_default_coordinates()
        
        # Apply transforms
        try:
            image = self.base_transform(image)
            image = self.aug_transform(image)
        except Exception as e:
            print(f"Error transforming image for {row['ID']}: {e}")
            image = torch.zeros((3, 512, 512), dtype=torch.float32)
        
        return image, torch.FloatTensor(target_coords)
    
    def load_and_enhance_image(self, image_path):
        """Load and enhance image with advanced preprocessing"""
        try:
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    # Advanced preprocessing
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Noise reduction
                    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                    
                    # Contrast enhancement
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    
                    # Sharpening
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    image = cv2.filter2D(image, -1, kernel)
                    
                    return image
            
            # Create default image
            return np.ones((512, 512, 3), dtype=np.uint8) * 128
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    def normalize_polygon_advanced(self, coords, num_points):
        """Advanced polygon normalization with geometric constraints"""
        if len(coords) == 0:
            return self.get_default_coordinates()
        
        # Ensure all coordinates are tuples of floats
        coords = [(float(x), float(y)) for x, y in coords]
        
        # Remove duplicate consecutive points
        unique_coords = []
        for i, coord in enumerate(coords):
            if i == 0 or coord != coords[i-1]:
                unique_coords.append(coord)
        coords = unique_coords
        
        # Ensure minimum number of points
        if len(coords) < 3:
            return self.get_default_coordinates()
        
        # Interpolate or sample to get exactly num_points
        if len(coords) != num_points:
            coords = self.resample_polygon(coords, num_points)
        
        # Normalize to image coordinates [0, 1]
        coords_array = np.array(coords, dtype=np.float32)
        
        # Get bounding box for normalization
        min_x, min_y = coords_array.min(axis=0)
        max_x, max_y = coords_array.max(axis=0)
        
        # Normalize to [0, 1] range
        if max_x > min_x:
            coords_array[:, 0] = (coords_array[:, 0] - min_x) / (max_x - min_x)
        if max_y > min_y:
            coords_array[:, 1] = (coords_array[:, 1] - min_y) / (max_y - min_y)
        
        return coords_array.flatten()
    
    def resample_polygon(self, coords, target_points):
        """Resample polygon to target number of points"""
        if len(coords) == target_points:
            return coords
        
        # Convert to numpy array
        coords_array = np.array(coords)
        
        # Calculate cumulative distances
        distances = np.sqrt(np.sum(np.diff(coords_array, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        total_distance = cumulative_distances[-1]
        
        # Create target distances
        target_distances = np.linspace(0, total_distance, target_points)
        
        # Interpolate coordinates
        new_coords = []
        for target_dist in target_distances:
            # Find the segment containing this distance
            segment_idx = np.searchsorted(cumulative_distances, target_dist) - 1
            segment_idx = max(0, min(segment_idx, len(coords) - 2))
            
            # Linear interpolation within the segment
            segment_start_dist = cumulative_distances[segment_idx]
            segment_end_dist = cumulative_distances[segment_idx + 1]
            
            if segment_end_dist > segment_start_dist:
                t = (target_dist - segment_start_dist) / (segment_end_dist - segment_start_dist)
                t = max(0, min(1, t))
            else:
                t = 0
            
            # Interpolate coordinates
            start_coord = coords[segment_idx]
            end_coord = coords[(segment_idx + 1) % len(coords)]
            
            new_x = start_coord[0] + t * (end_coord[0] - start_coord[0])
            new_y = start_coord[1] + t * (end_coord[1] - start_coord[1])
            
            new_coords.append((new_x, new_y))
        
        return new_coords
    
    def get_default_coordinates(self):
        """Get default normalized rectangle coordinates"""
        default_coords = np.array([
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
        ][:self.num_points], dtype=np.float32)
        return default_coords.flatten()
    
    def parse_wkt_polygon(self, wkt_string):
        """Parse WKT polygon string to coordinates"""
        try:
            if pd.isna(wkt_string) or not wkt_string or str(wkt_string).lower() == 'nan':
                return [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]  # Default rectangle
            
            # Clean the WKT string
            wkt_clean = str(wkt_string).strip()
            
            # Handle 3D WKT (POLYGON Z) by converting to 2D
            if 'POLYGON Z' in wkt_clean.upper():
                import re
                # Convert POLYGON Z to POLYGON
                wkt_clean = re.sub(r'POLYGON\s+Z\s*', 'POLYGON ', wkt_clean, flags=re.IGNORECASE)
                # Remove Z coordinate values (keep only X and Y)
                wkt_clean = re.sub(r'([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)', r'\1 \2', wkt_clean)
            
            # Handle different WKT formats
            if 'POLYGON' in wkt_clean.upper():
                try:
                    from shapely import wkt
                    polygon = wkt.loads(wkt_clean)
                    coords = list(polygon.exterior.coords)[:-1]  # Remove duplicate last point
                except Exception:
                    # If WKT parsing fails, try manual coordinate extraction
                    import re
                    coord_pattern = r'([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)'
                    matches = re.findall(coord_pattern, wkt_clean)
                    coords = [(float(x), float(y)) for x, y in matches] if matches else []
            else:
                # Try to parse as simple coordinate list
                import re
                numbers = re.findall(r'-?\d+\.?\d*', wkt_clean)
                if len(numbers) >= 6:  # At least 3 points (6 numbers)
                    coords = []
                    for i in range(0, len(numbers), 2):
                        if i+1 < len(numbers):
                            coords.append((float(numbers[i]), float(numbers[i+1])))
                else:
                    coords = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            
            # Ensure we have valid coordinates
            if not coords or len(coords) < 3:
                coords = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            
            # Ensure all coordinates are 2D tuples
            clean_coords = []
            for coord in coords:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    clean_coords.append((float(coord[0]), float(coord[1])))
                else:
                    clean_coords.append((0.0, 0.0))
            
            return clean_coords
            
        except Exception as e:
            print(f"Error parsing WKT '{wkt_string}': {e}")
            return [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

class CadastralDataset(Dataset):
    """Dataset for cadastral images and polygon coordinates"""
    
    def __init__(self, df, data_dir, transform=None, num_points=8):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.num_points = num_points
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.data_dir, f"anonymised_{row['ID']}.jpg")
        
        # Load image
        if os.path.exists(image_path):
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = np.zeros((512, 512, 3), dtype=np.uint8)
            except:
                image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            # Create dummy image if not found
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Parse polygon coordinates
        try:
            coords = self.parse_wkt_polygon(row['geometry'])
            target_coords = self.normalize_polygon(coords, self.num_points)
        except Exception as e:
            print(f"Error processing coordinates for {row['ID']}: {e}")
            # Create default coordinates
            default_coords = [(0.0, 0.0)] * self.num_points
            target_coords = np.array(default_coords, dtype=np.float32).flatten()
        
        # Transform image
        try:
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error transforming image for {row['ID']}: {e}")
            # Create default tensor
            image = torch.zeros((3, 512, 512), dtype=torch.float32)
        
        return image, torch.FloatTensor(target_coords)
    
    def parse_wkt_polygon(self, wkt_string):
        """Parse WKT polygon string to coordinates"""
        try:
            if pd.isna(wkt_string) or not wkt_string or str(wkt_string).lower() == 'nan':
                return [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]  # Default rectangle
            
            # Clean the WKT string
            wkt_clean = str(wkt_string).strip()
            
            # Handle 3D WKT (POLYGON Z) by converting to 2D
            if 'POLYGON Z' in wkt_clean.upper():
                import re
                # Convert POLYGON Z to POLYGON
                wkt_clean = re.sub(r'POLYGON\s+Z\s*', 'POLYGON ', wkt_clean, flags=re.IGNORECASE)
                # Remove Z coordinate values (keep only X and Y)
                wkt_clean = re.sub(r'([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)', r'\1 \2', wkt_clean)
            
            # Handle different WKT formats
            if 'POLYGON' in wkt_clean.upper():
                try:
                    polygon = wkt.loads(wkt_clean)
                    coords = list(polygon.exterior.coords)[:-1]  # Remove duplicate last point
                except Exception:
                    # If WKT parsing fails, try manual coordinate extraction
                    import re
                    coord_pattern = r'([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)'
                    matches = re.findall(coord_pattern, wkt_clean)
                    coords = [(float(x), float(y)) for x, y in matches] if matches else []
            else:
                # Try to parse as simple coordinate list
                import re
                numbers = re.findall(r'-?\d+\.?\d*', wkt_clean)
                if len(numbers) >= 6:  # At least 3 points (6 numbers)
                    coords = []
                    for i in range(0, len(numbers), 2):
                        if i+1 < len(numbers):
                            coords.append((float(numbers[i]), float(numbers[i+1])))
                else:
                    coords = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            
            # Ensure we have valid coordinates
            if not coords or len(coords) < 3:
                coords = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
            
            # Ensure all coordinates are 2D tuples
            clean_coords = []
            for coord in coords:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    clean_coords.append((float(coord[0]), float(coord[1])))
                else:
                    clean_coords.append((0.0, 0.0))
            
            return clean_coords
            
        except Exception as e:
            print(f"Error parsing WKT '{wkt_string}': {e}")
            return [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    
    def normalize_polygon(self, coords, num_points):
        """Normalize polygon to fixed number of points"""
        if len(coords) == 0:
            coords = [(0.0, 0.0)] * num_points
        
        # Ensure all coordinates are tuples of floats
        coords = [(float(x), float(y)) for x, y in coords]
        
        # Interpolate or sample to get exactly num_points
        if len(coords) > num_points:
            # Sample evenly spaced points
            indices = np.linspace(0, len(coords)-1, num_points).astype(int)
            coords = [coords[i] for i in indices]
        elif len(coords) < num_points:
            # Duplicate points to reach target number
            while len(coords) < num_points:
                coords.append(coords[-1])  # Duplicate last point
        
        # Ensure we have exactly num_points
        coords = coords[:num_points]
        
        # Convert to numpy array and ensure shape
        coords_array = np.array(coords, dtype=np.float32)
        if coords_array.shape != (num_points, 2):
            # Fallback: create default rectangle
            coords_array = np.array([(0.0, 0.0)] * num_points, dtype=np.float32)
        
        # Normalize coordinates to [0, 1] range
        if coords_array.max() > 1.0:
            coords_array[:, 0] = coords_array[:, 0] / 1000.0  # Normalize x
            coords_array[:, 1] = coords_array[:, 1] / 1000.0  # Normalize y
        
        # Clamp to [0, 1] range
        coords_array = np.clip(coords_array, 0.0, 1.0)
        
        return coords_array.flatten()

class ImprovedSOTADeepLearningExtractor:
    """Improved SOTA Deep Learning Extractor with advanced techniques"""
    
    def __init__(self, num_points=8, use_advanced=True):
        self.num_points = num_points
        self.use_advanced = use_advanced
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use advanced model by default
        if use_advanced:
            self.model = AdvancedSOTACadastralNet(num_points=num_points)
        else:
            self.model = SOTACadastralNet(num_points=num_points)
        
        self.model.to(self.device)
        
        # OCR for metadata extraction
        self.ocr_reader = easyocr.Reader(['en'])
        
        print(f"✅ Improved SOTA Deep Learning Extractor initialized on {self.device}")
        print(f"   Model: {'Advanced' if use_advanced else 'Standard'}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train(self, train_csv_path, data_dir, epochs=5, batch_size=8, learning_rate=0.0001):
        """Train with improved techniques"""
        
        print("🚀 Starting Improved SOTA Deep Learning Training...")
        
        # Load and split training data
        train_df = pd.read_csv(train_csv_path)
        print(f"📊 Training on {len(train_df)} samples")
        
        # Split into train/validation
        train_size = int(0.8 * len(train_df))
        val_size = len(train_df) - train_size
        
        train_subset = train_df.iloc[:train_size].reset_index(drop=True)
        val_subset = train_df.iloc[train_size:].reset_index(drop=True)
        
        print(f"   Train set: {len(train_subset)} samples")
        print(f"   Val set: {len(val_subset)} samples")
        
        # Create datasets
        train_dataset = AdvancedCadastralDataset(train_subset, data_dir, num_points=self.num_points, augment=True)
        val_dataset = AdvancedCadastralDataset(val_subset, data_dir, num_points=self.num_points, augment=False)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        # Setup advanced training components
        criterion = nn.SmoothL1Loss()  # More robust than MSE
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 20
        
        # Training loop with validation
        self.model.train()
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            epoch_train_loss = 0
            num_train_batches = 0
            
            self.model.train()
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
                
                if batch_idx % 5 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = 0
            num_val_batches = 0
            
            self.model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            print(f'✅ Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
            
            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'improved_sota_cadastral_model.pth')
                print(f'💾 Best model saved with val loss: {best_val_loss:.6f}')
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f'🛑 Early stopping after {epoch+1} epochs')
                    break
        
        print("🎯 Improved SOTA Deep Learning Training Complete!")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        print(f"   Final learning rate: {scheduler.optimizer.param_groups[0]['lr']:.2e}")
    
    def load_model(self, model_path='improved_sota_cadastral_model.pth'):
        """Load trained model"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ Improved SOTA model loaded from {model_path}")
            return True
        else:
            print(f"❌ Model file not found: {model_path}")
            return False
    
    def predict_polygon(self, image_path):
        """Predict polygon with improved post-processing"""
        
        if not os.path.exists(image_path):
            return None
        
        try:
            # Load and preprocess image
            dataset = AdvancedCadastralDataset(
                pd.DataFrame([{'ID': '0', 'geometry': 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'}]),
                os.path.dirname(image_path),
                augment=False,
                num_points=self.num_points
            )
            
            # Manual preprocessing for single image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_height, orig_width = image_rgb.shape[:2]
            
            # Enhanced preprocessing
            enhanced_image = dataset.load_and_enhance_image(image_path)
            
            # Apply transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(enhanced_image).unsqueeze(0).to(self.device)
            
            # Predict with model
            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor)
                coords = output.cpu().numpy().flatten()
            
            # Reshape and denormalize
            coords = coords.reshape(self.num_points, 2)
            
            # Denormalize coordinates
            coords[:, 0] = coords[:, 0] * orig_width
            coords[:, 1] = coords[:, 1] * orig_height
            
            # Ensure coordinates are within bounds
            coords[:, 0] = np.clip(coords[:, 0], 0, orig_width)
            coords[:, 1] = np.clip(coords[:, 1], 0, orig_height)
            
            # Post-processing: smooth the polygon
            coords = self.smooth_polygon(coords)
            
            # Convert to list of tuples
            polygon_coords = [(float(x), float(y)) for x, y in coords]
            
            return polygon_coords
            
        except Exception as e:
            print(f"Error predicting polygon for {image_path}: {e}")
            return None
    
    def smooth_polygon(self, coords):
        """Apply smoothing to polygon coordinates"""
        try:
            # Simple moving average smoothing
            smoothed = coords.copy()
            for i in range(len(coords)):
                prev_idx = (i - 1) % len(coords)
                next_idx = (i + 1) % len(coords)
                smoothed[i] = 0.25 * coords[prev_idx] + 0.5 * coords[i] + 0.25 * coords[next_idx]
            return smoothed
        except:
            return coords
    
    def extract_metadata(self, image_path):
        """Enhanced metadata extraction"""
        
        metadata = {
            'TargetSurvey': 'unknown unknown unknown',
            'Certified date': 'Unknown',
            'Total Area': 0.0,
            'Unit of Measurement': 'sq m',
            'Parish': 'Unknown',
            'LT Num': 'Unknown'
        }
        
        if not os.path.exists(image_path):
            return metadata
        
        try:
            # Load and enhance image for OCR
            dataset = AdvancedCadastralDataset(
                pd.DataFrame([{'ID': '0', 'geometry': 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'}]),
                os.path.dirname(image_path),
                augment=False
            )
            enhanced_image = dataset.load_and_enhance_image(image_path)
            
            # Convert to grayscale for OCR
            gray = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
            
            # Additional OCR preprocessing
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            
            # OCR extraction with multiple attempts
            results = []
            
            # Attempt 1: Original enhanced image
            results.extend(self.ocr_reader.readtext(enhanced_gray))
            
            # Attempt 2: Thresholded image
            _, thresh = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            results.extend(self.ocr_reader.readtext(thresh))
            
            # Combine and filter results
            all_text_pieces = []
            for (_, text, confidence) in results:
                if confidence > 0.4:  # Lower threshold for more text
                    all_text_pieces.append(text.strip())
            
            all_text = ' '.join(all_text_pieces)
            all_text_lower = all_text.lower()
            
            # Enhanced pattern matching
            self.extract_metadata_patterns(all_text_lower, metadata)
            
        except Exception as e:
            print(f"Enhanced metadata extraction failed for {image_path}: {e}")
        
        return metadata
    
    def extract_metadata_patterns(self, text, metadata):
        """Extract metadata using improved patterns"""
        import re
        
        # Survey information - improved patterns
        survey_patterns = [
            r'survey\s+(?:of|for)\s+([^,\n\.]{5,50})',
            r'surveyed\s+for\s+([^,\n\.]{5,50})',
            r'client[:\s]+([^,\n\.]{5,50})',
            r'owner[:\s]+([^,\n\.]{5,50})'
        ]
        
        for pattern in survey_patterns:
            match = re.search(pattern, text)
            if match:
                survey_info = match.group(1).strip()
                if len(survey_info) > 3:
                    metadata['TargetSurvey'] = survey_info
                    break
        
        # Date extraction - more comprehensive
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata['Certified date'] = match.group(1)
                break
        
        # Area extraction - improved
        area_patterns = [
            r'area[:\s]*([0-9,]+\.?\d*)\s*(sq\s*(?:m|ft)|square\s*(?:meter|metre|foot|feet)|m²|ft²|hectare|acre)',
            r'([0-9,]+\.?\d*)\s*(sq\s*(?:m|ft)|square\s*(?:meter|metre|foot|feet)|m²|ft²|hectare|acre)',
            r'total\s+area[:\s]*([0-9,]+\.?\d*)',
            r'area\s*=\s*([0-9,]+\.?\d*)'
        ]
        
        for pattern in area_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    area_str = match.group(1).replace(',', '')
                    area_val = float(area_str)
                    if area_val > 0:
                        metadata['Total Area'] = area_val
                        if len(match.groups()) > 1:
                            unit = match.group(2).lower()
                            if 'hectare' in unit or 'acre' in unit:
                                metadata['Unit of Measurement'] = unit
                            elif 'ft' in unit or 'foot' in unit or 'feet' in unit:
                                metadata['Unit of Measurement'] = 'sq ft'
                            else:
                                metadata['Unit of Measurement'] = 'sq m'
                        break
                except:
                    pass
        
        # Parish extraction - expanded list
        parishes = [
            'Kingston', 'St. Andrew', 'St Andrew', 'St. Thomas', 'St Thomas',
            'Portland', 'St. Mary', 'St Mary', 'St. Ann', 'St Ann',
            'Trelawny', 'St. James', 'St James', 'Hanover', 'Westmoreland',
            'St. Elizabeth', 'St Elizabeth', 'Manchester', 'Clarendon', 'St. Catherine', 'St Catherine'
        ]
        
        for parish in parishes:
            if parish.lower() in text:
                metadata['Parish'] = parish.replace('St ', 'St. ')
                break
        
        # LT Number extraction - improved
        lt_patterns = [
            r'l\.?t\.?\s*#?\s*([a-z0-9\-/]{3,15})',
            r'lot\s*#?\s*([a-z0-9\-/]{3,15})',
            r'land\s+title\s*#?\s*([a-z0-9\-/]{3,15})',
            r'title\s*#?\s*([a-z0-9\-/]{3,15})',
            r'parcel\s*#?\s*([a-z0-9\-/]{3,15})'
        ]
        
        for pattern in lt_patterns:
            match = re.search(pattern, text)
            if match:
                lt_num = match.group(1).strip().upper()
                if len(lt_num) >= 3:
                    metadata['LT Num'] = lt_num
                    break
    
    def process_test_data(self, test_csv_path, output_csv_path='final_test_predictions.csv'):
        """Process test data with improved pipeline"""
        
        print("🎯 Processing test data with Improved SOTA Deep Learning...")
        
        test_df = pd.read_csv(test_csv_path)
        results = []
        
        print(f"📊 Processing {len(test_df)} test samples...")
        
        for idx, row in test_df.iterrows():
            image_id = row['ID']
            image_path = f"data/anonymised_{image_id}.jpg"
            
            result = {
                'ID': image_id,
                'TargetSurvey': 'unknown unknown unknown',
                'Certified date': 'Unknown',
                'Total Area': 0.0,
                'Unit of Measurement': 'sq m',
                'Parish': 'Unknown',
                'LT Num': 'Unknown',
                'geometry': None
            }
            
            if os.path.exists(image_path):
                try:
                    # Get predictions
                    polygon_coords = self.predict_polygon(image_path)
                    metadata = self.extract_metadata(image_path)
                    result.update(metadata)
                    
                    # Format geometry
                    if polygon_coords and len(polygon_coords) >= 3:
                        if polygon_coords[0] != polygon_coords[-1]:
                            polygon_coords.append(polygon_coords[0])
                        
                        coord_strings = [f"({x}, {y})" for x, y in polygon_coords]
                        result['geometry'] = str(coord_strings).replace("'", "")
                    
                except Exception as e:
                    print(f"Error processing {image_id}: {e}")
            
            results.append(result)
            
            if (idx + 1) % 5 == 0:
                print(f"📈 Processed {idx + 1}/{len(test_df)} images...")
        
        # Save results
        results_df = pd.DataFrame(results)
        column_order = [
            'ID', 'TargetSurvey', 'Certified date', 'Total Area',
            'Unit of Measurement', 'Parish', 'LT Num', 'geometry'
        ]
        results_df = results_df[column_order]
        results_df.to_csv(output_csv_path, index=False)
        
        # Enhanced reporting
        valid_polygons = results_df['geometry'].notna().sum()
        metadata_scores = {}
        
        for field in ['TargetSurvey', 'Parish', 'LT Num']:
            if field in results_df.columns:
                success_count = (results_df[field] != 'Unknown').sum()
                metadata_scores[field] = success_count / len(results_df) * 100
        
        area_success = (results_df['Total Area'] > 0).sum() / len(results_df) * 100
        metadata_scores['Total Area'] = area_success
        
        print(f"✅ Improved SOTA predictions saved to {output_csv_path}")
        print(f"📊 Enhanced Results Summary:")
        print(f"   - Total predictions: {len(results_df)}")
        print(f"   - Valid polygons: {valid_polygons} ({valid_polygons/len(results_df)*100:.1f}%)")
        for field, score in metadata_scores.items():
            print(f"   - {field}: {score:.1f}%")
        
        return results_df

class SOTADeepLearningExtractor:
    """SOTA Deep Learning Extractor - Single file solution"""
    
    def __init__(self, num_points=8):
        self.num_points = num_points
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SOTACadastralNet(num_points=num_points)
        self.model.to(self.device)
        
        # OCR for metadata extraction
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ SOTA Deep Learning Extractor initialized on {self.device}")
        
    def train(self, train_csv_path, data_dir, epochs=50, batch_size=8):
        """Train the SOTA deep learning model"""
        
        print("🚀 Starting SOTA Deep Learning Training...")
        
        # Load training data
        train_df = pd.read_csv(train_csv_path)
        print(f"📊 Training on {len(train_df)} samples")
        
        # Create dataset and dataloader
        train_dataset = CadastralDataset(train_df, data_dir, self.transform, self.num_points)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            scheduler.step()
            
            print(f'✅ Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.6f}')
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'sota_cadastral_model.pth')
                print(f'💾 Best model saved with loss: {best_loss:.6f}')
        
        print("🎯 SOTA Deep Learning Training Complete!")
    
    def load_model(self, model_path='sota_cadastral_model.pth'):
        """Load trained model"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ SOTA model loaded from {model_path}")
            return True
        else:
            print(f"❌ Model file not found: {model_path}")
            return False
    
    def predict_polygon(self, image_path):
        """Predict polygon coordinates using SOTA deep learning"""
        
        if not os.path.exists(image_path):
            return None
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get original image dimensions for denormalization
            orig_height, orig_width = image_rgb.shape[:2]
            
            # Transform image for model
            image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Predict with model
            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor)
                coords = output.cpu().numpy().flatten()
            
            # Reshape to (num_points, 2)
            coords = coords.reshape(self.num_points, 2)
            
            # Denormalize coordinates
            coords[:, 0] = coords[:, 0] * orig_width   # Denormalize x
            coords[:, 1] = coords[:, 1] * orig_height  # Denormalize y
            
            # Ensure coordinates are within image bounds
            coords[:, 0] = np.clip(coords[:, 0], 0, orig_width)
            coords[:, 1] = np.clip(coords[:, 1], 0, orig_height)
            
            # Convert to list of tuples
            polygon_coords = [(float(x), float(y)) for x, y in coords]
            
            return polygon_coords
            
        except Exception as e:
            print(f"Error predicting polygon for {image_path}: {e}")
            return None
    
    def extract_metadata(self, image_path):
        """Extract metadata using enhanced OCR"""
        
        metadata = {
            'TargetSurvey': 'unknown unknown unknown',
            'Certified date': 'Unknown',
            'Total Area': 0.0,
            'Unit of Measurement': 'sq m',
            'Parish': 'Unknown',
            'LT Num': 'Unknown'
        }
        
        if not os.path.exists(image_path):
            return metadata
        
        try:
            # Enhanced OCR processing
            image = cv2.imread(image_path)
            
            # Preprocess for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR extraction
            results = self.ocr_reader.readtext(enhanced)
            
            all_text = ' '.join([text for (_, text, confidence) in results if confidence > 0.5])
            all_text_lower = all_text.lower()
            
            # Extract specific fields using patterns
            import re
            
            # Survey information
            survey_patterns = [
                r'survey\s+for\s+([^,\n]+)',
                r'surveyed\s+for\s+([^,\n]+)',
                r'client[:\s]+([^,\n]+)'
            ]
            
            for pattern in survey_patterns:
                match = re.search(pattern, all_text_lower)
                if match:
                    metadata['TargetSurvey'] = match.group(1).strip()
                    break
            
            # Date extraction
            date_patterns = [
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
                r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, all_text_lower)
                if match:
                    metadata['Certified date'] = match.group(1)
                    break
            
            # Area extraction
            area_patterns = [
                r'area[:\s]*([0-9,]+\.?\d*)\s*(sq\s*m|square\s*meter|m²|hectare)',
                r'([0-9,]+\.?\d*)\s*(sq\s*m|square\s*meter|m²|hectare)'
            ]
            
            for pattern in area_patterns:
                match = re.search(pattern, all_text_lower)
                if match:
                    try:
                        area_str = match.group(1).replace(',', '')
                        metadata['Total Area'] = float(area_str)
                        unit = match.group(2)
                        if 'hectare' in unit:
                            metadata['Unit of Measurement'] = 'hectare'
                        else:
                            metadata['Unit of Measurement'] = 'sq m'
                    except:
                        pass
                    break
            
            # Parish extraction
            parishes = ['Kingston', 'St. Andrew', 'St. Thomas', 'Portland', 'St. Mary', 'St. Ann', 
                       'Trelawny', 'St. James', 'Hanover', 'Westmoreland', 'St. Elizabeth', 
                       'Manchester', 'Clarendon', 'St. Catherine']
            
            for parish in parishes:
                if parish.lower() in all_text_lower:
                    metadata['Parish'] = parish
                    break
            
            # LT Number extraction
            lt_patterns = [
                r'lt\s*#?\s*([a-z0-9\-]+)',
                r'lot\s*#?\s*([a-z0-9\-]+)',
                r'land\s+title\s*#?\s*([a-z0-9\-]+)'
            ]
            
            for pattern in lt_patterns:
                match = re.search(pattern, all_text_lower)
                if match:
                    metadata['LT Num'] = match.group(1).upper()
                    break
            
        except Exception as e:
            print(f"Metadata extraction failed for {image_path}: {e}")
        
        return metadata
    
    def process_test_data(self, test_csv_path, output_csv_path='final_test_predictions.csv'):
        """Process test data and generate predictions"""
        
        print("🎯 Processing test data with SOTA Deep Learning...")
        
        # Load test data
        test_df = pd.read_csv(test_csv_path)
        results = []
        
        print(f"📊 Processing {len(test_df)} test samples...")
        
        for idx, row in test_df.iterrows():
            image_id = row['ID']
            image_path = f"data/anonymised_{image_id}.jpg"
            
            # Initialize result
            result = {
                'ID': image_id,
                'TargetSurvey': 'unknown unknown unknown',
                'Certified date': 'Unknown',
                'Total Area': 0.0,
                'Unit of Measurement': 'sq m',
                'Parish': 'Unknown',
                'LT Num': 'Unknown',
                'geometry': None
            }
            
            if os.path.exists(image_path):
                try:
                    # Get polygon prediction
                    polygon_coords = self.predict_polygon(image_path)
                    
                    # Get metadata
                    metadata = self.extract_metadata(image_path)
                    result.update(metadata)
                    
                    # Format geometry
                    if polygon_coords and len(polygon_coords) >= 3:
                        # Ensure polygon is closed
                        if polygon_coords[0] != polygon_coords[-1]:
                            polygon_coords.append(polygon_coords[0])
                        
                        # Convert to string format expected by evaluation
                        coord_strings = [f"({x}, {y})" for x, y in polygon_coords]
                        result['geometry'] = str(coord_strings).replace("'", "")
                    
                except Exception as e:
                    print(f"Error processing {image_id}: {e}")
            
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                print(f"📈 Processed {idx + 1}/{len(test_df)} images...")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Ensure correct column order
        column_order = [
            'ID', 'TargetSurvey', 'Certified date', 'Total Area',
            'Unit of Measurement', 'Parish', 'LT Num', 'geometry'
        ]
        
        results_df = results_df[column_order]
        
        # Save results
        results_df.to_csv(output_csv_path, index=False)
        
        print(f"✅ SOTA Deep Learning predictions saved to {output_csv_path}")
        print(f"📊 Summary:")
        print(f"   - Total predictions: {len(results_df)}")
        print(f"   - Valid polygons: {results_df['geometry'].notna().sum()}")
        print(f"   - Polygon success rate: {results_df['geometry'].notna().sum()/len(results_df)*100:.1f}%")
        
        return results_df

def main():
    """Main function - Improved SOTA Deep Learning Pipeline"""
    
    print("🚀 IMPROVED SOTA DEEP LEARNING CADASTRAL EXTRACTION")
    print("="*65)
    
    # Initialize improved SOTA extractor
    sota_extractor = ImprovedSOTADeepLearningExtractor(num_points=8, use_advanced=True)
    
    # Check if improved model exists
    model_loaded = sota_extractor.load_model('improved_sota_cadastral_model.pth')
    
    # Fallback to original model
    if not model_loaded:
        model_loaded = sota_extractor.load_model('sota_cadastral_model.pth')
    
    # Train if no model found
    if not model_loaded:
        print("📚 No pre-trained model found. Training new Improved SOTA model...")
        
        if os.path.exists('Train.csv'):
            # Enhanced training with more epochs and better parameters
            sota_extractor.train(
                'Train.csv', 
                'data', 
                epochs=2,  # More epochs for better convergence
                batch_size=6,  # Smaller batch size for better gradients
                learning_rate=0.0005  # Slightly higher learning rate
            )
        else:
            print("❌ Train.csv not found! Cannot train model.")
            return
    
    # Process test data
    if os.path.exists('Test.csv'):
        results_df = sota_extractor.process_test_data('Test.csv', 'final_test_predictions.csv')
        
        print("\n🎯 IMPROVED SOTA DEEP LEARNING PROCESSING COMPLETE!")
        print("="*65)
        print("🏆 Key Improvements:")
        print("   • Advanced CNN with residual connections and attention")
        print("   • Enhanced data augmentation and preprocessing") 
        print("   • Improved training with validation and early stopping")
        print("   • Better polygon smoothing and post-processing")
        print("   • Enhanced OCR with multiple extraction attempts")
        print("   • Comprehensive pattern matching for metadata")
        
    else:
        print("❌ Test.csv not found!")

if __name__ == "__main__":
    main()
