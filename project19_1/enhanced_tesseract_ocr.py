#!/usr/bin/env python3
"""
Enhanced OCR Integration for SOTA Deep Learning Pipeline

This script enhances the existing SOTA pipeline with improved OCR
specifically tuned for cadastral documents.
"""

import os
import cv2
import numpy as np
import pandas as pd
import subprocess
import re
from pathlib import Path

class EnhancedCadastralOCR:
    """Enhanced OCR specifically tuned for cadastral documents"""
    
    def __init__(self):
        self.cadastral_vocabulary = [
            'survey', 'surveyed', 'surveyor', 'land', 'title', 'lot', 'parish',
            'area', 'square', 'meter', 'metre', 'hectare', 'acre', 'boundary',
            'kingston', 'andrew', 'thomas', 'portland', 'mary', 'ann', 'trelawny',
            'james', 'hanover', 'westmoreland', 'elizabeth', 'manchester',
            'clarendon', 'catherine', 'certified', 'date', 'client', 'owner',
            'plan', 'drawing', 'sheet', 'reference', 'scale', 'north', 'bearing'
        ]
        
        # Check OCR availability
        self.ocr_method = self.detect_best_ocr()
        print(f"✅ Enhanced Cadastral OCR initialized using: {self.ocr_method}")
    
    def detect_best_ocr(self):
        """Detect the best available OCR method"""
        
        # Try Tesseract first
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return "tesseract"
        except:
            pass
        
        # Try EasyOCR as fallback
        try:
            import easyocr
            return "easyocr"
        except ImportError:
            pass
        
        return "basic"
    
    def preprocess_for_ocr(self, image_path, aggressive=True):
        """Advanced preprocessing specifically for cadastral documents"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if aggressive:
            # Aggressive preprocessing for difficult text
            
            # 1. Noise reduction with stronger parameters
            denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # 2. Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. Gamma correction for better contrast
            gamma = 1.2
            enhanced = np.power(enhanced / 255.0, 1/gamma) * 255
            enhanced = enhanced.astype(np.uint8)
            
            # 4. Unsharp masking for text sharpening
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            
            # 5. Morphological operations to clean text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
        else:
            # Basic preprocessing
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return enhanced
    
    def extract_text_tesseract(self, image_path, custom_config=None):
        """Extract text using enhanced Tesseract OCR"""
        
        try:
            # Preprocess image
            processed_image = self.preprocess_for_ocr(image_path)
            if processed_image is None:
                return ""
            
            # Save temporary file
            temp_path = "/tmp/temp_ocr_image.png"
            cv2.imwrite(temp_path, processed_image)
            
            # Custom Tesseract configuration for cadastral documents
            if custom_config is None:
                custom_config = [
                    "--oem", "3",  # Use LSTM OCR Engine Mode
                    "--psm", "6",  # Uniform block of text
                    "-c", "tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,():-/ ",
                    "-c", "load_system_dawg=false",
                    "-c", "load_freq_dawg=false"
                ]
            
            # Run Tesseract
            cmd = ["tesseract", temp_path, "stdout"] + custom_config
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return ""
                
        except Exception as e:
            print(f"Tesseract OCR error: {e}")
            return ""
    
    def extract_text_easyocr(self, image_path):
        """Extract text using EasyOCR with preprocessing"""
        
        try:
            import easyocr
            
            # Initialize reader
            reader = easyocr.Reader(['en'])
            
            # Preprocess image
            processed_image = self.preprocess_for_ocr(image_path)
            if processed_image is None:
                return ""
            
            # Extract text
            results = reader.readtext(processed_image)
            
            # Combine text with confidence filtering
            text_parts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.4 and len(text.strip()) > 1:
                    text_parts.append(text.strip())
            
            return " ".join(text_parts)
            
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def extract_text_multi_attempt(self, image_path):
        """Extract text using multiple attempts with different preprocessing"""
        
        all_text = []
        
        # Attempt 1: Basic preprocessing
        if self.ocr_method == "tesseract":
            text1 = self.extract_text_tesseract(image_path)
            if text1:
                all_text.append(text1)
        elif self.ocr_method == "easyocr":
            text1 = self.extract_text_easyocr(image_path)
            if text1:
                all_text.append(text1)
        
        # Attempt 2: Different Tesseract PSM modes
        if self.ocr_method == "tesseract":
            for psm in ["3", "4", "6", "8"]:
                config = ["--oem", "3", "--psm", psm]
                text = self.extract_text_tesseract(image_path, config)
                if text and text not in all_text:
                    all_text.append(text)
        
        # Attempt 3: Different preprocessing
        try:
            # Try with different image processing
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            temp_path = "/tmp/temp_binary.png"
            cv2.imwrite(temp_path, binary)
            
            if self.ocr_method == "tesseract":
                binary_text = self.extract_text_tesseract(temp_path)
                if binary_text and binary_text not in all_text:
                    all_text.append(binary_text)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Multi-attempt OCR error: {e}")
        
        # Combine all attempts
        combined_text = " ".join(all_text)
        return combined_text
    
    def post_process_text(self, raw_text):
        """Post-process extracted text for cadastral documents"""
        
        if not raw_text:
            return ""
        
        # Clean up common OCR errors
        corrections = {
            r'\b0\b': 'O',  # Zero to O
            r'\bI\b': '1',  # I to 1 in numeric contexts
            r'\bl\b': '1',  # l to 1 in numeric contexts
            r'rn': 'm',     # Common rn->m error
            r'vv': 'w',     # vv->w error
            r'©': 'e',      # Copyright symbol to e
        }
        
        processed_text = raw_text
        for pattern, replacement in corrections.items():
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        # Enhance cadastral-specific terms
        for term in self.cadastral_vocabulary:
            # Use fuzzy matching for common terms
            pattern = self.create_fuzzy_pattern(term)
            if re.search(pattern, processed_text, re.IGNORECASE):
                processed_text = re.sub(pattern, term, processed_text, flags=re.IGNORECASE)
        
        return processed_text.strip()
    
    def create_fuzzy_pattern(self, word):
        """Create fuzzy regex pattern for common OCR errors"""
        
        char_substitutions = {
            'o': '[o0]',
            '0': '[o0]',
            'i': '[il1]',
            'l': '[il1]',
            '1': '[il1]',
            'rn': '[rn|m]',
            'm': '[m|rn]'
        }
        
        fuzzy_word = ''
        for char in word.lower():
            if char in char_substitutions:
                fuzzy_word += char_substitutions[char]
            else:
                fuzzy_word += char
        
        return r'\b' + fuzzy_word + r'\b'
    
    def extract_enhanced_metadata(self, image_path):
        """Extract metadata using enhanced OCR techniques"""
        
        metadata = {
            'TargetSurvey': 'unknown unknown unknown',
            'Certified date': 'Unknown',
            'Total Area': 0.0,
            'Unit of Measurement': 'sq m',
            'Parish': 'Unknown',
            'LT Num': 'Unknown'
        }
        
        # Extract text with multiple attempts
        raw_text = self.extract_text_multi_attempt(image_path)
        
        # Post-process text
        processed_text = self.post_process_text(raw_text)
        
        if processed_text:
            # Apply enhanced pattern matching
            self.extract_metadata_patterns(processed_text.lower(), metadata)
        
        return metadata, processed_text
    
    def extract_metadata_patterns(self, text, metadata):
        """Enhanced pattern extraction for metadata"""
        
        # Survey information patterns
        survey_patterns = [
            r'survey\s+(?:of|for)\s+([^,\n\.]{5,60})',
            r'surveyed\s+for\s+([^,\n\.]{5,60})',
            r'client[:\s]+([^,\n\.]{5,60})',
            r'owner[:\s]+([^,\n\.]{5,60})',
            r'prepared\s+for\s+([^,\n\.]{5,60})'
        ]
        
        for pattern in survey_patterns:
            match = re.search(pattern, text)
            if match:
                survey_info = match.group(1).strip()
                # Clean up the result
                survey_info = re.sub(r'\s+', ' ', survey_info)  # Normalize spaces
                if len(survey_info) > 3 and not re.match(r'^\d+$', survey_info):
                    metadata['TargetSurvey'] = survey_info
                    break
        
        # Enhanced date patterns
        date_patterns = [
            r'(?:date[:\s]*)?(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})',
            r'(?:date[:\s]*)?(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
            r'(?:certified|surveyed|prepared)?\s*(?:on|date)?[:\s]*(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})',
            r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1).strip()
                if len(date_str) > 4:  # Reasonable date length
                    metadata['Certified date'] = date_str
                    break
        
        # Enhanced area patterns
        area_patterns = [
            r'(?:total\s+)?area[:\s]*([0-9,]+\.?\d*)\s*(sq(?:\s*m|uare\s*m)|m²|hectare|acre)',
            r'([0-9,]+\.?\d*)\s*(sq(?:\s*m|uare\s*m)|m²|hectare|acre)',
            r'area\s*=\s*([0-9,]+\.?\d*)',
            r'([0-9,]+\.?\d*)\s*(?:square\s*)?(?:meter|metre)s?'
        ]
        
        for pattern in area_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    area_str = match.group(1).replace(',', '')
                    area_val = float(area_str)
                    if 0.01 <= area_val <= 1000000:  # Reasonable range
                        metadata['Total Area'] = area_val
                        
                        # Determine unit
                        if len(match.groups()) > 1:
                            unit = match.group(2).lower()
                            if 'hectare' in unit:
                                metadata['Unit of Measurement'] = 'hectare'
                            elif 'acre' in unit:
                                metadata['Unit of Measurement'] = 'acre'
                            else:
                                metadata['Unit of Measurement'] = 'sq m'
                        break
                except:
                    pass
        
        # Enhanced parish extraction
        parishes = [
            'Kingston', 'St. Andrew', 'St Andrew', 'Saint Andrew',
            'St. Thomas', 'St Thomas', 'Saint Thomas',
            'Portland', 'St. Mary', 'St Mary', 'Saint Mary',
            'St. Ann', 'St Ann', 'Saint Ann',
            'Trelawny', 'St. James', 'St James', 'Saint James',
            'Hanover', 'Westmoreland', 'St. Elizabeth', 'St Elizabeth', 'Saint Elizabeth',
            'Manchester', 'Clarendon', 'St. Catherine', 'St Catherine', 'Saint Catherine'
        ]
        
        for parish in parishes:
            pattern = r'\b' + re.escape(parish.lower()) + r'\b'
            if re.search(pattern, text):
                # Standardize format
                standard_name = parish.replace('Saint ', 'St. ')
                metadata['Parish'] = standard_name
                break
        
        # Enhanced LT/Lot number patterns
        lt_patterns = [
            r'l\.?t\.?\s*#?\s*([a-z0-9\-/]{3,20})',
            r'lot\s*(?:no\.?|number|#)?\s*([a-z0-9\-/]{3,20})',
            r'land\s+title\s*#?\s*([a-z0-9\-/]{3,20})',
            r'title\s*(?:no\.?|number|#)?\s*([a-z0-9\-/]{3,20})',
            r'parcel\s*(?:no\.?|number|#)?\s*([a-z0-9\-/]{3,20})',
            r'plan\s*(?:no\.?|number|#)?\s*([a-z0-9\-/]{3,20})'
        ]
        
        for pattern in lt_patterns:
            match = re.search(pattern, text)
            if match:
                lt_num = match.group(1).strip().upper()
                # Validate format
                if 3 <= len(lt_num) <= 20 and re.match(r'^[A-Z0-9\-/]+$', lt_num):
                    metadata['LT Num'] = lt_num
                    break

# Integration function for SOTA pipeline
def integrate_enhanced_ocr_to_sota():
    """Create integration code for SOTA pipeline"""
    
    integration_code = '''
# Add this to your ImprovedSOTADeepLearningExtractor class

def __init__(self, num_points=8, use_advanced=True):
    # ... existing initialization code ...
    
    # Add enhanced OCR
    self.enhanced_ocr = EnhancedCadastralOCR()
    print("✅ Enhanced OCR integrated")

def extract_metadata(self, image_path):
    """Enhanced metadata extraction using fine-tuned OCR"""
    
    # Use enhanced OCR instead of basic OCR
    metadata, raw_text = self.enhanced_ocr.extract_enhanced_metadata(image_path)
    
    return metadata
'''
    
    print("🔗 Integration code for SOTA pipeline:")
    print(integration_code)
    
    # Save integration file
    with open("enhanced_ocr_integration.py", 'w') as f:
        f.write(integration_code)
    
    print("📁 Integration code saved to: enhanced_ocr_integration.py")

def main():
    """Demonstration of enhanced OCR"""
    
    print("🔧 ENHANCED TESSERACT OCR FOR CADASTRAL DOCUMENTS")
    print("="*60)
    
    # Initialize enhanced OCR
    enhanced_ocr = EnhancedCadastralOCR()
    
    # Test on available images
    if os.path.exists('data'):
        test_images = [f for f in os.listdir('data') if f.endswith('.jpg')][:3]
        
        print(f"\\n🧪 Testing enhanced OCR on {len(test_images)} images...")
        
        for image_file in test_images:
            image_path = os.path.join('data', image_file)
            print(f"\\n📄 Processing: {image_file}")
            
            try:
                # Extract metadata with enhanced OCR
                metadata, raw_text = enhanced_ocr.extract_enhanced_metadata(image_path)
                
                print("   Enhanced OCR Results:")
                for key, value in metadata.items():
                    if value not in ['Unknown', 'unknown unknown unknown', 0.0]:
                        print(f"     {key}: {value}")
                
                print(f"   Raw text length: {len(raw_text)} characters")
                
            except Exception as e:
                print(f"   Error: {e}")
    
    # Create integration code
    integrate_enhanced_ocr_to_sota()
    
    print("\\n✅ ENHANCED OCR DEMONSTRATION COMPLETE!")
    print("\\n💡 Benefits of Enhanced OCR:")
    print("   • Multi-attempt text extraction")
    print("   • Cadastral-specific preprocessing") 
    print("   • Advanced pattern matching")
    print("   • OCR error correction")
    print("   • Fuzzy matching for common terms")

if __name__ == "__main__":
    main()
