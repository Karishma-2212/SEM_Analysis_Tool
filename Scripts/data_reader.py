# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:12:23 2024

@author:  Karishma
"""

import cv2
from rsciio.digitalmicrograph import file_reader as dm_read
from rsciio.emd import file_reader as emd_read
from rsciio.tia import file_reader as tia_read
from rsciio.tiff import file_reader as tiff_read
import os
import numpy as np

def load(path):
    _, extension = os.path.splitext(path)
    img = None
    pixel_size = 1.0 # Default fallback

    # 1. HANDLE TIF/TIFF (including FEI/Thermo Fisher metadata)
    if extension.lower() in ('.tif', '.tiff'):
        try:
            from rsciio.tiff import file_reader as tiff_read
            tiff = tiff_read(path)
            # Ensure we get the data array out of the list
            if isinstance(tiff, list) and len(tiff) > 0:
                img = tiff[0]["data"]
                try:
                    # Attempt to extract automated pixel size
                    pixel_size = round(float(tiff[0]['axes'][0]['scale'])*1e6, 3)
                except:
                    pixel_size = 1.0
        except Exception:
            img = None # Force fallback to OpenCV below

    # 2. HANDLE DIGITAL MICROGRAPH (dm3/dm4)
    elif extension.lower() in ('.dm3', '.dm4'):
        try:
            from rsciio.digitalmicrograph import file_reader as dm_read
            dm = dm_read(path)
            img = dm[0]["data"]
            try:
                pixel_size = round(dm[0]['original_metadata']["ImageList"]["TagGroup0"]["ImageData"]["Calibrations"]["Dimension"]["TagGroup0"]["Scale"], 4)
            except:
                pixel_size = 1.0
        except Exception:
            img = None

    # 3. EMERGENCY FALLBACK (If specialized readers failed)
    if img is None:
        # This will read almost any image format as raw pixels
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # 4. FINAL STANDARDIZATION (Convert to 8-bit RGB for the AI)
    if img is not None:
        # Convert to float for normalization to avoid clipping
        img_float = img.astype(np.float32)
        
        # Normalize to 0-255 (Darkest pixel = 0, Brightest = 255)
        img_norm = (img_float - np.min(img_float)) / (np.max(img_float) - np.min(img_float) + 1e-8) * 255
        img_8bit = img_norm.astype(np.uint8)
        
        # Ensure it is 3-channel RGB (SAM AI requires this)
        if len(img_8bit.shape) == 2:
            img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
        else:
            # If it's BGR from OpenCV, convert to RGB
            img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB)
            
        img_x = img_rgb.copy()
        # Returns: (Processed image for AI, Image for display, Pixel scale)
        return img_x, img_rgb, pixel_size
    else:
        # If we reach here, the file is likely not an image or is corrupted
        raise ValueError(f"CRITICAL ERROR: Could not load image data from {path}")