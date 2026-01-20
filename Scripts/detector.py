# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 18:57:48 2023
Updated: Jan 2026 for Mac (MPS) + Robust Label Cropping + Safer Paths
@author: Karishma
"""

import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor


# ---------------------------
# Robust label-bar detector
# ---------------------------
def find_label_crop_row(
    img_rgb,
    scan_bottom_frac=0.40,   # scan bottom 40% to be safe across SEM formats
    dark_thresh=55,          # "dark" threshold in 8-bit grayscale
    min_dark_ratio=0.55,     # row considered label-like if >=55% pixels are dark
    min_label_rows_frac=0.18 # must have enough label-like rows inside scan region
):
    """
    Returns:
        crop_row (int): y index to crop at (keep img[:crop_row, :])
        label_found (bool)

    Why this is robust:
    - SEM label bars are typically a horizontal band with many dark pixels
    - Mean intensity fails because label bars contain bright text/scale lines
    """
    if img_rgb is None:
        return None, False

    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()

    h, w = gray.shape[:2]
    y0 = int(h * (1.0 - scan_bottom_frac))
    bottom = gray[y0:, :]

    # Fraction of dark pixels per row
    dark_ratio_per_row = (bottom < dark_thresh).mean(axis=1)

    # Decide if a label is present
    label_like_rows = (dark_ratio_per_row > min_dark_ratio).mean()
    label_found = label_like_rows > min_label_rows_frac
    if not label_found:
        return h, False

    # Find transition: first sustained run of label-like rows
    run = 0
    run_needed = max(12, int(0.02 * h))  # ~2% of image height, minimum 12 rows

    for i, r in enumerate(dark_ratio_per_row):
        if r > min_dark_ratio:
            run += 1
            if run >= run_needed:
                crop_row = y0 + i - run_needed + 1
                crop_row = int(np.clip(crop_row, 0, h))
                return crop_row, True
        else:
            run = 0

    # Fallback: crop at the scan start if detected but couldn't localize edge
    return y0, True


def _pick_torch_device(prefer_mps=True):
    """Choose torch device with safe fallback."""
    if prefer_mps:
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


def object_detection(
    img_x,
    img,
    path_to_image=None,
    img_size=1024,
    pred_score=0.25,
    overlap_thr=0.5,
    save=False,
    s_txt=False,
):
    """
    Robust crop (remove SEM label bar) + YOLO detection + SAM segmentation.

    IMPORTANT:
    - YOLO runs on analysis_img (cropped)
    - SAM runs on analysis_img (cropped)
    - Masks returned align with analysis_img coordinates

    Returns:
        masks3D_array: (N, Hc, Wc) uint8 masks aligned with analysis_img
        bbox: list of [x1,y1,x2,y2] boxes in analysis_img coordinates
        analysis_img: cropped image actually used for analysis
        crop_row: int (where crop happened; equals original height if no label)
        label_found: bool
    """
    from ultralytics import YOLO

    device = _pick_torch_device(prefer_mps=True)

    if img_x is None:
        return np.array([]), [], None, None, False

    h, w = img_x.shape[:2]

    # --- 1) ROBUST LABEL CROP ---
    crop_row, label_found = find_label_crop_row(img_x)

    if label_found and crop_row is not None and crop_row < h:
        print(f"Label detected. Cropping at row y={crop_row} (removing bottom bar)...")
        analysis_img = img_x[:crop_row, :]
    else:
        print("No label detected. Analyzing full image...")
        analysis_img = img_x
        crop_row = h
        label_found = False

    # --- 2) PATHS (match your folder structure) ---
    # detector.py is in scripts/, weights best12x.pt is in project root.
    base_dir = os.path.dirname(os.path.abspath(__file__))        # .../scripts
    project_root = os.path.dirname(base_dir)                     # .../ (root)

    yolo_ckpt = os.path.join(project_root, "best12x.pt")
    vit_h_ckpt = os.path.join(base_dir, "sam_vit_h_4b8939.pth")  # in scripts/
    mobile_sam_ckpt = os.path.join(base_dir, "mobile_sam.pt")    # in scripts/

    if not os.path.exists(yolo_ckpt):
        raise FileNotFoundError(
            f"YOLO weights not found at: {yolo_ckpt}\n"
            f"Expected best12x.pt in project root."
        )

    # --- 3) YOLO DETECTION on analysis_img ---
    model = YOLO(yolo_ckpt)
    results = model.predict(
        source=analysis_img,
        imgsz=img_size,
        conf=pred_score,
        iou=overlap_thr,
        device=device,   # force device
        verbose=False,
    )

    bbox = results[0].boxes.xyxy.tolist() if (results and len(results) > 0) else []

    if len(bbox) == 0:
        return np.array([]), [], analysis_img, crop_row, label_found

    # --- 4) SAM SEGMENTATION on analysis_img ---
    # Prefer vit_h if available, otherwise MobileSAM fallback
    masks = None

    if os.path.exists(vit_h_ckpt):
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=vit_h_ckpt)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(analysis_img)

        input_boxes = torch.tensor(bbox, device=predictor.device, dtype=torch.float32)
        transformed_boxes = predictor.transform.apply_boxes_torch(
            input_boxes, analysis_img.shape[:2]
        )

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    else:
        # Fallback: MobileSAM via Ultralytics
        if not os.path.exists(mobile_sam_ckpt):
            raise FileNotFoundError(
                f"Neither vit_h nor mobile_sam checkpoint found.\n"
                f"Missing: {vit_h_ckpt}\n"
                f"Missing: {mobile_sam_ckpt}"
            )

        from ultralytics import SAM
        sam_model = SAM(mobile_sam_ckpt)
        res = sam_model.predict(analysis_img, bboxes=bbox, save=False, device=device, verbose=False)

        if not res or res[0].masks is None:
            return np.array([]), bbox, analysis_img, crop_row, label_found

        masks = res[0].masks.data  # torch tensor (N,H,W)

    # --- 5) Convert masks to numpy uint8 (N,H,W), aligned with analysis_img ---
    masks_array = masks.detach().to("cpu").numpy()

    # shapes can be (N,1,H,W) or (N,H,W) or (1,H,W)
    if masks_array.ndim == 4:
        masks_array = np.squeeze(masks_array, axis=1)  # (N,H,W)
    elif masks_array.ndim == 2:
        masks_array = np.expand_dims(masks_array, axis=0)

    masks3D_array = (masks_array > 0).astype("uint8")

    return masks3D_array, bbox, analysis_img, crop_row, label_found


def segmentation(img, bbox, erode=False, dilate=False, kernel_size=(5, 5)):
    """
    Standalone segmentation using vit_h SAM.
    """
    device = _pick_torch_device(prefer_mps=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
    sam_checkpoint = os.path.join(base_dir, "sam_vit_h_4b8939.pth")

    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"vit_h checkpoint not found at: {sam_checkpoint}")

    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(img)

    input_boxes = torch.tensor(bbox, device=predictor.device, dtype=torch.float32)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    masks_array = masks.detach().to("cpu").numpy()
    if masks_array.ndim == 4:
        masks_array = np.squeeze(masks_array, axis=1)
    elif masks_array.ndim == 2:
        masks_array = np.expand_dims(masks_array, axis=0)

    masks3D_array = (masks_array > 0).astype("uint8")

    if erode or dilate:
        kernel = np.ones(kernel_size, np.uint8)
        processed = []
        for mask in masks3D_array:
            if erode:
                mask = cv2.erode(mask, kernel)
            if dilate:
                mask = cv2.dilate(mask, kernel)
            processed.append(mask)
        masks3D_array = np.asarray(processed)

    return masks3D_array, bbox


def segmentation_fast(img, bbox):
    """
    Fast segmentation using MobileSAM via Ultralytics.
    """
    device = _pick_torch_device(prefer_mps=True)

    from ultralytics import SAM
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mobile_ckpt = os.path.join(base_dir, "mobile_sam.pt")

    if not os.path.exists(mobile_ckpt):
        raise FileNotFoundError(f"mobile_sam checkpoint not found at: {mobile_ckpt}")

    model = SAM(mobile_ckpt)
    results = model.predict(img, bboxes=bbox, save=False, device=device, verbose=False)

    if not results or results[0].masks is None:
        return np.array([]), bbox

    masks = results[0].masks.data.detach().to("cpu").numpy()
    if masks.ndim == 4:
        masks = np.squeeze(masks, axis=1)
    elif masks.ndim == 2:
        masks = np.expand_dims(masks, axis=0)

    masks3D_array = (masks > 0).astype("uint8")
    return masks3D_array, bbox
