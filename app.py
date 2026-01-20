import streamlit as st
import sys
import os
from PIL import Image
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2

# Make sure imports work when scripts are in the scripts folder
sys.path.append(os.path.join(os.getcwd(), "scripts"))

st.set_page_config(layout="wide", page_title="SEM Analysis Tool")

# -----------------------------
# Session state init
# -----------------------------
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "seg_path" not in st.session_state:
    st.session_state.seg_path = None
if "img_used" not in st.session_state:
    st.session_state.img_used = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "img_x" not in st.session_state:
    st.session_state.img_x = None
if "img" not in st.session_state:
    st.session_state.img = None

# ROI tool state
if "roi_mode" not in st.session_state:
    st.session_state.roi_mode = "NAVIGATE"  # NAVIGATE | ROI | RECT | CIRCLE
if "roi_json" not in st.session_state:
    st.session_state.roi_json = None
if "roi_scale" not in st.session_state:
    st.session_state.roi_scale = 1.0  # display_scale = display_w / orig_w

# -----------------------------
# Load scripts
# -----------------------------
try:
    from detector import object_detection, find_label_crop_row
    from data_reader import load
    from analysis import particle_analysis
    from sam_visualize import visualize_mask
    scripts_loaded = True
except ImportError as e:
    st.sidebar.error(f"Error loading scripts: {e}")
    scripts_loaded = False


# -----------------------------
# Helper: crop based on canvas json (with scaling)
# -----------------------------
def crop_from_canvas_scaled(img_rgb, json_data, scale):
    """
    img_rgb: original numpy image (H,W,3)
    json_data: st_canvas json_data
    scale: display_width / original_width

    Returns:
        cropped_img (np.ndarray), roi_info (dict or None)
    """
    if json_data is None:
        return img_rgb, None

    objs = json_data.get("objects", [])
    if not objs:
        return img_rgb, None

    obj = objs[-1]  # last drawn object
    h, w = img_rgb.shape[:2]

    def unscale(v):
        return float(v) / float(scale)

    if obj.get("type") == "rect":
        left = int(max(0, unscale(obj.get("left", 0))))
        top = int(max(0, unscale(obj.get("top", 0))))
        width = int(max(1, unscale(obj.get("width", 1))))
        height = int(max(1, unscale(obj.get("height", 1))))

        right = int(min(w, left + width))
        bottom = int(min(h, top + height))

        if right <= left or bottom <= top:
            return img_rgb, None

        crop = img_rgb[top:bottom, left:right].copy()
        return crop, {"shape": "rect", "x1": left, "y1": top, "x2": right, "y2": bottom}

    if obj.get("type") == "circle":
        # Fabric circle: left/top is bbox top-left; radius in px (display coords)
        left_d = float(obj.get("left", 0))
        top_d = float(obj.get("top", 0))
        r_d = float(obj.get("radius", 1))

        cx = unscale(left_d + r_d)
        cy = unscale(top_d + r_d)
        r = unscale(r_d)

        x1 = int(max(0, cx - r))
        y1 = int(max(0, cy - r))
        x2 = int(min(w, cx + r))
        y2 = int(min(h, cy + r))

        if x2 <= x1 or y2 <= y1:
            return img_rgb, None

        crop = img_rgb[y1:y2, x1:x2].copy()

        # Optional: a mask for the circular region inside the crop
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.circle(mask, (int(cx - x1), int(cy - y1)), int(r), 255, -1)

        # Apply mask by setting outside circle to background (optional)
        # Here we keep crop as-is; you can use mask later if you want.
        return crop, {"shape": "circle", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "mask": mask}

    return img_rgb, None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("SEM ANALYSIS TOOL")
    if scripts_loaded:
        st.success("AI Engine Ready")

    st.header("IMAGE LOADING")
    uploaded_file = st.file_uploader("IMPORT", type=["jpg", "png", "tif", "dm3", "dm4"])

    # ---- NEW: Tools buttons (above calibration, as you wanted) ----
    st.header("TOOLS")

    if st.button("Navigate", use_container_width=True):
       st.session_state.roi_mode = "NAVIGATE"
       st.session_state.roi_json = None

    if st.button("Select ROI", use_container_width=True):
       st.session_state.roi_mode = "ROI"

    if st.button("Circle Area", use_container_width=True):
       st.session_state.roi_mode = "CIRCLE"

    if st.button("Rectangle Area", use_container_width=True):
       st.session_state.roi_mode = "RECT"

    if st.button("Clear Selection", use_container_width=True):
       st.session_state.roi_json = None

       st.caption(f"Active Tool: {st.session_state.roi_mode}")

    st.header("CALIBRATION")
    pixel_val = st.number_input("Pixel [PX]", value=1024)
    actual_size = st.number_input("Actual Size", value=30.0)
    unit = st.selectbox("Unit", ["um", "nm"])

    ratio = round(actual_size / pixel_val, 6) if pixel_val else 0.0
    st.write(f"Ratio: {ratio} {unit}/px")


# -----------------------------
# Layout
# -----------------------------
col_left, col_right = st.columns([3, 1])


with col_left:
    st.subheader("IMAGE WORKSPACE")

    if uploaded_file is None:
        st.info("Please import an SEM image from the sidebar to begin.")
    else:
        # Save locally so rsciio/cv2 can load from disk
        with open("temp_image_file", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Reset cached results if new upload
        if st.session_state.uploaded_name != uploaded_file.name:
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.df_results = None
            st.session_state.seg_path = None
            st.session_state.img_used = None
            st.session_state.img_x = None
            st.session_state.img = None
            st.session_state.roi_json = None
            st.session_state.roi_mode = "NAVIGATE"

        # Read once (cached in session state)
        if scripts_loaded and st.session_state.img_x is None:
            img_x, img, auto_pixel_size = load("temp_image_file")
            st.session_state.img_x = img_x
            st.session_state.img = img

        # If loading failed
        if st.session_state.img_x is None:
            st.error("Image could not be loaded. Check your data_reader/load() function.")
        else:
            # ----- Show image / ROI canvas -----
            img_np = st.session_state.img_x
            h0, w0 = img_np.shape[:2]

            # Fit canvas to a reasonable width
            max_display_w = 950  # adjust if you want smaller/larger
            display_w = int(min(max_display_w, w0))
            scale = display_w / float(w0)
            display_h = int(h0 * scale)
            st.session_state.roi_scale = scale

            # Build a display image for the canvas background
            bg = cv2.resize(img_np, (display_w, display_h), interpolation=cv2.INTER_AREA)
            bg_pil = Image.fromarray(bg)

            use_canvas = st.session_state.roi_mode in ["ROI", "RECT", "CIRCLE"]

            if use_canvas:
                draw_mode = "rect"
                if st.session_state.roi_mode == "CIRCLE":
                    draw_mode = "circle"

                canvas_res = st_canvas(
                    fill_color="rgba(0, 255, 255, 0.20)",
                    stroke_width=2,
                    stroke_color="#00ffff",
                    background_image=bg_pil,
                    update_streamlit=True,
                    height=display_h,
                    width=display_w,
                    drawing_mode=draw_mode,
                    key="roi_canvas",
                )
                st.session_state.roi_json = canvas_res.json_data
                st.caption("Draw on the image, then click RUN ANALYSIS.")
            else:
                st.image(Image.fromarray(img_np), caption=f"Loaded: {uploaded_file.name}", use_column_width=True)

            # Show label preview 
            if scripts_loaded and st.session_state.img_x is not None:
                crop_row, label_found = find_label_crop_row(st.session_state.img_x)
                if label_found and crop_row is not None and crop_row < st.session_state.img_x.shape[0]:
                    st.caption("Label detected")
                else:
                    st.caption("Label not detected.")

            # ----- Run analysis -----
            if st.button("RUN ANALYSIS", type="primary"):
                if not scripts_loaded:
                    st.error("Cannot run: Scripts are missing or errored.")
                else:
                    with st.spinner("AI is scanning and segmenting particles..."):
                        full_img = st.session_state.img_x
                        img_meta = st.session_state.img

                        # Apply ROI crop if any tool selected + ROI exists
                        img_for_ai = full_img
                        roi_info = None

                        if st.session_state.roi_mode in ["ROI", "RECT", "CIRCLE"] and st.session_state.roi_json:
                            img_for_ai, roi_info = crop_from_canvas_scaled(
                                full_img, st.session_state.roi_json, st.session_state.roi_scale
                            )
                            if roi_info is None:
                                st.warning("ROI tool selected but no valid shape found. Running full image.")
                                img_for_ai = full_img
                            else:
                                st.info(f"Running analysis on ROI only: {roi_info['shape']}")

                        # Run YOLO+SAM on the ROI crop (or full image)
                        masks, boxes, img_used, crop_row, label_found = object_detection(
                            img_for_ai, img_meta, "temp_image_file"
                        )
                        st.session_state.img_used = img_used

                        st.image(img_used, caption="Image analyzed (ROI crop if selected)", use_column_width=True)

                        st.session_state.df_results = particle_analysis(
                            masks, boxes, ratio, "results.csv", save=True
                        )

                        visualize_mask(masks, img_used, "output_seg.png", save=True, img_size=None)
                        st.session_state.seg_path = "output_seg.png"

                    # Show results
                    if st.session_state.seg_path is not None:
                        st.image(st.session_state.seg_path, caption="AI Segmentation Result", use_column_width=True)

                    if st.session_state.df_results is not None:
                        st.dataframe(st.session_state.df_results)

                    st.success(f"Analysis Complete! Detected {len(masks) if hasattr(masks, '__len__') else 0} particles.")


with col_right:
    st.subheader("DATA STREAM")

    if st.session_state.df_results is not None and len(st.session_state.df_results) > 0:
        st.dataframe(st.session_state.df_results)
    else:
        st.write("No results yet. Run analysis to populate the table.")
