# SEM_Analysis_Tool
A simple, interactive SEM image analysis tool built with Streamlit, YOLO, and SAM (Segment Anything) to detect, segment, and analyze particles from SEM images.
This tool is designed for materials science / microscopy users.

# What This App Does
Upload SEM images (.tif, .png, .jpg, .dm3, .dm4)

Automatically:
Detect particles using YOLO
Segment particles using SAM

Compute particle statistics:
Diameter
Area

Display:
Segmentation overlay
Results table (Data Stream)
Supports Mac (Apple Silicon / MPS)

⚠️ Label-bar cropping is currently disabled by design
(recommended to upload images without SEM metadata bars)

🧰 Requirements
System:  macOS (Intel or Apple Silicon)

Python 3.9 – 3.10 recommended
Python packages

Key dependencies:
streamlit
torch
ultralytics
segment-anything
opencv-python
pillow
streamlit-drawable-canvas

# Installation:
Create virtual environment:
python3 -m venv venv
source venv/bin/activate

Install compatible versions:
pip install streamlit==1.31.1

pip install streamlit-drawable-canvas==0.9.3

pip install pillow==10.2.0

pip install -r requirements.txt

# Running the App:
streamlit run app.py

