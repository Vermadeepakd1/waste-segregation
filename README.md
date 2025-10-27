# AI-Driven Waste Segregation System

## Overview
This project implements a real-time waste detection and classification system using the YOLOv8 deep learning model. It is trained on the Roboflow Garbage Classification dataset to classify waste into multiple categories such as glass, metal, biodegradable, cardboard, plastic, and paper.

Developed as part of the Inter IIIT Hackathon 2025 by IIITDM Kurnool.

## Features
- Real-time detection on images, videos, and live webcam feed
- Custom-trained YOLOv8 model with high accuracy on garbage classification
- Interactive Streamlit frontend with confidence tuning and analytics dashboard
- Video processing with detection statistics and sample frames
- Clean, user-friendly interface

## Demo Video
(OPTIONAL: Add link or path to demo video showing real-time detection)

## Dataset
- Source: Roboflow Garbage Classification
- Link: https://universe.roboflow.com/material-identification/garbage-classification-3
- Classes: GLASS, METAL, BIODEGRADABLE, CARDBOARD, PLASTIC, PAPER

## Installation

1. Clone the repository:
git clone https://github.com/YOUR-GITHUB-USERNAME/waste-segregation.git
cd waste-segregation

text

2. Create and activate a virtual environment:
python -m venv venv

Windows
venv\Scripts\activate

Mac/Linux
source venv/bin/activate

text

3. Install required libraries:
pip install -r requirements.txt

text

## Usage

### Train the Model
python train_model.py

text

### Test on Images
streamlit run app.py

text

- Select "Image Detection" in the sidebar and upload images.

### Test on Videos
- Select "Video Detection" and upload video files for detection.

### Live Webcam Detection
- Select "Live Webcam" mode and start/stop webcam detection.

## Repository Structure

waste-segregation/
├── app.py # Streamlit frontend code
├── train_model.py # YOLOv8 training script
├── requirements.txt # Project dependencies
├── README.md # This file
├── runs/ # Output of training and inference
│ ├── detect/ # Trained weights, inference outputs
│ └── ...
├── data.yaml # Dataset config file for training
└── demo/ # Demo videos and screenshots (optional)

text

## Model Performance

| Metric         | Score  |
| -------------- | ------ |
| mAP@0.5        | 54.6%  |
| Precision      | 59.4%  |
| Recall         | 49.4%  |
| Inference FPS  | 435+   |

Best-performing classes:
- GLASS (80.4% mAP50)
- METAL (70.3% mAP50)
- BIODEGRADABLE (63.2% mAP50)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
MIT License

## Acknowledgements
- Roboflow for dataset
- Ultralytics for YOLOv8 library
- Streamlit for UI