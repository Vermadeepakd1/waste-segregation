## Waste Segregation — AI-powered garbage classification

A compact, reproducible repository for training and running a YOLO-based waste-segregation model. The project includes dataset config, training scripts, example weights, and a Streamlit-based demo app for quick inference.

Key features:
- YOLO-based object detection (YOLOv8 compatible)
- Streamlit demo for image/video/webcam inference (`app.py`)
- Training scripts and dataset config (`data.yaml`, `train_model.py`, `train/`)
- Example model weights and training runs in `runs/`

Contents in this README:
- Quick start (install & run)
- Dataset & format
- Train & evaluate
- Run inference (Streamlit / script)
- File layout and notes

## Quick start

Prerequisites: Python 3.8+ and Git. We recommend using a virtual environment.

# Waste Segregation — AI-powered garbage classification 🚮🤖

A compact, reproducible repository for training and running a YOLO-based waste-segregation model. The project includes dataset config, training scripts, example weights, and a Streamlit-based demo app for quick inference.

✨ Key features
- 🚀 YOLOv8-based object detection
- 🖼️ Streamlit demo for image/video/webcam inference (`app.py`)
- 🧰 Training scripts and dataset config (`data.yaml`, `train_model.py`, `train/`)
- 📦 Example model weights and training runs in `runs/`

📚 Contents in this README
- Quick start (install & run)
- Dataset & format
- Train & evaluate
- Run inference (Streamlit / script)
- File layout and notes

## Quick start — Run locally (Windows / macOS / Linux) 🚀

Prerequisites: Python 3.8+ and Git. We recommend using a virtual environment.

1) Clone and enter the repository

```bash
git clone https://github.com/Vermadeepakd1/waste-segregation.git
cd waste-segregation
```

2) Create & activate a virtual environment

Windows (PowerShell):

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3) Install dependencies

```bash
pip install -r requirements.txt
```

Note: If you plan to use Ultralytics' YOLO training APIs directly, ensure `ultralytics` is present in `requirements.txt` or installed separately.

## Dataset & label format 📦

This repository uses the YOLO (Darknet) text-label format. For each image there is a `.txt` file containing lines in the form:

```
class_id center_x center_y width height
```

All coordinates are normalized to the range [0,1]. Typical dataset layout: `train/`, `valid/`, and `test/` with `images/` and `labels/` subfolders.

The project originally used the Roboflow Garbage Classification dataset. Check `data.yaml` for class names and paths used for training.

## Training 🧠

Two common ways to train the model:

1) Use the included training wrapper:

```bash
python train_model.py
```

2) Use Ultralytics YOLOv8 API directly (example):

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or your starting weights
model.train(data='data.yaml', epochs=50, imgsz=640)
```

During training, artifacts and logs are saved to `runs/` (for example `runs/train/exp*/`). Inspect `results.png`, `metrics.csv` and saved weights inside that folder.

Tips
- 🔁 To resume training, pass the checkpoint path as the `weights` argument.
- 📈 Monitor progress using the generated plots in `runs/train/exp*/`.

## Inference / Demo (Streamlit) 🎯

A Streamlit demo is provided in `app.py` for quick testing of images, videos, and webcam streams. Run it with:

```bash
streamlit run app.py
```

App capabilities:
- 🖼️ Image upload and batch processing
- 🎞️ Video file processing with simple stats
- 📷 Live webcam detection (if a camera is available)

Programmatic inference example:

```python
from ultralytics import YOLO
model = YOLO('runs/train/your_weights.pt')
results = model('test/images/example.jpg')
results.show()
```

## Evaluation & metrics 📊

Training outputs include mAP, precision, and recall. You can find CSVs and plots in `runs/train/exp*/` (replace `exp*` with your experiment folder).

Example reported metrics (from a run):
- mAP@0.5: 54.6%
- Precision: 59.4%
- Recall: 49.4%
- Inference FPS: 435+

Top classes from that run: GLASS, METAL, BIODEGRADABLE (see `runs/` for per-class breakdown).

## Project structure 🗂️

```
.
├─ app.py               # Streamlit demo app
├─ train_model.py       # High-level training wrapper
├─ data.yaml            # YOLO data config (classes, train/val paths)
├─ requirements.txt     # Python dependencies
├─ runs/                # Training/inference outputs and weights
├─ train/               # Optional training helpers / images / labels
├─ valid/               # Validation dataset images/labels
├─ test/                # Test images and labels
├─ yolov8n.pt           # Example base weights (Ultralytics)
└─ yolo11n.pt           # Example custom weights (if present)
```

## Troubleshooting & notes 🛠️

- ⚠️ Streamlit fails to start? Ensure the virtual environment is active and `streamlit` is installed.
- 🧯 GPU training needed? Install CUDA drivers and a CUDA-enabled `torch` build.
- 🔍 Label format errors? Check `train/labels/*.txt` for normalized `class x y w h` values.

## Contributing 🤝

Contributions are welcome! Open an issue or submit a PR with a clear description and small, testable changes.

Ideas for improvements
- Add example inference notebooks
- Provide a Dockerfile for reproducible runs
- Add a lightweight CI workflow for linting and a smoke test

## License �

This repository is released under the MIT License. See `LICENSE` if present.

## Acknowledgements 🙏

- Roboflow (dataset exports)
- Ultralytics (YOLOv8)
- Streamlit (UI)

---

