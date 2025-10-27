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

1) Clone and enter the repository

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/waste-segregation.git
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

If you plan to use Ultralytics' YOLO training APIs directly, make sure `ultralytics` is installed (it's typically present in `requirements.txt`).

## Dataset & format

This repository uses the YOLO (Darknet) text-label format: for each image there is a corresponding .txt file with lines formatted as:

```
class_id center_x center_y width height
```

All coordinates are normalized to [0,1]. Example dataset folders are `train/`, `valid/`, and `test/` with subfolders `images/` and `labels/`.

The project originally used the Roboflow Garbage Classification dataset. See `data.yaml` for class names and paths used for training.

## Training

Two ways to train:

- Use the included training script (simple wrapper):

```bash
python train_model.py
```

- Or use Ultralytics YOLOv8 directly (example):

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or use your starting weights
model.train(data='data.yaml', epochs=50, imgsz=640)
```

During training, model artifacts and logs are saved to `runs/` (check `runs/train/` or `runs/detect/` for weights and results).

Tips:
- To resume training from a checkpoint, pass the checkpoint path as the `weights` argument in the training call.
- Monitor training with the generated `results.png` and the `runs/train/exp*/` folder.

## Inference / Demo (Streamlit)

The repo contains `app.py`, a Streamlit app that demonstrates detection on images, videos, and webcam. Run it with:

```bash
streamlit run app.py
```

App features:
- Image upload and batch processing
- Video file processing with basic stats
- Live webcam detection (if camera available)

Alternatively, you can run a simple detection script that loads a weight and runs inference on an image:

```python
from ultralytics import YOLO
model = YOLO('runs/train/your_weights.pt')
results = model('test/images/example.jpg')
results.show()
```

## Evaluation

After training, metrics (mAP, precision, recall) are available in the training output (console and `runs/train/exp*/metrics.csv`). Replace `exp*` with the actual experiment folder name.

## Project structure

Top-level layout (important files/folders):

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

## Notes & troubleshooting

- If Streamlit fails to start, ensure your virtual environment is activated and `streamlit` is installed.
- If GPU training is required, ensure CUDA drivers + `torch` with CUDA are installed and visible to Python.
- If you get label-format errors during training, check `train/labels/*.txt` for correct `class x y w h` normalized values.

## Contributing

Contributions welcome. Opening issues with reproducible details or a small PR with tests/documentation is ideal.

Suggested small tasks:
- Add example inference notebooks
- Add a Dockerfile for reproducible runs
- Add CI checks for linting or simple smoke tests

## License

This repository is released under the MIT License. See `LICENSE` if present.

## Acknowledgements

- Roboflow (dataset exports)
- Ultralytics (YOLOv8)
- Streamlit (UI)

---

If you'd like, I can also:
- Add a short CONTRIBUTING.md and issue/PR templates
- Add a minimal GitHub Actions workflow to run a fast lint/test
