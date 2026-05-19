# Outfit Detection — YOLOv8 & YOLO11

Detects clothing items from images or webcam using YOLOv8 / YOLO11. Trained on a custom dataset with 7 clothing categories pulled from Roboflow.

I made this to practice training YOLO on a domain-specific dataset (fashion/clothes). Works decently for detecting individual garments — not as reliable when clothes overlap or image quality is low.

---

## What it detects

7 classes: **dress · pants · shirt · short · skirt · sweater · tshirt**

---

## Setup

Python 3.8+ is required.

```bash
pip install -r requirements.txt
```

No other setup needed. The pretrained weights (`yolov8n.pt` / `yolo11n.pt`) are included in the repo.

---

## Scripts

| File | What it does |
|------|-------------|
| `train.py` | Quick training run using COCO128 (just for testing the pipeline) |
| `trainoutfit.py` | Actual training on the outfit dataset — this is the main one |
| `detect_image.py` | Run detection on a saved image |
| `detect_webcam.py` | Live detection through webcam |

Run any of them directly:

```bash
python trainoutfit.py
python detect_image.py
python detect_webcam.py
```

---

## Training config

Default settings in `trainoutfit.py`:
- Model: `yolov8n.pt`
- Epochs: 50
- Image size: 640

Feel free to swap the model to `yolo11n.pt` or increase epochs if you have the hardware for it.

---

## Dataset

Source: [Clothes Detection — Roboflow Universe](https://universe.roboflow.com/clothes-5a2kp/clothes-detection-i80lw/dataset/2)  
License: CC BY 4.0  
7 classes, split into `train/`, `valid/`, `test/` folders.

The `data.yaml` points to these folders — if you move them, update the paths there.

---

## Folder structure

```
outfit-detection-yolov8/
├── train.py               # pipeline test with coco128
├── trainoutfit.py         # main training script
├── detect_image.py        # image detection
├── detect_webcam.py       # webcam / real-time detection
├── data.yaml              # dataset config
├── yolov8n.pt             # YOLOv8 nano weights
├── yolo11n.pt             # YOLO11 nano weights
├── train/                 # training images + labels
├── valid/                 # validation images + labels
└── test/                  # test images
```

---

## Requirements

```
ultralytics
opencv-python
matplotlib
```

Or just run `pip install -r requirements.txt`.

---

Trained results are saved under `runs/detect/`. The best weights end up at `runs/detect/trainX/weights/best.pt` — that's what `detect_image.py` and `detect_webcam.py` load.
