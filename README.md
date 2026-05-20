# Outfit Detection — YOLOv8 + In-Browser Demo

**Live:** https://outfit-rieltzx.vercel.app/

I built this to learn how to train YOLO on a custom fashion dataset, then deploy it as an in-browser app with no backend. The model runs locally in your browser — webcam frames never leave your device.

---

## What it detects

35 wearable classes pulled from OpenImages V7 and Fashionpedia. Grouped roughly like this:

| Category | Classes |
|----------|---------|
| Tops | shirt, t-shirt, tank-top, sweater, cardigan, vest |
| Outerwear | jacket, coat |
| Bottoms | pants, jeans, shorts, skirt |
| One-piece | dress, jumpsuit |
| Legwear | tights, sock |
| Footwear | shoe, sandal, boot, high-heels |
| Headwear | hat, helmet, hair-accessory |
| Eyewear | glasses, sunglasses |
| Neckwear | tie, scarf |
| Jewelry | necklace, earrings, watch, bracelet |
| Accessories | belt, glove, bag, mask |

---

## Repo layout

```
outfit-detection/
├── app/                         # React + TypeScript frontend (Vercel)
│   ├── src/                     # inference worker, class labels, UI
│   ├── public/
│   │   └── models/              # ONNX model files (served statically)
│   │       ├── outfit-v4-35cls-yolov8s.onnx
│   │       └── shoe-specialist-v1.onnx
│   ├── vercel.json              # COOP/COEP headers for SharedArrayBuffer
│   └── package.json
└── training/                    # Python training pipeline
    ├── trainoutfit.py           # main training script
    ├── train.py                 # quick test with COCO128
    ├── detect_image.py
    ├── detect_webcam.py
    ├── data.yaml                # 35-class dataset config
    ├── yolov8n.pt               # YOLOv8 nano base weights
    ├── yolo11n.pt               # YOLO11 nano base weights
    └── requirements.txt
```

---

## Running the web app locally

```bash
cd app
npm install
npm run dev
```

Opens at `http://localhost:5173`. Works in Chrome, Firefox, any browser with WASM + SharedArrayBuffer support.

`app/` holds the React + TypeScript frontend. All inference runs client-side via ONNX Runtime Web — there's a Web Worker handling the model so the UI doesn't block. The ONNX files sit in `app/public/models/` and get served statically.

> **Don't remove `vercel.json`** — it sets the `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers that SharedArrayBuffer needs. Without them, multi-threaded WASM falls back to single-threaded and gets slower.

---

## Training your own model

```bash
cd training
pip install -r requirements.txt
python trainoutfit.py
```

Defaults: `yolov8n.pt` base, 50 epochs, 640px. Swap to `yolo11n.pt` or crank up epochs if you have a decent GPU. Quick sanity check with COCO128 first:

```bash
python train.py
```

Output lands in `training/runs/detect/trainX/weights/best.pt`.

---

## Stack

**Frontend:** React · TypeScript · Vite · ONNX Runtime Web · WebAssembly

**Training:** Python · Ultralytics YOLOv8 · OpenCV · Matplotlib
