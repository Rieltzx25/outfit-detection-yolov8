from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/best.pt')

# deteksi real-time via webcam
model.predict(source=0, show=True)
