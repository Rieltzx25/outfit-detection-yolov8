from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    results = model.train(data='coco128.yaml', epochs=20, imgsz=640)
