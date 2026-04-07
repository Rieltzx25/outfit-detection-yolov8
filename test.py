from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('runs/detect/train/weights/best.pt')

# prediksi pada gambar baru
results = model.predict('gambar_test.jpg')

# tampilkan hasilnya
for result in results:
    img_plot = result.plot()
    plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

