from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import cv2

# --- Renk listesini tanımla ---
colors = {
    "red": [(255, 0, 0), (200, 0, 0), (180, 30, 30)],
    "green": [(0, 255, 0), (50, 200, 50), (30, 180, 30)],
    "blue": [(0, 0, 255), (30, 30, 200), (50, 80, 220)],
    "yellow": [(255, 255, 0), (240, 230, 50), (200, 200, 40)],
    "orange": [(255, 165, 0), (255, 140, 0), (255, 100, 10)],
    "purple": [(128, 0, 128), (150, 60, 180), (120, 30, 150)],
    "pink": [(255, 192, 203), (255, 105, 180), (255, 20, 147)],
    "brown": [(139, 69, 19), (160, 82, 45), (210, 105, 30)],
    "gray": [(128, 128, 128), (169, 169, 169), (192, 192, 192)],
    "black": [(0, 0, 0), (20, 20, 20), (40, 40, 40)],
    "white": [(255, 255, 255), (245, 245, 245), (230, 230, 230)],
    "cyan": [(0, 255, 255), (0, 200, 200), (60, 230, 230)],
    "magenta": [(255, 0, 255), (200, 0, 200), (230, 30, 230)],
    "lime": [(191, 255, 0), (173, 255, 47), (202, 255, 112)],
    "navy": [(0, 0, 128), (0, 0, 139), (25, 25, 112)],
}

def rgb_to_hue(rgb):
    bgr = np.uint8([[rgb[::-1]]])  # RGB → BGR
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0][0]  # Hue bileşeni

# Eğitim verisi oluştur
X = []
y = []

for label, rgb_list in colors.items():
    for rgb in rgb_list:
        hue = rgb_to_hue(rgb)
        X.append([hue])
        y.append(label)

X = np.array(X)
y = np.array(y)

# KNN Modeli
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Kaydet
with open("color_classifier.pkl", "wb") as f:
    pickle.dump(knn, f)

print("✅ Hue tabanlı model başarıyla eğitildi ve kaydedildi.")

