import cv2
import numpy as np
from PIL import Image
import pyttsx3
import pickle
import os

# --- Kamerayı aç ve parlaklık ayarlarını yap ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera açılamadı.")
    exit()

# Parlaklığı artırmayı dene (0.0 - 1.0 arasında deneyebilirsin)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)
cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
cap.set(cv2.CAP_PROP_SATURATION, 0.7)

print("📷 Kameradan görüntü alınıyor...")
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Görüntü alınamadı.")
    exit()

# --- Görüntüyü yapay parlaklıkla iyileştir ---
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

v = cv2.add(v, 60)
v = np.clip(v, 0, 255)

s = cv2.add(s, 50)
s = np.clip(s, 0, 255)

final_hsv = cv2.merge((h, s, v))
frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# --- Fotoğrafı kaydet ---
save_path = "captured.jpg"
success = cv2.imwrite(save_path, frame)
if success:
    print(f"📷 Fotoğraf başarıyla kaydedildi: {os.path.abspath(save_path)}")
else:
    print("❌ Fotoğraf kaydedilirken hata oluştu.")

# --- Renk analizi ---
img = Image.open(save_path)
width, height = img.size
x, y = width // 4, height // 4
w, h = width // 2, height // 2
cropped = img.crop((x, y, x + w, y + h))
cropped_np = np.array(cropped)
avg_color = tuple(np.mean(cropped_np.reshape(-1, 3), axis=0).astype(int))

print("🎨 Ortalama RGB:", avg_color)


# --- Modeli yükle ---
try:
    with open("color_classifier.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("❌ Model dosyası bulunamadı. Lütfen önce 'train_color_model.py' ile modeli eğit.")
    exit()

# --- Tahmin ve Sesli Çıktı ---
def rgb_to_hue(rgb):
    bgr = np.uint8([[rgb[::-1]]])  # RGB → BGR
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0][0]

hue_value = rgb_to_hue(avg_color)
predicted_color = model.predict([[hue_value]])[0]

print("🎯 Tahmin edilen renk:", predicted_color)

engine = pyttsx3.init()
engine.say(f"The color is {predicted_color}")
engine.runAndWait()
