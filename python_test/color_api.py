from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import pickle
import os

app = Flask(__name__)

# Modeli yükle
with open("color_classifier.pkl", "rb") as f:
    model = pickle.load(f)

def rgb_to_hue(rgb):
    bgr = np.uint8([[rgb[::-1]]])  # RGB → BGR
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0][0]

@app.route("/predict-color", methods=["POST"])
def predict_color():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "Resim dosyası gönderilmedi."}), 400

    img = Image.open(file)
    width, height = img.size
    x, y = width // 4, height // 4
    w, h = width // 2, height // 2
    cropped = img.crop((x, y, x + w, y + h))
    cropped_np = np.array(cropped)
    avg_color = tuple(np.mean(cropped_np.reshape(-1, 3), axis=0).astype(int))
    
    hue_value = rgb_to_hue(avg_color)
    predicted_color = model.predict([[hue_value]])[0]

    return jsonify({
        "predicted_color": predicted_color,
        "average_rgb": avg_color
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

 