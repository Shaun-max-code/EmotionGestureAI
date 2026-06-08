import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels
emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# CHANGE THIS TO YOUR IMAGE
img_path = "test/happy/PrivateTest_10077120.jpg"

# Load image
img = image.load_img(
    img_path,
    color_mode="grayscale",
    target_size=(48, 48)
)

# Preprocess
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array, verbose=0)[0]

print("\n===== Emotion Probabilities =====\n")

for emotion, prob in zip(emotions, prediction):
    print(f"{emotion:<10}: {prob*100:.2f}%")

predicted_emotion = emotions[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("\n==============================")
print(f"Predicted Emotion : {predicted_emotion}")
print(f"Confidence        : {confidence:.2f}%")
print("==============================")