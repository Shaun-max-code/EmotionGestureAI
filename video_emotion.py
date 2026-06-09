import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")

emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

video_path = "sample.mp4"

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % 30 != 0:
        continue

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        4
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (48, 48)
        )

        face = face / 255.0

        face = np.expand_dims(
            face,
            axis=-1
        )

        face = np.expand_dims(
            face,
            axis=0
        )

        prediction = model.predict(
            face,
            verbose=0
        )[0]

        emotion = emotions[
            np.argmax(prediction)
        ]

        print(
            f"Frame {frame_count}: {emotion}"
        )

cap.release()

print("Analysis Complete")