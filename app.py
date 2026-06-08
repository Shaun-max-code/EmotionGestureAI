import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("emotion_model.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

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

st.set_page_config(page_title="Emotion Recognition")

st.title("AI-Based Emotion Recognition System")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    pil_image = Image.open(uploaded_file)

    # Convert image to numpy array
    image_np = np.array(pil_image)

    # Convert grayscale image to RGB if needed
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_GRAY2RGB
        )

    st.image(
        image_np,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )

    if len(faces) == 0:

        st.error(
            "No face detected in the image."
        )

    else:

        # Select largest face
        largest_face = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )

        x, y, w, h = largest_face

        # Draw rectangle around face
        cv2.rectangle(
            image_np,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize face
        face = cv2.resize(
            face,
            (48, 48)
        )

        # Normalize
        face = face.astype(
            "float32"
        ) / 255.0

        # Reshape for CNN
        face = np.expand_dims(
            face,
            axis=-1
        )

        face = np.expand_dims(
            face,
            axis=0
        )

        # Predict emotion
        prediction = model.predict(
            face,
            verbose=0
        )[0]

        predicted_emotion = emotions[
            np.argmax(prediction)
        ]

        confidence = (
            np.max(prediction) * 100
        )

        st.subheader(
            "Detected Face"
        )

        st.image(
            image_np,
            use_container_width=True
        )

        st.success(
            f"Emotion: {predicted_emotion}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.subheader(
            "Confidence Scores"
        )

        for emotion, prob in zip(
            emotions,
            prediction
        ):
            st.write(
                f"{emotion}: {prob * 100:.2f}%"
            )