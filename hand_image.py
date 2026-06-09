import cv2
import mediapipe as mp

# Load image
image = cv2.imread("hand.jpg.png")

# Convert to RGB
rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        print(
            f"Hands Detected: {len(results.multi_hand_landmarks)}"
        )

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    else:
        print("No hands detected.")

cv2.imshow(
    "Hand Detection",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()