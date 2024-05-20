import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
#import imutils

# Load the trained model
model = load_model('asl_lstm_model.h5')

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

# Create a dictionary for all signs
signs = ['Yes','No','Thanks','Hello','Please','Goodbye','Sorry','You Are Welcome','I Love You']
labels_dict = {i: signs[i] for i in range(len(signs))}

# Capture video
cap = cv2.VideoCapture(0)

sequence = deque(maxlen=20)  # Store the last 20 frames

while True:
    ret, frame = cap.read()
    #frame = imutils.resize(frame, 720)
    mirror_frame = cv2.flip(frame,1)
    if not ret:
        print("Failed to capture frame from camera. Check camera index in cv2.VideoCapture().")
        break

    frame_rgb = cv2.cvtColor(mirror_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                mirror_frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

    if len(data_aux) == 42:
        sequence.append(data_aux)

    if len(sequence) == 20:
        prediction = model.predict(np.expand_dims(sequence, axis=0))
        predicted_sign = labels_dict[np.argmax(prediction)]

        cv2.putText(mirror_frame, predicted_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('frame', mirror_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
