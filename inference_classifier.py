# step 4: loading pre-existing model and the model predicts in real time what sign is being shown

import cv2
import numpy as np
import mediapipe as mp
import imutils
import pickle
import time

# Load the trained model
model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.6)

# Create a dictionary for all signs
signs = ['Hello','I_Love_You','No','Thanks','Yes','You_Are_Welcome']
labels_dict = {i: signs[i] for i in range(len(signs))}

# Capture video
cap = cv2.VideoCapture(0)

# sequence = deque(maxlen=35)  # Store the last 35 frames

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    mirror = cv2.flip(frame,1)
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(mirror, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                mirror,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

    if len(data_aux)==42:
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[np.argmax(prediction)]
        print(f"Prediction: {prediction}, Predicted Index: {np.argmax(prediction)}, Predicted Character: {predicted_character}")

        cv2.rectangle(mirror, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(mirror, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)

    cv2.imshow('frame', mirror)
    if cv2.waitKey(25) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
