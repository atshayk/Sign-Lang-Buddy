import pickle
import cv2
import mediapipe.python.solutions.hands as hands_
import mediapipe.python.solutions.drawing_utils as drawing_utils
import mediapipe.python.solutions.drawing_styles as drawing_styles
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS as connect
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)  # Change to the correct camera index if needed

mp_hands = hands_
mp_drawing = drawing_utils
mp_drawing_styles = drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7,min_tracking_confidence=0.7)

labels_dict = {i: chr(65 + i) for i in range(26)}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from camera. Check camera index in cv2.VideoCapture().")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:  # type: ignore
        hand_landmarks = results.multi_hand_landmarks[0]  # type: ignore
        mp_drawing.draw_landmarks(
            frame,  # image to draw
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

        if len(data_aux) == 42:  # Ensure data_aux is not empty
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)
        else:
            print(f"Detected landmarks are inconsistent: expected 42, got {len(data_aux)}")

    cv2.putText(frame, "Press Q to close", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 124, 0), 3,cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
