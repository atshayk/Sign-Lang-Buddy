import os
import pickle
import mediapipe.python.solutions.hands as hands_
import mediapipe.python.solutions.drawing_utils as drawing_utils
import mediapipe.python.solutions.drawing_styles as drawing_styles
import cv2
import numpy as np

mp_hands = hands_
mp_drawing = drawing_utils
mp_drawing_styles = drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

home_dir = './data'
sequence_length = 20

data = []
labels = []

for sign_name in os.listdir(home_dir):
    for seq_dir in os.listdir(os.path.join(home_dir, sign_name)):
        sequence_data = []
        for img_path in sorted(os.listdir(os.path.join(home_dir,sign_name,seq_dir))):
            data_aux = []
            X = []
            Y = []

            img = cv2.imread(os.path.join(home_dir, sign_name, seq_dir,img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks: # type: ignore
                for hand_landmarks in results.multi_hand_landmarks: # type: ignore
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        X.append(x)
                        Y.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        
                        data_aux.append(x - min(X))
                        data_aux.append(y - min(Y))
                
                sequence_data.append(data_aux)
                
        if(len(sequence_data)==sequence_length):
            data.append(sequence_data)
            labels.append(sign_name)

f = open('data_sequences.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
