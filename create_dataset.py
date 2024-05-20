import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

home = './data'
sequence_length = 20

data = []
labels = []

for sign_name in os.listdir(home):
    for seq_dir in os.listdir(os.path.join(home, sign_name)):
        sequence_data = []
        for img_path in sorted(os.listdir(os.path.join(home, sign_name, seq_dir))):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(home, sign_name, seq_dir, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
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

                sequence_data.append(data_aux)

        if len(sequence_data) == sequence_length:
            data.append(sequence_data)
            labels.append(sign_name)

f = open('data_sequences.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
