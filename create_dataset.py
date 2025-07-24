# step 2: go through the dataset and create a pickle file with the data and labels

import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

home = './data'
#sequence_length = 35

data = []
labels = []

for sign_name in os.listdir(home):
        for seq_name in os.listdir(os.path.join(home,sign_name)):
            for img_path in os.listdir(os.path.join(home, sign_name, seq_name)):
                data_aux = []
                x_ = []
                y_ = []

                img = cv2.imread(os.path.join(home, sign_name,seq_name,img_path))
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

            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(sign_name)

f = open('data_sequences.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
