# step 1: collect images for each sign
# turns on the camera and records every frame for each sign
# logs every sign in a separate folder

import os
import cv2

# Create a directory to store the images if it doesn't exist
home = './data'
if not os.path.exists(home):
    os.makedirs(home)

#signs = ['J','Z']
#signs = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
signs = ['Yes','No','Thanks','Hello','You_Are_Welcome','I_Love_You']

number_of_classes = len(signs) #len = 6
dataset_size = 50
#sequence_length = 35 #num of frames in each sequence

cap = cv2.VideoCapture(0)

for i in range(number_of_classes):
    sign_name = signs[i]
    sign_dir = os.path.join(home,sign_name)
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

    print(f'Collecting data for sign: {sign_name}') #debugging

    while True:
        ret, frame = cap.read()
        mirror_frame = cv2.flip(frame,1)
        cv2.putText(mirror_frame, f'Collecting for {sign_name}. Press "Q" to start', (0, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', mirror_frame)
        if cv2.waitKey(25) == ord('q'):
            break

    for num in range(dataset_size):
        ret, frame = cap.read()
        mirror_frame = cv2.flip(frame,1)
        cv2.imshow('frame', mirror_frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(sign_dir,f'{signs[i]} Image {num}.jpg'),mirror_frame)

cap.release()
cv2.destroyAllWindows()