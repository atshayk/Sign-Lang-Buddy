import os
import cv2

home = './data'
if not os.path.exists(home):
    os.makedirs(home)

#signs = ['A','B','C','D','E','F','G','H''I','J','K','L','M','N','O',
#         'P','Q','R','S','T','U','V','W','X','Y','Z','Yes','No',
#         'Thank You','Hello','Please','Goodbye','Sorry']
signs = ['A','B','C']
number_of_classes = len(signs)
dataset_size = 100
sequence_length = 20 #num of frames in each sequence

cap = cv2.VideoCapture(0)

for i in range(number_of_classes):
    sign_name = signs[i]
    sign_dir = os.path.join(home,sign_name)
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

    print(f'Collecting data for sign: {sign_name}')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Collecting for {sign_name} Ready? Press "Q" to start', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    for seq in range(dataset_size):
        sequence = []
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            sequence.append(frame)
        
        sequence_dir = os.path.join(sign_dir,f'Sequence {seq}')
        if not os.path.exists(sequence_dir):
            os.makedirs(sequence_dir)
        
        for frame_num, frame in enumerate(sequence):
            cv2.imwrite(os.path.join(sequence_dir,f'Frame {frame_num}.jpg'),frame)

cap.release()
cv2.destroyAllWindows()
