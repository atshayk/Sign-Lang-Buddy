from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from flask_cors import CORS  # Import the CORS class

app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)
# Load the trained model
model = load_model('asl_lstm_model.h5')

# Create a dictionary for all signs
signs = ['Yes', 'No', 'Thanks', 'Hello', 'Please', 'Goodbye', 'Sorry', 'You\'re Welcome', 'I Love You']
labels_dict = {i: signs[i] for i in range(len(signs))}

sequence = []
sequence_length = 20

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    global sequence

    try:
        # Decode the image
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        
        if np_arr.size == 0:
            print("Received empty frame data")
            return

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode the frame")

        # Process the frame with MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []

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
            sequence.append(data_aux)
            print(f"Hand landmarks data added to sequence: {data_aux}")

        if len(sequence) == sequence_length:
            print(f"Full sequence collected: {sequence}")
            prediction = model.predict(np.expand_dims(sequence, axis=0))
            predicted_sign = labels_dict[np.argmax(prediction)]
            print(f"Predicted sign: {predicted_sign}")
            sequence = []
            emit('sign', predicted_sign)
        else:
            print(f"Current sequence length: {len(sequence)}")
    except Exception as e:
        print(f'Error processing frame: {e}')

if __name__ == '__main__':
    socketio.run(app, debug=True)
