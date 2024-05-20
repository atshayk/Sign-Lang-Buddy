import pickle
from tkinter import SE
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

data_dict = pickle.load(open('./data_sequences.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)#, stratify=labels)

model = Sequential()
model.add(LSTM(64, return_sequences=True,input_shape=(20,42)))
model.add(LSTM(64))
model.add(Dense(9,activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train,epochs=50,validation_data=(x_test,y_test))

#y_predict = model.predict(x_test)

#score = accuracy_score(y_predict, y_test)
#print('{}% of samples were classified correctly !'.format(score * 100))

model.save('asl_lstm_model.h5')
