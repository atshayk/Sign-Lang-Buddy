import pickle
import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import GaussianNB

# Load data
data_dict = pickle.load(open('./data_sequences.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True,stratify=labels) # type: ignore

# Define model
model = RandomForestClassifier()
# model = Sequential()
# model.add(Dense(50,input_shape=(None,50,42)))
# model.add(Dense(30))
# model.add(Dense(15))
# model.add(Dense(6, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Define callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# model_checkpoint = ModelCheckpoint('best_asl_model.h5', save_best_only=True)

# Train model
model.fit(x_train, y_train)

# Save final model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()