import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import datetime
import pyaudio
import os

# Set up the audio stream from the inbuilt microphone
chunk_size = 8000
audio_format = pyaudio.paInt16
channels = 1
rate = 16000
stream = pyaudio.PyAudio().open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)

# Create a directory to store the labeled audio files
label_dir = 'labeled_audio'
os.makedirs(label_dir, exist_ok=True)

# Create a label dictionary
label_dict = {}

# Generate a dataset from the labeled audio files
def generate_dataset():
    X = []
    y = []
    if not label_dict:  # Check if label_dict is empty
        return X, y
    for label in label_dict:
        files = os.listdir(os.path.join(label_dir, label))
        for file in files:
            audio_path = os.path.join(label_dir, label, file)
            audio_data = np.fromfile(audio_path, dtype=np.int16)
            audio_data = preprocess_audio_data(audio_data)  # Preprocess the audio data
            X.append(audio_data)
            y.append(label_dict[label])
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y)
    return X, y

# Define the model architecture
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(8000, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(label_dict), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Function to preprocess audio data for training
def preprocess_audio_data(audio_data):
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))
    audio_data = np.expand_dims(audio_data, axis=0)
    audio_data = np.expand_dims(audio_data, axis=2)
    return audio_data

# Train the model
def train_model():
    while True:
        # Generate the dataset
        X_train, y_train = generate_dataset()

        # Check if dataset is empty
        if not X_train or not y_train:
            print("No labeled audio files found. Please collect labeled audio files first.")
            break

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[ModelCheckpoint('sound_model.h5', save_best_only=True)])

        # Save the model at each iteration
        model.save('sound_model.h5')

# Detect and label sound category
def detect_and_label_sound():
    label = input("Enter the label for the sound category: ")
    label_dir_path = os.path.join(label_dir, label)
    os.makedirs(label_dir_path, exist_ok=True)

    count = 0
    while True:
        audio_frames = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(label_dir_path, f"{label}_{timestamp}_{count}.raw")
        audio_frames.tofile(file_path)
        count += 1
        print(f"Audio saved: {file_path}")

# Start the training or detection loop based on user input
if not os.listdir(label_dir):
    print("No labeled audio files found. Starting in detection mode.")
    detect_and_label_sound()
else:
    while True:
        choice = input("Enter 't' to train the model, 'd' to detect and label sounds, or 'q' to quit: ")
        if choice == 't':
            train_model()
        elif choice == 'd':
            detect_and_label_sound()
        elif choice == 'q':
            break
        else:
            print("Invalid choice. Please try again.")
