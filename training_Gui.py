import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import datetime
import pyaudio
import os
import tkinter as tk
from tkinter import messagebox

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
    for label in label_dict:
        files = os.listdir(os.path.join(label_dir, label))
        for file in files:
            audio_path = os.path.join(label_dir, label, file)
            audio_data = np.fromfile(audio_path, dtype=np.int16)
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
    # Generate the dataset
    X_train, y_train = generate_dataset()

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[ModelCheckpoint('sound_model.h5', save_best_only=True)])

# Detect and label sound category
def detect_and_label_sound():
    label = label_entry.get()
    if not label:
        messagebox.showinfo("Error", "Please enter a label.")
        return

    label_dir_path = os.path.join(label_dir, label)
    os.makedirs(label_dir_path, exist_ok=True)

    count = 0
    while count < 100:  # Limit the number of raw data collected to 100
        audio_frames = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(label_dir_path, f"{label}_{timestamp}_{count}.raw")
        audio_frames.tofile(file_path)
        count += 1
        messagebox.showinfo("Info", f"Data collected: {count}/100")

    messagebox.showinfo("Info", "Data collection complete.")

# GUI
window = tk.Tk()
window.title("Sound Classification")
window.geometry("700x700")

# Add a background image to the window
background_image = tk.PhotoImage(file="defo.png")
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Add the label entry field
label_entry = tk.Entry(window)
label_entry.place(x=10, y=10)

# Add the train button
def train_button_hover(event):
    train_button.config(bg="light green")

def train_button_leave(event):
    train_button.config(bg="white")

train_button = tk.Button(window, text="Train Model", command=train_model, bg="white")
train_button.place(x=10, y=50)
train_button.bind("<Enter>", train_button_hover)
train_button.bind("<Leave>", train_button_leave)

# Add the detect and label button
def detect_button_hover(event):
    detect_button.config(bg="light green")

def detect_button_leave(event):
    detect_button.config(bg="white")

detect_button = tk.Button(window, text="Detect and Label Sound", command=detect_and_label_sound, bg="white")
detect_button.place(x=10, y=90)
detect_button.bind("<Enter>", detect_button_hover)
detect_button.bind("<Leave>", detect_button_leave)

window.mainloop()
