import numpy as np
from keras.models import load_model
import os
import pyaudio

# Set up the audio stream from the inbuilt microphone
chunk_size = 8000
audio_format = pyaudio.paInt16
channels = 1
rate = 16000
stream = pyaudio.PyAudio().open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)

# Load the trained model
model = load_model('sound_model.h5')

# Create a label dictionary (should match the labels used during training)
label_dict = {
    0: 'sound1',
    1: 'sound2',
    # Add more labels as needed
}

# Preprocess audio data for prediction
def preprocess_audio_data(audio_data):
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))
    audio_data = np.expand_dims(audio_data, axis=0)
    audio_data = np.expand_dims(audio_data, axis=2)
    return audio_data

# Detect and label sound category
def detect_and_label_sound():
    while True:
        audio_frames = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
        audio_data = preprocess_audio_data(audio_frames)
        predicted_label = np.argmax(model.predict(audio_data), axis=1)
        predicted_category = label_dict[predicted_label[0]]
        print(f"Detected sound category: {predicted_category}")

# Start the sound detection loop
detect_and_label_sound()
