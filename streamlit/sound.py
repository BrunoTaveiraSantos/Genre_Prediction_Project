import os 
import librosa
import math
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import keras_tuner as kt
import pickle
from scipy import stats
import random
import sounddevice as sd
import pygame

# Set the sample rate to 22050 samples per second
SAMPLE_RATE = 22050 #Sample rate refers to the number of samples, or measurements, taken per second to represent an audio signal.

# Set the desired duration of the audio track to 30 seconds
DURATION = 30

# Calculate the total number of samples in the audio track
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

import pygame
#play music
def play_sound(sound):
    pygame.init()
    pygame.mixer.init()

    pygame.mixer.music.load(sound)
    start_position = 45  # Starting position in seconds
    pygame.mixer.music.play(start=start_position)

    duration = 10  # Duration to play in seconds
    pygame.time.delay(int(duration * 1000))  # Delay for the specified duration

    pygame.mixer.music.stop()

def process_new_song(file_sample): 
  # dictionary to store data 
    data= { 
        "mfcc": []
        
       }
    duration = 60
    SAMPLE_RATE = 22050
    SAMPLES_PER_TRACK = SAMPLE_RATE * duration
    n_fft = 2048 
    hop_length = 512
    num_samples_per_segment = int(SAMPLES_PER_TRACK / 20)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)
    #file_path = file_sample
    #signal, sr = librosa.load(file_path, sr = SAMPLE_RATE,offset=150 ,duration=duration)
                # PROCESS segments extrating mfccs and storing data 
    for s in range(40):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment

        #store mfcc for segment if it has the expected lenght
        mfcc = librosa.feature.mfcc(y=file_sample[start_sample:finish_sample], n_mfcc=13, n_fft = 2048, hop_length=512)
        mfcc = mfcc.T
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc.tolist())
                #print("{}, segment:{}".format(file_path, s))

    inputs = np.array(data["mfcc"])
    inputs = inputs[..., np.newaxis]
    
    return inputs

import pickle
model = pickle.load(open('model.pkl','rb'))

def predict_new(model, X):
    X = X[np.newaxis, ...]
    prediction = model.predict(X) # X-> (1, 130,13,1)

    #extract index with max value
    predict_index = np.argmax(prediction, axis=1) #
    print("Predicted index:  {}".format(predict_index))
    return predict_index

def extract_numbers(data):
    numbers = []
    for arr in data:
        numbers.append(arr.tolist())
    
    mode = stats.mode(numbers, keepdims=True)
    return mode.mode[0]

def final_score(sound_file):
    genres_list = ["Blues","Classical","Country", "Disco", "Hiphop", "Jazz","Metal","Pop","Raggae","Rock"]
    inputs = process_new_song(sound_file)
    prediction = []
    #play_sound(file2)
    for i in range(6):
        pred = predict_new(model, inputs[i+1])
        prediction.append(pred[0])

        result = extract_numbers(prediction)
    return genres_list[result]

def process_recorded_song(audio):
        data= { 
        "mfcc": []
        
       }
        duration = 30
        SAMPLE_RATE = 22050
        SAMPLES_PER_TRACK = SAMPLE_RATE * duration
        n_fft = 2048 
        hop_length = 512
        num_samples_per_segment = int(SAMPLES_PER_TRACK / 10)
        expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)
        print("Recording started...")

        # Record audio
        #audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, blocking=True)
        #audio = audio.flatten()
        print("Recording finished.")
        for s in range(10):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment

                #store mfcc for segment if it has the expected lenght
                mfcc = librosa.feature.mfcc(y=audio[start_sample:finish_sample], n_mfcc=13, hop_length=512)
                mfcc = mfcc.T
                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        #print("{}, segment:{}".format(s))

        inputs = np.array(data["mfcc"])
        inputs = inputs[..., np.newaxis]
    
        return inputs

def final_score_recorded(model, inputs):
    genres_list = ["Blues","Classical","Country", "Disco", "Hiphop", "Jazz","Metal","Pop","Raggae","Rock"]
    #inputs = process_recorded_song()
    prediction = []
    for i in range(7):
        pred = predict_new(model, inputs[i])
        prediction.append(pred[0])

        result = extract_numbers(prediction)
    return genres_list[result]

# Image Generator

# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Sequential
# Bring in the layers for the neural network
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D

def build_generator(): 
    model = Sequential()
    
    # Takes in random values and reshapes it to 7x7x128
    # Beginnings of a generated image
    model.add(Dense(7*7*128, input_dim=600))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    
    #Upsampling block 1 
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Upsampling block 2 
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Convolutional block 1
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))
    
   # # Convolutional block 2
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Conv layer to get to one channel
    model.add(Conv2D(3, 4, padding='same', activation='sigmoid'))
    
    return model