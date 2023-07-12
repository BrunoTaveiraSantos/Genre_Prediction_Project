import streamlit as st
from sound import final_score, predict_new,extract_numbers,play_sound,process_new_song, process_recorded_song, final_score_recorded
import librosa
import pickle
import tempfile
import os
import sounddevice as sd
import time
import threading
from pydub import AudioSegment
from pydub.playback import play
from sound import build_generator
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import pygame

model = pickle.load(open('model.pkl','rb'))



#config of webapp
st.set_page_config( page_title="GenreGenius",
                   page_icon ="🎵",
                   layout="centered"
                   
                   )
#Lets start to build our web app
page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://tm.ibxk.com.br/2016/08/10/10174548760645.jpg?ims=1200x675");
    background-size: cover;
    }

    [data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
    }

    [data-testid="stToolbar"] {
    right: 2rem;
    }
    </style>
    """

st.markdown(page_bg_img, unsafe_allow_html=True)

#title, headr, subheader

st.title("Welcome to GenreGenius")
image = Image.open("DG75FRTP9mD8AI8BDMma--1--6i9cy.jpg")
    
# Display the image in Streamlit app
st.header("Hi! My name is Melody Maestro")
st.image(image)
st.header("Melody Maestro will listen to your song and predict the Genre of the music")
st.write("You can submit your song via a file or by recording with your microphone")

st.write("---")


#Line, sapces and columns
st.header("Upload your Song")
col1, col2 = st.columns(2)
with col1:
    temp_dir = tempfile.TemporaryDirectory()
    uploaded_file = st.file_uploader("Upload", type=["mp3", "wav","au"])
    if st.button("Start playing"):
        
        if uploaded_file is not None:
        
    # Save the uploaded file to the temporary directory
        
            file_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            album_name = uploaded_file.name
    # Load the audio data using librosa
            audio, sample_rate = librosa.load(file_path,sr = 22050,offset=150 ,duration=60)
            st.audio(uploaded_file, format='audio/wav',start_time=45)
    # Process the audio data as needed
            result = final_score(audio)
    # Clean up the temporary file
        #play_sound(uploaded_file)
            #time.sleep(45)
            os.remove(file_path)
            time.sleep(10)
            st.header("The genre of the song is:")
            st.header(result)
    # Clean up the temporary directory
            temp_dir.cleanup()
        else:
            st.write("Upload a File.")

st.write("---")
st.header("Record your Song")
if st.button("Start Recording"):
    
    st.write("Recording...")
    audio = sd.rec(int(30 * 22050), samplerate=22050, channels=1, blocking=True)
    audio = audio.flatten()
    st.audio(audio, format='audio/wav',start_time=45, sample_rate=22050)
    audio = process_recorded_song(audio)
    result = final_score_recorded(model,audio)
    st.write("Recording over")
    st.header("The genre of the song is:")
    audio = 0
    st.subheader(result)
    result=0

#lets add widgets for data collection (text, date, time inputs; text area) in columns
st.write("---")



#with st.sidebar:
st.title("Ask Melody Maestro to Generate an Album Cover for you")
a =st.button("Generate")

def generator():
    generator = build_generator()
    generator.load_weights('generator2.h5')
    imgs = generator.predict(tf.random.normal((1, 600, 1)))
    return imgs
def upscale_image(image, desired_resolution):
    # Get the current height and width of the image
    height, width = image.shape[:2]

    # Extract the desired width and height
    desired_width, desired_height = desired_resolution

    # Upscale the image using cv2.resize
    upscaled_image = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_CUBIC)

    return upscaled_image

if a >= 1:
    st.subheader("Your personalized album cover:")
    imgs = generator()
    imgs =1-imgs
    st.image(imgs, caption='Album', use_column_width=True)

    image_byte_array = io.BytesIO()
    Image.fromarray((imgs[0] * 255).astype('uint8')).save(image_byte_array, format='JPEG')
    image_byte_array = image_byte_array.getvalue()
    download_button_str = "Download Images"
    st.download_button(download_button_str, data=image_byte_array, file_name='generated_images.jpg', mime='image/jpeg')
    

