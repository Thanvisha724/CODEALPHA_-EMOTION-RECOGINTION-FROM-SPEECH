# Emotion Recognition From Speech

Speech Emotion Recognition project using:
- Librosa
- MFCC Features
- TensorFlow CN

## Dataset

This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

Download Dataset:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

After downloading, extract the dataset and place the Actor folders inside the dataset directory:

dataset/
├── Actor_01
├── Actor_02
├── Actor_03
...
├── Actor_24

## Run
pip install -r requirements.txt
python src/emotion_recognition.py
