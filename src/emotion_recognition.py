import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dataset Path
DATASET_PATH = "dataset"

features = []
labels = []

# Emotion Mapping
emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

print("Loading audio files...")

# Read all WAV files
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            try:
                audio, sr = librosa.load(
                    file_path,
                    duration=3,
                    offset=0.5
                )

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=40
                )

                mfcc_scaled = np.mean(mfcc.T, axis=0)

                emotion_code = file.split("-")[2]

                if emotion_code in emotion_map:
                    features.append(mfcc_scaled)
                    labels.append(emotion_map[emotion_code])

            except Exception as e:
                print("Error:", file_path)

print("Feature Extraction Completed")

# Convert to numpy arrays
X = np.array(features)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("Training Model...")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n====================")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("====================\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))
