import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

def extract_features(audio_path, max_pad_len=174):
    """
    Extract MFCC features from audio file.
    """
    try:
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def build_cnn_model(input_shape, num_classes):
    """
    Build 1D/2D CNN model for audio classification.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("=== Task 2: Speech Emotion Recognition ===")
    print("Dummy script template initialized. Replace input datasets (e.g. RAVDESS / TESS) in workflow.")
    
    # Synthetic feature example for setup verification
    X_dummy = np.random.rand(100, 40, 174, 1)
    y_dummy = np.random.randint(0, 4, 100) # 4 emotions: happy, sad, angry, neutral
    
    model = build_cnn_model((40, 174, 1), num_classes=4)
    model.summary()
