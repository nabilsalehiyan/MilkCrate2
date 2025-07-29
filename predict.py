import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib

<<<<<<< HEAD
# SETTINGS
DATASET_PATH = "/Users/nabilsalehiyan/dataset1"
SAMPLE_RATE = 22050
DURATION = 30

# FEATURE EXTRACTION
def extract_features(file_path):
    pass

def load_dataset(dataset_path):
    pass

# ✅ PREDICT FUNCTION
def predict_genre(file_path):
    try:
        feature = extract_features(file_path)
        if feature is None:
            return None
        feature = feature.reshape(1, -1)
        model = joblib.load('genre_classification_model_xgb.pkl')
        prediction = model.predict(feature)
        predicted_class = prediction[0]
        predicted_genre = label_encoder.inverse_transform([predicted_class])
        return predicted_genre[0]
=======

# Settings
DATASET_PATH = "/Users/nabilsalehiyan/dataset1"
SAMPLE_RATE = 22050
DURATION = 30  # seconds

# Extract multiple features from an audio file
def extract_features(file_path, sample_rate=SAMPLE_RATE, duration=DURATION):
    try:
        signal, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        
        # MFCCs (Mel Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        # Chroma feature
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)

        # Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal)
        zero_crossing_rate_scaled = np.mean(zero_crossing_rate.T, axis=0)

        # Root Mean Square Energy
        rmse = librosa.feature.rms(y=signal)
        rmse_scaled = np.mean(rmse.T, axis=0)

        # Combine all features into a single vector
        features = np.hstack([mfccs_scaled, chroma_scaled, spectral_contrast_scaled, zero_crossing_rate_scaled, rmse_scaled])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset and extract features
def load_dataset(dataset_path):
    features = []
    labels = []
    
    # Iterate through each genre folder
    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)
        
        # Check if the item in dataset_path is a directory
        if not os.path.isdir(genre_path):
            continue
        
        # Iterate through each .wav file in the genre folder
        for file in os.listdir(genre_path):
            if not file.lower().endswith(".wav"):
                continue
            
            file_path = os.path.join(genre_path, file)
            feature = extract_features(file_path)
            
            if feature is not None:
                features.append(feature)
                labels.append(genre)
    
    return np.array(features), np.array(labels)

# Load the dataset
X, y = load_dataset(DATASET_PATH)

# Encode the labels (genres)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = xgb.XGBClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"XGBoost Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, 'genre_classification_model_xgb.pkl')

# Function to predict genre from a .wav file
def predict_genre(file_path):
    try:
        # Extract features from the audio file
        feature = extract_features(file_path)
        
        if feature is None:
            return None
        
        # Reshape the feature for prediction (since it's a single sample)
        feature = feature.reshape(1, -1)
        
        # Load the saved model
        model = joblib.load('genre_classification_model_xgb.pkl')
        
        # Make predictions
        prediction = model.predict(feature)
        predicted_class = prediction[0]
        
        # Decode the predicted class back to genre
        predicted_genre = label_encoder.inverse_transform([predicted_class])
        
        return predicted_genre[0]
    
>>>>>>> 221dd39fc3cfd63da23cd126e3948e9ea9335902
    except Exception as e:
        print(f"Error predicting genre: {e}")
        return None

<<<<<<< HEAD
# ✅ TRAINING CODE GOES HERE ONLY IF FILE IS RUN DIRECTLY
if __name__ == "__main__":
    X, y = load_dataset(DATASET_PATH)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    joblib.dump(model, 'genre_classification_model_xgb.pkl')

    file_path = input("Enter path to audio file: ")
    predicted_genre = predict_genre(file_path)

    if predicted_genre:
        print(f"The predicted genre of '{file_path}' is: {predicted_genre}")
    else:
        print(f"Failed to predict genre for '{file_path}'.")

        
=======
# Allow the user to input a file path
file_path = input("/Users/nabilsalehiyan/dataset1/rock/rock.00014.wav")

# Example usage with user input
predicted_genre = predict_genre(file_path)

if predicted_genre:
    print(f"The predicted genre of '{file_path}' is: {predicted_genre}")
else:
    print(f"Failed to predict genre for '{file_path}'.")
>>>>>>> 221dd39fc3cfd63da23cd126e3948e9ea9335902
