import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib

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
    except Exception as e:
        print(f"Error predicting genre: {e}")
        return None

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

        
