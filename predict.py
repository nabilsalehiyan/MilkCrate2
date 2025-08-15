# train_model_subset.py
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# === CONFIG ===
AUDIO_PATH = "/Volumes/Nabil 15T/jamendo_audio"
METADATA_FILE = "autotagging_genre.tsv"
SAMPLE_RATE = 22050
DURATION = 30  # seconds

# Subset controls (FAST MODE)
USE_SUBSET = False          # set to False when you say "pineapple2"
PER_CLASS_LIMIT = None       # how many tracks max per genre in fast mode
MAX_TOTAL = None           # optional global cap (None = no cap)

RANDOM_STATE = 42

# === FEATURE EXTRACTION ===
def extract_features(file_path, sample_rate=SAMPLE_RATE, duration=DURATION):
    try:
        signal, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=signal, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal).T, axis=0)
        rmse = np.mean(librosa.feature.rms(y=signal).T, axis=0)
        return np.hstack([mfccs, chroma, contrast, zcr, rmse])
    except Exception as e:
        print(f"❌ Feature extraction failed for {file_path}: {e}")
        return None

# === DATA LOADING / SAMPLING ===
def load_metadata():
    md = pd.read_csv(METADATA_FILE, sep="\t", on_bad_lines="skip")
    # Keep only genre tags
    md = md[md["TAGS"].astype(str).str.startswith("genre---")]
    md = md.assign(genre=md["TAGS"].str.replace("genre---", "", regex=False))
    # Build absolute path
    md = md.assign(abs_path=md["PATH"].apply(lambda p: os.path.join(AUDIO_PATH, p)))
    return md[["abs_path", "genre"]]

def sample_subset(md: pd.DataFrame) -> pd.DataFrame:
    if not USE_SUBSET:
        return md
    # per-genre sampling (reproducible)
    sampled = (
        md.groupby("genre", group_keys=False)
          .apply(lambda g: g.sample(n=min(PER_CLASS_LIMIT, len(g)), random_state=RANDOM_STATE))
          .reset_index(drop=True)
    )
    if MAX_TOTAL is not None and len(sampled) > MAX_TOTAL:
        sampled = sampled.sample(n=MAX_TOTAL, random_state=RANDOM_STATE).reset_index(drop=True)
    return sampled

def filter_existing_files(md: pd.DataFrame) -> pd.DataFrame:
    exists_mask = md["abs_path"].apply(os.path.exists)
    missing = (~exists_mask).sum()
    if missing:
        print(f"⚠️ Skipping {missing} entries (files not found).")
    return md[exists_mask].reset_index(drop=True)

def build_dataset(md: pd.DataFrame):
    features, labels = [], []
    for i, row in md.iterrows():
        fp = row["abs_path"]
        feat = extract_features(fp)
        if feat is not None:
            features.append(feat)
            labels.append(row["genre"])
        # Small heartbeat every ~50 files
        if (i + 1) % 50 == 0:
            print(f"   ...processed {i+1}/{len(md)} files")
    X = np.array(features)
    y = np.array(labels)
    print(f"✅ Features extracted: {len(X)}")
    return X, y

# === MAIN TRAINING ===
if __name__ == "__main__":
    print("📦 Loading dataset and training model...")
    print(f"📍 Metadata path: {METADATA_FILE}")
    print(f"📍 Audio base path: {AUDIO_PATH}")
    print(f"⚡ Fast mode (subset) = {USE_SUBSET} | PER_CLASS_LIMIT={PER_CLASS_LIMIT} | MAX_TOTAL={MAX_TOTAL}")

    # 1) Read + subset
    md_all = load_metadata()
    print(f"🧾 Metadata rows with genre tags: {len(md_all)}")
    md_sub = sample_subset(md_all)
    print(f"🎯 Selected rows (after subset): {len(md_sub)}")
    
    md_sub = filter_existing_files(md_sub)
    print(f"🎧 Rows with existing audio files: {len(md_sub)}")

    # 🔎 Drop rare classes in the subset so stratify works
    MIN_SAMPLES_PER_CLASS = 3  # try 3; set to 2 if needed
    vc = md_sub["genre"].value_counts()
    keep_genres = vc[vc >= MIN_SAMPLES_PER_CLASS].index
    dropped = len(md_sub) - len(md_sub[md_sub["genre"].isin(keep_genres)])
    md_sub = md_sub[md_sub["genre"].isin(keep_genres)].reset_index(drop=True)
    print(f"🧹 Dropped {dropped} rows from rare classes (<{MIN_SAMPLES_PER_CLASS}).")
    print(f"🎯 Rows after rare-class filter: {len(md_sub)}")


    if len(md_sub) < 10:
        print("❌ Not enough files to train. Increase PER_CLASS_LIMIT or disable subset.")
        raise SystemExit

    # 2) Build dataset
    X, y = build_dataset(md_sub)
    if len(X) == 0:
        print("❌ No features extracted. Exiting.")
        raise SystemExit

    # 3) Split FIRST on raw labels, then encode
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test  = label_encoder.transform(y_test_raw)

    # Show the classes we’ll actually learn/predict in this FAST run
    print("🎶 Genres this model can predict (subset run):")
    print(label_encoder.classes_)

    # 4) Train model (no eval_metric in .fit for your xgboost version)
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test)
    print("✅ Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"✅ Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

    # 6) Save model + encoder (these are just for the subset; we’ll overwrite after pineapple2)
    joblib.dump(model, "genre_classification_model_xgb.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("💾 Model and encoder saved (subset run).")
