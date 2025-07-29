import streamlit as st
from predict import predict_genre

st.set_page_config(page_title="MilkCrate Genre Classifier", layout="centered")

st.title("🎵 MilkCrate: Music Genre Classifier")

st.markdown("Upload a `.wav` audio file and get a predicted genre using our trained XGBoost model.")

uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])

if uploaded_file:
    # Save the uploaded file to a temp location
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav", format="audio/wav")

    if st.button("🔍 Predict Genre"):
        genre = predict_genre("temp.wav")
        if genre:
            st.success(f"🎧 Predicted Genre: **{genre}**")
        else:
            st.error("🚫 Could not process the audio file. Please try a different one.")
else:
    st.info("👈 Upload a .wav file to begin.")