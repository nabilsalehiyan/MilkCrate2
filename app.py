import streamlit as st
from predict import predict_genre  # your function

st.title("🎵 Music Genre Classifier")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio("temp.wav", format="audio/wav")

    if st.button("Predict Genre"):
        genre = predict_genre("temp.wav")
        if genre:
            st.success(f"Predicted Genre: **{genre}**")
        else:
            st.error("Could not process the audio.")
