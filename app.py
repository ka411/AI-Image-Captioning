import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from feature_extraction import load_feature_extractor, extract_features

st.title("AI Image Captioning")
st.write("Upload an image and get an AI-generated caption.")

feature_model = load_feature_extractor()
model = tf.keras.models.load_model("image_caption_model.h5")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    features = extract_features(uploaded_file, feature_model)
    st.write("**Caption generation coming soon (requires tokenizer & vocab).**")
