import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model(r'BestModel_Mobilenet_Bokeh.h5')
class_names = ['Beras Hitam', 'Beras Merah', 'Beras Putih']

def classify_image(image_path):
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

def custom_progress_bar(confidence, class_colors):
    progress_html = "<div style='border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;'>"
    for i, (conf, color) in enumerate(zip(confidence, class_colors)):
        percentage = conf * 100
        progress_html += f"""
        <div style="width: {percentage:.2f}%; background: {color}; color: white; text-align: center; height: 24px; float: left;">
            {class_names[i]}: {percentage:.2f}%
        </div>
        """
    progress_html += "</div>"
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)

st.title("Klasifikasi Jenis Beras")

uploaded_files = st.file_uploader("Unggah Gambar Beras (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            label, confidence = classify_image(uploaded_file.name)

            if label != "Error":
                class_colors = ["#007BFF", "#FF4136", "#28A745"]  

                st.sidebar.write(f"Nama File: {uploaded_file.name}")
                st.sidebar.markdown(f"<h4 style='color: #007BFF;'>Prediksi: {label}</h4>", unsafe_allow_html=True)

                st.sidebar.write("Confidence:")
                for i, class_name in enumerate(class_names):
                    st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")

                custom_progress_bar(confidence, class_colors)

                st.sidebar.write("---")
            else:
                st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
