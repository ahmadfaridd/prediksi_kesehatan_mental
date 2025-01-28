import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# Muat model yang dilatih
MODEL_PATH = "random_forest_model.joblib"
try:
    model = load(MODEL_PATH)
    st.sidebar.success("Model berhasil dimuat!")
except FileNotFoundError:
    model = None
    st.sidebar.error("File model tidak ditemukan. Harap pastikan model disimpan sebagai random_forest_model.joblib.")

# Muat kumpulan data
CSV_PATH = "student_mental_health.csv"
try:
    data = pd.read_csv(CSV_PATH)
    st.sidebar.success("Dataset berhasil dimuat!")
except FileNotFoundError:
    data = None
    st.sidebar.error("Dataset tidak ditemukan. Harap pastikan model disimpan sebagai student_mental_health.csv.")

# Judul Aplikasi Streamlit
st.title("Aplikasi Prediksi Kesehatan mental")

# Tampilkan pratinjau kumpulan data
if data is not None:
    st.header("Pratinjau Kumpulan Data")
    st.write(data.head())

    # Bagian Prediksi
    if model is not None:
        st.header("Buat Prediksi")
        st.write("Memberikan fitur masukan untuk memprediksi status kesehatan mental.")

        # Menghasilkan kolom input secara dinamis berdasarkan kolom kumpulan data
        input_data = {}
        for feature in data.columns[:-1]:  # Kecualikan kolom target
            input_data[feature] = st.number_input(f"{feature}", value=0.0)

        input_df = pd.DataFrame([input_data])

        if st.button("Prediksi"):
            prediction = model.predict(input_df)
            st.success(f"Prediksi: {prediction[0]}")
    else:
        st.warning("Model tidak dimuat. Silakan periksa file model.")
else:
    st.info("Silakan unggah dataset 'student_mental_health.csv' untuk di proses.")
