import streamlit as st
import pandas as pd
from joblib import load

# Muat model yang dilatih
MODEL_PATH = "random_forest_model.joblib"
try:
    model = load(MODEL_PATH)
    st.sidebar.success("Model successfully loaded!")
except FileNotFoundError:
    st.sidebar.error("Model file not found. Please make sure it's named 'random_forest_model.joblib'.")
# Judul aplikasi
st.title("Applikasi Prediksi Mental Health")

# Input sederhana untuk fitur pengguna (ganti dengan kolom dataset Anda)
st.header("Masukan Informasi Anda")

## Contoh fitur berdasarkan kolom yang terlihat pada kesalahan Anda
age = st.number_input("Age", min_value=1, max_value=100, value=18)
gender = st.selectbox("Choose your gender", ["Male", "Female"])
anxiety = st.selectbox("Do you have Anxiety?", ["Yes", "No"])
depression = st.selectbox("Do you have Depression?", ["Yes", "No"])
panic_attack = st.selectbox("Do you have Panic attack?", ["Yes", "No"])
marital_status = st.selectbox("Marital status", ["Yes", "No"])

# Petakan masukan ke kerangka data untuk mencocokkan nama fitur model yang dilatih
input_data = {
    'Age': age,
    'Choose your gender_Male': 1 if gender == "Male" else 0,
    'Do you have Anxiety?_Yes': 1 if anxiety == "Yes" else 0,
    'Do you have Depression?_Yes': 1 if depression == "Yes" else 0,
    'Do you have Panic attack?_Yes': 1 if panic_attack == "Yes" else 0,
    'Marital status_Yes': 1 if marital_status == "Yes" else 0
}

# Ubah masukan menjadi DataFrame
input_df = pd.DataFrame([input_data])

# Hapus kolom Stempel Waktu jika ada (tampaknya tidak relevan)
input_df = input_df.loc[:, ~input_df.columns.str.startswith('Timestamp')]

# Tampilkan prediksi saat tombol diklik
if st.button("Predict"):
    if model is not None:
        # Indeks ulang agar sesuai dengan kumpulan fitur pelatihan model
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(input_df)
        # Dengan asumsi 1 positif dan 0 negatif pada model Anda
        if prediction[0] == 1:
            st.success("Prediksi: Positif (Menunjukkan masalah kesehatan mental)")
        else:
            st.warning("Prediksi: Negatif (Tidak ada masalah kesehatan mental yang terdeteksi)")
    else:
        st.error("Model tidak dimuat, tidak dapat membuat prediksi.")
