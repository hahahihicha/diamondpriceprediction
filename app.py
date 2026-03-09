# ================================
# 1. IMPORT LIBRARY
# ================================
import streamlit as st
import numpy as np
import joblib


# ================================
# 2. LOAD MODEL DAN SCALER
# ================================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# ================================
# 3. JUDUL APLIKASI
# ================================
st.title("Diamond Price Prediction 💎")


# ================================
# 4. INPUT DATA USER
# ================================
st.header("Input Data")

carat = st.number_input("Carat", min_value=0.0, step=0.01)

cut = st.selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
)

color = st.selectbox(
    "Color",
    ["J","I","H","G","F","E","D"]
)

clarity = st.selectbox(
    "Clarity",
    ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
)

depth = st.number_input("Depth", step=0.1)
table = st.number_input("Table", step=0.1)

x = st.number_input("Length (x)", step=0.01)
y = st.number_input("Width (y)", step=0.01)
z = st.number_input("Height (z)", step=0.01)


# ================================
# 5. KONVERSI DATA KATEGORI KE ANGKA
# ================================
cut_map = {
    "Fair":0,
    "Good":1,
    "Very Good":2,
    "Premium":3,
    "Ideal":4
}

color_map = {
    "J":0,
    "I":1,
    "H":2,
    "G":3,
    "F":4,
    "E":5,
    "D":6
}

clarity_map = {
    "I1":0,
    "SI2":1,
    "SI1":2,
    "VS2":3,
    "VS1":4,
    "VVS2":5,
    "VVS1":6,
    "IF":7
}


# ================================
# 6. PROSES PREDIKSI
# ================================
if st.button("Predict Price"):

    # menyusun data input
    data = np.array([[
        carat,
        cut_map[cut],
        color_map[color],
        clarity_map[clarity],
        depth,
        table,
        x,
        y,
        z
    ]])

    # scaling data menggunakan scaler
    data_scaled = scaler.transform(data)

    # prediksi harga
    prediction = model.predict(data_scaled)

    # tampilkan hasil
    st.success(f"Perkiraan Harga Diamond: ${prediction[0]:,.2f}")