# streamlit run app.py
import streamlit as st
import pandas as pd
import joblib
from babel.numbers import format_currency

# Menggunakan cache streamlit agar model tidak diload berkali2
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("models/random_forest_regressor_model.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        feature_scaler = joblib.load("models/feature_scaler.pkl")
        price_scaler = joblib.load("models/price_scaler.pkl")
        return model, feature_columns, feature_scaler, price_scaler
    except FileNotFoundError:
        return None, None, None, None, None

# Creating asset
rf_model, feature_columns, feature_scaler, price_scaler = load_assets()

st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")
st.title("Aplikasi Prediksi Harga Rumah")

# cek apakah aset berhasil di-load
if rf_model is None:
    st.error("Gagal load model")
else:
    st.markdown("""
    Masukan detail property di bawah ini untuk mendapatkan estimasi harga.
    Aplikasi ini menggunakan model *machine learning* untuk memberikan prediksi harga.
    """)

    with st.form("prediction_form"):
        st.header("Masukan Detail Property")

        # input numerik
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Luas Tanah (m²)", min_value=30.0, max_value=1000.0, value=120.0, step=10.0)
            bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3, step=1)
            garage = st.number_input("Kapasitas Garasi (mobil)", min_value=0, max_value=5, value=1, step=1)

        with col2:
            building_area = st.number_input("Luas Bangunan (m²)", min_value=30.0, max_value=800.0, value=90.0, step=10.0)
            bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=8, value=2, step=1)
            city = st.selectbox("Kota", ('Jakarta Selatan', 'Jakarta Timur', 'Jakarta Pusat', 'Jakarta Barat', 'Depok', 'Bogor', 'Tangerang', 'Tangerang Selatan'))

        submit_button = st.form_submit_button(label="Prediksi")

    if submit_button:
        try:
            input_data = {
                'area': [area],
                'building_area': [building_area],
                'bedrooms': [bedrooms],
                'bathrooms': [bathrooms],
                'garage': [garage],
                'city': [city]
            }
            
            print('Processing input data...')
            input_df = pd.DataFrame(input_data)
            input_df[['area', 'building_area']] = feature_scaler.transform(input_df[['area', 'building_area' ]])
            input_df = pd.get_dummies(input_df, columns=['city', 'bedrooms', 'bathrooms', 'garage'], prefix=['City', 'Bedroom', 'Bathroom', 'Garage'])

            print('Input data processed successfully')

            print('Reindexing input data...')
            input_processed = input_df.reindex(columns=feature_columns, fill_value=0)
            print('Input data reindexed successfully')

            print('Predicting price...')
            scaled_prediction = rf_model.predict(input_processed)
            print('Price predicted successfully')

            print('Scaling price...')
            original_price_prediction = price_scaler.inverse_transform(scaled_prediction.reshape(-1,1))[0][0] * 1000000
            print('Price scaled successfully')

            # format price rupiah
            original_price_prediction = format_currency(original_price_prediction, 'IDR', locale='id_ID')
            st.success("Prediksi Berhasil")
            st.metric(
                label="Estimasi Harga Properti",
                value=original_price_prediction
            )
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
