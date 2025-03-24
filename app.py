import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Modelleri yükle
@st.cache_resource
def load_models():
    lgbm_model = joblib.load('fraud_detection_lgbm.pkl')
    autoencoder = load_model('autoencoder_model.h5')
    return lgbm_model, autoencoder

lgbm_model, autoencoder = load_models()

# Streamlit arayüz
st.title('🚨 Credit Card Fraud Detection')
st.write("Lütken işlem verilerini girin:")

# Kullanıcı girdileri (Sadece kritik özellikler)
with st.form("fraud_form"):
    time = st.number_input('Zaman (saniye)', min_value=0)
    amount = st.number_input('Miktar', min_value=0.0)
    v14 = st.number_input('V14', value=0.0)
    v17 = st.number_input('V17', value=0.0)
    submitted = st.form_submit_button("Tahmin Yap")

if submitted:
    # Tüm özellikler için varsayılan 0 değerli bir dizi oluştur
    input_data = np.zeros(30)  # 30 özellik (Time, V1-V28, Amount)
    input_data[0] = time
    input_data[1:29] = 0  # V1-V28'deki diğer değerler (Demo için 0)
    input_data[29] = amount
    
    # Autoencoder ile MSE hesapla
    input_df = pd.DataFrame([input_data], columns=X_train.columns.tolist() + ['Amount'])
    reconstruction = autoencoder.predict(input_df)
    mse = np.mean(np.square(input_df - reconstruction))
    input_df['MSE'] = mse
    
    # Tahmin yap
    prediction = lgbm_model.predict(input_df)
    proba = lgbm_model.predict_proba(input_df)[0][1]
    
    # Sonuç
    st.write("## Sonuç")
    if prediction[0] == 1:
        st.error(f'🚨 **Dolandırıcılık Şüphesi!** (Olasılık: {proba:.2%})')
    else:
        st.success(f'✅ **Normal İşlem** (Olasılık: {proba:.2%})')

        