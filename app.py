import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Sabit dosya yolları
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_detection_lgbm.pkl")

# Modeli yükle
@st.cache_resource
def load_model():
    try:
        # LightGBM modelini yükle
        lgbm_model = joblib.load(MODEL_PATH)
        return lgbm_model
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None

# Ana fonksiyon
def main():
    st.title("Kredi Kartı Dolandırıcılık Tespiti")
    st.write("Bu uygulama, kredi kartı işlemlerinin dolandırıcılık olup olmadığını tespit eder.")
    
    # Modeli yükle
    lgbm_model = load_model()
    
    if lgbm_model is None:
        st.warning("Model yüklenemedi. Lütfen dosya yollarını kontrol edin.")
        return
    
    st.success("Model başarıyla yüklendi!")
    
    st.subheader("İşlem Bilgilerini Girin")
    
    tab1, tab2 = st.tabs(["Temel Özellikler", "Gelişmiş Özellikler"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            v17 = st.number_input("V17", value=-1.0, format="%.6f", key="v17_tab1")
            v14 = st.number_input("V14", value=-1.5, format="%.6f", key="v14_tab1")
            v12 = st.number_input("V12", value=1.0, format="%.6f", key="v12_tab1")
            v10 = st.number_input("V10", value=-0.5, format="%.6f", key="v10_tab1")
        
        with col2:
            amount = st.number_input("İşlem Tutarı (TL)", value=120.50, format="%.2f", key="amount_tab1")
            time = st.slider("İşlem Zamanı (Gün içinde, saat)", 0.0, 24.0, 12.0, key="time_tab1")
            v4 = st.number_input("V4", value=1.2, format="%.6f", key="v4_tab1")
            v3 = st.number_input("V3", value=-0.3, format="%.6f", key="v3_tab1")
    
    # Tüm özellikleri içeren veri sözlüğü
    feature_data = {
        'Time': time * 3600,  # Saati saniyelere çevir
        'Amount': amount
    }
    
    # V1-V28 özelliklerini ekle
    with tab2:
        st.write("Tüm V1-V28 özellikleri")
        
        # 4 sütunlu düzen
        cols = st.columns(4)
        for i in range(1, 29):
            feature_name = f'V{i}'
            # Özel durumları kontrol et
            if feature_name in ['V3', 'V4', 'V10', 'V12', 'V14', 'V17']:
                if feature_name == 'V3':
                    feature_data[feature_name] = v3
                elif feature_name == 'V4':
                    feature_data[feature_name] = v4
                elif feature_name == 'V10':
                    feature_data[feature_name] = v10
                elif feature_name == 'V12':
                    feature_data[feature_name] = v12
                elif feature_name == 'V14':
                    feature_data[feature_name] = v14
                elif feature_name == 'V17':
                    feature_data[feature_name] = v17
            else:
                # Burada 4 sütunlu gösterim için index hesaplama
                col_idx = (i - 1) % 4
                with cols[col_idx]:
                    default_val = 0.0
                    feature_data[feature_name] = st.number_input(
                        feature_name, 
                        value=default_val, 
                        format="%.6f",
                        key=f"{feature_name}_tab2"
                    )
    
    # Tahmin butonu
    if st.button("Dolandırıcılık Analizi Yap"):
        # DataFrame oluştur
        input_df = pd.DataFrame([feature_data])
        
        # Model girdisi için doğru sütun sıralamasını sağla
        if 'Class' in input_df.columns:
            input_df = input_df.drop('Class', axis=1)
        
        # LightGBM ile tahmin yap
        prediction = lgbm_model.predict(input_df)[0]
        probability = lgbm_model.predict_proba(input_df)[0][1]
        
        # Sonuçları göster
        st.subheader("Analiz Sonuçları")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dolandırıcılık Olasılığı", f"{probability:.2%}")
        
        with col2:
            if prediction == 1:
                st.error("⚠️ Bu işlem DOLANDIRICILIK olarak tespit edildi!")
            else:
                st.success("✅ Bu işlem normal görünüyor.")
            
            # Risk seviyesi
            risk_level = probability * 100
            if risk_level < 30:
                risk_text = "Düşük Risk"
            elif risk_level < 70:
                risk_text = "Orta Risk"
            else:
                risk_text = "Yüksek Risk"
            
            st.write(f"Risk Seviyesi: {risk_text}")
        
        # Değerlendirme çubuğu
        st.progress(int(risk_level))
        
        # Açıklama
        with st.expander("Sonuç Açıklaması"):
            st.write("""
            - **Dolandırıcılık Olasılığı**: LightGBM modelinin tahmin ettiği dolandırıcılık olasılığı.
            - Yüksek değerler dolandırıcılık riskinin daha fazla olduğunu gösterir.
            """)
            
            # Feature importance basit gösterimi
            st.write("En etkin faktörler:")
            importance_data = {
                'V17': abs(v17) * 0.15,
                'V14': abs(v14) * 0.12,
                'V12': abs(v12) * 0.10,
                'V10': abs(v10) * 0.08,
                'Amount': amount * 0.001
            }
            
            # En etkin faktörleri sırala
            sorted_factors = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
            for factor, value in sorted_factors:
                st.write(f"- {factor}: {value:.2f}")

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()