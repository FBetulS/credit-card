# Credit Card Fraud Detection Project

## 📖 Proje Açıklaması
Bu proje, kredi kartı dolandırıcılığını tespit etmek amacıyla geliştirilmiştir. Veri seti, kredi kartı işlemlerinin normal ve dolandırıcılık olarak sınıflandırılmasını sağlamak için kullanılmıştır.

## 🔗 Veri Kümesi
Veri kümesi, aşağıdaki kaynaklardan elde edilmiştir:
- Kaggle'daki [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
 
-  [Credit Card Fraud Detection Hugging Face](https://huggingface.co/spaces/btulftma/credit-card-detection)
  

## 🛠️ Kullanılan Kütüphaneler
- `pandas`: Veri analizi ve manipülasyonu için.
- `numpy`: Sayısal işlemler için.
- `matplotlib` ve `seaborn`: Verilerin görselleştirilmesi için.
- `imblearn`: Dengesiz veri setleri için SMOTE uygulaması.
- `sklearn`: Model değerlendirme ve metrikler için.
- `lightgbm`: Hızlı ve etkili bir makine öğrenimi modeli için.
- `tensorflow`: Sinir ağı modelleri için.

## 📊 Model Eğitimi ve Değerlendirme
Proje, aşağıdaki adımları içermektedir:
1. **Veri Yükleme**: `creditcard.csv` dosyası okunur ve ilk beş gözlem gösterilir.
2. **Sınıf Dağılımı**: Normal işlemler ve dolandırıcılık oranları görselleştirilir.
3. **Veri Ön İşleme**: SMOTE uygulanarak dengesiz veri seti dengelenir.
4. **Modelleme**: Autoencoder ve LightGBM kullanılarak dolandırıcılık tespiti yapılır.
5. **Değerlendirme**: Modelin performansı F1 skoru, doğruluk ve karışıklık matrisleri ile değerlendirilir.

### Model Performans Sonuçları
- **F1-Score**: 0.87
- **Doğruluk**: 1.00
- Karışıklık matrisi ve sınıflandırma raporu ile detaylı analiz yapılmıştır.

## 📈 Sonuçlar
Model, dolandırıcılık tespitinde yüksek doğruluk oranları ve F1 skoru ile başarılı sonuçlar göstermektedir.

## 👤 İletişim
**Ad**: [Adınızı Buraya Ekleyin]  
**GitHub**: [GitHub Profilinizin Linki]

## 📄 Lisans
Bu proje MIT Lisansı altında lisanslanmıştır.
