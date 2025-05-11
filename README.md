# 📊 YZTA Datathon 2024 

## 📝 Proje Özeti

Bu proje, **YZTA Datathon 2024** yarışması kapsamında  geliştirilen ürün fiyat tahmin modelini içermektedir. Modelim , çeşitli ürünlerin fiyatlarını tahmin etmek için **derin öğrenme teknikleri** kullanmakta ve **USD/TRY döviz kuru verileri** gibi dış etkenleri de hesaba katmaktadır.

---

## 📂 Veri Setleri

Projede kullanılan temel veri setleri:

- `train.csv`: Eğitim için kullanılan ana veri seti. Ürün özellikleri, kategorileri, market bilgileri ve fiyat bilgilerini içerir.
- `testFeatures.csv`: Test veri seti. Eğitilen modelin tahmin yapması için kullanılır.
- `usd_clean.csv`: 2019-2025 yılları arasındaki USD/TRY günlük döviz kuru verilerini içerir.
- `sample_submission.csv`: Yarışma tarafından sağlanan örnek gönderim formatı.

---

## ⚙️ Metodoloji

### 🔧 Veri Ön İşleme

- Tarih bilgilerinden yeni özellikler türetildi (yıl, ay, haftanın günü)
- Kategorik değişkenler (ürün, ürün kategorisi, market, şehir vb.) sayısallaştırıldı
- Sayısal değerlere logaritmik dönüşüm uygulandı
- USD/TRY döviz kuru verileri ana veri setiyle birleştirildi
- Veriler, eğitim ve doğrulama setlerine bölündü
- Özellikler `StandardScaler` ile standartlaştırıldı

### 🧠 Model Mimarisi

- **Giriş katmanı**: Ürün özellikleri, kategorik bilgiler ve döviz kuru verilerini alır
- **Gizli katmanlar**: 2 adet, her biri 32-128 nöron arasında (dinamik boyutlu)
- **Aktivasyon fonksiyonu**: ReLU
- **Çıkış katmanı**: Tek nöron (ürün fiyatı tahmini)

### 🔍 Hiperparametre Optimizasyonu

- Katmanlardaki nöron sayısı
- Optimizer seçimi (Adam, RMSprop)
- Erken durdurma mekanizması (EarlyStopping)

---

## 📈 Sonuçlar

En iyi model, USD/TRY döviz kuru verilerini de dahil ederek oluşturulmuş ve sonuçlar `en-iyi-sonuc-4.6025400670.csv` dosyasına kaydedilmiştir.

- **RMSE (Root Mean Square Error)**: `4.60254`

---

## Kurulum ve Kullanım

### Gereksinimler

Projeyi çalıştırmak için gerekli kütüphaneler `requirements.txt` dosyasında belirtilmiştir. Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

### Kullanım Talimatları

Projeyi kendi ortamınızda çalıştırmak için:

1. Bu repository'yi klonlayın:

   ```bash
   git clone https://github.com/kullanici-adi/yzta-datathon-grup103.git
   cd yzta-datathon-grup103
   ```

2. Gereksinimleri yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

3. Veri setlerini temin edin:
   - train.csv
   - testFeatures.csv
   - usd_clean.csv

4. Jupyter Notebook dosyasını (`grup-103.ipynb`) çalıştırın:

   ```bash
   jupyter notebook grup-103.ipynb
   ```

5. Notebook içerisindeki hücreleri sırasıyla çalıştırarak modelin eğitimini ve tahminleri gerçekleştirin.

## Model Özellikleri

- **Özellik Mühendisliği**: Tarih verileri işlenerek yeni özellikler yaratıldı
- **Dış Faktörler**: USD/TRY döviz kuru verileri modele dahil edildi
- **Log Dönüşümü**: Fiyat verileri için logaritmik dönüşüm uygulandı
- **Hiperparametre Optimizasyonu**: Keras Tuner ile optimal model mimarisi bulundu
- **Erken Durdurma**: Aşırı uyumu (overfitting) önlemek için EarlyStopping kullanıldı

## Dosya Yapısı

```
yzta-datathon-grup103/
├── README.md               # Proje açıklaması
├── LICENSE                 # MIT Lisansı
├── requirements.txt        # Gerekli kütüphaneler
├── best_model.ipynb          # Ana notebook dosyası
├── usd_clean.csv           # USD/TRY kur verileri
├── sample_submission.csv   # Örnek submission dosyası
└── en-iyi-sonuc-4.6025400670.csv # En iyi tahmin sonuçları
```
