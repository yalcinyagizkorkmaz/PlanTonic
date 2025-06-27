# User Authentication & ML-Driven Prediction API

PLANTONIC 
![Uploading Ekran Resmi 2025-06-27 15.22.25.png…]()


## 🔧 Özellikler

- Kullanıcı kayıt ve giriş işlemleri (JWT token tabanlı)
- Aile ID bazlı çoklu kullanıcı desteği
- SQLite tabanlı veritabanı
- TFLite makine öğrenimi modeli ile tahminleme
- Swagger arayüzü ile test edilebilir API
- Dockerfile ile kolay deployment

## 🗂️ Proje Yapısı

```
.
├── auth.py              # Kimlik doğrulama ve token üretimi
├── database.py          # Veritabanı bağlantısı ve oturumu
├── main.py              # FastAPI ana uygulama dosyası
├── models.py            # Pydantic ve SQLAlchemy modelleri
├── ml_model.py          # TensorFlow Lite model yükleme ve tahmin fonksiyonu
├── model.tflite         # Eğitimli ML modeli (TFLite formatında)
├── Dockerfile           # Docker yapılandırma dosyası
└── docker build .dockerfile  # Alternatif Docker build komutu içeren dosya
```

## 🚀 Kurulum ve Başlatma

### 1. Klonla

```bash
git clone https://github.com/kullaniciAdi/proje-adi.git
cd proje-adi
```

### 2. Gereksinimleri Yükle (Opsiyonel - Docker kullanmıyorsan)

```bash
pip install -r requirements.txt
```

> Not: `requirements.txt` dosyası oluşturulmamışsa, kullandığınız kütüphaneler:
> `fastapi`, `uvicorn`, `sqlalchemy`, `pydantic`, `python-jose`, `bcrypt`, `tensorflow`, `tflite-runtime`, `passlib` vs.

### 3. Uygulamayı Çalıştır

#### a) Geliştirme için:

```bash
uvicorn main:app --reload
```

#### b) Docker ile:

```bash
docker build -t auth-ml-app .
docker run -d -p 8000:8000 auth-ml-app
```

### 4. API Arayüzü

Tarayıcıdan eriş:
```
http://localhost:8000/docs
```

## 🔐 Kimlik Doğrulama

- `/register`: Kullanıcı kaydı
- `/login`: Kullanıcı girişi (JWT token döner)
- Diğer uç noktalara erişim için JWT token'ı `Authorization: Bearer <token>` başlığı ile gönderin.

## 🧠 ML Modeli Kullanımı

- `/predict`: Eğitimli TFLite model ile tahmin yapılır.
  - Gövdeye uygun veri göndererek sonuç alınabilir.
