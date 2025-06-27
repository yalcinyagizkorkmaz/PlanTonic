

# 🌿 PlanTonic

**PlanTonic** is a mobile application designed to assist amateur gardeners and plant enthusiasts by leveraging image-based machine learning and large language models (LLMs) to identify plant species, detect diseases, and provide personalized care recommendations.

---

## 🚀 Project Overview

PlanTonic simplifies plant care through a user-friendly mobile interface that enables:
- 📷 Image-based **plant species recognition**
- 🦠 Detection of **visible plant diseases**
- 🧠 AI-driven **personalized plant care suggestions**

---

## 🧠 Technologies & Tools

- **Frontend**: Swift & SwiftUI (iOS)
- **Backend**: Python with FastAPI
- **Machine Learning**: PyTorch, TensorFlow  
  - Models: MobileNetV2, ResNet50
- **NLP Integration**: Large Language Model (LLM) for custom care guidance
- **Database**: PostgreSQL (hosted via Neon.tech)
- **Infrastructure**: AWS EC2, Docker
- **DevOps**: Bitbucket, JIRA (Agile/Scrum, CI/CD workflows)

---

## 🌱 Application Areas

PlanTonic is built for:
- Home and indoor gardeners with little to no expertise
- Urban plant lovers seeking quick, intelligent care solutions
- Anyone looking to automate and simplify their plant care routines

---

## 🏗️ System Architecture

The app is structured in a modular, scalable architecture:
- iOS frontend for real-time image capture and care feedback
- RESTful backend APIs for data processing
- ML inference layer for species/disease detection
- LLM integration for contextual text generation

---

## 🧪 Research & References

- Mohanty et al. (2016) – Deep learning for plant disease detection  
- Goëau et al. (2017) – PlantCLEF dataset  
- Howard et al. (2017) – MobileNet for mobile vision  
- Tan & Le (2019) – EfficientNet model scaling

---


---

## 📍 Conclusion

PlanTonic bridges the gap between amateur plant enthusiasts and expert-level care using the power of AI. By combining modern machine learning with user-centric design, it provides a smarter, greener solution for plant health and happiness.

---

> _“Grow smart. Care better. Live green.”_


<img width="500" alt="Ekran Resmi 2025-06-27 15 23 02" src="https://github.com/user-attachments/assets/485295ad-1f86-4d49-9074-5c3f8a78de07" />



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
