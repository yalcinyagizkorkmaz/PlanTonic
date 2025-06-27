

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


## 🔧 Features

* User registration and login (JWT-based authentication)
* Multi-user support based on Family ID
* SQLite-based database
* Predictions using a TFLite machine learning model
* Testable API with Swagger UI
* Easy deployment with Dockerfile

## 📂 Project Structure

```
.
├── auth.py              # Authentication and token generation
├── database.py          # Database connection and session
├── main.py              # Main FastAPI application
├── models.py            # Pydantic and SQLAlchemy models
├── ml_model.py          # TensorFlow Lite model loading and prediction function
├── model.tflite         # Trained ML model in TFLite format
├── Dockerfile           # Docker configuration file
└── docker build .dockerfile  # File with alternative Docker build command
```

## 🚀 Setup & Run

### 1. Clone the Repository

```bash
git clone https://github.com/username/project-name.git
cd project-name
```

### 2. Install Requirements (Optional - if not using Docker)

```bash
pip install -r requirements.txt
```

> Note: If `requirements.txt` is not available, the main libraries used are:
> `fastapi`, `uvicorn`, `sqlalchemy`, `pydantic`, `python-jose`, `bcrypt`, `tensorflow`, `tflite-runtime`, `passlib`, etc.

### 3. Run the Application

#### a) For Development:

```bash
uvicorn main:app --reload
```

#### b) With Docker:

```bash
docker build -t auth-ml-app .
docker run -d -p 8000:8000 auth-ml-app
```

### 4. API Interface

Accessible via browser:

```
http://localhost:8000/docs
```

## 🔐 Authentication

* `/register`: User registration
* `/login`: User login (returns JWT token)
* For other endpoints, include the JWT token in the `Authorization: Bearer <token>` header.

## 🧠 ML Model Usage

* `/predict`: Makes a prediction using the trained TFLite model.

  * Submit appropriate data in the request body to receive a result.

