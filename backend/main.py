# main.py

import sys
import os
import io
import uuid
import base64
import json

import numpy as np
from PIL import Image
import tensorflow as tf

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from database import db, init_db
from auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM
from models import User, UserCreate
from jose import JWTError, jwt
from pydantic import BaseModel
from typing import Optional

# ----------------------------------------
# 1) Classifier sınıfları
# ----------------------------------------

class TFLiteModelBase:
    def __init__(self, model_path: str, label_map_path: str, input_size: tuple = (224, 224)):
        self.input_size = input_size

        # Interpreter yükle
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        inp, out = self.interpreter.get_input_details()[0], self.interpreter.get_output_details()[0]
        self.input_index = inp['index']
        self.output_index = out['index']
        self.quantization = inp.get('quantization', (0.0, 1.0))

        # label_map → index_to_class
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        self.index_to_class = {v: k for k, v in label_map.items()}

    def _preprocess(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert('RGB').resize(self.input_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, img_path: str) -> (str, float):
        data = self._preprocess(img_path)
        scale, zero_point = self.quantization

        if scale != 0:
            data = data / scale + zero_point
            data = data.astype(np.uint8)

        self.interpreter.set_tensor(self.input_index, data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)[0]

        if output.dtype == np.uint8:
            output = (output.astype(np.float32) - zero_point) * scale

        idx = int(np.argmax(output))
        conf = float(output[idx])
        cls = self.index_to_class.get(idx, "Unknown")
        return cls, conf


class PlantDiseaseClassifier(TFLiteModelBase):
    def __init__(self, model_path: str, label_map_path: str, input_size: tuple = (224, 224)):
        super().__init__(model_path, label_map_path, input_size)

    def disease(self, image_path: str, unhealthy_threshold: float = 0.45) -> dict:
        lbl, conf = self.predict(image_path)
        
        # Confidence kontrolü - 0.55 threshold
        if conf < unhealthy_threshold:
            lbl = "Unhealthy"
        else:
            lbl = "Healthy"
            
        return {
            "label": lbl,
            "confidence": conf
        }


class PlantIdentificationClassifier(TFLiteModelBase):
    def __init__(self, model_path: str, label_map_path: str, launch_data: dict, input_size: tuple = (224, 224)):
        super().__init__(model_path, label_map_path, input_size)
        self.launch_data = launch_data

    def identify(self, image_path: str, unhealthy_threshold: float = 0.40) -> dict:
        cls, conf = self.predict(image_path)
        status = "healthy"
        if conf < unhealthy_threshold:
            status = "unhealthy"
        return {
            "label": cls,
            "confidence": conf,
            "health_status": status
        }


# ----------------------------------------
# 2) Model & label dosyalarını yükle
# ----------------------------------------

# launch_data: identify_labels.json içindeki ek bilgiler
with open("models/labels_atakan.json", "r", encoding="utf-8") as f:
    launch_data = json.load(f)

plant_classifier = PlantIdentificationClassifier(
    model_path="models/plant_species_model_atakan.tflite",
    label_map_path="models/labels_atakan.json",
    launch_data=launch_data,
    input_size=(224, 224)
)

disease_classifier = PlantDiseaseClassifier(
    model_path="models/disease_model.tflite",
    label_map_path="models/disease_labels.json",
    input_size=(224, 224)
)


# ----------------------------------------
# 3) FastAPI app ve auth ayarları
# ----------------------------------------

app = FastAPI(
    title="Plantonic API",
    description="Plantonic API",
    version="1.0.0",
    openapi_tags=[
        {"name": "health", "description": "Sağlık kontrolü endpoint'leri"},
        {"name": "auth", "description": "Kimlik doğrulama ve yetkilendirme endpoint'leri"},
        {"name": "users", "description": "Kullanıcı yönetimi endpoint'leri"},
        {"name": "identification", "description": "Bitki görüntüleri endpoint'leri"},
    ]
)

# CORS ayarları
origins = [
    "http://localhost:3000",
    "http://localhost:8001",
    "http://13.60.85.186",
    "http://13.60.85.186:8001",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "message": "Yetkilendirme başlığı bulunamadı"
            }
        )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "status": "error",
                    "message": "Geçersiz kimlik bilgileri"
                }
            )
        
        user = db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
        if user is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "status": "error",
                    "message": "Kullanıcı bulunamadı"
                }
            )
        return user
        
    except JWTError as e:
        print(f"JWT Error: {e}")
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "message": "Geçersiz token"
            }
        )
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "message": "Kimlik doğrulama hatası"
            }
        )


# ----------------------------------------
# 4) Pydantic modelleri
# ----------------------------------------

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    status: str
    data: dict

class RegisterResponse(BaseModel):
    status: str
    message: str
    data: dict


# ----------------------------------------
# 5) Auth & user endpoint'leri
# ----------------------------------------

@app.get("/", tags=["health"])
async def root():
    return {"status": "healthy"}

@app.post("/register", tags=["auth"], response_model=RegisterResponse)
async def register(user: UserCreate):
    """
    Yeni kullanıcı kaydı oluşturur.
    
    Args:
        user (UserCreate): Email ve şifre bilgilerini içeren JSON verisi
        
    Returns:
        JSON: Kayıt durumu ve kullanıcı bilgileri
    """
    try:
        # Email kontrolü
        existing_user = db.fetch_one("SELECT * FROM users WHERE email = %s", (user.email,))
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": "Bu email adresi zaten kayıtlı"
                }
            )

        # Yeni kullanıcı oluştur
        user_id = str(uuid.uuid4())
        hashed_password = get_password_hash(user.password)

        db.execute_query(
            "INSERT INTO users (id, email, password_hash) VALUES (%s, %s, %s)",
            (user_id, user.email, hashed_password),
        )
        
        return {
            "status": "success",
            "message": "Kullanıcı başarıyla oluşturuldu",
            "data": {
                "user_id": user_id,
                "email": user.email
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Kayıt işlemi sırasında bir hata oluştu"
            }
        )

@app.post("/login", tags=["auth"], response_model=LoginResponse)
async def login(data: LoginRequest):
    """
    Kullanıcı girişi yapar ve token döndürür.
    
    Args:
        data (LoginRequest): Email ve şifre bilgilerini içeren JSON verisi
        
    Returns:
        JSON: Token ve kullanıcı bilgileri
    """
    try:
        user = db.fetch_one("SELECT * FROM users WHERE email = %s", (data.email,))
        if not user:
            raise HTTPException(
                status_code=401,
                detail={
                    "status": "error",
                    "message": "Hatalı email veya şifre"
                }
            )

        if not verify_password(data.password, user["password_hash"]):
            raise HTTPException(
                status_code=401,
                detail={
                    "status": "error",
                    "message": "Hatalı email veya şifre"
                }
            )

        access_token = create_access_token(data={"sub": user["id"], "email": user["email"]})
        return {
            "status": "success",
            "data": {
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": user["id"],
                    "email": user["email"]
                }
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Login hatası: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Giriş işlemi sırasında bir hata oluştu"
            }
        )

@app.get("/users/me", response_model=User, tags=["users"])
async def read_users_me(current_user=Depends(get_current_user)):
    return current_user

@app.get("/users", tags=["users"])
async def get_users(current_user=Depends(get_current_user)):
    users = db.fetch_all("SELECT id, email, created_at FROM users")
    return users


# ----------------------------------------
# 6) /identification endpoint
# ----------------------------------------

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/identification", tags=["identification"])
async def identify(plant_image: UploadFile = File(...), disease_image: UploadFile = File(...)):
    # Dosyaları kaydet
    plant_id = str(uuid.uuid4())
    plant_path = os.path.join(UPLOAD_DIR, f"{plant_id}.jpg")
    plant_pil = Image.open(io.BytesIO(await plant_image.read())).convert("RGB")
    plant_pil.save(plant_path)

    disease_id = str(uuid.uuid4())
    disease_path = os.path.join(UPLOAD_DIR, f"{disease_id}.jpg")
    disease_pil = Image.open(io.BytesIO(await disease_image.read())).convert("RGB")
    disease_pil.save(disease_path)

    # Tahminler
    plant_result   = plant_classifier.identify(plant_path)
    disease_result = disease_classifier.disease(disease_path)

    # Base64 encode
    with open(plant_path, "rb") as f:
        plant_b64 = base64.b64encode(f.read()).decode()
    with open(disease_path, "rb") as f:
        disease_b64 = base64.b64encode(f.read()).decode()

    # DB kaydı (opsiyonel)
    session_id = str(uuid.uuid4())
    db.execute_query("INSERT INTO images (image_id) VALUES (%s),(%s)", (plant_id, disease_id))
    db.execute_query(
        "INSERT INTO identification (sessionId,image_data,image_id) VALUES (%s,%s,%s),(%s,%s,%s)",
        (session_id, plant_b64, plant_id, session_id, disease_b64, disease_id)
    )

    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "plant_image_id": plant_id,
            "disease_image_id": disease_id,
            "plant_prediction": plant_result,
            "disease_prediction": disease_result
        }
    }


@app.get("/identification", tags=["identification"])
async def all_identifications():
    return db.fetch_all("SELECT * FROM images")

@app.get("/identification/{image_id}", tags=["identification"])
async def one_identification(image_id: str):
    rec = db.fetch_one("SELECT * FROM images WHERE image_id = %s", (image_id,))
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return rec


@app.get("/version", tags=["health"])
async def get_version():
    return {"version": "1.4.0"}


# ----------------------------------------
# 7) Uvicorn çalıştırma
# ----------------------------------------

if __name__ == "__main__":
    if "--dev" in sys.argv:
        uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8001)