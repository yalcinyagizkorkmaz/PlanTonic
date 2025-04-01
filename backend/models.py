from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str


class User(BaseModel):
    id: str
    email: str
    created_at: datetime



class ImageCreate(BaseModel):
    image_data: str
    plant_type: str
    session_id: str
    user_id: str

class Image(BaseModel):
    image_id: str
    image_data: str
    plant_type: str
    session_id: str
    user_id: str

class DiseaseCreate(BaseModel):
    patojen_name: str
    symptom: str

class Disease(BaseModel):
    patojen_id: str
    patojen_name: str
    symptom: str

class LabelCreate(BaseModel):
    image_id: str
    disease_name: str

class Label(BaseModel):
    label_id: str
    image_id: str
    disease_name: str
