import tensorflow as tf
import numpy as np
from PIL import Image
import json
import openai
import os
import warnings
from dotenv import load_dotenv

# TensorFlow Lite uyarısını gizle
warnings.filterwarnings('ignore', message='.*tf.lite.Interpreter is deprecated.*')


class TFLiteModelBase:
    def __init__(self, model_path: str, label_map_path: str, input_size: tuple = (224, 224)):
        self.input_size = input_size
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        inp, out = self.interpreter.get_input_details()[0], self.interpreter.get_output_details()[0]
        self.input_index = inp['index']
        self.output_index = out['index']
        self.quantization = inp.get('quantization', (0.0, 1.0))

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

    def disease(self, image_path: str) -> dict:
        lbl, conf = self.predict(image_path)
        return {"label": lbl, "confidence": conf}


class PlantIdentificationClassifier(TFLiteModelBase):
    def __init__(self, model_path: str, label_map_path: str, launch_data: dict, input_size: tuple = (224, 224)):
        super().__init__(model_path, label_map_path, input_size)
        self.launch_data = launch_data

    def identify(self, image_path: str, confidence_threshold: float = 0.30, unhealthy_threshold: float = 0.45) -> dict:
        cls, conf = self.predict(image_path)
        
        # Confidence çok düşükse tanımlanamadı
        if conf < confidence_threshold:
            return {
                "class_id": "Unknown",
                "confidence": conf,
                "display_label": "Tanımlanamadı",
                "names": ["Tanımlanamadı"],
                "health_status": "unknown",
                "identified": False
            }
        
        # Eğer confidence yeterliyse
        display_label = cls
        if conf < unhealthy_threshold:
            display_label = "Unhealthy"

        return {
            "class_id": cls,
            "confidence": conf,
            "display_label": display_label,
            "names": [cls],
            "health_status": "healthy" if conf >= unhealthy_threshold else "unhealthy",
            "identified": True
        }


class PlantLLMGenerator:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY environment variable is not set!")
            self.client = None
        else:
            openai.api_key = api_key
            self.client = openai

    def ask_gpt(self, prompt):
        try:
            if not self.client:
                return "OpenAI API key is not configured. Please set OPENAI_API_KEY environment variable."
                
            print(f"DEBUG: Gelen prompt: {prompt}")
            # Prompt'tan bitki bilgilerini çıkar
            if "Bitki türü:" in prompt:
                plant_info = prompt.split("Bitki türü:")[1].split(",")[0].strip()
                disease_info = prompt.split("Hastalık:")[1].strip() if "Hastalık:" in prompt else "Unknown"
                
                print(f"DEBUG: plant_info: '{plant_info}'")
                print(f"DEBUG: disease_info: '{disease_info}'")
                
                # Tanımlanamadı durumu için özel mesaj
                if "Tanımlanamadı" in plant_info:
                    print("DEBUG: Tanımlanamadı durumu tespit edildi")
                    return "We couldn't identify your plant, but in general, plants are living organisms that produce energy through photosynthesis, take root in soil, and contribute to the ecosystem by providing oxygen. With their various species, they maintain the balance of nature and provide both food and habitat for humans."
                else:
                    print("DEBUG: Normal bitki durumu")
                    # Hastalık durumunu kontrol et
                    is_diseased = "unhealthy" in prompt or "Hastalık:" in prompt and "Unknown" not in disease_info
                    
                    if is_diseased:
                        system_prompt = """You are a plant expert. Provide detailed care and treatment recommendations for diseased plants.
                        Focus especially on: Watering frequency and amount, Light requirements (direct/indirect sunlight), Soil type, Temperature preferences, Disease treatment, General care recommendations
                        
                        Write your sentences as plain text, don't use numbering. Give each recommendation in separate sentences.
                        Use the plant name and disease name in sentences. For example: "Your Aloe Vera plant appears to have Rust disease."."""
                        
                        user_prompt = f"Plant type: {plant_info}, Disease: {disease_info}. Provide detailed care and treatment recommendations for this {plant_info} plant's {disease_info} disease."
                    else:
                        system_prompt = """You are a plant expert. Provide detailed information about the given plant.
                        Focus especially on: Watering frequency and amount, Light requirements (direct/indirect sunlight), Soil type, Temperature preferences, General care recommendations
                        
                        Write your sentences in directive style, not in question-answer format."""
                    
                        user_prompt = f"Plant type: {plant_info}, Disease status: {disease_info}. Provide detailed care information about this plant."
                
                print(f"DEBUG: system_prompt: {system_prompt}")
                print(f"DEBUG: user_prompt: {user_prompt}")
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                # \n ve - karakterlerini kaldır ve temizle
                result = response.choices[0].message.content.replace('\n', ' ').replace(' - ', '. ').replace('- ', '').replace('  ', ' ').strip()
                
                # Hastalıklı bitkiler için numaralandırmayı kaldır
                if is_diseased:
                    # Numaralandırılmış liste formatını kaldır (1. 2. 3. gibi)
                    import re
                    result = re.sub(r'^\d+\.\s*', '', result, flags=re.MULTILINE)
                    result = re.sub(r'\s+\d+\.\s*', '. ', result)
                    result = result.replace('..', '.').strip()
                
                print(f"DEBUG: Sonuç: {result}")
                return result
            else:
                print("DEBUG: 'Bitki türü:' bulunamadı")
                return "Plant information could not be obtained."
                
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            print(f"DEBUG: Exception detayı: {type(e).__name__}")
            return "Sorry, I cannot provide plant information at the moment. Please try again later."
