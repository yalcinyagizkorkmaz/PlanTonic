import tensorflow as tf
import numpy as np
from PIL import Image
import json


class TFLiteModelBase:
    def __init__(self, model_path: str, label_map_path: str, input_size: tuple = (224, 224)):
        """
        model_path: .tflite dosya yolu
        label_map_path: JSON dosyası, {"class_name": index, ...}
        """
        self.input_size = input_size

        # Modeli yükle
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        inp, out = self.interpreter.get_input_details()[0], self.interpreter.get_output_details()[0]
        self.input_index = inp['index']
        self.output_index = out['index']
        self.quantization = inp.get('quantization', (0.0, 1.0))

        # label_map → index_to_class dönüşümü
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        self.index_to_class = {v: k for k, v in label_map.items()}

    def _preprocess(self, img_path: str) -> np.ndarray:
        # PIL ile aç, RGB, yeniden boyutla ve float32 → [0,1]
        img = Image.open(img_path).convert('RGB').resize(self.input_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, img_path: str) -> (str, float):
        """
        Tek bir görüntü için yorum yapar:
        - döndürür: (class_name, confidence)
        """
        data = self._preprocess(img_path)
        scale, zero_point = self.quantization

        # Eğer quantize edilmişse dönüştür
        if scale != 0:
            data = data / scale + zero_point
            data = data.astype(np.uint8)

        self.interpreter.set_tensor(self.input_index, data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)[0]

        # Uint8 → float dönüşümü
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
        """
        launch_data: {class_name: {"name": [...], "health_status": ...}, ...}
        """
        super().__init__(model_path, label_map_path, input_size)
        self.launch_data = launch_data

    def identify(self, image_path: str, unhealthy_threshold: float = 0.40) -> dict:
        """
        - `unhealthy_threshold`: eşiğin altındaki confidence'ları 'Unhealthy' olarak işaretler.
        """
        cls, conf = self.predict(image_path)
        info = self.launch_data.get(cls, {"name": [cls], "health_status": "unknown"})
        
        # Eğer confidence çok düşükse
        display_label = cls
        if conf < unhealthy_threshold:
            display_label = "Unhealthy"

        return {
            "class_id": cls,
            "confidence": conf,
            "display_label": display_label,
            "names": info["name"],
            "health_status": info["health_status"]
        }