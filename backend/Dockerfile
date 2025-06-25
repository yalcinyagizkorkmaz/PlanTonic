FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (pycairo ve benzeri için)
RUN apt-get update && apt-get install -y \
    gcc \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Environment variables


# Gereksinim dosyasını kopyala
COPY requirements.txt .

# Python bağımlılıklarını yükle
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Proje dosyalarını kopyala (models klasörü dahil)
COPY . .

# (Opsiyonel) Build aşamasında models klasörünü doğrula
RUN ls -l models/

# Uygulamayı başlat
CMD ["python3", "main.py"]
