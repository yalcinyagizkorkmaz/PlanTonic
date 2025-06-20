FROM python:3.12-slim

WORKDIR /app

# Paket kurulumu ve pip güncelleme tek RUN içinde, apt temizliği de yapılıyor
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Gereksiz dosya kopyalamadan önce sadece ihtiyaç duyulanları taşı
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --upgrade pip

# .env ve python dosyalarını kopyala
COPY .env ./
COPY *.py ./

# models klasörünü kopyala
COPY models/ ./models/



CMD ["python3", "main.py"]