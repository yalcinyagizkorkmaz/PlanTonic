FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY .env .
COPY *.py .
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --upgrade pip

CMD ["python3", "main.py"]
