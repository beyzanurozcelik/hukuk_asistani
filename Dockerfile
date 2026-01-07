# Dosya adı: Dockerfile

# Hafif ve uyumlu bir Python sürümü
FROM python:3.13-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksiz .pyc dosyalarının oluşmasını engelle ve logları anlık bas
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Import hatalarını önlemek için PYTHONPATH'e /app ekle
ENV PYTHONPATH=/app

# Sistem paketlerini güncelle (gerekirse build araçları için)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Önce requirements.txt kopyala (Cache avantajı için)
COPY requirements.txt .

# Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje kodlarını kopyala
COPY . .
