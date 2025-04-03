FROM python:3.10-slim

# Instalacja bibliotek systemowych potrzebnych do działania OpenCV i Streamlit
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ustawienie katalogu roboczego
WORKDIR /app

# Skopiowanie plików projektu do kontenera
COPY . .

# Instalacja zależności Pythona
RUN pip install --upgrade pip
RUN pip install streamlit opencv-python-headless numpy pyyaml

# Otworzenie portu, na którym działa Streamlit
EXPOSE 8501

# Domyślne polecenie uruchamiające aplikację
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
