FROM python:3.12-slim-bullseye

WORKDIR /app

# why the hell are there so many dependencies bruh Python libs man
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    python3-dev \
    libpng-dev \
    libfreetype6-dev \
    libhdf5-dev \
    libjpeg-dev \
    libffi-dev \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "-m", "flask", "--app", "MLApp", "run", "--debug", "--host=0.0.0.0"]