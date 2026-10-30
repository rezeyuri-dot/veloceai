FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    MODEL_PATH=/app/models/RealESRGAN_x4plus.pth \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY index.html /app/index.html
COPY colab_server.py /app/colab_server.py

RUN mkdir -p /app/models

EXPOSE 7860

CMD ["sh", "-c", "uvicorn colab_server:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
