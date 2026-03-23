FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

RUN pip install --no-cache-dir \
    git+https://github.com/MoonshotAI/Kimi-Audio.git#subdirectory=kimia_infer

COPY app/ ./app/

VOLUME ["/root/autodl-tmp/models"]

EXPOSE 8000

CMD ["gunicorn", "app.main:app", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "600", \
     "--keep-alive", "75", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
