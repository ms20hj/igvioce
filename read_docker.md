# FunASR 语音识别 —— Docker 服务化部署方案

## 方案概述

基于 FastAPI + FunASR 构建 HTTP 服务，支持：
- 文件上传触发语音识别
- **Server-Sent Events (SSE) 流式逐句返回**识别结果
- Gunicorn + Uvicorn 多进程高并发处理
- Docker 容器化部署，GPU 直通

---

## 目录结构

```
asr-service/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app/
│   ├── main.py          # FastAPI 主服务
│   └── asr_engine.py    # FunASR 模型封装
└── nginx/
    └── nginx.conf       # 反向代理 + 负载均衡
```

---

## 步骤1：创建 FunASR 模型封装

**`app/asr_engine.py`**

```python
import asyncio
from funasr import AutoModel

MODEL_DIR = '/root/autodl-tmp/models'

def load_model():
    return AutoModel(
        model='paraformer-zh',
        vad_model='fsmn-vad',
        punc_model='ct-punc',
        spk_model='cam++',
        device='cuda',
    )

def ms_to_time(ms):
    h = int(ms // 3600000)
    m = int((ms % 3600000) // 60000)
    s = int((ms % 60000) // 1000)
    return f'{h:02d}:{m:02d}:{s:02d}' if h > 0 else f'{m:02d}:{s:02d}'

def recognize_stream(model, audio_path: str):
    """
    生成器函数：逐句 yield 识别结果，供 SSE 流式推送。
    每条结果格式：[开始时间]-[结束时间] 用户N： 文字
    """
    result = model.generate(input=audio_path, batch_size_s=300)
    for res in result:
        sentences = res.get('sentence_info', [])
        if sentences:
            current_spk = None
            current_text = ''
            current_start = 0
            current_end = 0
            for sent in sentences:
                spk = sent.get('spk', 0)
                text = sent.get('text', '')
                start = sent.get('start', 0)
                end = sent.get('end', start)
                if spk != current_spk:
                    if current_text:
                        yield f'[{ms_to_time(current_start)}]-[{ms_to_time(current_end)}] 用户{current_spk + 1}： {current_text}'
                    current_spk = spk
                    current_text = text
                    current_start = start
                    current_end = end
                else:
                    current_text += text
                    current_end = end
            if current_text:
                yield f'[{ms_to_time(current_start)}]-[{ms_to_time(current_end)}] 用户{current_spk + 1}： {current_text}'
        else:
            full_text = res.get('text', '')
            if full_text:
                yield full_text
```

---

## 步骤2：创建 FastAPI 主服务

**`app/main.py`**

```python
import os
import uuid
import asyncio
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.asr_engine import load_model, recognize_stream

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print('正在加载 FunASR 模型...')
    model = load_model()
    print('模型加载完成，服务就绪。')
    yield
    print('服务关闭。')

app = FastAPI(title='FunASR 语音识别服务', lifespan=lifespan)

ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


@app.get('/health')
async def health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/asr/stream')
async def asr_stream(file: UploadFile = File(...)):
    """
    上传音频文件，以 SSE 流式返回逐句识别结果。

    响应格式（text/event-stream）：
      data: [00:16]-[00:21] 用户1： 你好，欢迎使用语音识别服务。\n\n
      data: [DONE]\n\n
    """
    if model is None:
        raise HTTPException(status_code=503, detail='模型尚未加载完成，请稍后重试')

    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f'不支持的文件格式：{ext}，支持格式：{", ".join(ALLOWED_EXTENSIONS)}'
        )

    # 将上传文件写入临时目录
    tmp_path = os.path.join(tempfile.gettempdir(), f'{uuid.uuid4()}{ext}')
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail='文件大小超过 500MB 限制')
        with open(tmp_path, 'wb') as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'文件保存失败：{str(e)}')

    async def event_generator():
        loop = asyncio.get_event_loop()
        try:
            # 在线程池中运行同步推理，避免阻塞事件循环
            def run_inference():
                return list(recognize_stream(model, tmp_path))

            sentences = await loop.run_in_executor(None, run_inference)
            for sentence in sentences:
                yield f'data: {sentence}\n\n'
                await asyncio.sleep(0)  # 让出控制权，保证 SSE 帧实时推送
        except Exception as e:
            yield f'data: [ERROR] {str(e)}\n\n'
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            yield 'data: [DONE]\n\n'

    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',   # 关闭 Nginx 缓冲，确保 SSE 实时推送
        },
    )
```

---

## 步骤3：依赖文件

**`requirements.txt`**

```
funasr>=1.0.0
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
gunicorn>=22.0.0
python-multipart>=0.0.9
```

---

## 步骤4：编写 Dockerfile

**`Dockerfile`**

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 拷贝应用代码
COPY app/ ./app/

# 模型目录（宿主机挂载）
VOLUME ["/root/autodl-tmp/models"]

EXPOSE 8000

# Gunicorn 多进程 + Uvicorn Worker，实现高并发
# --workers 建议设置为 CPU 核心数，GPU 推理建议 1~2（避免显存不足）
CMD ["gunicorn", "app.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "600", \
     "--keep-alive", "75", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

> ⚠️ **GPU 显存说明**：FunASR 模型（paraformer + VAD + punc + cam++）约占用 **3~4GB 显存**。
> RTX 4090（24GB）最多可安全运行 **4 个 Worker**，每个 Worker 独立加载模型。
> 如显存不足，可将 `--workers` 改为 `1`，通过 Nginx 横向扩展多容器实例。

---

## 步骤5：编写 docker-compose.yml

**`docker-compose.yml`**

```yaml
version: '3.9'

services:
  asr:
    build: .
    image: funasr-asr-service:latest
    container_name: asr-service
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - '8000:8000'
    volumes:
      # 挂载已下载的模型目录，避免容器内重复下载
      - /root/autodl-tmp/models:/root/autodl-tmp/models:ro
      # 临时文件目录
      - /tmp:/tmp
    environment:
      - CUDA_VISIBLE_DEVICES=0
    shm_size: '2gb'

  nginx:
    image: nginx:1.25-alpine
    container_name: asr-nginx
    restart: unless-stopped
    ports:
      - '80:80'
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - asr
```

---

## 步骤6：Nginx 反向代理配置（支持 SSE）

**`nginx/nginx.conf`**

```nginx
worker_processes auto;

events {
    worker_connections 4096;
}

http {
    upstream asr_backend {
        # 如横向扩展多个容器实例，在此添加多行 server
        server asr:8000;
        keepalive 64;
    }

    server {
        listen 80;

        # 上传文件大小限制（与服务端保持一致）
        client_max_body_size 500M;

        # SSE 专用 location：关闭缓冲，保证实时推送
        location /asr/stream {
            proxy_pass         http://asr_backend;
            proxy_http_version 1.1;
            proxy_set_header   Host $host;
            proxy_set_header   X-Real-IP $remote_addr;

            # 关闭 Nginx 缓冲，SSE 必须配置
            proxy_buffering        off;
            proxy_cache            off;
            proxy_read_timeout     600s;
            proxy_send_timeout     600s;
            chunked_transfer_encoding on;
        }

        location / {
            proxy_pass         http://asr_backend;
            proxy_http_version 1.1;
            proxy_set_header   Host $host;
            proxy_set_header   X-Real-IP $remote_addr;
            proxy_read_timeout 600s;
        }
    }
}
```

---

## 步骤7：构建并启动服务

```bash
# 1. 确保宿主机已安装 NVIDIA Container Toolkit
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 2. 构建镜像
docker compose build

# 3. 后台启动服务
docker compose up -d

# 4. 查看启动日志（等待模型加载完成）
docker compose logs -f asr

# 5. 健康检查
curl http://localhost/health
# 期望返回：{"status":"ok","model_loaded":true}
```

---

## 步骤8：接口调用示例

### curl 调用（流式接收）

```bash
curl -N -X POST http://localhost/asr/stream \
  -F "file=@/path/to/your/audio.mp3" \
  --no-buffer
```

**流式响应示例：**
```
data: [00:16]-[00:21] 用户1： 辣椒精来吧，老婆，你终于从国外回来了，

data: [00:22]-[00:23] 用户2： 怎么想我了？

data: [00:23]-[00:33] 用户1： 是呀，你就答应我吧，在公司做一个闲职，

data: [DONE]
```

### Python 客户端（流式读取）

```python
import requests

url = 'http://localhost/asr/stream'

with open('/path/to/audio.mp3', 'rb') as f:
    with requests.post(url, files={'file': f}, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    content = decoded[6:]
                    if content == '[DONE]':
                        print('=== 识别完成 ===')
                        break
                    elif content.startswith('[ERROR]'):
                        print(f'识别错误：{content}')
                        break
                    else:
                        print(content)
```

### JavaScript / 浏览器（EventSource + FormData）

```javascript
async function recognizeAudio(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/asr/stream', {
        method: 'POST',
        body: formData,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const content = line.slice(6);
                if (content === '[DONE]') {
                    console.log('识别完成');
                    return;
                }
                if (content && !content.startsWith('[ERROR]')) {
                    console.log(content); // 实时打印每句话
                }
            }
        }
    }
}
```

---

## 高并发优化说明

| 优化项 | 说明 |
|--------|------|
| **Gunicorn 多 Worker** | 每个 Worker 是独立进程，互不阻塞，充分利用多核 CPU |
| **UvicornWorker** | 异步事件循环，单 Worker 内可处理多路 I/O 并发 |
| **run_in_executor** | GPU 推理在线程池运行，不阻塞事件循环，SSE 帧可实时推送 |
| **Nginx keepalive** | 长连接复用，减少连接建立开销 |
| **proxy_buffering off** | Nginx 不缓冲 SSE 响应，保证实时性 |
| **模型复用** | 模型在服务启动时加载一次，后续请求直接复用，避免重复加载 |
| **Volume 只读挂载** | 模型目录以 `:ro` 挂载，避免误写，提升安全性 |

---

## 横向扩展（多 GPU / 多实例）

如服务器有多块 GPU 或需要更高并发，可通过 `docker compose scale` 横向扩展：

```yaml
# docker-compose.yml 中为每个实例分配不同 GPU
services:
  asr-0:
    build: .
    environment:
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - '8001:8000'

  asr-1:
    build: .
    environment:
      - CUDA_VISIBLE_DEVICES=1
    ports:
      - '8002:8000'
```

在 `nginx.conf` 中添加多个 upstream server：

```nginx
upstream asr_backend {
    server asr-0:8000;
    server asr-1:8000;
    keepalive 64;
}
```

---

## 常见问题

**Q：容器内找不到 GPU？**
```bash
# 检查 NVIDIA Container Toolkit 是否安装
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

**Q：模型加载很慢？**
> 首次启动需从 ModelScope 下载模型，建议提前在宿主机下载好（参考 README.md 步骤4），
> 再通过 Volume 挂载进容器，启动时仅加载不下载，速度大幅提升。

**Q：SSE 在 Nginx 后不实时？**
> 确认 `nginx.conf` 中 `/asr/stream` location 已设置 `proxy_buffering off`，
> 以及服务端响应头包含 `X-Accel-Buffering: no`。

**Q：并发请求显存不足？**
> 将 Dockerfile 中 `--workers` 调整为 `1`，通过多容器 + 多 GPU 方式横向扩展。
