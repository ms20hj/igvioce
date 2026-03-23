# 语音转文字 + 说话人分离 —— Kimi-Audio-7B-Instruct 私有化离线部署方案

# AutoDL 服务器快速部署（RTX 4090 / PyTorch 2.1.0 / Python 3.10 / CUDA 12.1）

## 环境说明
  镜像：PyTorch 2.1.0 / Python 3.10 / Ubuntu 22.04 / CUDA 12.1
  GPU：RTX 4090 (24GB) × 1
  内存：90GB

---

## 步骤1：SSH 登录服务器后，激活 base 环境

```bash
conda activate base
python --version   # 应显示 Python 3.10.x
```

---

## 步骤2：下载 Kimi-Audio 模型权重

模型约 16GB，建议下载到数据盘：

```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "
from modelscope import snapshot_download
snapshot_download(
    'moonshotai/Kimi-Audio-7B-Instruct',
    cache_dir='/root/autodl-tmp/models'
)
print('模型下载完成')
"
```

下载完成后模型路径为：
`/root/autodl-tmp/models/moonshotai/Kimi-Audio-7B-Instruct`

---

## 步骤3：下载 pyannote 说话人分离模型

pyannote/speaker-diarization-3.1 需要 HuggingFace 访问令牌（免费申请）。

1. 注册 https://huggingface.co 并申请访问 `pyannote/speaker-diarization-3.1` 模型
2. 在 https://huggingface.co/settings/tokens 创建 Access Token
3. 将 Token 填入 docker-compose.yml 的环境变量，或在启动时传入：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx
```

也可手动预下载到本地（推荐，离线运行）：

```bash
pip install pyannote.audio -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "
import os
os.environ['HF_TOKEN'] = 'hf_xxxxxxxxxxxxxxxxxxxxxx'
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',
    use_auth_token=os.environ['HF_TOKEN']
)
print('pyannote 模型下载完成')
"
```

---

## 步骤4：安装 kimia_infer 推理包

Kimi-Audio 必须通过官方 `kimia_infer` 包推理：

```bash
apt-get update && apt-get install -y ffmpeg libsndfile1 git

pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchaudio==2.1.0 soundfile librosa pyannote.audio \
    fastapi uvicorn[standard] gunicorn python-multipart \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

git clone https://github.com/MoonshotAI/Kimi-Audio.git /tmp/Kimi-Audio
cd /tmp/Kimi-Audio && git submodule update --init
pip install -e kimia_infer/
```

---

## 步骤5：验证安装

```bash
python -c "from kimia_infer.api.kimia import KimiAudio; print('kimia_infer OK')"
python -c "from pyannote.audio import Pipeline; print('pyannote OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 步骤6：Docker 方式启动服务

```bash
# 在服务器上
cd /path/to/kimi-asr-service

# 设置 HF Token（用于 pyannote 模型，若已预下载可留空）
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx

docker compose up -d --build
```

服务启动后访问：
- 健康检查：`GET http://localhost/health`
- 语音识别：`POST http://localhost/asr/stream`

---

## 步骤7：裸机方式启动服务（无 Docker）

```bash
cd /path/to/kimi-asr-service

export KIMI_AUDIO_MODEL_PATH=/root/autodl-tmp/models/moonshotai/Kimi-Audio-7B-Instruct
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx

gunicorn app.main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 600
```

---

## API 使用示例

### curl 调用

```bash
curl -N -X POST http://localhost:8000/asr/stream \
     -F "file=@/path/to/audio.mp3"
```

### Python 调用

```python
import requests

with open('/path/to/audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/asr/stream',
        files={'file': ('audio.mp3', f, 'audio/mpeg')},
        stream=True,
    )

for line in response.iter_lines():
    if line:
        text = line.decode('utf-8')
        if text.startswith('data: '):
            content = text[6:]
            if content == '[DONE]':
                break
            print(content)
```

### 输出示例

```
[00:16]-[00:21] 用户1： 辣椒精来吧，老婆，你终于从国外回来了，
[00:22]-[00:23] 用户2： 怎么想我了？
[00:23]-[00:33] 用户1： 是呀，你就答应我吧，在公司做一个闲职，每天看不到你我都心慌难受。是啊，你就答应十年吧。
```

---

## 注意事项

- **workers 只能设为 1**：Kimi-Audio 模型占用约 16GB 显存，RTX 4090 (24GB) 仅支持单进程推理。
- **pyannote 首次运行需联网**：若服务器无法访问 HuggingFace，请提前在有网络的机器上预下载模型，再上传到服务器。
- **推理时间**：说话人分离 + 逐段转写，1分钟音频约需 30~90 秒（依说话人数量而定）。
- **支持格式**：mp3 / wav / m4a / flac / ogg / aac，最大 500MB。
