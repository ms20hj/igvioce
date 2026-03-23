# 语音转文字 + 说话人分离 —— Paraformer-Large 私有化离线部署方案

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

## 步骤2：下载 Paraformer-Large 模型权重

FunASR 会在首次运行时自动从 ModelScope 下载模型，也可提前手动下载：

```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "
from modelscope import snapshot_download
# 主 ASR 模型
snapshot_download(
    'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    cache_dir='/root/autodl-tmp/models'
)
# VAD 模型
snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', cache_dir='/root/autodl-tmp/models')
# 标点模型
snapshot_download('iic/punc_ct-transformer_cn-en-common-vocab471067-large', cache_dir='/root/autodl-tmp/models')
# 说话人分离模型
snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common', cache_dir='/root/autodl-tmp/models')
print('所有模型下载完成')
"
```

模型默认缓存路径（FunASR 自动管理）：
`~/.cache/modelscope/hub/iic/`

---

## 步骤3：安装依赖

```bash
apt-get update && apt-get install -y ffmpeg libsndfile1

pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install funasr>=1.0.0 fastapi uvicorn[standard] gunicorn python-multipart \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 步骤4：验证安装

```bash
python -c "from funasr import AutoModel; print('funasr OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 步骤5：Docker 方式启动服务

```bash
# 在服务器上
cd /path/to/asr-service

docker compose up -d --build
```

服务启动后访问：
- 健康检查：`GET http://localhost/health`
- 语音识别（流式）：`POST http://localhost/asr/stream`
- 语音识别（非流式）：`POST http://localhost/asr`

---

## 步骤6：裸机方式启动服务（无 Docker）

```bash
cd /path/to/asr-service
mkdir -p logs

# 后台启动（SSH 断开后服务不会停止）
nohup gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:6006 \
    --timeout 3000 \
    --keep-alive 75 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --pid logs/gunicorn.pid \
    --daemon

# 查看状态
cat logs/gunicorn.pid          # 查看主进程 PID
tail -f logs/error.log         # 实时查看日志

# 停止服务
kill $(cat logs/gunicorn.pid)

```

---

## API 使用示例

### curl 调用

```bash
# 流式返回
curl -N -X POST http://localhost:6006/asr/stream \
     -F "file=@/path/to/audio.mp3"

# 流式 + 热词
curl -N -X POST http://localhost:6006/asr/stream \
     -F "file=@/path/to/audio.mp3" \
     -F "hotword=阿里巴巴 达摩院 人工智能"

# 非流式返回（JSON 列表）
curl -X POST http://localhost:6006/asr \
     -F "file=@/root/autodl-tmp/audio/3.mp3"
```

### Python 调用

#### 流式调用

```python
import requests

with open('/path/to/audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:6000/asr/stream',
        files={'file': ('audio.mp3', f, 'audio/mpeg')},
        data={'hotword': '阿里巴巴 达摩院'},
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

#### 非流式调用

```python
import requests

with open('/path/to/audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:6000/asr',
        files={'file': ('audio.mp3', f, 'audio/mpeg')},
        data={'hotword': '阿里巴巴 达摩院'},
    )

result = response.json()
for item in result:
    print(f"[{item['start']}]-[{item['end']}] {item['speaker']}： {item['text']}")
```

### 输出示例

#### 流式（SSE）

```
data: [00:16]-[00:21] 用户1： 辣椒精来吧，老婆，你终于从国外回来了，
data: [00:22]-[00:23] 用户2： 怎么想我了？
data: [00:23]-[00:33] 用户1： 是呀，你就答应我吧，在公司做一个闲职，每天看不到你我都心慌难受。
data: [DONE]
```

#### 非流式（JSON）

```json
[
  {"start": "00:16", "end": "00:21", "speaker": "用户1", "text": "辣椒精来吧，老婆，你终于从国外回来了，"},
  {"start": "00:22", "end": "00:23", "speaker": "用户2", "text": "怎么想我了？"},
  {"start": "00:23", "end": "00:33", "speaker": "用户1", "text": "是呀，你就答应我吧，在公司做一个闲职，每天看不到你我都心慌难受。"}
]
```

---

## 注意事项

- **workers 可设为 4**：Paraformer-Large 模型占用显存较少（约 1~2GB），RTX 4090 (24GB) 支持多进程并发推理。
- **模型首次启动自动下载**：FunASR 会自动从 ModelScope 拉取模型，首次启动需联网；离线环境请提前手动下载。
- **热词支持**：通过 `hotword` 参数传入空格分隔的词语，可提升特定词汇的识别准确率。
- **推理时间**：VAD + 标点 + 说话人分离全流程，1分钟音频约需 10~30 秒（依硬件性能而定）。
- **支持格式**：mp3 / wav / m4a / flac / ogg / aac，最大 500MB。
