
# 语音转文字 + 说话人分离 —— Kimi-Audio 私有化离线部署方案（详细操作指南）

# AutoDL 服务器快速部署（RTX 4090 / PyTorch 2.8.0 / Python 3.12 / CUDA 12.8）


## 环境说明
  镜像：PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8
  GPU：RTX 4090 (24GB) × 1
  内存：90GB

注意：AutoDL 镜像已内置 PyTorch，无需重新安装 torch，直接安装依赖即可。

---

## 步骤1：SSH 登录服务器后，激活 base 环境（conda 24.4 已预装）

```bash
conda activate base
python --version   # 应显示 Python 3.12.x
```

---

## 步骤2：安装系统依赖与 Python 包

**重要**：Kimi-Audio 必须通过官方 `kimia_infer` 包推理，不能直接用 `transformers` 的标准接口。

1. 安装系统依赖（ffmpeg 支持 mp3/m4a/ogg 等格式）
```bash
apt-get update && apt-get install -y ffmpeg libsndfile1 git
```

2. 升级 pip
```bash
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 安装音频处理依赖
```bash
pip install torchaudio==2.8.0 soundfile librosa \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 安装 pyannote.audio（用于说话人分离）
```bash
pip install pyannote.audio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

5. （可选）若 torchcodec 报 UserWarning，可直接卸载（pyannote.audio 不强依赖它）
```bash
pip uninstall torchcodec -y
```

6. 安装 kimia_infer 推理包（Kimi-Audio 官方推理库）

**方案A（推荐）：AutoDL 开启学术加速后直接 pip 安装**
```bash
# 先在 AutoDL 控制台开启「学术资源加速」，再执行：
pip install git+https://github.com/MoonshotAI/Kimi-Audio.git
```

**方案B：关闭 HTTP/2 后 git clone**
```bash
git config --global http.version HTTP/1.1
git clone https://github.com/MoonshotAI/Kimi-Audio.git /tmp/Kimi-Audio
cd /tmp/Kimi-Audio && git submodule update --init --recursive
pip install -e /tmp/Kimi-Audio/kimia_infer/
```

**方案C：通过 gitee 镜像 clone（无需翻墙）**
```bash
git clone https://gitee.com/mirrors/Kimi-Audio.git /tmp/Kimi-Audio
cd /tmp/Kimi-Audio && git submodule update --init --recursive
pip install -e /tmp/Kimi-Audio/kimia_infer/
```

**方案D：服务器无法访问 GitHub，通过本地下载 ZIP 后上传到服务器离线安装**

> **背景**：Kimi-Audio 仓库中 `kimia_infer/` 是一个 git submodule，下载 ZIP 包时 submodule 内容不会被包含，
> 导致 `kimia_infer/` 目录为空（没有 `setup.py` / `pyproject.toml`），无法 pip install。
> 同时 `kimia_infer/models/tokenizer/glm4` 也是一个 submodule，指向 `THUDM/glm-4-voice-tokenizer`。
> 解决方法：在本地分别下载主仓库和所有 submodule 的 ZIP，上传到服务器后手动组装目录结构。

**在本地电脑上操作（可访问 GitHub）：**

需要下载以下两个 ZIP 包：
1. Kimi-Audio 主仓库：https://github.com/MoonshotAI/Kimi-Audio/archive/refs/heads/master.zip
2. GLM-4 语音 tokenizer（submodule）：https://huggingface.co/THUDM/glm-4-voice-tokenizer/resolve/main/speech_tokenizer/speech_tokenizer.onnx
   或整个 tokenizer 仓库 ZIP：https://github.com/THUDM/glm-4-voice-tokenizer/archive/refs/heads/main.zip
   （注意：如果该仓库不存在独立 ZIP，可通过 `git clone https://github.com/THUDM/glm-4-voice-tokenizer` 后打包上传）

实际操作步骤（推荐用 git clone 在本地完整获取后打包上传）：

```bash
# === 在本地电脑执行（可访问 GitHub）===

# 1. 完整 clone（含 submodule）
git clone https://github.com/MoonshotAI/Kimi-Audio.git /tmp/Kimi-Audio-full
cd /tmp/Kimi-Audio-full
git submodule update --init --recursive

# 2. 打包为 tar.gz（包含所有 submodule 内容）
cd /tmp
tar czf Kimi-Audio-full.tar.gz Kimi-Audio-full/

# 3. 上传到服务器（替换为你的服务器地址）
scp Kimi-Audio-full.tar.gz root@<服务器IP>:/tmp/
```

**在服务器上操作：**

```bash
# === 在服务器上执行 ===

# 1. 解压
cd /tmp
tar xzf Kimi-Audio-full.tar.gz
mv Kimi-Audio-full Kimi-Audio

# 2. 验证 submodule 内容已存在
ls /tmp/Kimi-Audio/kimia_infer/
# 应能看到 __init__.py, api/, models/ 等文件

ls /tmp/Kimi-Audio/kimia_infer/models/tokenizer/glm4/
# 应能看到 speech_tokenizer/ 等文件

# 3. 安装依赖（flash_attn 需要特殊处理，先跳过）
grep -v "flash_attn" /tmp/Kimi-Audio/requirements.txt > /tmp/requirements_no_flash.txt
pip install -r /tmp/requirements_no_flash.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 单独安装 flash_attn（--no-build-isolation 让编译时能找到已装的 torch）
#    编译需要 5~15 分钟，请耐心等待
pip install flash_attn==2.7.4.post1 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

todo
# 5. 安装 kimia_infer
pip install -e /tmp/Kimi-Audio/kimia_infer/

# 5. 验证安装成功
python -c "from kimia_infer.api.kimia import KimiAudio; print('kimia_infer OK')"
```

---

## 步骤3：下载 Kimi-Audio 模型权重

模型约 16GB（BF16），建议下载到数据盘：

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

## 步骤4：下载 pyannote 说话人分离模型

通过 ModelScope 下载，无需 HuggingFace Token，国内访问更稳定。

```bash
python -c "
from modelscope import snapshot_download
snapshot_download(
    'pyannote/speaker-diarization-3.1',
    cache_dir='/root/autodl-tmp/models'
)
print('pyannote 模型下载完成')
"
```

下载完成后模型路径为：
`/root/autodl-tmp/models/pyannote/speaker-diarization-3.1`

加载时指定本地路径：

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    '/root/autodl-tmp/models/pyannote/speaker-diarization-3.1'
)
```

---

## 步骤5：验证安装

```bash
python -c "from kimia_infer.api.kimia import KimiAudio; print('kimia_infer OK')"
python -c "from pyannote.audio import Pipeline; print('pyannote OK')"
python -c "import torch; print('torch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

期望输出示例：
```
kimia_infer OK
pyannote OK
torch 版本: 2.8.0+cu128
CUDA 可用: True
GPU: NVIDIA GeForce RTX 4090 D
```

---

## 步骤6：运行 ASR 脚本（语音转文字 + 说话人分离）

将以下脚本保存为 `asr_kimi.py`，修改 `AUDIO_FILE`、`MODEL_PATH`、`HF_TOKEN` 后运行。

```python
import os
import torch
import soundfile as sf
import librosa
from kimia_infer.api.kimia import KimiAudio
from pyannote.audio import Pipeline

# ====== 配置区 ======
AUDIO_FILE      = '/root/autodl-tmp/audio/3.mp3'         # 输入音频/视频文件路径
MODEL_PATH      = '/root/autodl-tmp/models/moonshotai/Kimi-Audio-7B-Instruct'
PYANNOTE_PATH   = '/root/autodl-tmp/models/pyannote/speaker-diarization-3.1'
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
SEGMENT_SECS    = 30                                      # 每段最大秒数（防止显存溢出）
# ====================


def ms_to_time(ms):
    h = int(ms // 3600000)
    m = int((ms % 3600000) // 60000)
    s = int((ms % 60000) // 1000)
    return f'{h:02d}:{m:02d}:{s:02d}' if h > 0 else f'{m:02d}:{s:02d}'


def load_audio(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav, sr


def main():
    print('=== 加载 Kimi-Audio 模型 ===')
    model = KimiAudio(model_path=MODEL_PATH, load_detokenizer=False)

    sampling_params = {
        'audio_temperature': 0.8,
        'audio_top_k': 10,
        'text_temperature': 0.0,
        'text_top_k': 5,
        'audio_repetition_penalty': 1.0,
        'audio_repetition_window_size': 64,
        'text_repetition_penalty': 1.0,
        'text_repetition_window_size': 16,
    }

    print('=== 加载说话人分离模型 ===')
    diarization_pipeline = Pipeline.from_pretrained(
        PYANNOTE_PATH
    ).to(torch.device(DEVICE))

    print('=== 执行说话人分离 ===')
    diarization = diarization_pipeline(AUDIO_FILE)

    wav, sr = load_audio(AUDIO_FILE, sr=16000)
    total_samples = len(wav)

    print('=== 逐段 ASR 识别 ===')
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_s = turn.start
        end_s   = turn.end
        duration = end_s - start_s
        if duration < 0.3:
            continue

        # 分段截取，防止过长片段导致显存溢出
        seg_start = start_s
        while seg_start < end_s:
            seg_end = min(seg_start + SEGMENT_SECS, end_s)

            start_idx = int(seg_start * sr)
            end_idx   = min(int(seg_end * sr), total_samples)
            segment_wav = wav[start_idx:end_idx]

            tmp_path = '/tmp/_kimia_segment.wav'
            sf.write(tmp_path, segment_wav, sr)

            messages = [
                {'role': 'user', 'message_type': 'text',  'content': '请转写以下音频内容：'},
                {'role': 'user', 'message_type': 'audio', 'content': tmp_path},
            ]
            _, text = model.generate(messages, **sampling_params, output_type='text')
            text = text.strip()

            if text:
                results.append({
                    'start_ms': int(seg_start * 1000),
                    'end_ms':   int(seg_end   * 1000),
                    'speaker':  speaker,
                    'text':     text,
                })

            seg_start = seg_end

    # 合并同一说话人连续片段
    speaker_map = {}
    merged = []
    prev = None
    for r in results:
        spk = r['speaker']
        if spk not in speaker_map:
            speaker_map[spk] = len(speaker_map) + 1
        r['spk_idx'] = speaker_map[spk]

        if prev and prev['spk_idx'] == r['spk_idx'] and r['start_ms'] - prev['end_ms'] < 1500:
            prev['text']   += r['text']
            prev['end_ms']  = r['end_ms']
        else:
            if prev:
                merged.append(prev)
            prev = dict(r)
    if prev:
        merged.append(prev)

    print('\n=== 识别结果 ===')
    for r in merged:
        ts = f"[{ms_to_time(r['start_ms'])}]-[{ms_to_time(r['end_ms'])}]"
        print(f"{ts} 用户{r['spk_idx']}： {r['text']}")


if __name__ == '__main__':
    main()
```

运行命令：
```bash
python asr_kimi.py
```

---

## 输出示例

```
=== 识别结果 ===
[00:16]-[00:21] 用户1： 辣椒精来吧，老婆，你终于从国外回来了，
[00:22]-[00:23] 用户2： 怎么想我了？
[00:23]-[00:33] 用户1： 是呀，你就答应我吧，在公司做一个闲职，每天看不到你我都心慌难受。
[00:34]-[00:38] 用户2： 好吧，不过你得给我安排个适合的部门，没问题。
```

---

## 注意事项

- **workers 只能设为 1**：Kimi-Audio 模型约占 16GB 显存，RTX 4090 (24GB) 仅支持单进程推理。
- **pyannote 首次运行需联网**：若服务器无法访问 HuggingFace，请提前预下载模型后离线使用。
- **推理时间**：说话人分离 + 逐段转写，1分钟音频约需 30～90 秒（依说话人数量而定）。
- **支持格式**：mp3 / wav / m4a / flac / ogg / aac（需安装 ffmpeg）。
- **视频文件**：先用 ffmpeg 提取音轨再输入：
  ```bash
  ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
  ```
