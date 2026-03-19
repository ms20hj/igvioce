
   语音转文字 + 说话人分离 —— 私有化离线部署方案（详细操作指南）

   AutoDL 服务器快速部署（RTX 4090 / PyTorch 2.8.0 / Python 3.12 / CUDA 12.8）

【环境说明】
  镜像：PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8
  GPU：RTX 4090 (24GB) × 1
  内存：90GB

注意：AutoDL 镜像已内置 PyTorch，无需重新安装 torch，直接安装 funasr 即可。

------------------------------------------------------------
步骤1：SSH 登录服务器后，激活 base 环境（conda 24.4 已预装）
------------------------------------------------------------

conda activate base
python --version   # 应显示 Python 3.12.x

------------------------------------------------------------
步骤2：安装 FunASR 及依赖
------------------------------------------------------------

# 升级 pip
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 funasr（使用清华镜像加速）
pip install -U funasr -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装音频处理依赖（torchaudio 必须与 torch 版本严格匹配）
pip install torchaudio==2.8.0 soundfile librosa -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 modelscope（用于下载模型）
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 ffmpeg（支持 mp3/m4a 等格式）
# 注意：必须先更新包列表，否则会报 Unable to locate package ffmpeg
apt-get update && apt-get install -y ffmpeg

------------------------------------------------------------
步骤3：验证安装
------------------------------------------------------------

python -c "import funasr; print('funasr 版本:', funasr.__version__)"
python -c "import torch; print('torch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# 期望输出示例：
# funasr 版本: 1.x.x
# torch 版本: 2.8.0+cu128
# CUDA 可用: True
# GPU: NVIDIA GeForce RTX 4090

------------------------------------------------------------
步骤4：下载模型（首次需联网，后续离线可用）
------------------------------------------------------------

# AutoDL 服务器建议将模型下载到数据盘（/root/autodl-tmp 为免费数据盘挂载点）
python -c "
from funasr import AutoModel
model = AutoModel(
    model='paraformer-zh',
    vad_model='fsmn-vad',
    punc_model='ct-punc',
    spk_model='cam++',
    device='cuda',
)
print('模型下载并加载成功')
"

# 如需手动下载到指定目录（使用最新模型ID）：
python -c "
from modelscope import snapshot_download
snapshot_download('iic/paraformer-zh', cache_dir='/root/autodl-tmp/models')
snapshot_download('iic/fsmn-vad', cache_dir='/root/autodl-tmp/models')
snapshot_download('iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', cache_dir='/root/autodl-tmp/models')
snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common', cache_dir='/root/autodl-tmp/models')
"

# 下载完成后查看模型目录
ls /root/autodl-tmp/models/

------------------------------------------------------------
步骤5：快速功能测试（下载官方示例音频后测试）
------------------------------------------------------------

# 下载 FunASR 官方中文测试音频
wget -O /tmp/test_zh.wav "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"

# 运行测试（使用在线模型名称，自动匹配已缓存模型）
python -c "
from funasr import AutoModel

def ms_to_time(ms):
    h = int(ms // 3600000)
    m = int((ms % 3600000) // 60000)
    s = int((ms % 60000) // 1000)
    return f'{h:02d}:{m:02d}:{s:02d}' if h > 0 else f'{m:02d}:{s:02d}'

model = AutoModel(
    model='paraformer-zh',
    vad_model='fsmn-vad',
    punc_model='ct-punc',
    spk_model='cam++',
    device='cuda',  # 使用 RTX 4090 GPU 推理
)
result = model.generate(input='/root/autodl-tmp/audio/3.mp3', batch_size_s=300)
print('=== 识别结果 ===')
for res in result:
    sentences = res.get('sentence_info', [])
    if sentences:
        current_spk = None
        current_text = ''
        current_start = 0
        for sent in sentences:
            spk = sent.get('spk', 0)
            text = sent.get('text', '')
            start = sent.get('start', 0)
            if spk != current_spk:
                if current_text:
                    print(f'[{ms_to_time(current_start)}] 用户{current_spk + 1}： {current_text}')
                current_spk = spk
                current_text = text
                current_start = start
            else:
                current_text += text
        if current_text:
            print(f'[{ms_to_time(current_start)}] 用户{current_spk + 1}： {current_text}')
    else:
        print('识别结果：', res.get('text', ''))
"
