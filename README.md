
# 语音转文字 + 说话人分离 —— 私有化离线部署方案（详细操作指南）

# AutoDL 服务器快速部署（RTX 4090 / PyTorch 2.8.0 / Python 3.12 / CUDA 12.8）


## 环境说明
  镜像：PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8
  GPU：RTX 4090 (24GB) × 1
  内存：90GB

注意：AutoDL 镜像已内置 PyTorch，无需重新安装 torch，直接安装 funasr 即可。

------------------------------------------------------------
步骤1：SSH 登录服务器后，激活 base 环境（conda 24.4 已预装）
------------------------------------------------------------

conda activate base
python --version   # 应显示 Python 3.12.x


## 步骤2：安装 FunASR 及依赖

1. 升级 pip  
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 安装 funasr（使用清华镜像加速）  
pip install -U funasr -i https://pypi.tuna.tsinghua.edu.cn/simple

3. 安装音频处理依赖（torchaudio 必须与 torch 版本严格匹配）  
pip install torchaudio==2.8.0 soundfile librosa -i https://pypi.tuna.tsinghua.edu.cn/simple

4. 安装 modelscope（用于下载模型）  
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

5. 安装 ffmpeg（支持 mp3/m4a 等格式）  
 注意：必须先更新包列表，否则会报 Unable to locate package ffmpeg

    apt-get update && apt-get install -y ffmpeg


## 步骤3：验证安装
```python
python -c "import funasr; print('funasr 版本:', funasr.__version__)"
python -c "import torch; print('torch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

 期望输出示例：
 ```
 funasr 版本: 1.x.x
 torch 版本: 2.8.0+cu128
 CUDA 可用: True
 GPU: NVIDIA GeForce RTX 4090
 ```


## 步骤4：下载模型（首次需联网，后续离线可用）

 AutoDL 服务器建议将模型下载到数据盘（/root/autodl-tmp 为免费数据盘挂载点）  
 ```python
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
```

 如需手动下载到指定目录（使用最新模型ID）：  
 ```python 
python -c "
from modelscope import snapshot_download
snapshot_download('iic/paraformer-zh', cache_dir='/root/autodl-tmp/models')
snapshot_download('iic/fsmn-vad', cache_dir='/root/autodl-tmp/models')
snapshot_download('iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', cache_dir='/root/autodl-tmp/models')
snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common', cache_dir='/root/autodl-tmp/models')
"
```

下载完成后查看模型目录 ls /root/autodl-tmp/models/

## 步骤5：快速功能测试（下载官方示例音频后测试）

1. 下载 FunASR 官方中文测试音频  
wget -O /tmp/test_zh.wav "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"

2. 上传本地音频文件到服务器目录：/root/autodl-tmp/audio

3. 运行测试（使用在线模型名称，自动匹配已缓存模型）
```python    
python -c "
from funasr import AutoModel

def ms_to_time(ms):
    h = int(ms // 3600000)
    m = int((ms % 3600000) // 60000)
    s = int((ms % 60000) // 1000)
    return f'{h:02d}:{m:02d}:{s:02d}' if h > 0 else f'{m:02d}:{s:02d}'

# 使用更高精度的 Paraformer 大模型 + VAD + 标点 + 说话人分离
model = AutoModel(
    model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model='fsmn-vad',
    punc_model='ct-punc',
    spk_model='cam++',
    device='cuda',
    disable_pbar=True,
    disable_log=False,
)

# input后面跟着服务器上面的音频文件
audio_file = '/root/autodl-tmp/audio/3.mp3'

result = model.generate(
    input=audio_file, 
    batch_size_s=300,
    hotword='',  # 可以添加热词提升特定词汇识别准确度，如 '阿里巴巴 达摩院'
)

print('=== 识别结果 ===')
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
            end = sent.get('end', 0)
            
            if spk != current_spk:
                if current_text:
                    print(f'[{ms_to_time(current_start)}]-[{ms_to_time(current_end)}] 用户{current_spk + 1}： {current_text}')
                current_spk = spk
                current_text = text
                current_start = start
                current_end = end
            else:
                current_text += text
                current_end = end
        
        if current_text:
            print(f'[{ms_to_time(current_start)}]-[{ms_to_time(current_end)}] 用户{current_spk + 1}： {current_text}')
    else:
        print('识别结果：', res.get('text', ''))
"
```

输出结果

=== 识别结果 ===
```
[00:16]-[00:21] 用户1： 辣椒精来吧，老婆，你终于从国外回来了，
[00:22]-[00:23] 用户2： 怎么想我了？
[00:23]-[00:33] 用户1： 是呀，你就答应我吧，在公司做一个闲职，每天看不到你我都心慌难受。是啊，你就答应十年吧，
[00:34]-[00:38] 用户2： 好吧，不过你得给我安排个适场的部门，没问题。
[00:39]-[01:03] 用户1： 我先进去啊，北方的这张小姐贵公司能有您这样的人才，真是令人佩服。这七千万的合同我签了，
[01:04]-[01:06] 用户2： 吴总合作愉快，
[01:07]-[01:14] 用户1： 合作愉快。老婆，多亏了你才拿下这个难搞的大客户。等你出差回来，我一定给你安排一个庆功宴。好，
[01:14]-[01:16] 用户2： 那我就等着你的惊喜吧。
[01:19]-[01:19] 用户1： 哦，
[01:21]-[01:46] 用户2： 公司出差的标准不是五百一晚吗？什么时候变得这么抠了，又脏又差的这怎么住啊？帮我预定一家总统套房，今晚没事很久，
[01:48]-[01:53] 用户3： 江想念把你这次出差的报销。
[01:55]-[01:58] 用户2： 芳姐，我自费住别的酒店了，不用报销饭。
[02:00]-[02:09] 用户3： 哎呦，小江，这公司不是已经开好房间了吗？你花那些冤枉钱干嘛呀？有那钱还不如好好的孝敬一下。你爹妈呢？
[02:09]-[02:11] 用户1： 是啊，小静，你钱多了，
[02:11]-[02:11] 用户3： 没地花了吗？
[02:13]-[02:19] 用户2： 这就不让你费心了，我爸妈还怕我公夫多花呢，而且我皮肤敏感，还有洁癖。
[02:21]-[02:35] 用户3： 哎，你刚拿了点工资就不得了了。五星期全天五千块钱一晚的行政套房，你都管住啊，我要是你妈，我非得被你活活气死不可。
[02:36]-[02:43] 用户2： 昨天我是单人出差，你是怎么知道我的酒店和刚拿到工资就不知道自己几斤几两了。
[02:44]-[02:55] 用户3： 这公司辛辛苦苦的培养你，你也没说买点下午茶，好好的孝敬一下公司的领导。现在这小年轻儿啊，一个个还真是的，
[03:02]-[03:09] 用户2： 你以为你是谁啊？我话我自己才说你什么事儿，不要对别人的钱占有欲力那么强。好，哎呀，
[03:09]-[03:16] 用户3： 我这是看你刚出社会，不懂事，帮你父母好好的管教一下，真是好心，当成人跟肺谢央。
[03:16]-[03:25] 用户2： 我父母还健在，轮不到你这个外人，人倒是你我怎么才发现这还有你这样的蛀处。
[03:25]-[03:26] 用户1： 你，
```