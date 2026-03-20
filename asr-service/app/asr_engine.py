from funasr import AutoModel


def load_model():
    return AutoModel(
        model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        vad_model='fsmn-vad',
        punc_model='ct-punc',
        spk_model='cam++',
        device='cuda',
        disable_pbar=True,
        disable_log=False,
    )


def ms_to_time(ms):
    h = int(ms // 3600000)
    m = int((ms % 3600000) // 60000)
    s = int((ms % 60000) // 1000)
    return f'{h:02d}:{m:02d}:{s:02d}' if h > 0 else f'{m:02d}:{s:02d}'


def recognize_stream(model, audio_path: str, hotword: str = ''):
    """
    生成器函数：逐句 yield 识别结果，供 SSE 流式推送。
    每条结果格式：[开始时间]-[结束时间] 用户X： 文字
    Paraformer-Large 模型，支持 VAD + 标点 + 说话人分离，按说话人分段输出。
    """
    result = model.generate(
        input=audio_path, 
        batch_size_s=300,
        hotword=hotword,
    )
    
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
                yield f'识别结果： {full_text}'
