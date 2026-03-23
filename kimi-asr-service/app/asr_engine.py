import os
import torch
import numpy as np
import soundfile as sf
import librosa
from kimia_infer.api.kimia import KimiAudio


_DEFAULT_MODEL_PATH = os.environ.get(
    'KIMI_AUDIO_MODEL_PATH',
    '/root/autodl-tmp/models/moonshotai/Kimi-Audio-7B-Instruct',
)

_SAMPLING_PARAMS = {
    'audio_temperature': 0.0,
    'audio_top_k': 1,
    'text_temperature': 0.0,
    'text_top_k': 1,
    'audio_repetition_penalty': 1.0,
    'audio_repetition_window_size': 64,
    'text_repetition_penalty': 1.0,
    'text_repetition_window_size': 16,
}


def load_model():
    model_path = _DEFAULT_MODEL_PATH
    print(f'正在加载 Kimi-Audio 模型：{model_path}')
    model = KimiAudio(model_path=model_path, load_detokenizer=False)
    print('Kimi-Audio 模型加载完成。')
    return model


def load_diarization_pipeline():
    from pyannote.audio import Pipeline
    hf_token = os.environ.get('HF_TOKEN', '')
    print('正在加载 pyannote 说话人分离 pipeline...')
    if hf_token:
        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization-3.1',
            use_auth_token=hf_token,
        )
    else:
        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization-3.1',
        )
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device('cuda'))
    print('pyannote pipeline 加载完成。')
    return pipeline


def sec_to_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f'{h:02d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'


def _load_audio_16k(audio_path: str):
    """Load audio as 16 kHz mono numpy array."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    return audio, sr


def _transcribe_segment(model, audio_16k: np.ndarray, sr: int = 16000) -> str:
    """Transcribe a single numpy audio segment using KimiAudio."""
    import tempfile, uuid
    tmp = os.path.join(tempfile.gettempdir(), f'kimi_seg_{uuid.uuid4()}.wav')
    try:
        sf.write(tmp, audio_16k, sr)
        messages = [
            {'role': 'user', 'message_type': 'text',
             'content': '请逐字转写以下音频，仅输出转写文字，不要加任何解释。'},
            {'role': 'user', 'message_type': 'audio', 'content': tmp},
        ]
        _, text = model.generate(messages, **_SAMPLING_PARAMS, output_type='text')
        return (text or '').strip()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def recognize_stream(model, diarization_pipeline, audio_path: str):
    """
    Generator: yields one formatted line per merged speaker segment.

    Format: [MM:SS]-[MM:SS] 用户X： <text>

    Steps:
    1. Run pyannote diarization to get (start, end, speaker) segments.
    2. For each segment, extract audio and transcribe with KimiAudio.
    3. Merge consecutive segments from the same speaker.
    4. Yield each merged segment as a formatted string.
    """
    audio_16k, sr = _load_audio_16k(audio_path)
    total_samples = len(audio_16k)

    print('开始说话人分离...')
    diarization = diarization_pipeline(audio_path)

    raw_segments = []
    speaker_map = {}
    speaker_counter = [0]

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_counter[0] += 1
            speaker_map[speaker] = speaker_counter[0]
        raw_segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker_map[speaker],
        })

    if not raw_segments:
        yield '未检测到语音内容。'
        return

    print(f'检测到 {len(raw_segments)} 个说话片段，开始逐段转写...')

    transcribed = []
    for seg in raw_segments:
        start_sample = int(seg['start'] * sr)
        end_sample = min(int(seg['end'] * sr), total_samples)
        if end_sample <= start_sample:
            continue
        chunk = audio_16k[start_sample:end_sample]
        text = _transcribe_segment(model, chunk, sr)
        if text:
            transcribed.append({
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker'],
                'text': text,
            })

    if not transcribed:
        yield '未识别到任何语音内容。'
        return

    merged = []
    cur = dict(transcribed[0])
    for seg in transcribed[1:]:
        if seg['speaker'] == cur['speaker']:
            cur['text'] += seg['text']
            cur['end'] = seg['end']
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)

    for seg in merged:
        line = (
            f'[{sec_to_time(seg["start"])}]-[{sec_to_time(seg["end"])}] '
            f'用户{seg["speaker"]}： {seg["text"]}'
        )
        yield line
