import os
import re
import uuid
import asyncio
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.asr_engine import load_model, recognize_stream

model = None

ALLOWED_EXTENSIONS = {
    '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac',
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.3gp'
}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print('正在加载 Paraformer-Large 模型（含 VAD + 标点 + 说话人分离）...')
    model = load_model()
    print('模型加载完成，服务就绪。')
    yield
    print('服务关闭。')


app = FastAPI(title='FunASR 语音识别服务 (Paraformer-Large + 说话人分离)', version='2.0.0', lifespan=lifespan)


@app.get('/health')
async def health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/asr/stream')
async def asr_stream(file: UploadFile = File(...), hotword: str = ''):
    """
    上传音频/视频文件，以 SSE 流式返回逐句识别结果（含说话人分离）。

    支持格式：
    - 音频：mp3 / wav / m4a / flac / ogg / aac
    - 视频：mp4 / avi / mov / mkv / wmv / flv / webm / mpeg / mpg / 3gp
    最大文件大小：500MB。

    参数：
        - file: 音频/视频文件
        - hotword: 热词（可选），用空格分隔多个词，如 "阿里巴巴 达摩院 人工智能"

    响应格式（text/event-stream）：
        data: [00:16]-[00:21] 用户1： 你好，欢迎使用语音识别服务。

        data: [00:22]-[00:25] 用户2： 谢谢，很高兴使用。

        data: [DONE]
    """
    if model is None:
        raise HTTPException(status_code=503, detail='模型尚未加载完成，请稍后重试')

    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f'不支持的文件格式：{ext}，支持格式：{", ".join(sorted(ALLOWED_EXTENSIONS))}',
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail='文件大小超过 500MB 限制')

    tmp_path = os.path.join(tempfile.gettempdir(), f'{uuid.uuid4()}{ext}')
    try:
        with open(tmp_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'文件保存失败：{str(e)}')

    async def event_generator():
        loop = asyncio.get_event_loop()
        try:
            def run_inference():
                return list(recognize_stream(model, tmp_path, hotword=hotword))

            sentences = await loop.run_in_executor(None, run_inference)
            for sentence in sentences:
                yield f'data: {sentence}\n\n'
                await asyncio.sleep(0)
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
            'X-Accel-Buffering': 'no',
        },
    )


@app.post('/asr')
async def asr(file: UploadFile = File(...), hotword: str = ''):
    """
    上传音频/视频文件，返回识别结果对象列表。

    支持格式：
    - 音频：mp3 / wav / m4a / flac / ogg / aac
    - 视频：mp4 / avi / mov / mkv / wmv / flv / webm / mpeg / mpg / 3gp
    最大文件大小：500MB。

    参数：
        - file: 音频/视频文件
        - hotword: 热词（可选），用空格分隔多个词，如 "阿里巴巴 达摩院 人工智能"

    响应格式（application/json）：
        [
            {"start": "00:16", "end": "00:21", "speaker": "用户1", "text": "你好，欢迎使用语音识别服务。"},
            {"start": "00:22", "end": "00:25", "speaker": "用户2", "text": "谢谢，很高兴使用。"}
        ]
    """
    if model is None:
        raise HTTPException(status_code=503, detail='模型尚未加载完成，请稍后重试')

    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f'不支持的文件格式：{ext}，支持格式：{", ".join(sorted(ALLOWED_EXTENSIONS))}',
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail='文件大小超过 500MB 限制')

    tmp_path = os.path.join(tempfile.gettempdir(), f'{uuid.uuid4()}{ext}')
    try:
        with open(tmp_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'文件保存失败：{str(e)}')

    try:
        loop = asyncio.get_event_loop()

        def run_inference():
            return list(recognize_stream(model, tmp_path, hotword=hotword))

        sentences = await loop.run_in_executor(None, run_inference)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'推理失败：{str(e)}')
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def parse_sentence(s: str) -> dict:
        """将 '[mm:ss]-[mm:ss] 用户X： text' 解析为字典。"""
        m = re.match(r'\[([^\]]+)\]-\[([^\]]+)\]\s+(.+?)：\s*(.*)', s)
        if m:
            return {'start': m.group(1), 'end': m.group(2), 'speaker': m.group(3), 'text': m.group(4)}
        return {'start': '', 'end': '', 'speaker': '', 'text': s}

    return [parse_sentence(s) for s in sentences]
