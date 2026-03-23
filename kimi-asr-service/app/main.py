import os
import uuid
import asyncio
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.asr_engine import load_model, load_diarization_pipeline, recognize_stream

kimi_model = None
diarization_pipeline = None

ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kimi_model, diarization_pipeline
    print('正在加载 Kimi-Audio 模型...')
    kimi_model = load_model()
    print('正在加载 pyannote 说话人分离 pipeline...')
    diarization_pipeline = load_diarization_pipeline()
    print('所有模型加载完成，服务就绪。')
    yield
    print('服务关闭。')


app = FastAPI(
    title='Kimi-Audio 语音识别服务（含说话人分离）',
    version='1.0.0',
    lifespan=lifespan,
)


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'kimi_model_loaded': kimi_model is not None,
        'diarization_loaded': diarization_pipeline is not None,
    }


@app.post('/asr/stream')
async def asr_stream(file: UploadFile = File(...)):
    """
    上传音频文件，以 SSE 流式返回逐句识别结果（含说话人分离）。

    支持格式：mp3 / wav / m4a / flac / ogg / aac，最大 500MB。

    响应格式（text/event-stream）：
        data: [00:16]-[00:21] 用户1： 辣椒精来吧，老婆，你终于从国外回来了，

        data: [00:22]-[00:23] 用户2： 怎么想我了？

        data: [DONE]
    """
    if kimi_model is None or diarization_pipeline is None:
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
                return list(recognize_stream(kimi_model, diarization_pipeline, tmp_path))

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
