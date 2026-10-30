"""VeloceAI inference API using Spandrel (no BasicSR dependency)."""
import os
import re
import uuid
import shutil
import subprocess, tempfile, time
import urllib.request
from functools import wraps
from pathlib import Path
from threading import Lock
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import uvicorn
from spandrel import ModelLoader

MODEL = Path(os.environ.get("MODEL_PATH", "/content/RealESRGAN_x4plus.pth"))
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
ANIME_MODEL = Path(os.environ.get("ANIME_MODEL_PATH", "/content/RealESRGAN_x4plus_anime_6B.pth"))
ANIME_MODEL_URL = os.environ.get(
    "ANIME_MODEL_URL",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Avoid CPU thread oversubscription; OpenCV must not spawn another worker pool.
if DEVICE == "cpu":
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
elif DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

app = FastAPI(title="VeloceAI Restoration Engine", version="2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

@app.middleware("http")
async def log_enhancement_requests(request, call_next):
    """Log upload arrival and completion so tunnel failures are distinguishable from AI failures."""
    is_enhancement = request.method == "POST" and (
        request.url.path in ("/api/enhance/image", "/api/enhance/video", "/api/enhance/video/chunked")
        or request.url.path == "/api/upload/chunk"
    )
    started = time.monotonic()
    if is_enhancement:
        print(
            f"[VeloceAI upload] received {request.method} {request.url.path} "
            f"content-length={request.headers.get('content-length', 'unknown')}",
            flush=True
        )
    try:
        response = await call_next(request)
    except Exception:
        if is_enhancement:
            print(f"[VeloceAI upload] request failed after {time.monotonic()-started:.1f}s", flush=True)
        raise
    if is_enhancement:
        print(
            f"[VeloceAI upload] completed status={response.status_code} "
            f"elapsed={time.monotonic()-started:.1f}s",
            flush=True
        )
    return response

descriptors = {}
MODEL_LOCK = Lock()
PROCESS_LOCK = Lock()
progress_state = {"percent": 0, "stage": "idle", "busy": False}

# Chunked mobile uploads and background jobs avoid one long multipart tunnel request.
UPLOAD_ROOT = Path(os.environ.get("VELOCEAI_UPLOAD_DIR", "/content/veloceai-uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
UPLOADS = {}
JOBS = {}
JOB_LOCK = Lock()
CHUNK_SIZE = 1024 * 1024

def set_progress(percent, stage):
    progress_state.update({"percent": max(0, min(100, int(percent))), "stage": str(stage)})


def track_progress(func):
    """Keep progress state recoverable when validation or processing fails."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not PROCESS_LOCK.acquire(blocking=False):
            raise HTTPException(409, "Another enhancement is already running. Please wait for it to finish.")
        try:
            return func(*args, **kwargs)
        except HTTPException as exc:
            detail = str(exc.detail).replace("\\n", " ")[:100]
            progress_state.update({"percent": 0, "stage": "Failed: " + detail, "busy": False})
            raise
        except Exception as exc:
            detail = str(exc).replace("\\n", " ")[:100] or exc.__class__.__name__
            progress_state.update({"percent": 0, "stage": "Failed: " + detail, "busy": False})
            raise HTTPException(500, "Processing failed: " + detail) from exc
        finally:
            progress_state["busy"] = False
            PROCESS_LOCK.release()
    return wrapper

@app.get("/")
def home():
    """Serve the frontend and API from one permanent origin."""
    return FileResponse(Path(__file__).with_name("index.html"), media_type="text/html")


@app.get("/progress")
def get_progress():
    return dict(progress_state)

@app.post("/api/upload/init")
def upload_init(payload: dict = Body(...)):
    filename = Path(str(payload.get("filename", "upload.bin"))).name
    kind = str(payload.get("kind", ""))
    size = int(payload.get("size", 0))
    allowed = (".png", ".jpg", ".jpeg", ".webp", ".bmp") if kind == "image" else (".mp4", ".mov", ".mkv", ".webm")
    limit = 25 * 1024 * 1024 if kind == "image" else 150 * 1024 * 1024
    if kind not in ("image", "video") or Path(filename).suffix.lower() not in allowed:
        raise HTTPException(400, "Unsupported media type.")
    if size <= 0 or size > limit:
        raise HTTPException(413, f"File size must be between 1 byte and {limit // (1024*1024)} MB.")
    upload_id = uuid4().hex
    folder = UPLOAD_ROOT / upload_id
    folder.mkdir(parents=True, exist_ok=False)
    with JOB_LOCK:
        UPLOADS[upload_id] = {"filename": filename, "kind": kind, "size": size,
                              "chunk_size": CHUNK_SIZE, "folder": folder, "chunks": set()}
    print(f"[VeloceAI chunk upload] started id={upload_id} kind={kind} size={size}", flush=True)
    return {"ok": True, "upload_id": upload_id, "chunk_size": CHUNK_SIZE}


@app.post("/api/upload/chunk")
def upload_chunk(upload_id: str = Form(...), index: int = Form(...), file: UploadFile = File(...)):
    with JOB_LOCK:
        state = UPLOADS.get(upload_id)
    if not state:
        raise HTTPException(404, "Upload session not found. Please restart the upload.")
    total_chunks = (state["size"] + state["chunk_size"] - 1) // state["chunk_size"]
    if index < 0 or index >= total_chunks:
        raise HTTPException(400, "Invalid upload chunk number.")
    data = file.file.read()
    expected = min(state["chunk_size"], state["size"] - index * state["chunk_size"])
    if len(data) != expected:
        raise HTTPException(400, f"Chunk {index} has {len(data)} bytes; expected {expected}.")
    (state["folder"] / f"{index:06d}.part").write_bytes(data)
    with JOB_LOCK:
        state["chunks"].add(index)
    print(f"[VeloceAI chunk upload] id={upload_id} chunk={index+1}/{total_chunks} bytes={len(data)}", flush=True)
    return {"ok": True, "index": index}


@app.post("/api/jobs/start")
def start_job(payload: dict = Body(...)):
    upload_id = str(payload.get("upload_id", ""))
    settings = payload.get("settings") or {}
    with JOB_LOCK:
        state = UPLOADS.get(upload_id)
        active = any(j.get("status") in ("queued", "processing") for j in JOBS.values())
        if active or progress_state.get("busy"):
            raise HTTPException(409, "Another enhancement is already running. Please wait.")
        if not state:
            raise HTTPException(404, "Upload session not found. Please upload the file again.")
        total_chunks = (state["size"] + state["chunk_size"] - 1) // state["chunk_size"]
        if len(state["chunks"]) != total_chunks:
            raise HTTPException(400, "Upload is incomplete. Please retry the missing chunks.")
        source = state["folder"] / state["filename"]
        with open(source, "wb") as merged:
            for index in range(total_chunks):
                part = state["folder"] / f"{index:06d}.part"
                if not part.is_file():
                    raise HTTPException(400, f"Upload chunk {index} is missing.")
                with open(part, "rb") as chunk:
                    shutil.copyfileobj(chunk, merged)
        if source.stat().st_size != state["size"]:
            raise HTTPException(400, "Uploaded file size does not match the original.")
        job_id = uuid4().hex
        job_folder = state["folder"]
        JOBS[job_id] = {"status": "queued", "percent": 0, "stage": "Queued",
                        "result": None, "error": None, "kind": state["kind"],
                        "filename": state["filename"]}
        del UPLOADS[upload_id]
    Thread(target=_run_enhancement_job,
           args=(job_id, state["kind"], state["filename"], source, job_folder, settings),
           daemon=True).start()
    print(f"[VeloceAI job] queued id={job_id} kind={state['kind']}", flush=True)
    return {"ok": True, "job_id": job_id}


def _run_enhancement_job(job_id, kind, filename, source, folder, settings):
    with JOB_LOCK:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["stage"] = "Starting AI engine"
    try:
        params = {"scale": int(settings.get("scale", 2)),
                  "denoise": int(settings.get("denoise", 0)),
                  "sharpness": float(settings.get("sharpness", 1.2)),
                  "crf": int(settings.get("crf", 17)),
                  "profile": str(settings.get("profile", "anime")),
                  "line_detail": int(settings.get("line_detail", 20)),
                  "anti_alias": int(settings.get("anti_alias", 0))}
        with open(source, "rb") as source_file:
            upload = StarletteUploadFile(filename=filename, file=source_file)
            if kind == "image":
                response = enhance_image(file=upload, **params)
                output_name, media_type = "VeloceAI-Restored.png", "image/png"
            else:
                crop = str(settings.get("crop_black_bars", "false")).lower() in ("1", "true", "yes")
                response = enhance_video(file=upload, crop_black_bars=crop, **params)
                output_name, media_type = "VeloceAI-Restored.mp4", "video/mp4"
        result_path = folder / output_name
        result_path.write_bytes(response.body)
        if not result_path.is_file() or result_path.stat().st_size == 0:
            raise RuntimeError("The enhancement result is empty.")
        with JOB_LOCK:
            JOBS[job_id].update({"status": "complete", "percent": 100, "stage": "Complete",
                                 "result": str(result_path), "media_type": media_type,
                                 "output_name": output_name})
        print(f"[VeloceAI job] complete id={job_id} bytes={result_path.stat().st_size}", flush=True)
    except Exception as exc:
        detail = str(getattr(exc, "detail", exc))[:500]
        with JOB_LOCK:
            JOBS[job_id].update({"status": "failed", "error": detail,
                                 "stage": "Failed: " + detail, "percent": 0})
        progress_state.update({"percent": 0, "stage": "Failed: " + detail[:100], "busy": False})
        print(f"[VeloceAI job] failed id={job_id}: {detail}", flush=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Enhancement job not found.")
        result = {k: job.get(k) for k in ("status", "percent", "stage", "error")}
    if result["status"] == "processing":
        result["percent"] = progress_state.get("percent", result["percent"])
        result["stage"] = progress_state.get("stage", result["stage"])
    return result


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Enhancement job not found.")
        if job.get("status") != "complete" or not job.get("result"):
            raise HTTPException(409, "Enhancement is not complete yet.")
        path, media_type, output_name = job["result"], job["media_type"], job["output_name"]
    return FileResponse(path, media_type=media_type, filename=output_name,
                        headers={"Cache-Control": "no-store"})



@app.post("/api/preview/video")
def preview_video(file: UploadFile = File(...)):
    """Extract a browser-friendly PNG thumbnail for videos whose codecs cannot preview locally."""
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, "Unsupported video format.")
    data = file.file.read()
    if not data:
        raise HTTPException(400, "The uploaded video is empty.")
    if len(data) > 150 * 1024 * 1024:
        raise HTTPException(413, "Video limit is 150 MB.")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / ("preview" + suffix)
        thumb = root / "thumbnail.png"
        src.write_bytes(data)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vf", "scale=w='min(960,iw)':h=-2",
            "-frames:v", "1", "-an", str(thumb)
        ]
        try:
            extracted = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise HTTPException(408, "Video preview extraction timed out.")
        if extracted.returncode != 0 or not thumb.is_file() or thumb.stat().st_size == 0:
            detail = extracted.stderr[-500:] if extracted.stderr else "No video frame could be extracted."
            raise HTTPException(400, "Could not create video preview: " + detail)
        return Response(thumb.read_bytes(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

def load_model(profile="general"):
    """Load and cache a model matched to the selected content profile."""
    key = "anime" if profile == "anime" else "standard"
    model_path = ANIME_MODEL if key == "anime" else MODEL
    model_url = ANIME_MODEL_URL if key == "anime" else MODEL_URL
    if key in descriptors:
        return descriptors[key]
    with MODEL_LOCK:
        if key in descriptors:
            return descriptors[key]
        if not model_path.is_file():
            if progress_state.get("busy"):
                set_progress(max(8, progress_state.get("percent", 1)),
                             "Downloading " + ("compact anime" if key == "anime" else "standard") + " model")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_model = model_path.with_suffix(model_path.suffix + ".download")
            try:
                request = urllib.request.Request(model_url, headers={"User-Agent": "VeloceAI/2.1"})
                with urllib.request.urlopen(request, timeout=180) as response, open(temporary_model, "wb") as target:
                    shutil.copyfileobj(response, target)
                minimum_size = 10 * 1024 * 1024 if key == "anime" else 40 * 1024 * 1024
                if temporary_model.stat().st_size < minimum_size:
                    raise RuntimeError("Downloaded model file is too small or incomplete.")
                temporary_model.replace(model_path)
            except Exception:
                if temporary_model.exists():
                    temporary_model.unlink()
                raise
        if progress_state.get("busy"):
            set_progress(max(10, progress_state.get("percent", 1)),
                         "Loading " + ("compact anime" if key == "anime" else "standard") + " model")
        loaded = ModelLoader().load_from_file(model_path)
        loaded.model.eval().to(DEVICE)
        loaded.model.to(memory_format=torch.channels_last)
        descriptors[key] = loaded
    return descriptors[key]

@app.post("/warmup")
def warmup(profile: str = "anime"):
    """Download and load the selected AI model during Colab startup."""
    try:
        if profile not in ("general", "anime", "real"):
            raise HTTPException(400, "Invalid processing profile.")
        load_model(profile)
        return {"ok": True, "model_loaded": True, "profile": profile, "device": DEVICE}
    except Exception as exc:
        raise HTTPException(503, "AI model warm-up failed: " + str(exc)) from exc


@app.get("/health")
def health():
    # Liveness only: do not download/load the model during every page open.
    return {
        "ok": True,
        "engine": "Real-ESRGAN x4plus via Spandrel",
        "device": DEVICE,
        "model_loaded": bool(descriptors),
        "models_loaded": sorted(descriptors.keys()),
        "message": "ready" if descriptors else "models load automatically on first enhancement"
    }

def validate(scale, sharpness, denoise, crf, profile, line_detail, anti_alias):
    if scale not in (2,4): raise HTTPException(400,"Scale must be 2 or 4.")
    if not 1 <= sharpness <= 2: raise HTTPException(400,"Sharpness must be between 1 and 2.")
    if not 0 <= denoise <= 100: raise HTTPException(400,"Denoise must be between 0 and 100.")
    if not 15 <= crf <= 28: raise HTTPException(400,"CRF must be between 15 and 28.")
    if profile not in ("general", "anime", "real"): raise HTTPException(400,"Invalid processing profile.")
    if not 0 <= line_detail <= 100: raise HTTPException(400,"Line detail must be between 0 and 100.")
    if not 0 <= anti_alias <= 100: raise HTTPException(400,"Anti-alias must be between 0 and 100.")
    try:
        model_key = "anime" if profile == "anime" else "standard"
        if progress_state.get("busy") and model_key not in descriptors:
            set_progress(max(8, progress_state.get("percent", 1)),
                         "Loading AI model · first startup may take longer")
        load_model(profile)
        if progress_state.get("busy"):
            set_progress(max(8, progress_state.get("percent", 1)), "AI model ready · preparing media")
    except Exception as e:
        raise HTTPException(503,"AI model unavailable: "+str(e))

@torch.inference_mode()
def enhance_frame(src: Path, dst: Path, scale: int, sharpness: float, denoise: int,
                  profile: str = "general", line_detail: int = 0, anti_alias: int = 0):
    model = load_model(profile)
    im = Image.open(src).convert("RGB")
    # For requested 2x output, feed half-size input to the native x4 model.
    # This produces the requested dimensions directly and uses about 1/4
    # the neural compute of x4-then-downsample (the old, unnecessarily slow path).
    original_size = im.size
    if scale == 2:
        work_size = (max(1, (im.width + 1) // 2), max(1, (im.height + 1) // 2))
        im = im.resize(work_size, Image.Resampling.LANCZOS)
    # Denoise at working resolution before upscaling, never on the 4x output.
    if denoise:
        import cv2
        source = np.asarray(im)
        lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
        luma, channel_a, channel_b = cv2.split(lab)
        luma = cv2.fastNlMeansDenoising(luma, None, denoise / 100 * 8, 7, 21)
        clean = cv2.cvtColor(cv2.merge((luma, channel_a, channel_b)), cv2.COLOR_LAB2RGB)
        im = Image.fromarray(clean)
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)
    # Match the model's channels-last layout on CPU as well as CUDA to avoid
    # repeated NCHW/NHWC conversions inside convolution layers.
    x = x.contiguous(memory_format=torch.channels_last)
    if DEVICE == "cuda":
        # FP16 autocast is supported by the CUDA conv layers in Real-ESRGAN
        # and reduces per-frame latency and VRAM use on Colab GPUs.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model.model(x).clamp(0,1)
    else:
        y = model.model(x).clamp(0,1)
    # The 2x path is already at target dimensions because its input was halved above.
    out = (y.squeeze(0).permute(1,2,0).cpu().numpy()*255.0).round().clip(0,255).astype(np.uint8)
    result = Image.fromarray(out)
    if scale == 2 and result.size != (original_size[0] * 2, original_size[1] * 2):
        result = result.resize((original_size[0] * 2, original_size[1] * 2), Image.Resampling.LANCZOS)
    # Classical edge-aware post-processing after Real-ESRGAN; not a separate AI model.
    if anti_alias > 0:
        import cv2
        rgb = np.asarray(result)
        strength = anti_alias / 100.0
        smooth = cv2.bilateralFilter(rgb, d=5, sigmaColor=18 + 52*strength,
                                     sigmaSpace=2 + 3*strength)
        result = Image.fromarray(cv2.addWeighted(rgb, 1.0 - 0.58*strength,
                                                  smooth, 0.58*strength, 0))
    # Selective luminance-edge enhancement for fine anime outlines.
    if line_detail > 0:
        import cv2
        rgb = np.asarray(result).astype(np.float32)
        u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        luma = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        base = cv2.GaussianBlur(luma, (0, 0), sigmaX=0.85)
        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        mask = np.clip((magnitude - 10.0) / 45.0, 0.0, 1.0)
        amount = (line_detail / 100.0) * (1.35 if profile == "anime" else 0.7)
        delta = (luma - base) * amount * mask
        result = Image.fromarray(np.clip(rgb + delta[:, :, None], 0, 255).astype(np.uint8))
    if sharpness > 1:
        from PIL import ImageEnhance
        result = ImageEnhance.Sharpness(result).enhance(sharpness)
    result.save(dst)

@app.post("/api/enhance/image")
@track_progress
def enhance_image(file: UploadFile=File(...), scale:int=2, denoise:int=25,
                        sharpness:float=1.2, crf:int=17, profile:str="anime",
                        line_detail:int=35, anti_alias:int=15):
    progress_state.update({"percent": 1, "stage": "Preparing image request", "busy": True})
    validate(scale,sharpness,denoise,crf,profile,line_detail,anti_alias)
    suffix=Path(file.filename or "image.png").suffix.lower()
    if suffix not in (".png",".jpg",".jpeg",".webp",".bmp"): raise HTTPException(400,"Unsupported image format.")
    data=file.file.read()
    if len(data)>25*1024*1024: raise HTTPException(413,"Image limit is 25 MB.")
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/("input"+suffix); dst=Path(td)/"VeloceAI-Restored.png"
        src.write_bytes(data)
        try:
            set_progress(15, "AI image enhancement · working")
            enhance_frame(src,dst,scale,sharpness,denoise,profile,line_detail,anti_alias)
            set_progress(95, "Preparing enhanced image")
            output = dst.read_bytes()
            if not output:
                raise RuntimeError("The enhanced image file is empty.")
        except Exception as e:
            raise HTTPException(500,"Image AI failed: "+str(e)) from e
        progress_state.update({"percent": 100, "stage": "Complete", "busy": False})
        return Response(output,media_type="image/png",
          headers={"Content-Disposition":'attachment; filename="VeloceAI-Restored.png"'})

# Chunked browser uploads avoid a single large multipart POST through mobile networks
# and Cloudflare Quick Tunnels. Each request is intentionally small and retryable.
CHUNK_UPLOAD_ROOT = Path("/content/veloceai-chunks")
MAX_UPLOAD_CHUNK = 3 * 1024 * 1024

@app.post("/api/upload/chunk")
def receive_video_chunk(upload_id: str, index: int, total: int, file: UploadFile = File(...)):
    if not re.fullmatch(r"[a-f0-9]{32}", upload_id or ""):
        raise HTTPException(400, "Invalid upload session.")
    if total < 1 or total > 100 or index < 0 or index >= total:
        raise HTTPException(400, "Invalid video chunk number.")
    data = file.file.read(MAX_UPLOAD_CHUNK + 1)
    if not data or len(data) > MAX_UPLOAD_CHUNK:
        raise HTTPException(413, "Video chunk is empty or too large.")
    folder = CHUNK_UPLOAD_ROOT / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (f"{index:04d}.part")).write_bytes(data)
    print(f"[VeloceAI chunk] saved {index + 1}/{total}; {len(data)} bytes", flush=True)
    return {"ok": True, "index": index, "received_bytes": len(data)}


@app.post("/api/enhance/video/chunked")
def enhance_video_chunked(upload_id: str, total_chunks: int, filename: str = "video.mp4",
                          scale: int = 2, denoise: int = 25, sharpness: float = 1.2,
                          crf: int = 17, crop_black_bars: bool = False,
                          profile: str = "anime", line_detail: int = 35,
                          anti_alias: int = 15):
    if not re.fullmatch(r"[a-f0-9]{32}", upload_id or ""):
        raise HTTPException(400, "Invalid upload session.")
    if total_chunks < 1 or total_chunks > 100:
        raise HTTPException(400, "Invalid video chunk count.")
    safe_name = Path(filename or "video.mp4").name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in (".mp4", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, "Unsupported video format.")
    folder = CHUNK_UPLOAD_ROOT / upload_id
    assembled = folder / ("assembled" + suffix)
    try:
        expected = [folder / f"{i:04d}.part" for i in range(total_chunks)]
        missing = [str(i) for i, part in enumerate(expected) if not part.is_file()]
        if missing:
            raise HTTPException(400, "Video upload is incomplete; missing chunks: " + ", ".join(missing[:8]))
        total_bytes = sum(part.stat().st_size for part in expected)
        if total_bytes <= 0 or total_bytes > 150 * 1024 * 1024:
            raise HTTPException(413, "Video limit is 150 MB.")
        with assembled.open("wb") as output:
            for part in expected:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output)
        print(f"[VeloceAI chunk] assembled {safe_name}; {total_bytes} bytes; processing", flush=True)
        source_file = assembled.open("rb")
        upload = UploadFile(filename=safe_name, file=source_file)
        try:
            return enhance_video(file=upload, scale=scale, denoise=denoise,
                                 sharpness=sharpness, crf=crf,
                                 crop_black_bars=crop_black_bars, profile=profile,
                                 line_detail=line_detail, anti_alias=anti_alias)
        finally:
            source_file.close()
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@app.post("/api/enhance/video")
@track_progress
def enhance_video(file:UploadFile=File(...), scale:int=2, denoise:int=25,
                        sharpness:float=1.2, crf:int=17, crop_black_bars:bool=False,
                        profile:str="anime", line_detail:int=35, anti_alias:int=15):
    progress_state.update({"percent": 1, "stage": "Preparing video request", "busy": True})
    validate(scale,sharpness,denoise,crf,profile,line_detail,anti_alias)
    suffix=Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4",".mov",".mkv",".webm"): raise HTTPException(400,"Unsupported video format.")
    data=file.file.read()
    if len(data)>150*1024*1024: raise HTTPException(413,"Video limit is 150 MB.")
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); src=root/("input"+suffix); frames=root/"frames"; out=root/"out"
        frames.mkdir(); out.mkdir(); src.write_bytes(data)
        set_progress(7, "Checking video duration and streams")
        probe=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(src)],capture_output=True,text=True)
        try: duration=float(probe.stdout.strip())
        except: duration=0
        if probe.returncode or duration<=0: raise HTTPException(400,"Could not read video.")
        if duration>45: raise HTTPException(413,"Video limit is 45 seconds.")
        fp=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=avg_frame_rate","-of","csv=p=0",str(src)],capture_output=True,text=True)
        fps=fp.stdout.strip() or "30/1"
        try:
            fps_parts = fps.split("/", 1)
            fps_value = float(fps_parts[0]) / (float(fps_parts[1]) if len(fps_parts) == 2 else 1.0)
            if fps_value <= 0:
                fps_value = 30.0
        except (ValueError, ZeroDivisionError):
            fps_value = 30.0
        estimated_frames = max(1, int(duration * fps_value))
        crop_filter = None
        if crop_black_bars:
            set_progress(7, "Detecting black bars · 0%")
            sample_duration = min(duration, 120 / max(fps_value, 1.0))
            detect_cmd=["ffmpeg","-hide_banner","-nostats","-stats_period","0.5",
                        "-progress","pipe:1","-i",str(src),"-vf","cropdetect=24:16:0",
                        "-frames:v","120","-f","null","-"]
            detect=subprocess.Popen(detect_cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                    text=True,bufsize=1)
            detect_lines=[]
            for raw_line in detect.stdout:
                line=raw_line.strip()
                if line:
                    detect_lines.append(line)
                    detect_lines=detect_lines[-160:]
                if line.startswith("out_time_ms="):
                    try:
                        elapsed=max(0,int(line.split("=",1)[1]))/1_000_000
                        ratio=min(1.0,elapsed/max(sample_duration,0.001))
                        set_progress(7 + int(ratio),
                                     f"Detecting black bars · {round(ratio*100)}%")
                    except ValueError:
                        pass
            detect_code=detect.wait()
            import re
            crops=re.findall(r"crop=(\d+:\d+:\d+:\d+)", "\\n".join(detect_lines))
            if detect_code == 0 and crops:
                crop_filter=crops[-1]
        # FFmpeg extraction used to block at a fixed 8% until every PNG was written.
        # Its progress pipe now advances the UI while frames are being decoded.
        set_progress(8, "Extracting video frames · 0%")
        extract=["ffmpeg","-y","-hide_banner","-loglevel","error","-nostats",
                 "-stats_period","0.5","-progress","pipe:1","-i",str(src)]
        if crop_filter:
            extract += ["-vf",f"crop={crop_filter}"]
        extract += ["-map","0:v:0","-vsync","0","-compression_level","1",str(frames/"%08d.png")]
        ex=subprocess.Popen(extract,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                            text=True,bufsize=1)
        extract_errors=[]
        for raw_line in ex.stdout:
            line=raw_line.strip()
            if line and ("=" not in line or line.startswith("Error")):
                extract_errors.append(line)
                extract_errors=extract_errors[-12:]
            if line.startswith("out_time_ms="):
                try:
                    elapsed=max(0,int(line.split("=",1)[1]))/1_000_000
                    ratio=min(1.0,elapsed/max(duration,0.001))
                    set_progress(8 + int(4*ratio),
                                 f"Extracting video frames · {round(ratio*100)}%")
                except ValueError:
                    pass
            elif line.startswith("frame="):
                try:
                    extracted_frames=max(0,int(line.split("=",1)[1]))
                    extraction_ratio = min(1.0, extracted_frames / estimated_frames)
                    set_progress(8 + int(4 * extraction_ratio),
                                 f"Extracting video frames · {extracted_frames}/{estimated_frames}")
                except (ValueError,ZeroDivisionError):
                    pass
        extract_code=ex.wait()
        if extract_code != 0:
            raise HTTPException(400,"Frame extraction failed: "+" ".join(extract_errors)[-800:])
        inputs=sorted(frames.glob("*.png"))
        if not inputs: raise HTTPException(400,"No video frames found.")
        try:
            total_frames = len(inputs)
            for i,p in enumerate(inputs,1):
                # Advance the stage before the expensive model call, not only after it.
                start_percent=12 + int(79*(i-1)/total_frames)
                set_progress(start_percent, f"AI enhancement · frame {i}/{total_frames}")
                enhance_frame(p,out/(f"{i:08d}.png"),scale,sharpness,denoise,profile,line_detail,anti_alias)
                set_progress(12 + int(79*i/total_frames), f"AI enhancement · frame {i}/{total_frames}")
        except Exception as e:
            raise HTTPException(500,"Video AI failed: "+str(e)) from e
        set_progress(92, "Encoding enhanced video and audio · 0%")
        target=root/"VeloceAI-Restored.mp4"
        encode=["ffmpeg","-y","-hide_banner","-loglevel","error","-nostats",
          "-stats_period","0.5","-progress","pipe:1","-framerate",fps,
          "-i",str(out/"%08d.png"),"-i",str(src),
          "-map","0:v:0","-map","1:a:0?","-c:v","libx264","-preset","veryfast","-crf",str(crf),
          "-pix_fmt","yuv420p","-c:a","aac","-b:a","192k","-movflags","+faststart",
          "-t",f"{duration:.6f}",str(target)]
        enc=subprocess.Popen(encode,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,bufsize=1)
        encode_errors=[]
        for raw_line in enc.stdout:
            line=raw_line.strip()
            if line and ("=" not in line or line.startswith("Error")):
                encode_errors.append(line)
                encode_errors=encode_errors[-12:]
            if line.startswith("out_time_ms="):
                try:
                    elapsed=max(0,int(line.split("=",1)[1]))/1_000_000
                    ratio=min(1.0,elapsed/max(duration,0.001))
                    set_progress(92 + int(6*ratio),
                                 f"Encoding video and audio · {round(ratio*100)}%")
                except ValueError:
                    pass
        encode_code=enc.wait()
        if encode_code != 0 or not target.exists():
            raise HTTPException(500,"Video export failed: "+" ".join(encode_errors)[-1000:])
        output = target.read_bytes()
        if not output:
            raise HTTPException(500, "The enhanced video file is empty.")
        progress_state.update({"percent": 100, "stage": "Complete", "busy": False})
        return Response(output,media_type="video/mp4",
          headers={"Content-Disposition":'attachment; filename="VeloceAI-Restored.mp4"'})

if __name__=="__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
