"""VeloceAI inference API using Spandrel (no BasicSR dependency)."""
import os
import shutil
import subprocess, tempfile, time
import urllib.request
from functools import wraps
from pathlib import Path
from threading import Lock
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import uvicorn
from spandrel import ModelLoader

MODEL = Path(os.environ.get("MODEL_PATH", "/content/RealESRGAN_x4plus.pth"))
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Avoid CPU thread oversubscription on Colab; use AMP on CUDA in enhance_frame.
if DEVICE == "cpu":
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
elif DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

app = FastAPI(title="VeloceAI Restoration Engine", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

@app.middleware("http")
async def log_enhancement_requests(request, call_next):
    """Log upload arrival and completion so tunnel failures are distinguishable from AI failures."""
    is_enhancement = request.method == "POST" and request.url.path in (
        "/api/enhance/image", "/api/enhance/video"
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

descriptor = None
MODEL_LOCK = Lock()
PROCESS_LOCK = Lock()
progress_state = {"percent": 0, "stage": "idle", "busy": False}

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

def load_model():
    global descriptor
    if descriptor is not None:
        return descriptor
    # A hosted Space downloads the model automatically on its first request.
    # Colab can keep using the model downloaded by its setup cell.
    with MODEL_LOCK:
        if descriptor is None:
            if not MODEL.is_file():
                if progress_state.get("busy"):
                    set_progress(max(8, progress_state.get("percent", 1)),
                                 "Downloading Real-ESRGAN model weights")
                MODEL.parent.mkdir(parents=True, exist_ok=True)
                temporary_model = MODEL.with_suffix(MODEL.suffix + ".download")
                try:
                    request = urllib.request.Request(
                        MODEL_URL,
                        headers={"User-Agent": "VeloceAI/1.0"}
                    )
                    with urllib.request.urlopen(request, timeout=120) as response, open(temporary_model, "wb") as target:
                        shutil.copyfileobj(response, target)
                    if temporary_model.stat().st_size < 20 * 1024 * 1024:
                        raise RuntimeError("Downloaded model file is too small or incomplete.")
                    temporary_model.replace(MODEL)
                except Exception:
                    if temporary_model.exists():
                        temporary_model.unlink()
                    raise
            if progress_state.get("busy"):
                set_progress(max(10, progress_state.get("percent", 1)),
                             "Loading Real-ESRGAN model")
            loaded = ModelLoader().load_from_file(MODEL)
            loaded.model.eval().to(DEVICE)
            if DEVICE == "cuda":
                loaded.model.to(memory_format=torch.channels_last)
            descriptor = loaded
    return descriptor

@app.get("/health")
def health():
    # Liveness only: do not download/load the model during every page open.
    return {
        "ok": True,
        "engine": "Real-ESRGAN x4plus via Spandrel",
        "device": DEVICE,
        "model_loaded": descriptor is not None,
        "message": "ready" if descriptor is not None else "model loads automatically on first enhancement"
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
        if progress_state.get("busy") and descriptor is None:
            set_progress(max(8, progress_state.get("percent", 1)),
                         "Loading AI model · first startup may take longer")
        load_model()
        if progress_state.get("busy"):
            set_progress(max(8, progress_state.get("percent", 1)), "AI model ready · preparing media")
    except Exception as e:
        raise HTTPException(503,"AI model unavailable: "+str(e))

@torch.inference_mode()
def enhance_frame(src: Path, dst: Path, scale: int, sharpness: float, denoise: int,
                  profile: str = "general", line_detail: int = 0, anti_alias: int = 0):
    model = load_model()
    im = Image.open(src).convert("RGB")
    # Denoise at source resolution before upscaling. Running NLM on the 4x
    # output was needlessly expensive (up to 16x as many pixels per frame).
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
    if DEVICE == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
        # FP16 autocast is supported by the CUDA conv layers in Real-ESRGAN
        # and reduces per-frame latency and VRAM use on Colab GPUs.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model.model(x).clamp(0,1)
    else:
        y = model.model(x).clamp(0,1)
    if scale == 2:
        h,w = y.shape[-2:]
        y = torch.nn.functional.interpolate(y, size=(h//2,w//2), mode="bicubic", align_corners=False)
    out = (y.squeeze(0).permute(1,2,0).cpu().numpy()*255.0).round().clip(0,255).astype(np.uint8)
    result = Image.fromarray(out)
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
