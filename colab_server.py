"""VeloceAI inference API using Spandrel (no BasicSR dependency)."""
import subprocess, tempfile
from functools import wraps
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
from spandrel import ModelLoader

MODEL = Path("/content/RealESRGAN_x4plus.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="VeloceAI Restoration Engine", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])
descriptor = None
progress_state = {"percent": 0, "stage": "idle", "busy": False}

def set_progress(percent, stage):
    progress_state.update({"percent": max(0, min(100, int(percent))), "stage": str(stage)})


def track_progress(func):
    """Keep progress state recoverable when validation or processing fails."""
    @wraps(func)
    def wrapper(*args, **kwargs):
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
    return wrapper

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
    if descriptor is None:
        if not MODEL.is_file():
            raise RuntimeError("Model file missing: " + str(MODEL))
        descriptor = ModelLoader().load_from_file(MODEL)
        descriptor.model.eval().to(DEVICE)
    return descriptor

@app.get("/health")
def health():
    try:
        m = load_model()
        return {"ok": True, "engine": "Real-ESRGAN x4plus via Spandrel", "device": DEVICE,
                "message": "ready"}
    except Exception as e:
        return {"ok": False, "engine": "Spandrel", "message": str(e)}

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
            set_progress(max(2, progress_state.get("percent", 1)),
                         "Loading AI model · first startup may take longer")
        load_model()
        if progress_state.get("busy"):
            set_progress(max(6, progress_state.get("percent", 1)), "AI model ready · preparing media")
    except Exception as e:
        raise HTTPException(503,"AI model unavailable: "+str(e))

@torch.inference_mode()
def enhance_frame(src: Path, dst: Path, scale: int, sharpness: float, denoise: int,
                  profile: str = "general", line_detail: int = 0, anti_alias: int = 0):
    model = load_model()
    im = Image.open(src).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)
    y = model.model(x).clamp(0,1)
    if scale == 2:
        h,w = y.shape[-2:]
        y = torch.nn.functional.interpolate(y, size=(h//2,w//2), mode="bicubic", align_corners=False)
    out = (y.squeeze(0).permute(1,2,0).cpu().numpy()*255.0).round().clip(0,255).astype(np.uint8)
    result = Image.fromarray(out)
    if denoise:
        import cv2
        a = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2LAB)
        l,aa,bb = cv2.split(a)
        l = cv2.fastNlMeansDenoising(l,None,denoise/100*8,7,21)
        result = Image.fromarray(cv2.cvtColor(cv2.merge((l,aa,bb)),cv2.COLOR_LAB2RGB))
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
        crop_filter = None
        if crop_black_bars:
            set_progress(8, "Detecting black bars")
            detect=subprocess.run(["ffmpeg","-hide_banner","-i",str(src),"-vf","cropdetect=24:16:0","-frames:v","120","-f","null","-"],capture_output=True,text=True)
            import re
            crops=re.findall(r"crop=(\d+:\d+:\d+:\d+)",detect.stderr)
            if crops:
                crop_filter=crops[-1]
        set_progress(8, "Extracting video frames")
        extract=["ffmpeg","-y","-v","error","-i",str(src)]
        if crop_filter:
            extract += ["-vf",f"crop={crop_filter}"]
        extract += ["-map","0:v:0","-vsync","0",str(frames/"%08d.png")]
        ex=subprocess.run(extract,capture_output=True,text=True)
        if ex.returncode: raise HTTPException(400,"Frame extraction failed: "+ex.stderr[-800:])
        inputs=sorted(frames.glob("*.png"))
        if not inputs: raise HTTPException(400,"No video frames found.")
        try:
            total_frames = len(inputs)
            for i,p in enumerate(inputs,1):
                enhance_frame(p,out/(f"{i:08d}.png"),scale,sharpness,denoise,profile,line_detail,anti_alias)
                set_progress(10 + int(80*i/total_frames), f"Enhancing frame {i}/{total_frames}")
        except Exception as e:
            progress_state.update({"stage": "Failed", "busy": False})
            raise HTTPException(500,"Video AI failed: "+str(e))
        set_progress(93, "Encoding video and audio")
        target=root/"VeloceAI-Restored.mp4"
        enc=subprocess.run(["ffmpeg","-y","-v","error","-framerate",fps,"-i",str(out/"%08d.png"),"-i",str(src),
          "-map","0:v:0","-map","1:a:0?","-c:v","libx264","-preset","veryfast","-crf",str(crf),
          "-pix_fmt","yuv420p","-c:a","aac","-b:a","192k","-movflags","+faststart","-t",f"{duration:.6f}",str(target)],
          capture_output=True,text=True,timeout=1200)
        if enc.returncode or not target.exists():
            progress_state.update({"stage": "Failed", "busy": False})
            raise HTTPException(500,"Video export failed: "+enc.stderr[-1000:])
        output = target.read_bytes()
        if not output:
            raise HTTPException(500, "The enhanced video file is empty.")
        progress_state.update({"percent": 100, "stage": "Complete", "busy": False})
        return Response(output,media_type="video/mp4",
          headers={"Content-Disposition":'attachment; filename="VeloceAI-Restored.mp4"'})

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
