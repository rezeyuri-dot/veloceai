"""VeloceAI Colab inference API. Requires prepared Real-ESRGAN, ffmpeg and FastAPI."""
import subprocess, tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn

ROOT = Path("/content/Real-ESRGAN")
MODEL = ROOT / "weights/RealESRGAN_x4plus.pth"
INFER = ROOT / "inference_realesrgan.py"
app = FastAPI(title="VeloceAI AI Restoration Engine", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

@app.get("/health")
def health():
    ok = INFER.is_file() and MODEL.is_file()
    return {"ok": ok, "engine": "Real-ESRGAN x4plus", "message": "ready" if ok else "model files missing"}

def validate(scale, sharpness, denoise, crf):
    if scale not in (2, 4): raise HTTPException(400, "Scale must be 2 or 4.")
    if not 1 <= sharpness <= 2: raise HTTPException(400, "Sharpness must be between 1 and 2.")
    if not 0 <= denoise <= 100: raise HTTPException(400, "Denoise must be between 0 and 100.")
    if not 15 <= crf <= 28: raise HTTPException(400, "CRF must be between 15 and 28.")
    if not INFER.is_file() or not MODEL.is_file():
        raise HTTPException(503, "Real-ESRGAN model is not installed in this Colab session.")

def upscale(source: Path, out: Path, scale: int):
    cmd = ["python", str(INFER), "-n", "RealESRGAN_x4plus", "-i", str(source),
           "-o", str(out), "--model_path", str(MODEL), "--outscale", str(scale),
           "--suffix", "veloceai", "--tile", "256"]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    if p.returncode:
        raise HTTPException(500, "AI upscaling failed: " + (p.stderr or p.stdout)[-1800:])
    outputs = sorted(out.glob("*_veloceai.png"))
    if not outputs: raise HTTPException(500, "The AI model produced no output.")
    return outputs

def finish_image(path: Path, sharpness: float, denoise: int):
    # Denoise only the LAB luminance channel to avoid intentional hue/saturation changes.
    try:
        import cv2
        import numpy as np
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: raise ValueError("Could not read enhanced frame")
        if denoise:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            strength = denoise / 100 * 10
            l = cv2.fastNlMeansDenoising(l, None, strength, 7, 21)
            image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        if sharpness > 1:
            blur = cv2.GaussianBlur(image, (0, 0), 1.0)
            image = cv2.addWeighted(image, sharpness, blur, 1-sharpness, 0)
        if not cv2.imwrite(str(path), image): raise ValueError("Could not save enhanced frame")
    except ImportError:
        if sharpness > 1:
            from PIL import Image, ImageEnhance
            im = Image.open(path).convert("RGB")
            ImageEnhance.Sharpness(im).enhance(sharpness).save(path)

@app.post("/api/enhance/image")
async def enhance_image(file: UploadFile = File(...), scale: int = 2,
                        denoise: int = 25, sharpness: float = 1.2, crf: int = 17):
    validate(scale, sharpness, denoise, crf)
    suffix = Path(file.filename or "image.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        raise HTTPException(400, "Use PNG, JPG, WEBP or BMP.")
    data = await file.read()
    if len(data) > 25*1024*1024: raise HTTPException(413, "Image limit is 25 MB.")
    with tempfile.TemporaryDirectory(prefix="veloceai_img_") as td:
        work=Path(td); src=work/("input"+suffix); out=work/"out"; out.mkdir()
        src.write_bytes(data)
        outputs=upscale(src,out,scale)
        result=outputs[0]; finish_image(result,sharpness,denoise)
        return Response(result.read_bytes(),media_type="image/png",
                        headers={"Content-Disposition":'attachment; filename="VeloceAI-Restored.png"'})

@app.post("/api/enhance/video")
async def enhance_video(file: UploadFile = File(...), scale: int = 2,
                        denoise: int = 25, sharpness: float = 1.2, crf: int = 17):
    validate(scale, sharpness, denoise, crf)
    suffix=Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4",".mov",".mkv",".webm"):
        raise HTTPException(400,"Use MP4, MOV, MKV or WEBM.")
    data=await file.read()
    if len(data)>150*1024*1024: raise HTTPException(413,"Video limit is 150 MB.")
    with tempfile.TemporaryDirectory(prefix="veloceai_vid_") as td:
        work=Path(td); src=work/("input"+suffix); frames=work/"frames"; out=work/"upscaled"
        frames.mkdir();out.mkdir();src.write_bytes(data)
        probe=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
          "-of","default=noprint_wrappers=1:nokey=1",str(src)],capture_output=True,text=True)
        try: duration=float(probe.stdout.strip())
        except ValueError: duration=0
        if probe.returncode or duration<=0: raise HTTPException(400,"Could not read this video.")
        if duration>45: raise HTTPException(413,"Video limit is 45 seconds.")
        fpsprobe=subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
          "-show_entries","stream=avg_frame_rate","-of","default=nw=1:nk=1",str(src)],
          capture_output=True,text=True)
        fps=fpsprobe.stdout.strip() or "30/1"
        extract=subprocess.run(["ffmpeg","-y","-v","error","-i",str(src),"-map","0:v:0",
          "-vsync","0",str(frames/"%08d.png")],capture_output=True,text=True,timeout=300)
        if extract.returncode: raise HTTPException(400,"Frame extraction failed: "+extract.stderr[-1000:])
        outputs=upscale(frames,out,scale)
        expected=len(list(frames.glob("*.png")))
        if len(outputs)!=expected: raise HTTPException(500,"AI output frame count does not match input.")
        for frame in outputs: finish_image(frame,sharpness,denoise)
        pattern=str(out/"%08d_veloceai.png")
        target=work/"VeloceAI-Restored.mp4"
        encode=subprocess.run(["ffmpeg","-y","-v","error","-framerate",fps,"-i",pattern,
          "-i",str(src),"-map","0:v:0","-map","1:a:0?","-c:v","libx264",
          "-preset","medium","-crf",str(crf),"-pix_fmt","yuv420p","-c:a","aac",
          "-b:a","192k","-movflags","+faststart","-shortest",str(target)],
          capture_output=True,text=True,timeout=1200)
        if encode.returncode or not target.is_file() or target.stat().st_size==0:
            raise HTTPException(500,"MP4 export failed: "+encode.stderr[-1400:])
        return Response(target.read_bytes(),media_type="video/mp4",
          headers={"Content-Disposition":'attachment; filename="VeloceAI-Restored.mp4"'})

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
