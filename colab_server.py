"""
VeloceAI temporary Google Colab API.
Run this file inside the already-prepared /content/Real-ESRGAN Colab runtime.
This is a temporary demo server, not permanent hosting.
"""
import os, shutil, subprocess, tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

ROOT = Path("/content/Real-ESRGAN")
MODEL = ROOT / "weights" / "RealESRGAN_x4plus.pth"
INFER = ROOT / "inference_realesrgan.py"
app = FastAPI(title="VeloceAI Colab Engine", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporary public demo only; restrict this for production.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    ready = ROOT.exists() and INFER.exists() and MODEL.exists()
    return {"ok": ready, "engine": "Real-ESRGAN x4plus", "message": "ready" if ready else "model files missing"}

def run_upscale(input_path: Path, output_dir: Path, outscale: int):
    if not INFER.exists():
        raise HTTPException(503, "Real-ESRGAN repository not found in this Colab session.")
    if not MODEL.exists():
        raise HTTPException(503, "RealESRGAN_x4plus.pth was not found at /content/Real-ESRGAN/weights/.")
    cmd = [
        "python", str(INFER), "-n", "RealESRGAN_x4plus",
        "-i", str(input_path), "-o", str(output_dir),
        "--model_path", str(MODEL), "--outscale", str(outscale),
        "--suffix", "veloceai", "--tile", "256"
    ]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise HTTPException(500, "AI processing failed: " + (p.stderr or p.stdout)[-1800:])
    outputs = [p for p in output_dir.iterdir() if p.is_file()]
    if not outputs:
        raise HTTPException(500, "The AI model completed but produced no output file.")
    return outputs[0]

@app.post("/api/enhance/image")
async def enhance_image(file: UploadFile = File(...), scale: int = 2):
    if scale not in (2, 4):
        raise HTTPException(400, "scale must be 2 or 4")
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        raise HTTPException(400, "Upload a PNG, JPG, JPEG, WEBP, or BMP image.")
    with tempfile.TemporaryDirectory(prefix="veloceai_") as tmp:
        work = Path(tmp); src = work / ("input" + suffix); out = work / "out"
        out.mkdir()
        data = await file.read()
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(413, "Image limit is 25 MB.")
        src.write_bytes(data)
        result = run_upscale(src, out, scale)
        # Return bytes before the temporary directory is removed.
        from fastapi.responses import Response
        mime = "image/png" if result.suffix.lower() == ".png" else "application/octet-stream"
        return Response(content=result.read_bytes(), media_type=mime,
                        headers={"Content-Disposition": 'attachment; filename="veloceai-enhanced.png"'})

@app.post("/api/enhance/video")
async def enhance_video(file: UploadFile = File(...), scale: int = 2):
    if scale not in (2, 4):
        raise HTTPException(400, "scale must be 2 or 4")
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in (".mp4", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, "Upload MP4, MOV, MKV, or WEBM.")
    with tempfile.TemporaryDirectory(prefix="veloceai_video_") as tmp:
        work = Path(tmp); src = work / ("input" + suffix); frames = work / "frames"
        enhanced = work / "enhanced"; frames.mkdir(); enhanced.mkdir()
        data = await file.read()
        if len(data) > 150 * 1024 * 1024:
            raise HTTPException(413, "Video limit is 150 MB.")
        src.write_bytes(data)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True
        )
        try: duration = float(probe.stdout.strip())
        except Exception: duration = 0
        if probe.returncode != 0 or duration <= 0:
            raise HTTPException(400, "Could not read this video. Please use a supported video file.")
        if duration > 30:
            raise HTTPException(413, "For this temporary Colab demo, videos are limited to 30 seconds.")
        # Preserve the source frame rate; extract numbered frames for AI inference.
        extract = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(src), "-vsync", "0",
             str(frames / "%08d.png")],
            capture_output=True, text=True, timeout=300
        )
        if extract.returncode != 0:
            raise HTTPException(400, "Could not extract video frames: " + extract.stderr[-1000:])
        run_upscale(frames, enhanced, scale)
        out_video = work / "veloceai-enhanced.mp4"
        # Real-ESRGAN outputs frame names with suffix; sort numerically and encode at source FPS.
        pattern = str(enhanced / "%08d_veloceai.png")
        fps = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1", str(src)],
            capture_output=True, text=True
        ).stdout.strip() or "30/1"
        encode = subprocess.run(
            ["ffmpeg", "-v", "error", "-framerate", fps, "-i", pattern,
             "-i", str(src), "-map", "0:v:0", "-map", "1:a?",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
             "-shortest", "-movflags", "+faststart", str(out_video)],
            capture_output=True, text=True, timeout=900
        )
        if encode.returncode != 0 or not out_video.exists():
            raise HTTPException(500, "Could not rebuild the enhanced video: " + encode.stderr[-1400:])
        from fastapi.responses import Response
        return Response(content=out_video.read_bytes(), media_type="video/mp4",
                        headers={"Content-Disposition": 'attachment; filename="veloceai-enhanced.mp4"'})

if __name__ == "__main__":
    # Cloudflare Tunnel or another tunnel must expose this port for the GitHub Pages site.
    uvicorn.run(app, host="0.0.0.0", port=8000)
