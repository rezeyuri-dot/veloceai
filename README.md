---
title: VeloceAI
emoji: 🎞️
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: AI image and video restoration with Real-ESRGAN
---

# VeloceAI

A self-hosted image and video restoration studio powered by Real-ESRGAN x4plus.

## What this deployment does

- Serves the English VeloceAI website and inference API from the same origin.
- Downloads the Real-ESRGAN x4plus weights automatically on the first enhancement request.
- Uses a CUDA GPU when available and falls back to CPU.
- Reports upload, extraction, frame-enhancement, and encoding progress.
- Supports PNG, JPG, WEBP, BMP images and MP4, MOV, MKV, WEBM videos.
- Does not require Google Colab or a Cloudflare quick tunnel after deployment.

## Hardware

For usable video performance, select **NVIDIA T4 small** (or a faster compatible GPU) in the Space Settings → Hardware after the first deployment. The current published T4-small rate is $0.40/hour while the Space is running. The default CPU hardware is suitable for smoke tests, not fast video processing.

## One-time deployment from the GitHub repository

This repository includes a GitHub Actions workflow that mirrors the project to a Hugging Face Docker Space.

1. In Hugging Face, create a fine-grained access token with write access to Spaces.
2. In the GitHub repository settings, add:
   - Repository variable `HF_SPACE_ID`: your Hugging Face Space ID, in the form `username/veloceai`.
   - Repository secret `HF_TOKEN`: the Hugging Face write token.
3. Run the **Deploy to Hugging Face Space** workflow from the Actions tab whenever you are ready to publish. This is manual so an unconfigured repository does not produce failed deploys.
4. In the new Space Settings → Hardware, select **NVIDIA T4 small** or faster, then wait for the Space to rebuild. Docker Spaces require a Hugging Face Pro plan (personal accounts), and GPU use is billed separately.
5. Open the Space URL. Keep the Space running while you process files; pause or downgrade the hardware when you are not using it to avoid ongoing GPU charges.

The token is stored as a GitHub Actions secret and is not part of the website or source code.

## Local container

```bash
docker build -t veloceai .
docker run --rm -p 7860:7860 --gpus all veloceai
```

Then open `http://localhost:7860`.

## Notes

- The model weights are downloaded from the official Real-ESRGAN GitHub release on first use and cached at `MODEL_PATH`.
- Video processing is computationally expensive. Runtime depends on clip length, frame rate, resolution, upscale factor, and available GPU.
- This project is independent and is not affiliated with Topaz Labs. It does not use the Topaz engine.
