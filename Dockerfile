FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt update && \
  apt install -y --no-install-recommends \
    ffmpeg \
    python3-pip \
    pulseaudio

RUN pip install -U \
  openai-whisper \
  pulsectl \
  scipy \
  --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /work
