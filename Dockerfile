FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    wget

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh |  bash
RUN apt-get install -y git-lfs
RUN git lfs install

# COPY download_models.sh .
# RUN chmod +x download_models.sh