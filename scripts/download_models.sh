#!/bin/bash

if [ ! -d "pretrained_models" ]; then
  if ! git clone https://huggingface.co/fudan-generative-ai/hallo pretrained_models; then
    echo "Error: Failed to clone pretrained_models repository."
  else
    cd pretrained_models
    git lfs pull
    cd ..
  fi
fi

if [ ! -f "pretrained_models/hallo/net.pth" ]; then
  if ! wget -P pretrained_models/hallo/ https://huggingface.co/fudan-generative-ai/hallo/resolve/main/hallo/net.pth; then
    echo "Error: Failed to download net.pth."
  fi
fi

if [ ! -d "HumanAssets" ]; then
  if ! git clone https://huggingface.co/spaces/MakiAi/HumanAssets; then
    echo "Error: Failed to clone HumanAssets repository."
  else
    cd HumanAssets
    git lfs pull
    cd ..
  fi
fi