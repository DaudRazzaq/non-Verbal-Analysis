#!/usr/bin/env bash
set -o errexit

echo "Installing system dependencies..."
apt-get update && apt-get install -y ffmpeg libsm6 libxext6 

# Make sure pip is updated first
pip install --upgrade pip

# Install core build tools
pip install setuptools==59.8.0 wheel==0.38.4

# Install main requirements
pip install -r requirements.txt

# Try to install MediaPipe specifically
pip install --no-deps mediapipe-silicon==0.10.0 || pip install --no-deps mediapipe==0.8.11
