#!/usr/bin/env bash
set -o errexit

echo "Upgrading pip and installing core build tools..."
pip install --upgrade pip setuptools wheel

echo "Installing system dependencies..."
apt-get update && apt-get install -y ffmpeg libsm6 libxext6 

echo "Installing Python requirements..."
pip install -r requirements.txt

echo "Installing MediaPipe..."
pip install --no-cache-dir mediapipe==0.10.5
