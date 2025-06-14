#!/usr/bin/env bash
# Exit on error
set -o errexit

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Then install requirements
pip install -r requirements.txt
