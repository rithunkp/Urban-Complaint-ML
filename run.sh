#!/bin/bash
set -e

echo "Installing dependencies and starting the Gradio app"
pip install -r requirements.txt
python app.py
