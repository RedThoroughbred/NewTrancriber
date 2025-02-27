#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch venv/installed
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg is not installed, which is required for audio processing."
    echo "Please install FFmpeg:"
    echo "  - On macOS: brew install ffmpeg"
    echo "  - On Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg"
    echo "  - On Windows: Download from ffmpeg.org and add to your PATH"
fi

# Start the application
echo "Starting Transcription App..."
echo "Open your browser and go to: http://127.0.0.1:5050"
python app.py