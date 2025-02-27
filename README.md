# Video Transcription App

A beautiful, modern application for transcribing videos using Whisper AI.

## Features

- Upload single or multiple video files for transcription
- Modern dark-themed UI built with Tailwind CSS
- Interactive transcript viewer with timeline navigation
- Search functionality within transcripts
- Download transcripts as text files
- Responsive design for mobile and desktop
- Dashboard for managing all transcripts
- Links to original video file locations

## Screenshots

[Screenshots will be added once the app is deployed]

## Tech Stack

- **Backend:** Python with Flask
- **AI Model:** OpenAI Whisper (large model)
- **Frontend:** Tailwind CSS, HTML, JavaScript
- **NLP:** NLTK integration for additional text processing
- **Storage:** Local file storage for uploads and transcripts

## Prerequisites

- Python 3.8 or higher
- FFmpeg (required by Whisper for audio processing)

## Installation

### Method 1: Easy Start (macOS/Linux)

1. Clone this repository:
   ```
   git clone <repository-url>
   cd transcription_app
   ```

2. Make sure FFmpeg is installed:
   - On macOS: `brew install ffmpeg`
   - On Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`

3. Run the start script:
   ```
   ./start.sh
   ```

4. Open your browser and go to: `http://127.0.0.1:5050`

### Method 2: Manual Local Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd transcription_app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install FFmpeg:
   - On macOS: `brew install ffmpeg`
   - On Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your PATH

5. Start the application:
   ```
   python app.py
   ```

6. Open your browser and go to: `http://127.0.0.1:5050`

### Method 3: Docker Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd transcription_app
   ```

2. Make sure Docker and Docker Compose are installed on your system.

3. Build and start the container:
   ```
   docker-compose up -d
   ```

4. Open your browser and go to: `http://localhost:5050`

5. To stop the application:
   ```
   docker-compose down
   ```

### Configuration Options

You can modify the Whisper model used for transcription by setting the `WHISPER_MODEL` environment variable:

- `tiny`: Fastest, lowest accuracy
- `base`: Good balance of speed and accuracy (default)
- `small`: Better accuracy, slower
- `medium`: High accuracy, much slower
- `large`: Highest accuracy, slowest

For Docker installation, uncomment the `WHISPER_MODEL` line in `docker-compose.yml` and set your preferred model size.

## Usage

1. On the home page, drag and drop video files or click to select files.
2. Optionally add a title and topic for the transcription.
3. Click "Transcribe Now" to begin processing.
4. Progress bars will show the transcription status for each file.
5. After processing, you can view or download the transcripts.
6. Use the dashboard to access all previously created transcripts.

## Future Enhancements

- Vector database integration for semantic search
- User authentication and multiple user support
- Enhanced metadata for better organization
- Direct video playback synchronized with transcript
- Export to various formats (SRT, VTT, etc.)
- Batch processing with background workers

## License

[MIT License](LICENSE)

## Acknowledgments

- OpenAI for the Whisper model
- Flask for the web framework
- Tailwind CSS for the styling
- NLTK for natural language processing capabilities