FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg - required for Whisper
RUN apt-get update && apt-get install -y ffmpeg git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for uploads and transcripts
RUN mkdir -p static/uploads static/transcripts

# Expose the port the app runs on
EXPOSE 5050

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "app:app"]