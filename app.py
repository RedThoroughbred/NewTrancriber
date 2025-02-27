from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
import json
from werkzeug.utils import secure_filename
import whisper
import nltk
from datetime import datetime
import shutil

# Download nltk data
nltk.download('punkt', quiet=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TRANSCRIPT_FOLDER'] = 'static/transcripts'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Make sure upload and transcript directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRANSCRIPT_FOLDER'], exist_ok=True)

# Load Whisper model (will download if not present)
model = None  # Lazy loading to avoid startup delay

def get_model():
    global model
    if model is None:
        # Use base model for faster loading
        model = whisper.load_model("base")
    return model

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Get list of transcripts
    transcripts = []
    for file in os.listdir(app.config['TRANSCRIPT_FOLDER']):
        if file.endswith('.json'):
            with open(os.path.join(app.config['TRANSCRIPT_FOLDER'], file), 'r') as f:
                data = json.load(f)
                transcripts.append(data)
    
    return render_template('dashboard.html', transcripts=transcripts)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('file')
    
    # Clean upload folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    results = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if file:
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            extension = os.path.splitext(original_filename)[1]
            filename = f"{file_id}{extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            
            # Get metadata
            title = request.form.get('title', original_filename)
            topic = request.form.get('topic', '')
            
            # Process with Whisper (in a real app, would use Celery for background processing)
            try:
                model = get_model()
                result = model.transcribe(filepath)
                
                # Save transcript as JSON with metadata
                transcript_data = {
                    'id': file_id,
                    'title': title,
                    'topic': topic,
                    'original_filename': original_filename,
                    'date': datetime.now().isoformat(),
                    'filepath': filepath,
                    'transcript': result['text'],
                    'segments': result['segments']
                }
                
                transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{file_id}.json")
                with open(transcript_path, 'w') as f:
                    json.dump(transcript_data, f)
                
                results.append({
                    'id': file_id,
                    'title': title,
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'filename': original_filename,
                    'success': False,
                    'error': str(e)
                })
    
    return jsonify(results)

@app.route('/transcript/<transcript_id>')
def view_transcript(transcript_id):
    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
    
    if not os.path.exists(transcript_path):
        return render_template('404.html', message="Transcript not found"), 404
    
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    return render_template('transcript.html', transcript=transcript_data)

@app.route('/download/<transcript_id>')
def download_transcript(transcript_id):
    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
    
    if not os.path.exists(transcript_path):
        return jsonify({'error': 'Transcript not found'}), 404
    
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Create a plain text version for download
    text_content = transcript_data['transcript']
    
    # Create a temporary file for download
    download_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.txt")
    with open(download_path, 'w') as f:
        f.write(text_content)
    
    return send_file(download_path, as_attachment=True, 
                    download_name=f"{transcript_data['title']}.txt")

if __name__ == '__main__':
    app.run(debug=True, port=5050)