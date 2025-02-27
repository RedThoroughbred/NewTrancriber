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
        # Get model size from environment variable or use "base" as default
        model_size = os.environ.get("WHISPER_MODEL", "base")
        print(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
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
                # Add a preview of the transcript (first 200 characters)
                data['preview'] = data['transcript'][:200] + "..." if len(data['transcript']) > 200 else data['transcript']
                transcripts.append(data)
    
    return render_template('dashboard.html', transcripts=transcripts)

@app.route('/search', methods=['POST'])
def search_transcripts():
    query = request.json.get('query', '').lower()
    if not query or len(query) < 2:
        return jsonify([])
    
    results = []
    for file in os.listdir(app.config['TRANSCRIPT_FOLDER']):
        if file.endswith('.json'):
            with open(os.path.join(app.config['TRANSCRIPT_FOLDER'], file), 'r') as f:
                data = json.load(f)
                transcript_text = data['transcript'].lower()
                
                if query in transcript_text:
                    # Find all occurrences of the query
                    matches = []
                    last_pos = 0
                    
                    # Get up to 3 snippet matches
                    for _ in range(3):
                        pos = transcript_text.find(query, last_pos)
                        if pos == -1:
                            break
                            
                        # Get a snippet of text around the match
                        start = max(0, pos - 50)
                        end = min(len(transcript_text), pos + len(query) + 50)
                        snippet = "..." + transcript_text[start:end] + "..."
                        
                        # Highlight the match
                        highlight_start = max(0, pos - start)
                        matches.append({
                            'snippet': snippet,
                            'highlight_pos': highlight_start,
                            'highlight_len': len(query)
                        })
                        
                        last_pos = pos + len(query)
                    
                    if matches:
                        results.append({
                            'id': data['id'],
                            'title': data['title'],
                            'date': data['date'],
                            'matches': matches,
                            'match_count': transcript_text.count(query)
                        })
    
    # Sort results by number of matches (most matches first)
    results.sort(key=lambda x: x['match_count'], reverse=True)
    
    return jsonify(results[:10])  # Limit to top 10 results

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