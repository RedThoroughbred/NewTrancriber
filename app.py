from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
import json
from werkzeug.utils import secure_filename
import whisper
import nltk
from datetime import datetime
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from modules.llm.summarize import summarize_transcript, extract_topics, extract_action_items
from modules.integration import feature_is_available
from modules.llm.meeting_intelligence import extract_questions_answers, extract_decisions, extract_commitments

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

# Integrate optional modules
from modules.integration import integrate_with_app
integrate_with_app(app)

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
    
@app.route('/delete/<transcript_id>', methods=['POST'])
def delete_transcript(transcript_id):
    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
    
    if not os.path.exists(transcript_path):
        return jsonify({'success': False, 'error': 'Transcript not found'}), 404
    
    try:
        # Load transcript to get the video file path
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
            
        # Delete JSON transcript file
        os.remove(transcript_path)
        
        # Delete text transcript file if it exists
        text_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.txt")
        if os.path.exists(text_path):
            os.remove(text_path)
            
        # Return success response
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
        
@app.route('/api/transcripts/<transcript_id>/update', methods=['POST'])
def update_transcript(transcript_id):
    """Update transcript metadata like topic, title, etc."""
    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
    
    if not os.path.exists(transcript_path):
        return jsonify({'success': False, 'error': 'Transcript not found'}), 404
    
    try:
        # Get the data to update
        data = request.json
        
        # Load existing transcript
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
            
        # Update fields that are provided
        if 'topic' in data:
            transcript_data['topic'] = data['topic']
            
        if 'title' in data:
            transcript_data['title'] = data['title']
            
        # Save updated transcript
        with open(transcript_path, 'w') as f:
            json.dump(transcript_data, f)
            
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
    """
    Enhanced transcribe endpoint that handles multiple files with individual metadata.
    
    For each file, it expects:
    - file_0, file_1, file_2, etc.
    - title_0, title_1, title_2, etc. (optional)
    - topic_0, topic_1, topic_2, etc. (optional)
    - file_count: total number of files
    
    It also handles YouTube URL and file path inputs as before.
    After transcription, generates summary, topics, and action items if LLM is available.
    """
    # Clean upload folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    results = []
    
    # Check if we're dealing with YouTube URL
    if 'youtube_url' in request.form:
        from youtube_downloader import download_youtube_video
        
        youtube_url = request.form.get('youtube_url')
        title = request.form.get('title', '')
        topic = request.form.get('topic', '')
        
        try:
            # Download the YouTube video
            video_info = download_youtube_video(youtube_url, app.config['UPLOAD_FOLDER'])
            file_id = video_info['file_id']
            filepath = video_info['file_path']
            if not title:
                title = video_info['title']
            
            # Process with Whisper
            model = get_model()
            result = model.transcribe(filepath)
            transcript_text = result['text']

            # LLM analysis (only if available)
            summary = ""
            topics = []
            action_items = []
            if feature_is_available('llm'):
                summary = summarize_transcript(transcript_text) or "Failed to generate summary."
                topics = extract_topics(transcript_text) or []
                action_items_result = extract_action_items(transcript_text) or {"action_items": []}
                action_items = action_items_result.get('action_items', [])
                qa_result = extract_questions_answers(transcript_text) or {"qa_pairs": []}
                qa_pairs = qa_result.get('qa_pairs', [])
                
                decisions_result = extract_decisions(transcript_text) or {"decisions": []}
                decisions = decisions_result.get('decisions', [])
                
                commitments_result = extract_commitments(transcript_text) or {"commitments": []}
                commitments = commitments_result.get('commitments', [])
            
            # Save transcript as JSON with metadata and LLM results
            transcript_data = {
                'id': file_id,
                'title': title,
                'topic': topic,
                'original_filename': f"{video_info['title']}.mp4",
                'date': datetime.now().isoformat(),
                'filepath': filepath,
                'transcript': transcript_text,
                'segments': result['segments'],
                'youtube_id': video_info['video_id'],
                'youtube_url': youtube_url,
                'youtube_channel': video_info['channel'],
                'summary': summary,
                'topics': topics,
                'action_items': action_items,
                'qa_pairs': qa_pairs,
                'decisions': decisions,
                'commitments': commitments
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
                'filename': 'YouTube Video',
                'success': False,
                'error': str(e)
            })
            
    # Check if we're dealing with a file path
    elif 'file_path' in request.form:
        import shutil
        
        file_path = request.form.get('file_path')
        title = request.form.get('title', os.path.basename(file_path))
        topic = request.form.get('topic', '')
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_id = str(uuid.uuid4())
            filename = os.path.basename(file_path)
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            shutil.copy2(file_path, dest_path)
            
            # Process with Whisper
            model = get_model()
            result = model.transcribe(dest_path)
            transcript_text = result['text']

            # LLM analysis (only if available)
            summary = ""
            topics = []
            action_items = []
            if feature_is_available('llm'):
                summary = summarize_transcript(transcript_text) or "Failed to generate summary."
                topics = extract_topics(transcript_text) or []
                action_items_result = extract_action_items(transcript_text) or {"action_items": []}
                action_items = action_items_result.get('action_items', [])
                qa_result = extract_questions_answers(transcript_text) or {"qa_pairs": []}
                qa_pairs = qa_result.get('qa_pairs', [])
                
                decisions_result = extract_decisions(transcript_text) or {"decisions": []}
                decisions = decisions_result.get('decisions', [])
                
                commitments_result = extract_commitments(transcript_text) or {"commitments": []}
                commitments = commitments_result.get('commitments', [])
            
            # Save transcript as JSON with metadata and LLM results
            transcript_data = {
                'id': file_id,
                'title': title,
                'topic': topic,
                'original_filename': filename,
                'date': datetime.now().isoformat(),
                'filepath': file_path,
                'transcript': transcript_text,
                'segments': result['segments'],
                'summary': summary,
                'topics': topics,
                'action_items': action_items,
                'qa_pairs': qa_pairs,
                'decisions': decisions,
                'commitments': commitments
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
                'filename': os.path.basename(file_path),
                'success': False,
                'error': str(e)
            })
            
    # Check if the post request has multiple files with per-file metadata
    elif 'file_count' in request.form:
        file_count = int(request.form.get('file_count', 0))
        
        for i in range(file_count):
            file_key = f'file_{i}'
            title_key = f'title_{i}'
            topic_key = f'topic_{i}'
            
            if file_key not in request.files:
                continue
                
            file = request.files[file_key]
            if file.filename == '':
                continue
                
            title = request.form.get(title_key, file.filename)
            topic = request.form.get(topic_key, '')
            
            original_filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            extension = os.path.splitext(original_filename)[1]
            filename = f"{file_id}{extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            
            try:
                model = get_model()
                result = model.transcribe(filepath)
                transcript_text = result['text']

                # LLM analysis (only if available)
                summary = ""
                topics = []
                action_items = []
                if feature_is_available('llm'):
                    summary = summarize_transcript(transcript_text) or "Failed to generate summary."
                    topics = extract_topics(transcript_text) or []
                    action_items_result = extract_action_items(transcript_text) or {"action_items": []}
                    action_items = action_items_result.get('action_items', [])
                    qa_result = extract_questions_answers(transcript_text) or {"qa_pairs": []}
                    qa_pairs = qa_result.get('qa_pairs', [])
                    
                    decisions_result = extract_decisions(transcript_text) or {"decisions": []}
                    decisions = decisions_result.get('decisions', [])
                    
                    commitments_result = extract_commitments(transcript_text) or {"commitments": []}
                    commitments = commitments_result.get('commitments', [])
                
                # Save transcript as JSON with metadata and LLM results
                transcript_data = {
                    'id': file_id,
                    'title': title,
                    'topic': topic,
                    'original_filename': original_filename,
                    'date': datetime.now().isoformat(),
                    'filepath': filepath,
                    'transcript': transcript_text,
                    'segments': result['segments'],
                    'summary': summary,
                    'topics': topics,
                    'action_items': action_items,
                    'qa_pairs': qa_pairs,
                    'decisions': decisions,
                    'commitments': commitments
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
    
    # Fallback to the standard file upload format
    elif 'file' in request.files:
        files = request.files.getlist('file')
        
        for file in files:
            if file.filename == '':
                continue
                
            if file:
                original_filename = secure_filename(file.filename)
                file_id = str(uuid.uuid4())
                extension = os.path.splitext(original_filename)[1]
                filename = f"{file_id}{extension}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                file.save(filepath)
                
                title = request.form.get('title', original_filename)
                topic = request.form.get('topic', '')
                
                try:
                    model = get_model()
                    result = model.transcribe(filepath)
                    transcript_text = result['text']

                    # LLM analysis (only if available)
                    summary = ""
                    topics = []
                    action_items = []
                    if feature_is_available('llm'):
                        summary = summarize_transcript(transcript_text) or "Failed to generate summary."
                        topics = extract_topics(transcript_text) or []
                        action_items_result = extract_action_items(transcript_text) or {"action_items": []}
                        action_items = action_items_result.get('action_items', [])
                        qa_result = extract_questions_answers(transcript_text) or {"qa_pairs": []}
                        qa_pairs = qa_result.get('qa_pairs', [])
                        
                        decisions_result = extract_decisions(transcript_text) or {"decisions": []}
                        decisions = decisions_result.get('decisions', [])
                        
                        commitments_result = extract_commitments(transcript_text) or {"commitments": []}
                        commitments = commitments_result.get('commitments', [])
                    
                    # Save transcript as JSON with metadata and LLM results
                    transcript_data = {
                        'id': file_id,
                        'title': title,
                        'topic': topic,
                        'original_filename': original_filename,
                        'date': datetime.now().isoformat(),
                        'filepath': filepath,
                        'transcript': transcript_text,
                        'segments': result['segments'],
                        'summary': summary,
                        'topics': topics,
                        'action_items': action_items,
                        'qa_pairs': qa_pairs,
                        'decisions': decisions,
                        'commitments': commitments
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
    else:
        return jsonify({'error': 'No file or YouTube URL provided'}), 400
    
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
    
    download_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}_report.pdf")
    
    # Create the PDF document with better styling
    doc = SimpleDocTemplate(download_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Add custom styles
    title_style = ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=12
    )
    
    heading2_style = ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#3498DB'),
        spaceAfter=10
    )
    
    normal_style = ParagraphStyle(
        name='CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    # Start building content
    content = []

    # Title and Date with better styling
    content.append(Paragraph(f"Transcript Report: {transcript_data['title']}", title_style))
    content.append(Paragraph(f"Date: {transcript_data['date'].split('T')[0]}", normal_style))
    content.append(Spacer(1, 12))
    
    # Summary section
    content.append(Paragraph("Summary", heading2_style))
    summary = transcript_data.get('summary', "No summary available.")
    content.append(Paragraph(summary.replace('\n', '<br/>'), normal_style))
    content.append(Spacer(1, 12))

    # Topics section
    content.append(Paragraph("Topics", heading2_style))
    topics = transcript_data.get('topics', [])
    if topics:
        for topic in topics:
            content.append(Paragraph(topic.get('name', 'Unnamed Topic'), ParagraphStyle(
                name='TopicHeading',
                parent=styles['Heading3'],
                textColor=colors.HexColor('#2980B9'),
                fontSize=12
            )))
            for point in topic.get('points', []):
                content.append(Paragraph(f"• {point}", normal_style))
    else:
        content.append(Paragraph("No topics identified.", normal_style))
    content.append(Spacer(1, 12))

    # Action Items with improved table
    content.append(Paragraph("Action Items", heading2_style))
    action_items = transcript_data.get('action_items', [])
    if action_items and isinstance(action_items, list):
        # Define table headers
        action_data = [[
            Paragraph("<b>Task</b>", normal_style),
            Paragraph("<b>Assignee</b>", normal_style),
            Paragraph("<b>Due</b>", normal_style),
            Paragraph("<b>Priority</b>", normal_style)
        ]]
        
        # Add wrapped action item rows
        for item in action_items:
            task = Paragraph(item.get('task', 'Unnamed Task'), normal_style)
            assignee = Paragraph(item.get('assignee', 'Unassigned'), normal_style)
            due = Paragraph(item.get('due', 'Not specified'), normal_style)
            
            priority = item.get('priority', 'Medium')
            priority_color = '#E74C3C' if priority.lower() == 'high' else '#F39C12' if priority.lower() == 'medium' else '#2ECC71'
            priority_cell = Paragraph(f'<font color="{priority_color}">{priority}</font>', normal_style)
            
            action_data.append([task, assignee, due, priority_cell])
        
        # Create table with adjusted widths and improved styling
        table = Table(action_data, colWidths=[200, 100, 100, 50])
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8F9F9'), colors.HexColor('#EAECEE')]),
        ]))
        content.append(table)
    else:
        content.append(Paragraph("No action items identified.", normal_style))
    content.append(Spacer(1, 12))
    
    # Questions & Answers Section (New)
    qa_pairs = transcript_data.get('qa_pairs', [])
    if qa_pairs:
        content.append(Paragraph("Questions & Answers", heading2_style))
        for qa in qa_pairs:
            qa_text = f"""
            <b>Q:</b> {qa.get('question', 'Unknown question')}<br/>
            <i>Asked by:</i> {qa.get('asker', 'Unknown')}<br/>
            <b>A:</b> {qa.get('answer', 'No answer provided')}<br/>
            <i>Answered by:</i> {qa.get('answerer', 'Unknown')}
            """
            content.append(Paragraph(qa_text, ParagraphStyle(
                name='QA',
                parent=normal_style,
                backColor=colors.HexColor('#EBF5FB'),
                borderPadding=10,
                borderWidth=1,
                borderColor=colors.HexColor('#AED6F1'),
                borderRadius=8,
            )))
            content.append(Spacer(1, 6))
        content.append(Spacer(1, 12))
    
    # Key Decisions Section (New)
    decisions = transcript_data.get('decisions', [])
    if decisions:
        content.append(Paragraph("Key Decisions", heading2_style))
        for decision in decisions:
            decision_text = f"""
            <b>{decision.get('decision', 'Unnamed Decision')}</b><br/>
            <i>Context:</i> {decision.get('context', 'No context provided')}<br/>
            <i>Stakeholders:</i> {', '.join(decision.get('stakeholders', ['Unknown']))}<br/>
            <i>Impact:</i> {decision.get('impact', 'Medium')}<br/>
            <i>Next Steps:</i> {decision.get('next_steps', 'None specified')}
            """
            content.append(Paragraph(decision_text, normal_style))
            content.append(Spacer(1, 6))
        content.append(Spacer(1, 12))
    
    # Personal Commitments Section (New)
    commitments = transcript_data.get('commitments', [])
    if commitments:
        content.append(Paragraph("Personal Commitments", heading2_style))
        for commitment in commitments:
            commitment_text = f"""
            <b>{commitment.get('person', 'Unknown')}:</b> {commitment.get('commitment', 'Unspecified')}<br/>
            <i>Timeframe:</i> {commitment.get('timeframe', 'Not specified')}<br/>
            <i>Confidence:</i> {commitment.get('confidence', 'Medium')}
            """
            content.append(Paragraph(commitment_text, normal_style))
            content.append(Spacer(1, 6))
        content.append(Spacer(1, 12))
    
    # Add a page break before the full transcript
    content.append(PageBreak())
    
    # Full transcript with timestamps
    content.append(Paragraph("Full Transcript", heading2_style))
    transcript_text = ""
    for segment in transcript_data['segments']:
        minutes = int(segment['start'] // 60)
        seconds = int(segment['start'] % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        content.append(Paragraph(
            f'<font color="#3498DB"><b>[{time_str}]</b></font> {segment["text"]}',
            normal_style
        ))
    
    # Build the document
    doc.build(content)
    
    return send_file(download_path, as_attachment=True, download_name=f"{transcript_data['title']}_report.pdf")

@app.route('/api/transcripts/<transcript_id>/save-topics', methods=['POST'])
def save_topics(transcript_id):
    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
    
    if not os.path.exists(transcript_path):
        return jsonify({'success': False, 'error': 'Transcript not found'}), 404
    
    try:
        # Get data from request
        data = request.json
        topics = data.get('topics', [])
        
        # Load existing transcript
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
            
        # Update topics
        transcript_data['topics'] = topics
        
        # Save updated transcript
        with open(transcript_path, 'w') as f:
            json.dump(transcript_data, f)
            
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5050)

