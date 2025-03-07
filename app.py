from flask import Flask, render_template, request, jsonify, send_file, redirect
import os
import uuid
import json
import threading
import tempfile
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
        if file.endswith('.json') and not file.startswith('comparison_'):
            try:
                with open(os.path.join(app.config['TRANSCRIPT_FOLDER'], file), 'r') as f:
                    data = json.load(f)
                    
                    # Check if this is a transcript file (has 'transcript' field)
                    if 'transcript' in data:
                        # Add a preview of the transcript (first 200 characters)
                        data['preview'] = data['transcript'][:200] + "..." if len(data['transcript']) > 200 else data['transcript']
                        transcripts.append(data)
            except Exception as e:
                print(f"Error loading transcript {file}: {e}")
                continue
    
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

# Track ongoing multi-transcript analysis jobs
analysis_jobs = {}

@app.route('/compare-transcripts')
def compare_transcripts():
    """Handle multi-transcript comparison view"""
    # Get transcript IDs from query string
    ids_param = request.args.get('ids', '')
    if not ids_param:
        return redirect('/dashboard')
        
    transcript_ids = ids_param.split(',')
    if len(transcript_ids) < 2:
        return redirect('/dashboard')
    
    # Load transcripts
    transcripts = []
    for transcript_id in transcript_ids:
        transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                try:
                    transcript_data = json.load(f)
                    transcripts.append(transcript_data)
                except:
                    continue
    
    if len(transcripts) < 2:
        # Not enough valid transcripts
        return redirect('/dashboard')
    
    # SIMPLIFIED: Always generate a new analysis
    print(f"Generating fresh analysis for {len(transcripts)} transcripts")
    
    # Import here to avoid circular imports
    from modules.analysis.multi_transcript import analyze_multiple_transcripts
    
    try:
        # Run analysis
        analysis_id = str(uuid.uuid4())
        analysis_results = analyze_multiple_transcripts(
            transcript_ids,
            app.config['TRANSCRIPT_FOLDER'],
            use_llm=False,
            analysis_id=analysis_id
        )
        
        # Save the analysis
        analysis_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"comparison_{analysis_id}.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f)
        
        # Print some details for debugging
        print(f"Analysis completed successfully:")
        print(f"- ID: {analysis_results.get('id')}")
        print(f"- Summary length: {len(analysis_results.get('comparative_summary', ''))}")
        print(f"- Topics found: {len(analysis_results.get('common_topics', []))}")
        
        # Debug what's in the summary
        summary = analysis_results.get('comparative_summary', '')
        if summary:
            print(f"- Summary starts with: {summary[:100]}...")
            
            # Check if the summary is rendered correctly
            if summary.startswith('##'):
                print("  (Summary uses markdown format)")
            else:
                print("  (Summary uses plain text format)")
                
        # Use the analysis results directly
        comparison_data = analysis_results
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide a fallback comparison
        comparison_data = {
            'id': str(uuid.uuid4()),
            'date': datetime.now().isoformat(),
            'transcript_ids': transcript_ids,
            'transcripts_metadata': [
                {
                    'id': t.get('id', ''),
                    'title': t.get('title', 'Untitled'),
                    'date': t.get('date', '')
                }
                for t in transcripts
            ],
            'comparative_summary': f"Error during analysis: {str(e)}",
            'common_topics': [],
            'evolving_topics': [],
            'conflicting_information': [],
            'action_item_status': []
        }
    
    # Debug what's being passed to the template
    print("Passing to template:")
    print(f"- Transcripts: {len(transcripts)}")
    print(f"- Comparison data: {type(comparison_data)}")
    print(f"- Summary in comparison data: {len(comparison_data.get('comparative_summary', ''))}")
    
    # Render the template with the analysis results
    return render_template(
        'transcript_comparison.html', 
        transcripts=transcripts, 
        comparison=comparison_data, 
        request=request
    )

@app.route('/compare-transcripts/status')
def compare_transcripts_status():
    """Check the status of a multi-transcript analysis job"""
    # Get the job ID from the request
    ids_param = request.args.get('ids', '')
    if not ids_param:
        return jsonify({'status': 'error', 'message': 'No transcript IDs provided'})
    
    transcript_ids = ids_param.split(',')
    
    # Look for a matching job
    for job_id, job_data in analysis_jobs.items():
        if sorted(job_data.get('transcript_ids', [])) == sorted(transcript_ids):
            print(f"Found job {job_id} with status: {job_data['status']}")
            if job_data['status'] == 'completed':
                # Return the results
                return jsonify({
                    'status': 'completed',
                    'job_id': job_id
                })
            elif job_data['status'] == 'failed':
                # Return the error
                return jsonify({
                    'status': 'failed',
                    'message': job_data.get('error', 'Unknown error')
                })
            else:
                # Still running
                return jsonify({
                    'status': 'running',
                    'job_id': job_id
                })
    
    # Check for existing saved analysis
    for file in os.listdir(app.config['TRANSCRIPT_FOLDER']):
        if file.startswith('comparison_') and file.endswith('.json'):
            analysis_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], file)
            try:
                with open(analysis_path, 'r') as f:
                    analysis_data = json.load(f)
                    # Check if this analysis contains our transcript IDs
                    if sorted(analysis_data.get('transcript_ids', [])) == sorted(transcript_ids):
                        print(f"Found saved analysis: {analysis_data.get('id')}")
                        return jsonify({
                            'status': 'completed',
                            'job_id': analysis_data.get('id')
                        })
            except Exception as e:
                print(f"Error reading analysis file {file}: {e}")
                continue
    
    print(f"No job or saved analysis found for transcript IDs: {transcript_ids}")
    
    # Check if we need to create a new job
    # This can happen if the page was refreshed and the job was lost
    comparison_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"comparison_{transcript_ids[0]}_{transcript_ids[1]}.json")
    
    # Load the transcripts
    transcripts = []
    for transcript_id in transcript_ids:
        transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, 'r') as f:
                    transcript_data = json.load(f)
                    transcripts.append(transcript_data)
            except:
                continue
    
    if len(transcripts) >= 2 and transcript_ids[0] not in analysis_jobs:
        print(f"Creating new analysis job for transcript IDs: {transcript_ids}")
        
        # Create basic comparison data
        comparison_data = {
            'id': str(uuid.uuid4()),
            'date': datetime.now().isoformat(),
            'transcript_ids': transcript_ids,
            'transcripts_metadata': [
                {
                    'id': t.get('id', ''),
                    'title': t.get('title', 'Untitled'),
                    'date': t.get('date', '')
                }
                for t in transcripts
            ],
            'comparative_summary': 'Basic comparison analysis.',
            'common_topics': [],
            'evolving_topics': [],
            'conflicting_information': [],
            'action_item_status': []
        }
        
        # Save it without LLM processing
        try:
            with open(comparison_path, 'w') as f:
                json.dump(comparison_data, f)
            
            print(f"Saved basic comparison data to {comparison_path}")
            return jsonify({
                'status': 'completed',
                'job_id': comparison_data['id']
            })
        except Exception as e:
            print(f"Error saving comparison data: {e}")
    
    # No matching job or saved analysis found
    return jsonify({
        'status': 'not_found'
    })

@app.route('/test-comparison')
def test_comparison():
    """Test route to debug the transcript comparison functionality"""
    # Get transcript IDs from query string
    ids_param = request.args.get('ids', '')
    if not ids_param:
        return jsonify({'error': 'No transcript IDs provided'})
        
    transcript_ids = ids_param.split(',')
    
    # Load transcripts
    transcripts = []
    for transcript_id in transcript_ids:
        transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, 'r') as f:
                    transcript_data = json.load(f)
                    transcripts.append(transcript_data)
            except Exception as e:
                return jsonify({'error': f'Error loading transcript {transcript_id}: {str(e)}'})
    
    # Run analysis directly
    try:
        print("TEST ROUTE: Running transcript analysis...")
        
        # Validate transcript data
        validation_results = []
        for idx, t in enumerate(transcripts):
            validation = {
                'title': t.get('title', 'Untitled'),
                'id': t.get('id', 'No ID'),
                'date': t.get('date', 'No date'),
                'has_text': 'transcript' in t and bool(t['transcript']),
                'word_count': len(t.get('transcript', '').split()) if 'transcript' in t and t['transcript'] else 0
            }
            validation_results.append(validation)
        
        # Don't import at the top level to avoid circular imports
        from modules.analysis.multi_transcript import analyze_multiple_transcripts
        
        # Run analysis with a new ID
        test_id = str(uuid.uuid4())
        analysis_results = analyze_multiple_transcripts(
            transcript_ids,
            app.config['TRANSCRIPT_FOLDER'],
            use_llm=False,
            analysis_id=test_id
        )
        
        # Return combined debug information
        return jsonify({
            'test_id': test_id,
            'transcript_validation': validation_results,
            'analysis_results': {
                'id': analysis_results.get('id'),
                'summary_length': len(analysis_results.get('comparative_summary', '')),
                'common_topics_count': len(analysis_results.get('common_topics', [])),
                'summary_excerpt': analysis_results.get('comparative_summary', '')[:200] + '...' if len(analysis_results.get('comparative_summary', '')) > 200 else analysis_results.get('comparative_summary', '')
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Error in analysis: {str(e)}',
            'transcript_count': len(transcripts),
            'transcript_ids': transcript_ids
        })

@app.route('/download-comparison')
def download_comparison():
    """Download a PDF report of a multi-transcript comparison"""
    # Get transcript IDs from query string
    ids_param = request.args.get('ids', '')
    if not ids_param:
        return redirect('/dashboard')
        
    transcript_ids = ids_param.split(',')
    
    # Load transcripts
    transcripts = []
    for transcript_id in transcript_ids:
        transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, 'r') as f:
                    transcript_data = json.load(f)
                    transcripts.append(transcript_data)
            except:
                continue
    
    if len(transcripts) < 2:
        # Not enough valid transcripts
        return redirect('/dashboard')
    
    # Generate a fresh analysis for the report
    try:
        # Import here to avoid circular imports
        from modules.analysis.multi_transcript import analyze_multiple_transcripts
        
        # Run analysis
        analysis_id = str(uuid.uuid4())
        print(f"Generating fresh analysis for report with ID: {analysis_id}")
        
        analysis_data = analyze_multiple_transcripts(
            transcript_ids,
            app.config['TRANSCRIPT_FOLDER'],
            use_llm=False,
            analysis_id=analysis_id
        )
        
        # Save it for future reference
        analysis_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"comparison_{analysis_id}.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f)
            
        print(f"Using fresh analysis for PDF report: {len(analysis_data.get('comparative_summary', ''))} chars")
        
    except Exception as e:
        print(f"Error generating analysis for PDF: {e}")
        # Try to find an existing analysis as fallback
        analysis_data = None
        for file in os.listdir(app.config['TRANSCRIPT_FOLDER']):
            if file.startswith('comparison_') and file.endswith('.json'):
                analysis_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], file)
                try:
                    with open(analysis_path, 'r') as f:
                        data = json.load(f)
                        # Check if this analysis contains our transcript IDs
                        if sorted(data.get('transcript_ids', [])) == sorted(transcript_ids):
                            analysis_data = data
                            break
                except:
                    continue
                    
        if not analysis_data:
            # No analysis found
            return redirect(f'/compare-transcripts?ids={ids_param}')
    
    # Generate PDF report
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "transcript_comparison.pdf")
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Set up styles
    title_style = styles["Title"]
    
    # Main heading style
    main_heading_style = ParagraphStyle(
        "MainHeading",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.blue,
        spaceAfter=10
    )
    
    # Sub heading style
    sub_heading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.navy,
        spaceAfter=8
    )
    
    # Normal text style
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["Normal"],
        fontSize=11,
        leading=14
    )
    
    # Bullet point style
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=normal_style,
        leftIndent=20,
        firstLineIndent=-15
    )
    
    # Sub-bullet style
    sub_bullet_style = ParagraphStyle(
        "SubBullet",
        parent=bullet_style,
        leftIndent=40,
        firstLineIndent=-15
    )
    
    # Bold text style
    bold_style = ParagraphStyle(
        "Bold",
        parent=normal_style,
        fontName="Helvetica-Bold"
    )
    
    # Date style
    date_style = ParagraphStyle(
        "Date",
        parent=normal_style,
        fontSize=10,
        textColor=colors.grey
    )
    
    # Title
    elements.append(Paragraph("Multi-Transcript Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Date
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", date_style))
    elements.append(Spacer(1, 24))
    
    # Transcript Information
    elements.append(Paragraph("Transcripts Included", main_heading_style))
    elements.append(Spacer(1, 12))
    
    # Create a nicer table for transcript info
    transcript_data = [["Title", "Date", "Topic"]]
    for t in analysis_data.get('transcripts_metadata', []):
        date_str = t.get('date', '').split('T')[0] if t.get('date') else 'N/A'
        topic_str = t.get('topic', 'No topic')
        transcript_data.append([t.get('title', 'Untitled'), date_str, topic_str])
    
    # Create a nicer looking table for PDF
    transcript_table = Table(transcript_data, colWidths=[250, 80, 150])
    transcript_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    elements.append(transcript_table)
    elements.append(Spacer(1, 24))
    
    # Comparative Summary - Use a prettier header
    elements.append(Paragraph("Comparative Analysis", main_heading_style))
    
    # Process the summary with proper formatting
    summary_text = analysis_data.get('comparative_summary', 'No summary available')
    
    # Parse main markdown content and skip the first two lines 
    # (which contain the title we already added)
    lines = summary_text.split('\n')
    line_index = 0
    
    # Skip initial title lines if they exist
    while line_index < len(lines) and (
        lines[line_index].startswith('## Comparative Analysis') or 
        lines[line_index].strip() == '' or
        lines[line_index].startswith('This analysis compares')
    ):
        line_index += 1
        
    # Process the rest of the content
    while line_index < len(lines):
        line = lines[line_index]
        
        if line.strip() == '':
            # Add spacing between paragraphs
            elements.append(Spacer(1, 6))
        elif line.startswith('## '):
            # Main heading (already handled above)
            pass
        elif line.startswith('### '):
            # Sub heading - use our custom style
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(line[4:], sub_heading_style))
            elements.append(Spacer(1, 4))
        elif line.startswith('- **'):
            # Bold bullet point
            elements.append(Paragraph(f"• {line[3:]}", bullet_style))
        elif line.startswith('  - '):
            # Indented bullet point
            elements.append(Paragraph(f"◦ {line[4:]}", sub_bullet_style))
        elif line.startswith('- '):
            # Regular bullet point
            elements.append(Paragraph(f"• {line[2:]}", bullet_style))
        else:
            # Regular paragraph
            elements.append(Paragraph(line, normal_style))
            
        line_index += 1
    elements.append(Spacer(1, 24))
    
    # Common Topics Section
    elements.append(Paragraph("Common Topics", main_heading_style))
    elements.append(Spacer(1, 12))
    
    common_topics = analysis_data.get('common_topics', [])
    if common_topics:
        # Create a styled table for common topics
        topic_data = []
        
        # Add table header
        topic_data.append(["Topic", "Description", "Appears In"])
        
        # Add data rows
        for topic in common_topics:
            name = topic.get('name', 'Unnamed Topic')
            description = topic.get('description', 'No description')
            frequency = topic.get('frequency', 0)
            
            # Format appearance information
            appears_in = f"{frequency} transcript"
            if frequency != 1:
                appears_in += "s"
                
            # Add row to table
            topic_data.append([name, description, appears_in])
            
        # Create styled topic table
        colWidths = [120, 280, 80]
        topic_table = Table(topic_data, colWidths=colWidths)
        topic_table.setStyle(TableStyle([
            # Header row styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            
            # Cell padding
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            
            # Borders and backgrounds
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            
            # Font styling for data cells
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))
        
        elements.append(topic_table)
    else:
        elements.append(Paragraph("No common topics were identified across these transcripts.", normal_style))
    
    elements.append(Spacer(1, 12))
    elements.append(PageBreak())
    
    # Topic Evolution with better styling
    elements.append(Paragraph("Topic Evolution Over Time", main_heading_style))
    elements.append(Spacer(1, 8))
    
    # Add a note about what this section shows
    elements.append(Paragraph(
        "This section tracks how topics appear and evolve across multiple transcripts over time.",
        normal_style
    ))
    elements.append(Spacer(1, 12))
    
    evolving_topics = analysis_data.get('evolving_topics', [])
    if evolving_topics:
        for topic in evolving_topics:
            topic_name = topic.get('name', 'Unnamed Topic')
            
            # Create topic header with box around it
            elements.append(Paragraph(topic_name, sub_heading_style))
            elements.append(Spacer(1, 8))
            
            evolution = topic.get('evolution', [])
            
            # Create timeline table for evolution
            timeline_data = []
            timeline_data.append(["Date", "Transcript", "Details"])
            
            for event in evolution:
                date_str = event.get('date', 'N/A')
                transcript_id = event.get('transcript_id', 'Unknown')
                summary = event.get('summary', 'No details')
                
                # Find transcript title
                transcript_title = "Unknown transcript"
                for t in analysis_data.get('transcripts_metadata', []):
                    if t.get('id') == transcript_id:
                        transcript_title = t.get('title', 'Untitled')
                        break
                
                timeline_data.append([date_str, transcript_title, summary])
            
            # Create styled timeline table
            colWidths = [80, 150, 250]
            timeline_table = Table(timeline_data, colWidths=colWidths)
            timeline_table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.navy),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                
                # Cell padding
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                
                # Borders
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.grey),
                
                # Alternate row colors for readability
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                
                # Make date column bold
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ]))
            
            elements.append(timeline_table)
            elements.append(Spacer(1, 16))
    else:
        elements.append(Paragraph(
            "No topic evolution could be tracked across these transcripts. This may be because there aren't enough common topics or the transcripts don't have sufficient content overlap.",
            normal_style
        ))
    
    elements.append(Spacer(1, 12))
    
    # Action Items with nice styling
    elements.append(Paragraph("Action Item Tracking", main_heading_style))
    elements.append(Spacer(1, 8))
    
    # Add descriptive text
    elements.append(Paragraph(
        "This section tracks action items mentioned across multiple transcripts and their current status.",
        normal_style
    ))
    elements.append(Spacer(1, 12))
    
    action_items = analysis_data.get('action_item_status', [])
    if action_items:
        # Create a table for action items
        action_data = []
        action_data.append(["Action Item", "Assignee", "First Mentioned", "Status", "Notes"])
        
        for item in action_items:
            description = item.get('description', 'Unnamed Action')
            assignee = item.get('assignee', 'Unassigned')
            first_mentioned = item.get('first_mentioned', 'Unknown')
            status = item.get('status', 'pending')
            notes = item.get('notes', '')
            
            # Convert status to readable format
            if status == 'completed':
                status_text = "Completed"
            elif status == 'in_progress':
                status_text = "In Progress"
            else:
                status_text = "Pending"
            
            action_data.append([description, assignee, first_mentioned, status_text, notes])
        
        # Create styled action item table
        colWidths = [150, 90, 80, 70, 90]
        action_table = Table(action_data, colWidths=colWidths)
        action_table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Cell padding
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            
            # Borders
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            
            # Row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            
            # Column alignments
            ('ALIGN', (2, 1), (2, -1), 'CENTER'),  # Center date
            ('ALIGN', (3, 1), (3, -1), 'CENTER'),  # Center status
            
            # Status colors
            ('TEXTCOLOR', (3, 1), (3, -1), colors.black),
        ]))
        
        # Add custom color to status cells
        for i, item in enumerate(action_items, 1):
            status = item.get('status', 'pending')
            if status == 'completed':
                action_table.setStyle(TableStyle([
                    ('BACKGROUND', (3, i), (3, i), colors.lightgreen),
                ]))
            elif status == 'in_progress':
                action_table.setStyle(TableStyle([
                    ('BACKGROUND', (3, i), (3, i), colors.lightyellow),
                ]))
        
        elements.append(action_table)
    else:
        elements.append(Paragraph(
            "No action items were found or tracked across these transcripts.",
            normal_style
        ))
    
    # Build the PDF
    doc.build(elements)
    
    # Return the PDF file
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="transcript_comparison.pdf"
    )

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Enhanced transcribe endpoint that handles video transcription,
    extracts key moments, and captures screenshots at those moments.
    """
    import traceback
    import logging
    
    # Set up logging
    logger = logging.getLogger(__name__)
    
    try:
        # Check if the post request has the file part
        if 'file' not in request.files and 'youtube_url' not in request.form and 'file_path' not in request.form:
            logger.warning("No input source provided (file/youtube/path)")
            return jsonify({'error': 'No input source provided'}), 400
        
        # Process uploaded file, YouTube URL, or file path
        video_path = None
        original_filename = None
        file_id = str(uuid.uuid4())  # Generate a unique ID for this transcription
        
        # Create necessary directories
        transcripts_dir = os.path.join(app.config['TRANSCRIPT_FOLDER'])
        uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'])
        screenshots_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"screenshots_{file_id}")
        
        for directory in [transcripts_dir, uploads_dir, screenshots_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                logger.warning("Empty filename provided")
                return jsonify({'error': 'No selected file'}), 400
            
            original_filename = secure_filename(file.filename)
            video_path = os.path.join(uploads_dir, f"{file_id}{os.path.splitext(original_filename)[1]}")
            
            # Save the uploaded file
            logger.info(f"Saving uploaded file to {video_path}")
            file.save(video_path)
            
            title = request.form.get('title', original_filename)
            topic = request.form.get('topic', '')
        
        # Handle YouTube URL
        elif 'youtube_url' in request.form:
            from youtube_downloader import download_youtube_video
            
            youtube_url = request.form.get('youtube_url')
            title = request.form.get('title', '')
            topic = request.form.get('topic', '')
            
            try:
                # Download the YouTube video
                logger.info(f"Downloading YouTube video: {youtube_url}")
                video_info = download_youtube_video(youtube_url, uploads_dir)
                file_id = video_info['file_id']
                video_path = video_info['file_path']
                if not title:
                    title = video_info['title']
                original_filename = f"{video_info['title']}.mp4"
            except Exception as e:
                logger.error(f"YouTube download failed: {str(e)}")
                return jsonify({'error': f'Failed to download YouTube video: {str(e)}'}), 400
        
        # Handle file path input
        elif 'file_path' in request.form:
            import shutil
            
            file_path = request.form.get('file_path')
            title = request.form.get('title', os.path.basename(file_path))
            topic = request.form.get('topic', '')
            
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return jsonify({'error': f'File not found: {file_path}'}), 400
            
            original_filename = os.path.basename(file_path)
            video_path = os.path.join(uploads_dir, original_filename)
            
            # Copy the file to our uploads directory
            logger.info(f"Copying file from {file_path} to {video_path}")
            shutil.copy2(file_path, video_path)
        
        # Validate video path
        if not video_path or not os.path.exists(video_path):
            logger.error(f"Video file not found at {video_path}")
            return jsonify({'error': 'Invalid video file'}), 400
        
        # Step 1: Transcribe the video with whisper
        try:
            logger.info(f"Transcribing video: {video_path}")
            model = get_model()
            result = model.transcribe(video_path)
            transcript_text = result['text']
            logger.info(f"Transcription complete: {len(transcript_text)} characters")
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
        
        # Step 2: LLM analysis (only if available)
        summary = ""
        topics = []
        action_items = []
        key_moments = []
        
        if feature_is_available('llm'):
            try:
                logger.info("Generating summary from transcript")
                summary = summarize_transcript(transcript_text) or "Failed to generate summary."
                
                logger.info("Extracting topics from transcript")
                topics = extract_topics(transcript_text) or []
                
                logger.info("Extracting action items from transcript")
                action_items_result = extract_action_items(transcript_text) or {"action_items": []}
                action_items = action_items_result.get('action_items', [])
                
                logger.info("Extracting Q&A from transcript")
                qa_result = extract_questions_answers(transcript_text) or {"qa_pairs": []}
                qa_pairs = qa_result.get('qa_pairs', [])
                
                logger.info("Extracting decisions from transcript")
                decisions_result = extract_decisions(transcript_text) or {"decisions": []}
                decisions = decisions_result.get('decisions', [])
                
                logger.info("Extracting commitments from transcript")
                commitments_result = extract_commitments(transcript_text) or {"commitments": []}
                commitments = commitments_result.get('commitments', [])
                
                # Step 3: Extract key visual moments
                try:
                    logger.info("Extracting key visual moments from transcript")
                    from modules.llm.meeting_intelligence import extract_key_visual_moments
                    key_moments_result = extract_key_visual_moments(
                        transcript_text, 
                        result['segments'] if 'segments' in result else None,
                        video_path
                    )
                    
                    if key_moments_result.get('success', False):
                        key_moments = key_moments_result.get('key_moments', [])
                        logger.info(f"Extracted {len(key_moments)} key moments")
                        
                        # Step 4: Extract screenshots for key moments
                        if key_moments and os.path.exists(video_path):
                            try:
                                logger.info(f"Extracting screenshots for {len(key_moments)} key moments")
                                
                                # Extract timestamps
                                timestamps = [moment.get('timestamp', 0) for moment in key_moments]
                                
                                # Extract screenshots
                                from modules.video_processing import extract_screenshots_for_transcript
                                screenshot_results = extract_screenshots_for_transcript(
                                    file_id,
                                    video_path,
                                    timestamps,
                                    screenshots_dir
                                )
                                
                                # Update key moments with screenshot paths
                                for i, result in enumerate(screenshot_results):
                                    if i < len(key_moments):
                                        # Convert the absolute path to a relative web path
                                        static_dir = 'static'
                                        if static_dir in result['screenshot_path']:
                                            web_path = '/' + result['screenshot_path'].split(static_dir)[1]
                                            web_path = '/' + static_dir + web_path
                                        else:
                                            web_path = result['screenshot_path']
                                        
                                        key_moments[i]['screenshot_path'] = web_path
                                        logger.info(f"Screenshot for moment {i+1}: {web_path}")
                            except Exception as e:
                                logger.error(f"Error extracting screenshots: {str(e)}")
                                traceback.print_exc()
                    else:
                        logger.warning(f"Key moments extraction failed: {key_moments_result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error in key moments extraction: {str(e)}")
                    traceback.print_exc()
            except Exception as e:
                logger.error(f"LLM analysis failed: {str(e)}")
                traceback.print_exc()
        
        # Step 5: Save transcript as JSON with metadata and LLM results
        try:
            transcript_data = {
                'id': file_id,
                'title': title,
                'topic': topic,
                'original_filename': original_filename,
                'date': datetime.now().isoformat(),
                'filepath': video_path,
                'transcript': transcript_text,
                'segments': result['segments'] if 'segments' in result else [],
                'summary': summary,
                'topics': topics,
                'action_items': action_items,
                'qa_pairs': qa_pairs if 'qa_pairs' in locals() else [],
                'decisions': decisions if 'decisions' in locals() else [],
                'commitments': commitments if 'commitments' in locals() else [],
                'key_moments': key_moments  # New: Add key moments with screenshots
            }
            
            # Save JSON transcript file with robust error handling
            transcript_path = os.path.join(transcripts_dir, f"{file_id}.json")
            temp_path = os.path.join(transcripts_dir, f"{file_id}_temp.json")
            
            try:
                # First write to a temporary file
                with open(temp_path, 'w') as f:
                    json.dump(transcript_data, f, indent=2)
                
                # If successful, rename to final path
                os.replace(temp_path, transcript_path)
                logger.info(f"Successfully saved transcript to {transcript_path}")
            except Exception as e:
                logger.error(f"Error saving transcript to {transcript_path}: {str(e)}")
                
                # Try an alternate approach
                try:
                    # Direct write to final path
                    with open(transcript_path, 'w') as f:
                        json.dump(transcript_data, f, indent=2)
                    logger.info(f"Saved transcript to {transcript_path} (direct write)")
                except Exception as e2:
                    logger.error(f"Fatal error saving transcript: {str(e2)}")
                    
                    # Save to a fallback location if all else fails
                    fallback_path = os.path.join(os.path.dirname(transcripts_dir), f"{file_id}_fallback.json")
                    with open(fallback_path, 'w') as f:
                        json.dump(transcript_data, f, indent=2)
                    logger.info(f"Saved transcript to fallback path: {fallback_path}")
        except Exception as e:
            logger.error(f"Error preparing transcript data: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Failed to save transcript: {str(e)}'}), 500
        
        # Return success response with the transcript ID
        return jsonify([{
            'id': file_id,
            'title': title,
            'success': True
        }])
            
    except Exception as e:
        logger.error(f"Unhandled exception in transcribe endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/transcript/<transcript_id>')
def view_transcript(transcript_id):
    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
    
    if not os.path.exists(transcript_path):
        return render_template('404.html', message="Transcript not found"), 404
    
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # DEBUG: Print transcript info and available keys
    print(f"Viewing transcript: {transcript_id}")
    print(f"Available keys in transcript data: {list(transcript_data.keys())}")
    
    # DEBUG: Print key moments info if available
    if 'key_moments' in transcript_data:
        print(f"Transcript has {len(transcript_data['key_moments'])} key moments")
        for i, moment in enumerate(transcript_data['key_moments']):
            print(f"Moment {i+1}: {moment.get('title', 'Untitled')} - {moment.get('timestamp')}")
            print(f"Has screenshot: {'screenshot_path' in moment}")
            if 'screenshot_path' in moment:
                print(f"Screenshot path: {moment['screenshot_path']}")
    else:
        print("No key_moments found in transcript data")
        # Create fallback key moments from screenshots folder
        screenshots_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"screenshots_{transcript_id}")
        if os.path.exists(screenshots_dir):
            print(f"Found screenshots directory: {screenshots_dir}")
            screenshot_files = [f for f in os.listdir(screenshots_dir) if f.endswith('.jpg') or f.endswith('.png')]
            print(f"Found {len(screenshot_files)} screenshot files")
            
            if screenshot_files:
                # Create fallback key moments
                key_moments = []
                for i, filename in enumerate(screenshot_files):
                    # Extract timestamp from filename (format: transcript_id_screenshot_index_timestamp.jpg)
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        try:
                            timestamp = float(parts[-1].replace('s.jpg', '').replace('s.png', ''))
                            
                            # Find closest segment to this timestamp
                            closest_segment = min(transcript_data['segments'], key=lambda s: abs(s.get('start', 0) - timestamp))
                            
                            moment = {
                                "timestamp": timestamp,
                                "title": f"Key Visual Moment {i+1}",
                                "description": f"Screenshot captured at {timestamp}s",
                                "transcript_text": closest_segment.get('text', ''),
                                "screenshot_path": f"/static/uploads/screenshots_{transcript_id}/{filename}"
                            }
                            key_moments.append(moment)
                            print(f"Created fallback moment: {moment['title']} at {timestamp}s")
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing filename {filename}: {e}")
                
                transcript_data['key_moments'] = key_moments
                print(f"Added {len(key_moments)} fallback key moments to transcript data")
    
    # Ensure all expected structures exist to prevent attribute errors
    for field in ['action_items', 'qa_pairs', 'topics', 'decisions', 'commitments', 'key_moments']:
        if field not in transcript_data or transcript_data[field] is None:
            transcript_data[field] = []
            print(f"Initialize empty {field} list")
        elif isinstance(transcript_data[field], dict) and field in transcript_data[field]:
            # If it's a dict with matching key, extract the list
            transcript_data[field] = transcript_data[field].get(field, [])
            print(f"Extracted {field} list from dict")
    
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
            # Safely handle None values in all fields
            task_text = item.get('task', 'Unnamed Task') if item.get('task') is not None else 'Unnamed Task'
            assignee_text = item.get('assignee', 'Unassigned') if item.get('assignee') is not None else 'Unassigned'
            due_text = item.get('due', 'Not specified') if item.get('due') is not None else 'Not specified'
            
            # Create paragraph objects with safe values
            task = Paragraph(task_text, normal_style)
            assignee = Paragraph(assignee_text, normal_style)
            due = Paragraph(due_text, normal_style)
            
            # Handle priority safely
            priority = item.get('priority', 'Medium') if item.get('priority') is not None else 'Medium'
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
            # Handle possible None values
            decision_title = decision.get('decision', 'Unnamed Decision') if decision.get('decision') is not None else 'Unnamed Decision'
            context = decision.get('context', 'No context provided') if decision.get('context') is not None else 'No context provided'
            
            # Safely handle stakeholders list
            stakeholders = decision.get('stakeholders', ['Unknown'])
            stakeholders_text = ', '.join(stakeholders) if stakeholders is not None else 'Unknown'
            
            decision_text = f"""
            <b>{decision_title}</b><br/>
            <i>Context:</i> {context}<br/>
            <i>Stakeholders:</i> {stakeholders_text}<br/>
            <i>Impact:</i> {decision.get('impact', 'Medium') if decision.get('impact') is not None else 'Medium'}<br/>
            <i>Next Steps:</i> {decision.get('next_steps', 'None specified') if decision.get('next_steps') is not None else 'None specified'}
            """
            content.append(Paragraph(decision_text, normal_style))
            content.append(Spacer(1, 6))
        content.append(Spacer(1, 12))
    
    # Personal Commitments Section (New)
    commitments = transcript_data.get('commitments', [])
    if commitments:
        content.append(Paragraph("Personal Commitments", heading2_style))
        for commitment in commitments:
            # Handle possible None values
            person = commitment.get('person', 'Unknown') if commitment.get('person') is not None else 'Unknown'
            commitment_desc = commitment.get('commitment', 'Unspecified') if commitment.get('commitment') is not None else 'Unspecified'
            timeframe = commitment.get('timeframe', 'Not specified') if commitment.get('timeframe') is not None else 'Not specified'
            confidence = commitment.get('confidence', 'Medium') if commitment.get('confidence') is not None else 'Medium'
            
            commitment_text = f"""
            <b>{person}:</b> {commitment_desc}<br/>
            <i>Timeframe:</i> {timeframe}<br/>
            <i>Confidence:</i> {confidence}
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
    app.run(debug=True, host='0.0.0.0', port=5050)

