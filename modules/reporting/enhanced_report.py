from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import os
from datetime import datetime

def generate_enhanced_report(transcript_data, output_path, logo_path=None):
    """
    Generate an enhanced PDF report with better styling and all meeting intelligence features.
    
    Args:
        transcript_data: Dictionary containing transcript and analysis data
        output_path: Path where to save the PDF
        logo_path: Optional path to company logo
    """
    # Create the document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    
    # Add custom styles
    styles.add(ParagraphStyle(
        name='Heading1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.HexColor('#2C3E50')
    ))
    
    styles.add(ParagraphStyle(
        name='Heading2',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=10,
        textColor=colors.HexColor('#3498DB')
    ))
    
    styles.add(ParagraphStyle(
        name='Heading3',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.HexColor('#2980B9')
    ))
    
    styles.add(ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=8
    ))
    
    styles.add(ParagraphStyle(
        name='Bullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        firstLineIndent=0,
        spaceBefore=0,
        bulletIndent=10,
        bulletFontName='Helvetica',
        bulletFontSize=10,
        bulletText='•'
    ))
    
    # Prepare content elements
    elements = []
    
    # Add logo and header
    header_data = [["", ""]]
    if logo_path and os.path.exists(logo_path):
        img = Image(logo_path, width=1.5*inch, height=0.5*inch)
        header_data[0][0] = img
    
    header_title = f"""
    <font size="14" color="#2C3E50"><b>Meeting Transcript Report</b></font><br/>
    <font size="10" color="#7F8C8D">{transcript_data['title']}</font><br/>
    <font size="8" color="#95A5A6">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</font>
    """
    header_data[0][1] = Paragraph(header_title, styles['Normal'])
    
    header = Table(header_data, colWidths=[2*inch, 4*inch])
    header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    elements.append(header)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add divider
    elements.append(Table([['']], colWidths=[7.5*inch], rowHeights=[1]))
    elements[-1].setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#3498DB')),
    ]))
    elements.append(Spacer(1, 0.25*inch))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", styles['Heading1']))
    
    if transcript_data.get('summary'):
        summary_text = transcript_data['summary'].replace('\n', '<br/>')
        elements.append(Paragraph(summary_text, styles['Normal']))
    else:
        elements.append(Paragraph("No summary available.", styles['Normal']))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Key Topics Section
    elements.append(Paragraph("Key Topics", styles['Heading1']))
    
    if transcript_data.get('topics') and len(transcript_data['topics']) > 0:
        for topic in transcript_data['topics']:
            elements.append(Paragraph(topic.get('name', 'Unnamed Topic'), styles['Heading3']))
            
            for point in topic.get('points', []):
                elements.append(Paragraph(f"• {point}", styles['Bullet']))
            
            elements.append(Spacer(1, 0.1*inch))
    else:
        elements.append(Paragraph("No topics identified.", styles['Normal']))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Action Items Section with enhanced table
    elements.append(Paragraph("Action Items", styles['Heading1']))
    
    if transcript_data.get('action_items') and len(transcript_data['action_items']) > 0:
        # Create action items table with improved styling
        action_data = [
            [
                Paragraph("<b>Task</b>", styles['Normal']),
                Paragraph("<b>Assignee</b>", styles['Normal']),
                Paragraph("<b>Due</b>", styles['Normal']),
                Paragraph("<b>Priority</b>", styles['Normal'])
            ]
        ]
        
        for item in transcript_data['action_items']:
            task = Paragraph(item.get('task', 'Unnamed Task'), styles['Normal'])
            assignee = Paragraph(item.get('assignee', 'Unassigned'), styles['Normal'])
            due = Paragraph(item.get('due', 'Not specified'), styles['Normal'])
            
            priority = item.get('priority', 'Medium')
            if priority.lower() == 'high':
                priority_color = '#E74C3C'  # Red for high priority
            elif priority.lower() == 'low':
                priority_color = '#2ECC71'  # Green for low priority
            else:
                priority_color = '#F39C12'  # Orange for medium priority
                
            priority_cell = Paragraph(f'<font color="{priority_color}">{priority}</font>', styles['Normal'])
            
            action_data.append([task, assignee, due, priority_cell])
        
        # Set column widths proportionally
        action_table = Table(action_data, colWidths=[3.5*inch, 1.5*inch, 1.25*inch, 1.25*inch])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9F9')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8F9F9'), colors.HexColor('#EAECEE')]),
        ]))
        elements.append(action_table)
    else:
        elements.append(Paragraph("No action items identified.", styles['Normal']))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Key Decisions Section (New)
    elements.append(Paragraph("Key Decisions", styles['Heading1']))
    
    if transcript_data.get('decisions') and len(transcript_data['decisions']) > 0:
        for i, decision in enumerate(transcript_data['decisions']):
            # Create a styled box for each decision
            decision_text = f"""
            <b>{decision.get('decision', 'Unnamed Decision')}</b><br/>
            <i>Context:</i> {decision.get('context', 'No context provided')}<br/>
            <i>Stakeholders:</i> {', '.join(decision.get('stakeholders', ['Unknown']))}<br/>
            <i>Impact:</i> <font color="{get_impact_color(decision.get('impact', 'Medium'))}">{decision.get('impact', 'Medium')}</font><br/>
            <i>Next Steps:</i> {decision.get('next_steps', 'None specified')}
            """
            
            decision_paragraph = Paragraph(decision_text, styles['Normal'])
            
            # Create a table to act as a container with background and border
            decision_table = Table([[decision_paragraph]], colWidths=[7.5*inch])
            
            # Alternating background colors
            bg_color = '#EBF5FB' if i % 2 == 0 else '#D6EAF8'
            
            decision_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg_color)),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#3498DB')),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ]))
            
            elements.append(decision_table)
            elements.append(Spacer(1, 0.1*inch))
    else:
        elements.append(Paragraph("No key decisions identified.", styles['Normal']))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Q&A Section (New)
    elements.append(Paragraph("Questions & Answers", styles['Heading1']))
    
    if transcript_data.get('qa_pairs') and len(transcript_data['qa_pairs']) > 0:
        for i, qa in enumerate(transcript_data['qa_pairs']):
            qa_text = f"""
            <b>Q:</b> {qa.get('question', 'Unknown question')}<br/>
            <i>Asked by:</i> {qa.get('asker', 'Unknown')}<br/>
            <b>A:</b> {qa.get('answer', 'No answer provided')}<br/>
            <i>Answered by:</i> {qa.get('answerer', 'Unknown')}
            """
            
            if i % 2 == 0:
                elements.append(Paragraph(qa_text, ParagraphStyle(
                    name=f'QA{i}',
                    parent=styles['Normal'],
                    backColor=colors.HexColor('#F5EEF8'),
                    borderPadding=10,
                    borderWidth=1,
                    borderColor=colors.HexColor('#D7BDE2'),
                    borderRadius=8,
                )))
            else:
                elements.append(Paragraph(qa_text, ParagraphStyle(
                    name=f'QA{i}',
                    parent=styles['Normal'],
                    backColor=colors.HexColor('#E8F8F5'),
                    borderPadding=10,
                    borderWidth=1,
                    borderColor=colors.HexColor('#A3E4D7'),
                    borderRadius=8,
                )))
            
            elements.append(Spacer(1, 0.1*inch))
    else:
        elements.append(Paragraph("No question and answer pairs identified.", styles['Normal']))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Personal Commitments Section (New)
    elements.append(Paragraph("Personal Commitments", styles['Heading1']))
    
    if transcript_data.get('commitments') and len(transcript_data['commitments']) > 0:
        # Create a table for commitments
        commitment_data = [
            [
                Paragraph("<b>Person</b>", styles['Normal']),
                Paragraph("<b>Commitment</b>", styles['Normal']),
                Paragraph("<b>Timeframe</b>", styles['Normal']),
                Paragraph("<b>Confidence</b>", styles['Normal'])
            ]
        ]
        
        for commitment in transcript_data['commitments']:
            person = Paragraph(commitment.get('person', 'Unknown'), styles['Normal'])
            commitment_text = Paragraph(commitment.get('commitment', 'Unspecified'), styles['Normal'])
            timeframe = Paragraph(commitment.get('timeframe', 'Not specified'), styles['Normal'])
            
            confidence = commitment.get('confidence', 'Medium')
            confidence_color = get_confidence_color(confidence)
            confidence_cell = Paragraph(f'<font color="{confidence_color}">{confidence}</font>', styles['Normal'])
            
            commitment_data.append([person, commitment_text, timeframe, confidence_cell])
        
        commitment_table = Table(commitment_data, colWidths=[1.5*inch, 3.5*inch, 1.25*inch, 1.25*inch])
        commitment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9B59B6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D7BDE2')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5EEF8')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F5EEF8'), colors.HexColor('#EBDEF0')]),
        ]))
        elements.append(commitment_table)
    else:
        elements.append(Paragraph("No personal commitments identified.", styles['Normal']))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # Add a page break before the transcript
    elements.append(PageBreak())
    
    # Add full transcript
    elements.append(Paragraph("Full Transcript", styles['Heading1']))
    
    if transcript_data.get('segments') and len(transcript_data['segments']) > 0:
        for segment in transcript_data['segments']:
            time_str = format_time(segment['start'])
            elements.append(Paragraph(
                f'<font color="#3498DB"><b>[{time_str}]</b></font> {segment["text"]}',
                styles['Normal']
            ))
            elements.append(Spacer(1, 0.05*inch))
    else:
        elements.append(Paragraph("No transcript segments available.", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    
    return output_path

def format_time(seconds):
    """Format seconds into MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_impact_color(impact):
    """Get color for impact level"""
    impact = impact.lower()
    if impact == 'high':
        return '#C0392B'  # Dark Red
    elif impact == 'low':
        return '#27AE60'  # Dark Green
    else:  # Medium or default
        return '#D35400'  # Dark Orange

def get_confidence_color(confidence):
    """Get color for confidence level"""
    confidence = confidence.lower()
    if confidence == 'high':
        return '#27AE60'  # Dark Green
    elif confidence == 'low':
        return '#C0392B'  # Dark Red
    else:  # Medium or default
        return '#D35400'  # Dark Orange

# Now update the download_transcript endpoint in app.py to use this new function
"""
@app.route('/download/<transcript_id>')
def download_transcript(transcript_id):
    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}.json")
    if not os.path.exists(transcript_path):
        return jsonify({'error': 'Transcript not found'}), 404
    
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    download_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{transcript_id}_report.pdf")
    
    # Path to logo if you have one
    logo_path = os.path.join(app.root_path, 'static', 'images', 'logo.png')
    if not os.path.exists(logo_path):
        logo_path = None
    
    # Generate the enhanced report
    generate_enhanced_report(transcript_data, download_path, logo_path)
    
    return send_file(download_path, as_attachment=True, download_name=f"{transcript_data['title']}_report.pdf")
"""