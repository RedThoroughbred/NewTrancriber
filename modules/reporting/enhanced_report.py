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
    
    # Modify existing styles directly
    styles['Heading1'].fontSize = 18
    styles['Heading1'].spaceAfter = 12
    styles['Heading1'].textColor = colors.HexColor('#2C3E50')
    
    styles['Heading2'].fontSize = 16
    styles['Heading2'].spaceAfter = 10
    styles['Heading2'].textColor = colors.HexColor('#3498DB')
    
    styles['Heading3'].fontSize = 14
    styles['Heading3'].spaceAfter = 8
    styles['Heading3'].textColor = colors.HexColor('#2980B9')
    
    styles['Normal'].fontSize = 10
    styles['Normal'].leading = 14
    styles['Normal'].spaceAfter = 8
    
    styles['Bullet'].leftIndent = 20
    styles['Bullet'].firstLineIndent = 0
    styles['Bullet'].spaceBefore = 0
    styles['Bullet'].bulletIndent = 10
    styles['Bullet'].bulletFontName = 'Helvetica'
    styles['Bullet'].bulletFontSize = 10
    styles['Bullet'].bulletText = '•'
    
    # Add a custom caption style with unique name
    custom_caption_style = ParagraphStyle(
        name='CustomCaption',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.gray,
        alignment=1,  # Center alignment
        spaceAfter=12
    )
    styles.add(custom_caption_style)
    
    print(f"Generating report for: {transcript_data.get('title')}")
    print(f"Key moments available: {transcript_data.get('key_moments') is not None}")
    if transcript_data.get('key_moments'):
        print(f"Number of key moments: {len(transcript_data.get('key_moments', []))}")
        for i, moment in enumerate(transcript_data['key_moments']):
            print(f"  Moment {i+1}: {moment.get('title')}, ts={moment.get('timestamp')}")
            print(f"  Has screenshot: {'screenshot_path' in moment}")
            if 'screenshot_path' in moment:
                print(f"  Screenshot path: {moment['screenshot_path']}")

    # Prepare content elements
    elements = []
    
    # Add logo and header
    header_data = [["", ""]]
    if logo_path and os.path.exists(logo_path):
        img = Image(logo_path, width=1.5*inch, height=0.5*inch)
        header_data[0][0] = img
    
    title_text = transcript_data.get('title', 'Untitled')
    header_title = f"""
    <font size="14" color="#2C3E50"><b>Meeting Transcript Report</b></font><br/>
    <font size="10" color="#7F8C8D">{title_text}</font><br/>
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
        summary_text = str(transcript_data['summary']).replace('\n', '<br/>')
        elements.append(Paragraph(summary_text, styles['Normal']))
    else:
        elements.append(Paragraph("No summary available.", styles['Normal']))
    
    elements.append(Spacer(1, 0.25*inch))
    
    if transcript_data.get('key_moments') and len(transcript_data['key_moments']) > 0:
        elements.append(Paragraph("Key Visual Moments", styles['Heading1']))
        elements.append(Paragraph("Important moments from the meeting with visual context", styles['Normal']))
        elements.append(Spacer(1, 0.15*inch))
        
        for i, moment in enumerate(transcript_data['key_moments']):
            # Add moment title with timestamp
            timestamp_str = format_timestamp(moment.get('timestamp', 0))
            moment_title = f"{moment.get('title', f'Key Moment {i+1}')} ({timestamp_str})"
            elements.append(Paragraph(moment_title, styles['Heading3']))
            
            # Check if there's a screenshot
            if 'screenshot_path' in moment and moment['screenshot_path']:
                try:
                    # Process the screenshot path
                    screenshot_path = moment['screenshot_path']
                    
                    # Handle various path formats
                    if screenshot_path.startswith('/static/'):
                        # Try multiple methods to find the correct path
                        import os
                        
                        # Method 1: Try using Flask app's static folder if available
                        try:
                            from flask import current_app
                            if current_app:
                                screenshot_path = screenshot_path.replace('/static/', current_app.static_folder + os.path.sep)
                        except (ImportError, RuntimeError):
                            pass
                            
                        # Method 2: Try relative path from current directory
                        if not os.path.exists(screenshot_path):
                            relative_path = screenshot_path.replace('/static/', 'static/')
                            if os.path.exists(relative_path):
                                screenshot_path = relative_path
                                
                        # Method 3: Try using parent directory of script
                        if not os.path.exists(screenshot_path):
                            parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                            alt_path = os.path.join(parent_dir, screenshot_path.lstrip('/'))
                            if os.path.exists(alt_path):
                                screenshot_path = alt_path
                    
                    # Check for PNG version if JPG not found
                    if not os.path.exists(screenshot_path) and screenshot_path.endswith('.jpg'):
                        png_path = screenshot_path.replace('.jpg', '.png')
                        if os.path.exists(png_path):
                            screenshot_path = png_path
                    
                    # If the file exists, add it to the report
                    if os.path.exists(screenshot_path):
                        # Add screenshot image with proper sizing
                        img = Image(screenshot_path, width=6*inch, height=4*inch, kind='proportional')
                        elements.append(img)
                        
                        # Add caption
                        caption = f"Screenshot at {timestamp_str}"
                        elements.append(Paragraph(caption, styles['CustomCaption']))
                    else:
                        elements.append(Paragraph(f"[Screenshot not found: {screenshot_path}]", styles['Normal']))
                except Exception as e:
                    print(f"Error adding image: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add a placeholder if the image fails
                    elements.append(Paragraph(f"[Screenshot could not be included: {str(e)}]", styles['Normal']))
            
            # Add moment description
            if 'description' in moment and moment['description']:
                description_text = str(moment.get('description', ''))
                elements.append(Paragraph(description_text, styles['Normal']))
                
            # Add transcript text
            if 'transcript_text' in moment and moment['transcript_text']:
                transcript_text = str(moment.get('transcript_text', ''))
                # Create a styled text box
                text_box = Table([[Paragraph(transcript_text, styles['Normal'])]], 
                               colWidths=[7*inch], 
                               rowHeights=None)
                text_box.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F5F5')),
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ]))
                elements.append(text_box)
            
            elements.append(Spacer(1, 0.35*inch))
        
        # Page break after key moments
        elements.append(PageBreak())

    # Key Topics Section
    elements.append(Paragraph("Key Topics", styles['Heading1']))
    
    if transcript_data.get('topics') and len(transcript_data['topics']) > 0:
        for topic in transcript_data['topics']:
            topic_name = str(topic.get('name', 'Unnamed Topic'))
            elements.append(Paragraph(topic_name, styles['Heading3']))
            
            for point in topic.get('points', []):
                point_text = str(point) if point is not None else ''
                elements.append(Paragraph(f"• {point_text}", styles['Bullet']))
            
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
            # Safely handle None values in all fields
            task_text = item.get('task', 'Unnamed Task') if item.get('task') is not None else 'Unnamed Task'
            assignee_text = item.get('assignee', 'Unassigned') if item.get('assignee') is not None else 'Unassigned'
            due_text = item.get('due', 'Not specified') if item.get('due') is not None else 'Not specified'
            
            # Create paragraph objects with safe values
            task = Paragraph(task_text, styles['Normal'])
            assignee = Paragraph(assignee_text, styles['Normal'])
            due = Paragraph(due_text, styles['Normal'])
            
            # Handle priority safely
            priority = item.get('priority', 'Medium') if item.get('priority') is not None else 'Medium'
            priority_lower = priority.lower() if hasattr(priority, 'lower') else 'medium'
            
            if priority_lower == 'high':
                priority_color = '#E74C3C'  # Red for high priority
            elif priority_lower == 'low':
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
            # Handle possible None values
            decision_title = decision.get('decision', 'Unnamed Decision') if decision.get('decision') is not None else 'Unnamed Decision'
            context = decision.get('context', 'No context provided') if decision.get('context') is not None else 'No context provided'
            
            # Safely handle stakeholders list
            stakeholders = decision.get('stakeholders', ['Unknown'])
            stakeholders_text = ', '.join(stakeholders) if stakeholders is not None else 'Unknown'
            
            # Safe handling for impact and next steps
            impact = decision.get('impact', 'Medium') if decision.get('impact') is not None else 'Medium'
            next_steps = decision.get('next_steps', 'None specified') if decision.get('next_steps') is not None else 'None specified'
            
            decision_text = f"""
            <b>{decision_title}</b><br/>
            <i>Context:</i> {context}<br/>
            <i>Stakeholders:</i> {stakeholders_text}<br/>
            <i>Impact:</i> <font color="{get_impact_color(impact)}">{impact}</font><br/>
            <i>Next Steps:</i> {next_steps}
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
            # Safely handle possible None values
            question = qa.get('question', 'Unknown question') if qa.get('question') is not None else 'Unknown question'
            asker = qa.get('asker', 'Unknown') if qa.get('asker') is not None else 'Unknown'
            answer = qa.get('answer', 'No answer provided') if qa.get('answer') is not None else 'No answer provided'
            answerer = qa.get('answerer', 'Unknown') if qa.get('answerer') is not None else 'Unknown'
            
            qa_text = f"""
            <b>Q:</b> {question}<br/>
            <i>Asked by:</i> {asker}<br/>
            <b>A:</b> {answer}<br/>
            <i>Answered by:</i> {answerer}
            """
            
            # Create a ParagraphStyle without registering it in the stylesheet
            qa_style = ParagraphStyle(
                name=f'qa_style_{i}',  # Use a unique name just for safety
                parent=styles['Normal'],
                backColor=colors.HexColor('#F5EEF8' if i % 2 == 0 else '#E8F8F5'),
                borderPadding=10,
                borderWidth=1,
                borderColor=colors.HexColor('#D7BDE2' if i % 2 == 0 else '#A3E4D7'),
                borderRadius=8,
            )
            elements.append(Paragraph(qa_text, qa_style))
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
            # Handle possible None values
            person_text = commitment.get('person', 'Unknown') if commitment.get('person') is not None else 'Unknown'
            commitment_desc = commitment.get('commitment', 'Unspecified') if commitment.get('commitment') is not None else 'Unspecified'
            timeframe_text = commitment.get('timeframe', 'Not specified') if commitment.get('timeframe') is not None else 'Not specified'
            confidence_text = commitment.get('confidence', 'Medium') if commitment.get('confidence') is not None else 'Medium'
            
            person = Paragraph(person_text, styles['Normal'])
            commitment_text = Paragraph(commitment_desc, styles['Normal']) 
            timeframe = Paragraph(timeframe_text, styles['Normal'])
            
            confidence_color = get_confidence_color(confidence_text)
            confidence_cell = Paragraph(f'<font color="{confidence_color}">{confidence_text}</font>', styles['Normal'])
            
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
            time_str = format_time(segment.get('start', 0))
            segment_text = str(segment.get('text', ''))
            elements.append(Paragraph(
                f'<font color="#3498DB"><b>[{time_str}]</b></font> {segment_text}',
                styles['Normal']
            ))
            elements.append(Spacer(1, 0.05*inch))
    else:
        elements.append(Paragraph("No transcript segments available.", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    
    return output_path

def format_timestamp(seconds):
    """Format seconds into MM:SS format with leading zeros"""
    if seconds is None:
        return "00:00"
    
    try:
        seconds = float(seconds)
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    except (ValueError, TypeError):
        return "00:00"

def format_time(seconds):
    """Format seconds into MM:SS format"""
    try:
        minutes = int(float(seconds) // 60)
        seconds = int(float(seconds) % 60)
        return f"{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "00:00"

def get_impact_color(impact):
    """Get color for impact level"""
    try:
        impact_lower = impact.lower() if hasattr(impact, 'lower') else 'medium'
        if impact_lower == 'high':
            return '#C0392B'  # Dark Red
        elif impact_lower == 'low':
            return '#27AE60'  # Dark Green
        else:  # Medium or default
            return '#D35400'  # Dark Orange
    except (AttributeError, TypeError):
        return '#D35400'  # Default to medium/orange

def get_confidence_color(confidence):
    """Get color for confidence level"""
    try:
        confidence_lower = confidence.lower() if hasattr(confidence, 'lower') else 'medium'
        if confidence_lower == 'high':
            return '#27AE60'  # Dark Green
        elif confidence_lower == 'low':
            return '#C0392B'  # Dark Red
        else:  # Medium or default
            return '#D35400'  # Dark Orange
    except (AttributeError, TypeError):
        return '#D35400'  # Default to medium/orange