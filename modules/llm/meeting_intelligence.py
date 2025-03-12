"""
Enhanced transcript analysis for meeting intelligence features.

This module adds capabilities to detect questions, answers, decisions, and commitments
in meeting transcripts using LLM analysis. It also supports comparing multiple transcripts.
"""
import logging
from typing import List, Dict, Any, Optional, Union
import time
import re
import json
from .ollama import get_client, is_available

logger = logging.getLogger(__name__)
# System prompts for different analysis types
QUESTIONS_ANSWERS_SYSTEM_PROMPT = """
You are an expert at analyzing meeting transcripts. Your task is to identify all questions asked
and their corresponding answers (if available) in the transcript.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "qa_pairs": [
       {
         "question": "The exact question text from the transcript",
         "asker": "Person who asked (or 'Unknown' if not identifiable)",
         "answer": "The answer given (or 'No answer provided' if none was given)",
         "answerer": "Person who answered (or 'Unknown' if not identifiable)",
         "timestamp": "Approximate time in the meeting (if available)"
       }
     ]
   }
3. Only include clear questions that are explicitly asked
4. Include both direct and rhetorical questions
5. If no questions are found, return {"qa_pairs": []}
"""

DECISIONS_SYSTEM_PROMPT = """
You are an expert at analyzing meeting transcripts. Your task is to identify all decisions
and conclusions reached during the meeting.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "decisions": [
       {
         "decision": "Brief description of the decision made",
         "context": "The full context or quote containing the decision",
         "stakeholders": ["List of people involved in or affected by the decision"],
         "impact": "Brief assessment of the significance (High/Medium/Low)",
         "next_steps": "Any mentioned follow-up actions related to this decision"
       }
     ]
   }
3. Focus on clear decisions, agreements, or conclusions reached
4. Look for phrases like "we've decided", "let's go with", "we agree", etc.
5. If no decisions are found, return {"decisions": []}
"""

COMMITMENTS_SYSTEM_PROMPT = """
You are an expert at analyzing meeting transcripts. Your task is to identify personal
commitments and promises made by participants (distinct from action items which may be assigned).

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "commitments": [
       {
         "person": "Name of person making the commitment",
         "commitment": "Description of what they committed to do",
         "context": "The quote containing the commitment",
         "timeframe": "When they committed to do it (if mentioned)",
         "confidence": "High/Medium/Low based on certainty of the commitment"
       }
     ]
   }
3. Focus on personal commitments using language like "I will", "I'll take care of", "I promise", etc.
4. Distinguish from formal action items that might be assigned to someone
5. If no commitments are found, return {"commitments": []}
"""

def repair_json(text):
    """
    Attempt to repair common JSON syntax errors like missing commas.
    
    Args:
        text: String containing potentially malformed JSON
        
    Returns:
        Repaired JSON string
    """
    if not text or not isinstance(text, str):
        return text
        
    # Check if the content is empty or not JSON-like
    stripped_text = text.strip()
    if not (stripped_text.startswith('{') or stripped_text.startswith('[') or
            stripped_text.startswith('```')):
        return text
    
    # Remove any markdown code block syntax
    # This handles ```json at the start and ``` at the end
    stripped_text = re.sub(r'^```(?:json)?\s*', '', stripped_text)
    stripped_text = re.sub(r'\s*```$', '', stripped_text)
    
    # Fix missing commas between a string value and the next key
    # This pattern looks for: "key": "value" "nextKey":
    fixed_text = re.sub(r'("[^"]*")\s*("[^"]*")\s*:', r'\1,\2:', stripped_text)
    
    # Fix missing commas after numbers followed by property names
    # This pattern looks for: "key": 123 "nextKey":
    fixed_text = re.sub(r'(\d+)\s*("[^"]*")\s*:', r'\1,\2:', fixed_text)
    
    # Fix missing commas after boolean values followed by property names
    # This pattern looks for: "key": true "nextKey":
    fixed_text = re.sub(r'(true|false|null)\s*("[^"]*")\s*:', r'\1,\2:', fixed_text)
    
    # Fix missing commas between array elements
    # This pattern looks for: "item1" "item2"
    fixed_text = re.sub(r'("[^"]*")\s+("[^"]*")', r'\1,\2', fixed_text)
    
    # Fix missing commas in arrays after numbers
    # This pattern looks for: 123 "item2"
    fixed_text = re.sub(r'(\d+)\s+("[^"]*")', r'\1,\2', fixed_text)
    
    # Fix missing commas between entries in an object
    # This pattern looks for: "key": value }
    fixed_text = re.sub(r'("[^"]*")\s*:\s*([^,{\[\s][^,{\[]*?)(\s*})', r'\1: \2,\3', fixed_text)
    
    return fixed_text

def _truncate_text(text: str, max_length: int = 6000) -> str:
    """
    Truncate text to a maximum length while preserving sentence boundaries.
    
    Args:
        text: The text to truncate
        max_length: Maximum length in characters
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Find the last sentence boundary before max_length
    end_markers = ['. ', '! ', '? ']
    best_end = max_length
    
    for marker in end_markers:
        pos = text[:max_length].rfind(marker)
        if pos > 0:
            best_end = pos + len(marker)
            break
    
    return text[:best_end] + "..."

def safe_json_loads(json_str, default_value):
    """
    Safely parse JSON with extensive error handling and repair.
    
    Args:
        json_str: JSON string to parse
        default_value: Default value to return if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    if not json_str or not isinstance(json_str, str):
        return default_value
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Error parsing JSON. Attempting repair...")
        
        # Try to repair and parse
        repaired_json = repair_json(json_str)
        try:
            return json.loads(repaired_json)
        except json.JSONDecodeError:
            print(f"Repair failed. Trying regex extraction...")
            
            # Try to extract with regex
            json_match = re.search(r'(\{[\s\S]*\})', json_str)
            if json_match:
                try:
                    extracted_json = json_match.group(1)
                    repaired_extracted = repair_json(extracted_json)
                    return json.loads(repaired_extracted)
                except:
                    pass
                    
            print(f"All JSON parsing methods failed. Using default value.")
            return default_value

def extract_questions_answers(transcript_text: str) -> dict:
    """
    Extract questions and answers from a transcript using LLM.
    
    Args:
        transcript_text: The transcript text
        
    Returns:
        Dictionary with question-answer pairs
    """
    if not is_available():
        return {"qa_pairs": []}
        
    try:
        client = get_client()
        truncated_text = _truncate_text(transcript_text)
        
        prompt = (
            f"Identify all questions and their answers from this transcript:\n\n{truncated_text}"
        )
        
        result = client.generate(
            prompt=prompt,
            system=QUESTIONS_ANSWERS_SYSTEM_PROMPT,
            max_tokens=3000
        )
        
        # Use the safe_json_loads function
        qa_data = safe_json_loads(result, {"qa_pairs": []})
        
        # Ensure we have the expected structure
        if isinstance(qa_data, list) and len(qa_data) > 0:
            # If we got a list, check if it has qa_pairs
            if isinstance(qa_data[0], dict) and "qa_pairs" in qa_data[0]:
                return {"qa_pairs": qa_data[0]["qa_pairs"]}
            else:
                # Maybe the list itself is qa_pairs
                return {"qa_pairs": qa_data}
        elif isinstance(qa_data, dict):
            if "qa_pairs" in qa_data:
                return qa_data
            else:
                # Convert dict to qa_pairs if needed
                return {"qa_pairs": [qa_data] if qa_data else []}
        else:
            return {"qa_pairs": []}
        
    except Exception as e:
        print(f"Error extracting questions and answers: {e}")
        return {"qa_pairs": []}

def extract_decisions(transcript_text: str) -> dict:
    """
    Extract decisions and conclusions from a transcript using LLM.
    
    Args:
        transcript_text: The transcript text
        
    Returns:
        Dictionary with decisions
    """
    if not is_available():
        return {"decisions": []}
        
    try:
        client = get_client()
        truncated_text = _truncate_text(transcript_text)
        
        prompt = (
            f"Identify all decisions and conclusions reached in this transcript:\n\n{truncated_text}"
        )
        
        result = client.generate(
            prompt=prompt,
            system=DECISIONS_SYSTEM_PROMPT,
            max_tokens=2000
        )
        
        # Use the safe_json_loads function
        decisions_data = safe_json_loads(result, {"decisions": []})
        
        # Ensure we have the expected structure
        if isinstance(decisions_data, list):
            # If we got a list, wrap it
            return {"decisions": decisions_data}
        elif isinstance(decisions_data, dict):
            if "decisions" in decisions_data:
                return decisions_data
            else:
                # Convert dict to decisions if needed
                return {"decisions": [decisions_data] if decisions_data else []}
        else:
            return {"decisions": []}
        
    except Exception as e:
        print(f"Error extracting decisions: {e}")
        return {"decisions": []}

def extract_commitments(transcript_text: str) -> dict:
    """
    Extract personal commitments and promises from a transcript using LLM.
    
    Args:
        transcript_text: The transcript text
        
    Returns:
        Dictionary with commitments
    """
    if not is_available():
        return {"commitments": []}
        
    try:
        client = get_client()
        truncated_text = _truncate_text(transcript_text)
        
        prompt = (
            f"Identify all personal commitments and promises made in this transcript:\n\n{truncated_text}"
        )
        
        result = client.generate(
            prompt=prompt,
            system=COMMITMENTS_SYSTEM_PROMPT,
            max_tokens=2000
        )
        
        # Use the safe_json_loads function
        commitments_data = safe_json_loads(result, {"commitments": []})
        
        # Ensure we have the expected structure
        if isinstance(commitments_data, list):
            # If we got a list, wrap it
            return {"commitments": commitments_data}
        elif isinstance(commitments_data, dict):
            if "commitments" in commitments_data:
                return commitments_data
            else:
                # Convert dict to commitments if needed
                return {"commitments": [commitments_data] if commitments_data else []}
        else:
            return {"commitments": []}
        
    except Exception as e:
        print(f"Error extracting commitments: {e}")
        return {"commitments": []}

# Multi-transcript analysis prompts
COMPARATIVE_SUMMARY_PROMPT = """
You are a Chief of Staff preparing a briefing for the CEO about recent meetings.
Your task is to analyze multiple meeting transcripts and extract the most important business insights.

CREATE A STRUCTURED REPORT IN THIS FORMAT:
1. EXECUTIVE SUMMARY (2-3 sentences on the most critical insights)
2. KEY DECISIONS (bullet points of major decisions reached)
3. CRITICAL ISSUES (bullet points of problems requiring attention)
4. EVOLVING DISCUSSIONS (how key topics changed across meetings)
5. NEXT STEPS (clear action items extracted from the meetings)

IMPORTANT GUIDELINES:
- Be extremely concise - the CEO has limited time
- Focus only on business-critical information
- Highlight any conflicting decisions or inconsistencies
- Specifically note any financial, strategic, or personnel issues
- Mention specific people by name when they made important contributions
- Use data and specifics from the transcripts wherever possible
"""

COMMON_TOPICS_PROMPT = """
You are a business intelligence analyst preparing a report for executives.
Your task is to extract topics that appear across multiple meeting transcripts, focusing on business-critical themes.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "common_topics": [
       {
         "name": "Name of the common topic",
         "description": "Brief description of what this topic encompasses",
         "frequency": 3, // Number of transcripts where this appears
         "business_priority": "High/Medium/Low based on business impact",
         "transcripts": [
           {
             "id": "transcript-id-1", // From the transcript metadata
             "date": "2023-02-15" // From the transcript metadata
           },
           ...
         ]
       },
       ...
     ]
   }
3. Only include topics that appear in at least 2 transcripts
4. Prioritize topics related to strategic decisions, financial matters, project status, and client relationships
5. If no common topics are found, return {"common_topics": []}
"""

TOPIC_EVOLUTION_PROMPT = """
You are a strategic advisor to the CEO tracking how important business topics evolve over time.
Your task is to analyze how discussions, plans, and decisions change across a series of meetings.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "evolving_topics": [
       {
         "name": "Name of the evolving topic",
         "business_impact": "High/Medium/Low based on strategic importance",
         "evolution": [
           {
             "transcript_id": "id-of-transcript", // From the transcript metadata
             "date": "2023-02-15", // From the transcript metadata
             "summary": "Brief summary of the topic's state/discussion at this point",
             "key_changes": "What changed since the previous discussion"
           },
           ...
         ]
       },
       ...
     ]
   }
3. List evolution entries in chronological order
4. Only include topics that show meaningful evolution or change
5. Focus on topics with high business impact (financial, strategic, operational)
6. If no evolving topics are found, return {"evolving_topics": []}
"""

CONFLICTS_CHANGES_PROMPT = """
You are a risk management consultant identifying potential issues from meeting transcripts.
Your task is to flag where information, decisions, or plans changed or conflicted across meetings.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "conflicting_information": [
       {
         "topic": "The subject of the conflict or change",
         "risk_level": "High/Medium/Low based on potential business impact",
         "recommendation": "Brief recommendation on how to address this conflict",
         "changes": [
           {
             "transcript_id": "id-of-transcript", // From the transcript metadata
             "date": "2023-02-15", // From the transcript metadata 
             "description": "Description of the position or information at this point"
           },
           ...
         ]
       },
       ...
     ]
   }
3. List changes in chronological order
4. Prioritize conflicts that could impact finances, timelines, client relationships, or strategic goals
5. If no conflicts or changes are found, return {"conflicting_information": []}
"""

ACTION_ITEM_TRACKING_PROMPT = """
You are an executive project manager tracking action items across multiple meetings.
Your task is to identify action items, track their status, and flag any overdue or at-risk tasks.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "action_item_status": [
       {
         "description": "Description of the action item",
         "assignee": "Person assigned to the task",
         "first_mentioned": "2023-02-15", // Date when first mentioned
         "status": "completed", // One of: "pending", "in_progress", "completed", "canceled", "at_risk", "overdue" 
         "priority": "High/Medium/Low based on business impact",
         "mentions": [
           {
             "transcript_id": "id-of-transcript", // From the transcript metadata
             "date": "2023-02-15", // From the transcript metadata
             "status": "pending", // Status at this mention
             "notes": "Any additional context from this mention"
           },
           ...
         ]
       },
       ...
     ]
   }
3. List mentions in chronological order
4. Track the same action item across multiple transcripts
5. If a deadline was mentioned and passed without completion, mark as "overdue"
6. If completion is repeatedly delayed, mark as "at_risk"
7. If no action items are found, return {"action_item_status": []}
"""

def analyze_combined_transcripts(combined_text: str, transcripts_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze multiple transcripts combined into a single text.
    
    Args:
        combined_text: Text containing multiple transcripts with separators
        transcripts_metadata: List of metadata dictionaries for each transcript
        
    Returns:
        Dictionary with various analysis results
    """
    if not is_available():
        return {
            "comparative_summary": "LLM analysis not available.",
            "common_topics": [],
            "evolving_topics": [],
            "conflicting_information": [],
            "action_item_status": []
        }
    
    try:
        client = get_client()
        
        # Truncate combined text to avoid token limits
        truncated_text = _truncate_text(combined_text, max_length=12000)
        
        # Add metadata context
        metadata_context = "Transcripts metadata:\n"
        for i, meta in enumerate(transcripts_metadata):
            transcript_id = meta.get('id', f'transcript-{i}')
            title = meta.get('title', 'Untitled')
            date = meta.get('date', '').split('T')[0] if meta.get('date') else 'Unknown date'
            topic = meta.get('topic', 'No specific topic')
            metadata_context += f"- Transcript {i+1}: ID={transcript_id}, Title={title}, Date={date}, Topic={topic}\n"
        
        # Combined prompt with metadata
        context_and_text = f"{metadata_context}\n\n{truncated_text}"
        
        # Analysis results
        results = {}
        
        # Generate comparative summary
        print("Generating comparative summary...")
        summary_prompt = "Generate a comprehensive executive briefing of these meeting transcripts, highlighting key business insights:"
        results['comparative_summary'] = client.generate(
            prompt=summary_prompt + context_and_text,
            system=COMPARATIVE_SUMMARY_PROMPT,
            max_tokens=2000
        )
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
        
        # Extract common topics
        print("Extracting common topics...")
        topics_prompt = "Identify business-critical topics that appear across multiple transcripts:"
        topics_result = client.generate(
            prompt=topics_prompt + context_and_text,
            system=COMMON_TOPICS_PROMPT,
            max_tokens=2000
        )
        
        # Use the safe_json_loads function
        topics_data = safe_json_loads(topics_result, {"common_topics": []})
        if isinstance(topics_data, dict) and "common_topics" in topics_data:
            results['common_topics'] = topics_data["common_topics"]
        else:
            results['common_topics'] = []
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
        
        # Analyze topic evolution
        print("Analyzing topic evolution...")
        evolution_prompt = "Analyze how strategic topics and business decisions evolved across these meetings over time:"
        evolution_result = client.generate(
            prompt=evolution_prompt + context_and_text,
            system=TOPIC_EVOLUTION_PROMPT,
            max_tokens=2000
        )
        
        # Use the safe_json_loads function
        evolution_data = safe_json_loads(evolution_result, {"evolving_topics": []})
        if isinstance(evolution_data, dict) and "evolving_topics" in evolution_data:
            results['evolving_topics'] = evolution_data["evolving_topics"]
        else:
            results['evolving_topics'] = []
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
        
        # Find contradictions or conflicts
        print("Identifying conflicts and changes...")
        conflicts_prompt = "Identify any business risks, contradictions, or inconsistencies across these meetings:"
        conflicts_result = client.generate(
            prompt=conflicts_prompt + context_and_text,
            system=CONFLICTS_CHANGES_PROMPT,
            max_tokens=2000
        )
        
        # Use the safe_json_loads function
        conflicts_data = safe_json_loads(conflicts_result, {"conflicting_information": []})
        if isinstance(conflicts_data, dict) and "conflicting_information" in conflicts_data:
            results['conflicting_information'] = conflicts_data["conflicting_information"]
        else:
            results['conflicting_information'] = []
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
        
        # Track action items across meetings
        print("Tracking action items...")
        action_prompt = "Track business-critical action items across these meetings, with special attention to status and priority:"
        action_result = client.generate(
            prompt=action_prompt + context_and_text,
            system=ACTION_ITEM_TRACKING_PROMPT,
            max_tokens=2000
        )
        
        # Use the safe_json_loads function
        action_data = safe_json_loads(action_result, {"action_item_status": []})
        if isinstance(action_data, dict) and "action_item_status" in action_data:
            results['action_item_status'] = action_data["action_item_status"]
        else:
            results['action_item_status'] = []
        
        return results
    
    except Exception as e:
        print(f"Error in multi-transcript analysis: {e}")
        return {
            "comparative_summary": f"Error analyzing transcripts: {str(e)}",
            "common_topics": [],
            "evolving_topics": [],
            "conflicting_information": [],
            "action_item_status": []
        }

"""
Key Visual Moments extraction - IMPROVED VERSION
"""

KEY_MOMENTS_SYSTEM_PROMPT = """
You are an expert at analyzing meeting transcripts and identifying the MOST SIGNIFICANT moments that would be valuable to capture as screenshots.

LOOK FOR THESE TYPES OF KEY MOMENTS (in priority order):
1. VISUAL REFERENCES - When someone is explicitly showing or demonstrating something:
   - "As you can see on this slide..."
   - "Let me show you this diagram..."
   - "If you look at the screen..."

2. SIGNIFICANT CONTENT MOMENTS:
   - Decision points: When important decisions are made or finalized
   - Problem statements: Clear articulation of key challenges
   - Solution proposals: Detailed explanations of solutions
   - Action items: When specific tasks are assigned
   - Quantitative insights: When numbers, metrics, or data are discussed
   - Technical demonstrations: Explanations of how something works

3. STRUCTURAL ELEMENTS:
   - Meeting opening: Introduction, agenda setting
   - Topic transitions: Clear shifts to new important topics
   - Summarization points: "To recap what we've covered..."
   - Meeting conclusion: Final thoughts, next steps

YOUR TASK:
Identify the exact moments in the transcript where screenshots would provide maximum value for someone reviewing the meeting later.

YOU MUST RESPOND ONLY WITH JSON. DO NOT explain your reasoning or add any text that is not part of the JSON.
Return EXACTLY this JSON structure and nothing else:
{
  "key_moments": [
    {
      "timestamp": 123.5,
      "title": "Brief title for this moment",
      "description": "Short description of why this moment is valuable",
      "moment_type": "visual_reference|decision|problem|solution|action_item|technical|summary"
    }
  ]
}

The first timestamp should be very early in the video (introduction).
The last timestamp should be very late in the video (conclusion).
The middle timestamps should be the most significant moments based on the criteria above.
"""

def calculate_adaptive_context_window(video_duration_seconds: float) -> int:
    """
    Calculate an appropriate context window size based on video duration.
    
    Args:
        video_duration_seconds: Video duration in seconds
        
    Returns:
        Context window size (number of segments)
    """
    # Convert to minutes
    duration_minutes = video_duration_seconds / 60
    
    # Base context window of 5 segments
    # Add 1 segment for every 10 minutes of content
    # Cap at 15 segments maximum
    context_size = 5 + int(duration_minutes / 10)
    return max(5, min(context_size, 15))

def detect_natural_content_breaks(segments: List[Dict[str, Any]], min_segments_between: int = 10) -> List[Dict[str, Any]]:
    """
    Detect natural breaks in transcript content that might indicate significant moments.
    
    Args:
        segments: List of transcript segments
        min_segments_between: Minimum segments between detected breaks
        
    Returns:
        List of dicts with timestamp and reason for the break
    """
    if not segments or len(segments) < min_segments_between*2:
        return []
    
    content_breaks = []
    last_break_idx = -min_segments_between  # Allow first segment to be a break
    
    # Signals that often indicate topic transitions or important moments
    transition_phrases = [
        "moving on to", "let's talk about", "next item", "next topic",
        "to summarize", "in conclusion", "let me show you", 
        "as you can see", "now we'll", "turning to", "let's discuss",
        "i'd like to", "the next point", "another important",
        "to wrap up", "we need to decide", "the key question"
    ]
    
    # Track the rolling density of technical or quantitative terms
    technical_term_count = [0] * len(segments)
    quantitative_term_regex = r'\b\d+(?:\.\d+)?%?\b|\bdollars?\b|\beuro\b|\bpercent(?:age)?\b|\bratio\b|\bmetrics?\b|\bKPI\b'
    
    # Process segments for technical/quantitative term density
    for i, segment in enumerate(segments):
        text = segment.get("text", "").lower()
        
        # Count quantitative terms
        import re
        quant_terms = len(re.findall(quantitative_term_regex, text))
        
        # Calculate rolling sum (last 3 segments)
        window_start = max(0, i-2)
        technical_term_count[i] = sum(technical_term_count[window_start:i]) + quant_terms
    
    # Now detect breaks based on multiple signals
    for i, segment in enumerate(segments):
        # Skip if too close to last break
        if i - last_break_idx < min_segments_between:
            continue
        
        text = segment.get("text", "").lower()
        is_break = False
        break_reason = ""
        
        # Check for transition phrases
        for phrase in transition_phrases:
            if phrase in text:
                is_break = True
                break_reason = f"Transition phrase: '{phrase}'"
                break
        
        # Check for spike in technical/quantitative terms (if 3x average)
        if not is_break and i > 2:
            avg_terms = sum(technical_term_count[max(0, i-10):i]) / min(10, i)
            if technical_term_count[i] > avg_terms * 3 and technical_term_count[i] >= 3:
                is_break = True
                break_reason = "High density of quantitative/technical terms"
        
        # Check for long pause between segments (indicating potential topic change)
        if not is_break and i > 0:
            current_start = segment.get("start", 0)
            prev_end = segments[i-1].get("end", 0)
            pause_duration = current_start - prev_end
            
            # If pause is more than 2 seconds (and not just a sentence pause)
            if pause_duration > 2.0:
                is_break = True
                break_reason = f"Natural pause of {pause_duration:.1f} seconds"
        
        # If we detected a break, add it
        if is_break:
            content_breaks.append({
                "timestamp": segment.get("start", 0),
                "segment_idx": i,
                "reason": break_reason,
                "text": text[:50] + "..." if len(text) > 50 else text
            })
            last_break_idx = i
    
    return content_breaks

def extract_semantic_context(segments: List[Dict[str, Any]], 
                            center_idx: int, 
                            base_window_size: int) -> tuple:
    """
    Extract context using semantic boundaries rather than fixed window size.
    
    Args:
        segments: List of transcript segments
        center_idx: Index of the central segment
        base_window_size: Base number of segments to include
        
    Returns:
        Tuple of (start_idx, end_idx, context_text)
    """
    if not segments or center_idx < 0 or center_idx >= len(segments):
        return center_idx, center_idx, ""
    
    # Start with the base window approach
    start_idx = max(0, center_idx - base_window_size)
    end_idx = min(len(segments) - 1, center_idx + base_window_size)
    
    # Get the segments in this range
    window_segments = segments[start_idx:end_idx+1]
    
    # Join all text in the current window
    initial_text = " ".join([s.get("text", "") for s in window_segments])
    
    # Look for better start boundary - complete sentences
    sentence_endings = ['.', '!', '?']
    if start_idx > 0:
        # Look at the text of the first segment in our window
        first_segment_text = window_segments[0].get("text", "")
        
        # If it starts with lowercase letter and there is a previous segment,
        # it might be in the middle of a thought - expand backward
        if first_segment_text and first_segment_text[0].islower():
            # Try to include the previous segment
            new_start_idx = max(0, start_idx - 1)
            if new_start_idx < start_idx:
                prev_segment = segments[new_start_idx]
                prev_text = prev_segment.get("text", "")
                
                # Check if the previous segment ends with a sentence-ending punctuation
                if prev_text and prev_text[-1] not in sentence_endings:
                    # It doesn't end with a sentence boundary, so include it
                    start_idx = new_start_idx
    
    # Look for better end boundary - complete sentences  
    if end_idx < len(segments) - 1:
        # Look at the text of the last segment in our window
        last_segment_text = window_segments[-1].get("text", "")
        
        # If it doesn't end with a sentence-ending punctuation, 
        # it might be in the middle of a thought - expand forward
        if last_segment_text and last_segment_text[-1] not in sentence_endings:
            # Try to include the next segment
            new_end_idx = min(len(segments) - 1, end_idx + 1) 
            if new_end_idx > end_idx:
                next_segment = segments[new_end_idx]
                next_text = next_segment.get("text", "")
                end_idx = new_end_idx
    
    # Join the text from the adjusted range
    context_text = " ".join([s.get("text", "") for s in segments[start_idx:end_idx+1]])
    
    return start_idx, end_idx, context_text

def extract_key_visual_moments(
    transcript_text: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    video_path: Optional[str] = None,
    max_moments: int = 7, #changed from 5 to 7
    context_window: int = 5
) -> Dict[str, Any]:
    """
    Extract key visual moments from a transcript where screenshots would be valuable,
    including extended context around each moment.
    
    Args:
        transcript_text: Full transcript text
        segments: List of transcript segments with timestamps
        video_path: Path to the video file (optional)
        max_moments: Maximum number of key moments to identify
        context_window: Number of segments to include before and after for context
        
    Returns:
        Dictionary with key_moments list and metadata
    """
    logger.info(f"Extracting up to {max_moments} key visual moments from transcript with context window of {context_window} segments")
    
    try:
        # Define strategic points in the video for key moments if we don't have segments
        if not segments or len(segments) == 0:
            logger.warning("No segments provided, creating simple timestamp-based moments")
            return _create_default_key_moments(max_moments)
        
        # Get total duration from segments
        total_duration = segments[-1].get("end", 0) if segments else 0
        
        # Calculate strategic timestamps for key moments (intro, points, conclusion)
        timestamps = []
        
        # Beginning (5% into the video)
        timestamps.append(total_duration * 0.05)
        
        # Evenly distribute remaining timestamps up to 95% of the duration
        middle_points = max_moments - 2  # Subtract intro and conclusion
        if middle_points > 0:
            step = (total_duration * 0.9) / (middle_points + 1)
            start = total_duration * 0.05 + step
            for i in range(middle_points):
                timestamps.append(start + i * step)
        
        # End (95% into the video)
        timestamps.append(total_duration * 0.95)
        
        # Create key moments with extended context
        key_moments = []
        
        for i, timestamp in enumerate(timestamps[:max_moments]):
            # Find closest segment to this timestamp
            closest_segment_idx = _find_segment_index_by_timestamp(segments, timestamp)
            
            if closest_segment_idx == -1:
                continue
                
            # Determine segment indices for context window
            start_idx = max(0, closest_segment_idx - context_window)
            end_idx = min(len(segments) - 1, closest_segment_idx + context_window)
            
            # Get the actual timestamp from the segment
            actual_timestamp = segments[closest_segment_idx].get("start", timestamp)
            
            # Title based on position
            title = "Introduction" if i == 0 else "Conclusion" if i == len(timestamps) - 1 else f"Key point {i}"
            
            # Extract transcript text from all segments in the context window
            transcript_context = " ".join([segment.get("text", "") for segment in segments[start_idx:end_idx+1]])
            
            # Create the key moment with rich context
            key_moment = {
                "timestamp": actual_timestamp,
                "title": title,
                "description": f"Important moment at {_format_timestamp(actual_timestamp)}",
                "transcript_text": transcript_context,
                # Add timestamp range for UI highlighting
                "context_start_time": segments[start_idx].get("start", actual_timestamp),
                "context_end_time": segments[end_idx].get("end", actual_timestamp)
            }
            
            key_moments.append(key_moment)
        
        logger.info(f"Successfully extracted {len(key_moments)} key visual moments with rich context")
        
        # Return the key moments with success indication
        return {
            "key_moments": key_moments,
            "success": True,
            "count": len(key_moments)
        }
        
    except Exception as e:
        logger.error(f"Error extracting key visual moments: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return an empty list with error information
        return {
            "key_moments": [],
            "success": False,
            "error": str(e)
        }

def extract_smart_key_moments(
    transcript_text: str,
    segments: List[Dict[str, Any]],
    video_path: Optional[str] = None,
    fixed_count: int = None,
    dynamic_count: bool = False,
    min_count: int = 5,
    max_count: int = 20,
    context_window: int = 5,
    title_format: str = "numbered"  # Added parameter with default value
) -> Dict[str, Any]:
    """
    Extract key moments intelligently, with either a fixed count or dynamically based on video length.
    
    Args:
        transcript_text: The transcript text
        segments: List of transcript segments with timestamps
        video_path: Path to the video file (optional)
        fixed_count: If provided, always use this many key moments (legacy parameter)
        dynamic_count: If True, calculate key moment count based on video length
        min_count: Minimum number of key moments when using dynamic_count
        max_count: Maximum number of key moments when using dynamic_count
        context_window: Number of segments to include before and after for context
        title_format: Format for titles - "numbered" or "content"
        
    Returns:
        Dictionary with key_moments list and metadata
    """
    logger.info("Extracting smart key moments")
    
    # Calculate video duration
    total_duration = segments[-1].get("end", 0) if segments and len(segments) > 0 else 1800
    duration_minutes = total_duration / 60
    
    # Determine how many key moments to extract
    if dynamic_count:
        # Base formula: minimum + 1 moment per 10 minutes of content
        moment_count = min_count + int(duration_minutes / 10)
        # Apply maximum limit
        moment_count = min(moment_count, max_count)
        logger.info(f"Using dynamic count: {moment_count} moments for {duration_minutes:.1f} minute video")
    else:
        # Use legacy fixed count approach
        moment_count = fixed_count if fixed_count is not None else 7
        logger.info(f"Using fixed count: {moment_count} moments")
    
    # Generate standard titles framework
    standard_titles = ["Introduction"]
    for i in range(1, moment_count-1):
        standard_titles.append(f"Key point {i}")
    standard_titles.append("Conclusion")
    
    try:
        # Only attempt content analysis if LLM is available
        if is_available():
            # Try to find content-meaningful timestamps
            interesting_timestamps = []
            
            # 1. First look for explicit visual cues
            visual_cues = find_visual_cues_in_transcript(transcript_text)
            if visual_cues:
                for cue in visual_cues:
                    timestamp_str = cue.get('timestamp')
                    if timestamp_str:
                        try:
                            # Convert MM:SS to seconds
                            parts = timestamp_str.split(':')
                            if len(parts) == 2:
                                timestamp = int(parts[0]) * 60 + int(parts[1])
                                if 0 <= timestamp <= total_duration:
                                    interesting_timestamps.append({
                                        'timestamp': timestamp,
                                        'reason': f"Visual reference: {cue.get('visual_cue')}"
                                    })
                        except:
                            pass

            # If we don't have enough timestamps from visual cues, try content breaks
            if len(interesting_timestamps) < moment_count:
                content_breaks = detect_natural_content_breaks(segments)
                
                # Add these as potential timestamps
                for break_info in content_breaks:
                    timestamp = break_info.get('timestamp', 0)
                    # Skip if too close to existing timestamps
                    if not any(abs(ts.get('timestamp', 0) - timestamp) < 10 for ts in interesting_timestamps):
                        user_friendly_reason = "Key content point"
                        if "transition phrase" in break_info.get('reason', '').lower():
                            user_friendly_reason = "Topic transition"
                        elif "pause" in break_info.get('reason', '').lower():
                            user_friendly_reason = "Discussion point"
                        elif "technical terms" in break_info.get('reason', '').lower():
                            user_friendly_reason = "Technical details"

                        interesting_timestamps.append({
                            'timestamp': timestamp,
                            'reason': break_info.get('reason', 'Natural content break'),
                            'title': "Key point",  # Removed numbering - will add proper numbering later
                            'description': user_friendly_reason
                        })          
            
            # If we don't have enough timestamps from regex, use LLM
            if len(interesting_timestamps) < moment_count:
                truncated_text = _truncate_text(transcript_text, max_length=8000)
                
                # Include duration information
                prompt = f'''
                This video is {duration_minutes:.2f} minutes long and requires EXACTLY {moment_count} key moments.

                ANALYZE THIS TRANSCRIPT TO FIND:
                1. An INTRODUCTION moment within the first 5% of the video ({total_duration * 0.05:.0f} seconds)
                2. A CONCLUSION moment within the last 5% of the video (after {total_duration * 0.95:.0f} seconds)
                3. {moment_count-2} intermediate moments representing the MOST SIGNIFICANT points in the discussion

                PRIORITIZE THESE MOMENT TYPES (from most to least important):
                - Visual demonstrations ("Let me show you...", "On the screen...", "Here you can see...")
                - Decision points (where key decisions are made)
                - Problem statements (clear articulation of challenges)
                - Solution explanations
                - Quantitative insights (important numbers/metrics)
                - Technical demonstrations
                - Action item assignments
                - Topic transitions and summary points

                TRANSCRIPT:
                {truncated_text}

                For each moment, provide a timestamp (in seconds) between 0 and {total_duration}.
                '''
                
                # Get content-meaningful timestamps
                client = get_client()
                response = client.generate(
                    prompt=prompt,
                    system=KEY_MOMENTS_SYSTEM_PROMPT,
                    max_tokens=1500
                )
                
                llm_result = safe_json_loads(response, {"key_moments": []})
                if isinstance(llm_result, dict) and "key_moments" in llm_result:
                    for moment in llm_result["key_moments"]:
                        timestamp = moment.get("timestamp", 0)
                        # Convert string timestamp to seconds if needed
                        if isinstance(timestamp, str) and ":" in timestamp:
                            parts = timestamp.split(":")
                            if len(parts) == 2:  # MM:SS format
                                timestamp = int(parts[0]) * 60 + float(parts[1])
                            elif len(parts) == 3:  # HH:MM:SS format
                                timestamp = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                        
                        # Validate timestamp
                        if 0 <= timestamp <= total_duration:
                            interesting_timestamps.append({
                                'timestamp': timestamp,
                                'title': moment.get('title'),
                                'description': moment.get('description')
                            })
            
            # Sort by timestamp
            interesting_timestamps.sort(key=lambda x: x.get('timestamp', 0))
            
            # If we have content-aware timestamps, use them
            if len(interesting_timestamps) >= moment_count:
                # ENHANCED SELECTION ALGORITHM:
                # 1. Always ensure introduction (first 5% of video)
                # 2. Always ensure conclusion (last 5% of video)
                # 3. Prioritize visual moments in the middle
                # 4. Fill remaining slots with important content moments
                
                final_timestamps = []
                
                # Check if we have an introduction moment in the first 5% of the video
                intro_threshold = total_duration * 0.05
                has_intro = any(ts.get('timestamp', 0) <= intro_threshold for ts in interesting_timestamps)

                # Check if we have a conclusion moment in the last 5% of the video
                conclusion_threshold = total_duration * 0.95
                has_conclusion = any(ts.get('timestamp', 0) >= conclusion_threshold for ts in interesting_timestamps)
                
                # If no intro, add the earliest timestamp or force one at 5% mark
                if not has_intro:
                    if interesting_timestamps:
                        intro_ts = interesting_timestamps[0]
                        # Force it to be recognized as intro by setting timestamp to 5% mark
                        intro_ts['timestamp'] = total_duration * 0.05
                        intro_ts['title'] = "Introduction"
                        final_timestamps.append(intro_ts)
                    else:
                        final_timestamps.append({
                            'timestamp': total_duration * 0.05,
                            'title': "Introduction",
                            'description': "Start of the meeting"
                        })
                
                # Add middle timestamps - prioritize visual cues
                visual_cue_keywords = ['screen', 'show', 'look', 'see', 'image', 'picture', 'diagram', 'slide']
                
                # First pass: add timestamps with visual keywords
                visual_timestamps = []
                for ts in interesting_timestamps:
                    # Check if this is likely an intro or conclusion moment
                    ts_time = ts.get('timestamp', 0)
                    if ts_time <= intro_threshold or ts_time >= conclusion_threshold:
                        continue
                        
                    # Check if this contains a visual reference
                    title = ts.get('title', '').lower()
                    description = ts.get('description', '').lower()
                    text = title + " " + description
                    
                    if any(keyword in text for keyword in visual_cue_keywords):
                        visual_timestamps.append(ts)
                
                # Second pass: add other timestamps to fill in
                remaining_needed = moment_count - 2  # Excluding intro and conclusion
                
                # First add visual timestamps (up to the limit)
                if visual_timestamps:
                    # Sort by timestamp
                    visual_timestamps.sort(key=lambda x: x.get('timestamp', 0))
                    final_timestamps.extend(visual_timestamps[:remaining_needed])
                    remaining_needed -= len(visual_timestamps[:remaining_needed])
                
                # If we still need more timestamps, add remaining content moments
                if remaining_needed > 0:
                    # Filter out intro, conclusion, and already added visual timestamps
                    remaining_timestamps = []
                    for ts in interesting_timestamps:
                        ts_time = ts.get('timestamp', 0)
                        # Skip intro/conclusion ranges
                        if ts_time <= intro_threshold or ts_time >= conclusion_threshold:
                            continue
                            
                        # Skip if already added as a visual timestamp
                        if any(vts.get('timestamp') == ts.get('timestamp') for vts in visual_timestamps[:remaining_needed]):
                            continue
                            
                        remaining_timestamps.append(ts)
                    
                    # If we have remaining timestamps, evenly sample them
                    if remaining_timestamps:
                        # Sort by timestamp
                        remaining_timestamps.sort(key=lambda x: x.get('timestamp', 0))
                        
                        # Evenly sample
                        if len(remaining_timestamps) <= remaining_needed:
                            # Just add all of them
                            final_timestamps.extend(remaining_timestamps)
                        else:
                            # Evenly sample
                            step = len(remaining_timestamps) / remaining_needed
                            for i in range(remaining_needed):
                                idx = min(int(i * step), len(remaining_timestamps) - 1)
                                final_timestamps.append(remaining_timestamps[idx])
                
                # If no conclusion, add the latest timestamp or force one at 95% mark
                if not has_conclusion:
                    if interesting_timestamps:
                        conclusion_ts = interesting_timestamps[-1]
                        # Force it to be recognized as conclusion
                        conclusion_ts['timestamp'] = total_duration * 0.95
                        conclusion_ts['title'] = "Conclusion"
                        final_timestamps.append(conclusion_ts)
                    else:
                        final_timestamps.append({
                            'timestamp': total_duration * 0.95,
                            'title': "Conclusion",
                            'description': "End of the meeting"
                        })
                elif not any(ts.get('timestamp', 0) >= conclusion_threshold for ts in final_timestamps):
                    # We know there is a conclusion timestamp but we haven't added it yet
                    for ts in interesting_timestamps:
                        if ts.get('timestamp', 0) >= conclusion_threshold:
                            ts['title'] = "Conclusion"
                            final_timestamps.append(ts)
                            break
                
                # Sort by timestamp
                final_timestamps.sort(key=lambda x: x.get('timestamp', 0))
                
                # Ensure we have the right number of timestamps (should be moment_count)
                while len(final_timestamps) < moment_count:
                    # Add evenly spaced timestamps in gaps
                    current_timestamps = [ts.get('timestamp', 0) for ts in final_timestamps]
                    largest_gap = 0
                    largest_gap_start = 0
                    
                    for i in range(len(current_timestamps) - 1):
                        gap = current_timestamps[i+1] - current_timestamps[i]
                        if gap > largest_gap:
                            largest_gap = gap
                            largest_gap_start = current_timestamps[i]
                    
                    # Add a timestamp in the middle of the largest gap
                    new_timestamp = largest_gap_start + largest_gap / 2
                    final_timestamps.append({
                        'timestamp': new_timestamp,
                        'title': "Key point",  # Removed numbering - will add proper numbering later
                        'description': f"Important moment at {_format_timestamp(new_timestamp)}"
                    })
                    
                    # Re-sort by timestamp
                    final_timestamps.sort(key=lambda x: x.get('timestamp', 0))
                
                # Ensure we have exactly moment_count timestamps
                final_timestamps = final_timestamps[:moment_count]
                
                # Calculate adaptive context window size based on video duration
                adaptive_window_size = calculate_adaptive_context_window(total_duration)
                logger.info(f"Using adaptive context window size of {adaptive_window_size} segments for {duration_minutes:.1f} minute video")

                # Now process these smart timestamps into full key moments
                smart_moments = []
                for i, timestamp_info in enumerate(final_timestamps):
                    timestamp = timestamp_info.get('timestamp', 0)
                    
                    # Find closest segment
                    closest_segment_idx = _find_segment_index_by_timestamp(segments, timestamp)
                    
                    if closest_segment_idx == -1:
                        continue
                    
                    # Get context segments using semantic boundaries
                    start_idx, end_idx, transcript_context = extract_semantic_context(
                        segments, 
                        closest_segment_idx, 
                        adaptive_window_size
                    )
                    
                    # Get actual timestamp from segment
                    actual_timestamp = segments[closest_segment_idx].get("start", timestamp)
                    
                    # Create the key moment with rich context
                    smart_moment = {
                        "timestamp": actual_timestamp,
                        "title": "",  # Temporary blank title - will be fixed in a moment
                        "description": timestamp_info.get('description', f"Important moment at {_format_timestamp(actual_timestamp)}"),
                        "transcript_text": transcript_context,
                        "context_start_time": segments[start_idx].get("start", actual_timestamp),
                        "context_end_time": segments[end_idx].get("end", actual_timestamp)
                    }
                    
                    smart_moments.append(smart_moment)
                
                # Fix the titles with proper sequential numbering
                for i, moment in enumerate(smart_moments):
                    if i == 0:
                        moment["title"] = "Introduction"
                    elif i == len(smart_moments) - 1:
                        moment["title"] = "Conclusion"
                    else:
                        moment["title"] = f"Key point {i}"
                
                # If we successfully created moments, return them
                if smart_moments:
                    return {
                        "key_moments": smart_moments,
                        "success": True,
                        "count": len(smart_moments)
                    }
        
        # Fall back to the original method if anything above fails
        logger.info(f"Using original method to create {moment_count} evenly-spaced key moments")
        
    except Exception as e:
        logger.error(f"Error in smart key moments extraction: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Standard method (original approach) that always works
    key_moments = []
    
    # Beginning (5% into the video)
    timestamps = []
    timestamps.append(total_duration * 0.05)
    
    # Evenly distribute remaining timestamps up to 95% of the duration
    middle_points = moment_count - 2  # Subtract intro and conclusion
    if middle_points > 0:
        step = (total_duration * 0.9) / (middle_points + 1)
        start = total_duration * 0.05 + step
        for i in range(middle_points):
            timestamps.append(start + i * step)
    
    # End (95% into the video)
    timestamps.append(total_duration * 0.95)
    
    # Create key moments with extended context
    for i, timestamp in enumerate(timestamps):
        # Find closest segment to this timestamp
        closest_segment_idx = _find_segment_index_by_timestamp(segments, timestamp)
        
        if closest_segment_idx == -1:
            continue
            
        # Determine segment indices for context window
        start_idx = max(0, closest_segment_idx - context_window)
        end_idx = min(len(segments) - 1, closest_segment_idx + context_window)
        
        # Get the actual timestamp from the segment
        actual_timestamp = segments[closest_segment_idx].get("start", timestamp)
        
        # Extract transcript text from all segments in the context window
        transcript_context = " ".join([segment.get("text", "") for segment in segments[start_idx:end_idx+1]])
        
        # Create the key moment with rich context
        key_moment = {
            "timestamp": actual_timestamp,
            "title": "",  # Temporary blank title
            "description": f"Important moment at {_format_timestamp(actual_timestamp)}",
            "transcript_text": transcript_context,
            "context_start_time": segments[start_idx].get("start", actual_timestamp),
            "context_end_time": segments[end_idx].get("end", actual_timestamp)
        }
        
        key_moments.append(key_moment)
    
    # Fix the titles with proper sequential numbering
    for i, moment in enumerate(key_moments):
        if i == 0:
            moment["title"] = "Introduction"
        elif i == len(key_moments) - 1:
            moment["title"] = "Conclusion"
        else:
            moment["title"] = f"Key point {i}"
    
    # Return the key moments with success indication
    return {
        "key_moments": key_moments,
        "success": True,
        "count": len(key_moments)
    }

def _find_segment_index_by_timestamp(segments: List[Dict[str, Any]], timestamp: float) -> int:
    """
    Find the index of the segment closest to the given timestamp.
    
    Args:
        segments: List of transcript segments
        timestamp: Target timestamp in seconds
        
    Returns:
        Index of the closest segment, or -1 if segments is empty
    """
    if not segments:
        return -1
        
    closest_idx = 0
    min_distance = float('inf')
    
    for i, segment in enumerate(segments):
        start = segment.get("start", 0)
        distance = abs(start - timestamp)
        
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
    
    return closest_idx

def _create_default_key_moments(max_moments: int) -> Dict[str, Any]:
    """
    Create generic key moments when no segments are available.
    
    Args:
        max_moments: Maximum number of moments to create
        
    Returns:
        Dictionary with key_moments and metadata
    """
    # Create generic moments at regular intervals
    default_moments = []
    
    # Assume a typical 30-minute video for default timestamps
    assumed_duration = 1800  # 30 minutes in seconds
    
    for i in range(max_moments):
        # Calculate timestamp (evenly distributed)
        pct = (i / (max_moments - 1)) if max_moments > 1 else 0.5
        timestamp = assumed_duration * pct
        
        # Create title based on position
        title = "Introduction" if i == 0 else "Conclusion" if i == max_moments - 1 else f"Key point {i}"
        
        default_moments.append({
            "timestamp": timestamp,
            "title": title,
            "description": f"{title} at {_format_timestamp(timestamp)}",
            "transcript_text": f"Transcript context for {title} would appear here."
        })
    
    return {
        "key_moments": default_moments,
        "success": True,
        "count": len(default_moments)
    }

def _format_timestamp(seconds: float) -> str:
    """
    Format seconds as MM:SS.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"



def find_visual_cues_in_transcript(transcript: str) -> List[Dict[str, Any]]:
    """
    Find phrases in the transcript that indicate visual elements are being discussed.
    
    Args:
        transcript: Full transcript text
        
    Returns:
        List of potential visual moments with timestamps
    """
    visual_cues = []
    
    # Common phrases that indicate visual content
    cue_patterns = [
        r'(?:take a look at|looking at|on the screen|slide|diagram|chart|graph|image|picture|figure|screenshot|demo)',
        r'(?:let me show you|I\'m showing|as you can see|see here|you can see|can you see)',
        r'(?:pointing to|highlighting|circling|selecting|presenting|demonstrates|demonstration)'
    ]
    
    # Combine patterns into a single regex
    combined_pattern = '|'.join(cue_patterns)
    
    # Find timestamp patterns near these visual cues
    timestamp_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)'
    
    # Look for timestamps followed by visual cues
    matches = re.finditer(rf'{timestamp_pattern}[^\n]{{0,50}}({combined_pattern})[^\n]{{0,100}}', transcript, re.IGNORECASE)
    
    for match in matches:
        timestamp = match.group(1)
        cue = match.group(2)
        context = match.group(0)
        
        visual_cues.append({
            "timestamp": timestamp,
            "visual_cue": cue,
            "context": context,
            "description": f"Visual element: {cue}",
            "importance": "medium"
        })
    
    # Also check for visual cues followed by timestamps
    matches = re.finditer(rf'({combined_pattern})[^\n]{{0,50}}{timestamp_pattern}', transcript, re.IGNORECASE)
    
    for match in matches:
        cue = match.group(1)
        timestamp = match.group(2)
        context = match.group(0)
        
        visual_cues.append({
            "timestamp": timestamp,
            "visual_cue": cue,
            "context": context,
            "description": f"Visual element: {cue}",
            "importance": "medium"
        })
    
    return visual_cues