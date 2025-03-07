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

# Multi-transcript analysis prompts - IMPROVED FOR EXECUTIVES
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
Updated KEY_MOMENTS_SYSTEM_PROMPT and extract_key_visual_moments function
for modules/llm/meeting_intelligence.py
"""

KEY_MOMENTS_SYSTEM_PROMPT = """
You are an expert at analyzing meeting transcripts and identifying key moments where a screenshot would be valuable.
Look for moments where:
1. Someone is demonstrating or showing something
2. A speaker references visual content ("as you can see here...")
3. Someone is presenting slides or other visual media
4. Important visuals are being discussed

YOU MUST RESPOND ONLY WITH JSON. DO NOT explain your reasoning or add any text that is not part of the JSON.
Return EXACTLY this JSON structure and nothing else:
{
  "key_moments": [
    {
      "timestamp": 123.5,
      "title": "Brief title for this moment",
      "description": "Short description of why this visual moment is important"
    },
    {
      "timestamp": 456.7,
      "title": "Another key moment",
      "description": "Description of this moment"
    }
  ]
}

Limit to 5-10 most important visual moments. If no clear visual moments exist, return {"key_moments": []}.
"""

def extract_key_visual_moments(
    transcript: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    video_path: Optional[str] = None,
    max_moments: int = 5
) -> Dict[str, Any]:
    """
    Extract key visual moments from a transcript where screenshots would be valuable.
    
    Args:
        transcript: Full transcript text
        segments: List of transcript segments with timestamps (optional)
        video_path: Path to the video file (optional)
        max_moments: Maximum number of key moments to identify
        
    Returns:
        Dictionary with key_moments list and metadata
    """
    logger.info(f"Extracting up to {max_moments} key visual moments from transcript")
    
    try:
        # Try to use the LLM to identify key moments via Ollama
        try:
            from modules.llm.ollama import get_client
            
            # Get Ollama client
            ollama_client = get_client()
            
            if ollama_client.is_available():
                logger.info("Using Ollama client to extract key moments")
                
                # Prepare the prompt for the LLM
                prompt = f"""
You are an expert video editor. Identify {max_moments} key moments in this transcript where a screenshot would be valuable.

For each moment:
1. Identify timestamps where visual elements are important (slides, demonstrations, etc.)
2. Write a brief description of what's visually important
3. Rate importance as high, medium, or low

Format as JSON:
```json
[
  {{
    "timestamp": "MM:SS",
    "description": "Description of visual element",
    "importance": "high/medium/low"
  }}
]
```

TRANSCRIPT:
{transcript[:2000]}... (transcript continues)
"""
                # Call the Ollama API
                logger.info("Sending prompt to Ollama")
                llm_response = ollama_client.generate(
                    prompt=prompt,
                    system="You are a helpful assistant that outputs clean, valid JSON.",
                    temperature=0.3
                )
                
                if llm_response:
                    # Parse the response with the JSON parser
                    from modules.llm.json_parser import extract_key_moments
                    key_moments = extract_key_moments(llm_response)
                    
                    if key_moments:
                        logger.info(f"Successfully extracted {len(key_moments)} key moments from LLM")
                    else:
                        logger.info("No key moments found in LLM response, falling back to rule-based approach")
                        key_moments = find_visual_cues_in_transcript(transcript)
                else:
                    logger.warning("Empty response from LLM, falling back to rule-based approach")
                    key_moments = find_visual_cues_in_transcript(transcript)
            else:
                logger.warning("Ollama not available, falling back to rule-based approach")
                key_moments = find_visual_cues_in_transcript(transcript)
                
        except ImportError as e:
            logger.warning(f"Error importing Ollama client: {str(e)}")
            key_moments = find_visual_cues_in_transcript(transcript)
        except Exception as e:
            logger.warning(f"Error using Ollama client: {str(e)}")
            key_moments = find_visual_cues_in_transcript(transcript)
        
        # If we still don't have any key moments, create some based on transcript structure
        if not key_moments:
            logger.info("No key moments found using any method, creating generic moments")
            key_moments = []
            
            # If segments are provided, use them to create evenly spaced moments
            if segments and len(segments) > 0:
                # Get total transcript duration
                total_duration = segments[-1]["end"] if segments[-1].get("end") else 0
                
                if total_duration > 0:
                    # Create evenly spaced moments (beginning, 25%, 50%, 75%, near end)
                    spacing = [0.05, 0.25, 0.5, 0.75, 0.95]  # Avoid exact beginning/end
                    for i, pct in enumerate(spacing[:max_moments]):
                        timestamp_seconds = total_duration * pct
                        
                        # Format timestamp as MM:SS
                        minutes = int(timestamp_seconds // 60)
                        seconds = int(timestamp_seconds % 60)
                        formatted_timestamp = f"{minutes:02d}:{seconds:02d}"
                        
                        moment_type = "Introduction" if i == 0 else \
                                    "Conclusion" if i == len(spacing) - 1 else \
                                    f"Key point {i}"
                        
                        key_moments.append({
                            "timestamp": formatted_timestamp,
                            "description": f"{moment_type} at {formatted_timestamp}",
                            "importance": "high" if i in [0, len(spacing)-1] else "medium"
                        })
        
        # Process the extracted moments
        valid_moments = []
        for moment in key_moments:
            # Check for required fields
            if "timestamp" not in moment or not moment["timestamp"]:
                continue
                
            if "description" not in moment or not moment["description"]:
                continue
            
            # Ensure importance is one of the valid values
            if "importance" not in moment or moment["importance"] not in ["high", "medium", "low"]:
                moment["importance"] = "medium"
            
            # Add a title field derived from description
            if "title" not in moment:
                # Create a short title from the description
                title = moment["description"]
                if len(title) > 40:
                    title = title[:37] + "..."
                moment["title"] = title
            
            # If segments are provided, find the matching segment text
            if segments and "transcript_text" not in moment:
                # Convert timestamp to seconds for comparison
                try:
                    from modules.video_processing import parse_timestamp
                    timestamp_seconds = parse_timestamp(moment["timestamp"])
                except ImportError:
                    # Fallback if parse_timestamp is unavailable
                    parts = str(moment["timestamp"]).split(":")
                    if len(parts) == 2:
                        timestamp_seconds = int(parts[0]) * 60 + float(parts[1])
                    else:
                        timestamp_seconds = float(moment["timestamp"])
                
                # Find the segment that contains this timestamp
                matching_segment = None
                for segment in segments:
                    if "start" in segment and "end" in segment:
                        if segment["start"] <= timestamp_seconds <= segment["end"]:
                            matching_segment = segment
                            break
                
                # If we found a matching segment, add its text
                if matching_segment and "text" in matching_segment:
                    moment["transcript_text"] = matching_segment["text"]
                else:
                    # Otherwise, try to find the closest segment
                    closest_segment = min(segments, key=lambda s: 
                                        abs((s.get("start", 0) + s.get("end", 0))/2 - timestamp_seconds)
                                        if "start" in s and "end" in s else float('inf'))
                    if closest_segment and "text" in closest_segment:
                        moment["transcript_text"] = closest_segment["text"]
                    else:
                        moment["transcript_text"] = "No matching transcript text found"
            
            valid_moments.append(moment)
        
        # Sort by timestamp
        try:
            from modules.video_processing import parse_timestamp
            valid_moments.sort(key=lambda m: parse_timestamp(m["timestamp"]) if "timestamp" in m else 0)
        except ImportError:
            # Fallback sorting if parse_timestamp is unavailable
            def simple_parse(ts):
                if not ts:
                    return 0
                parts = str(ts).split(":")
                if len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                return float(ts)
            valid_moments.sort(key=lambda m: simple_parse(m.get("timestamp", 0)))
        
        # Ensure we don't exceed the max number of moments
        if len(valid_moments) > max_moments:
            valid_moments = valid_moments[:max_moments]
        
        logger.info(f"Successfully extracted {len(valid_moments)} key visual moments")
        return {
            "key_moments": valid_moments,
            "success": True,
            "count": len(valid_moments)
        }
        
    except Exception as e:
        logger.error(f"Error extracting key visual moments: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "key_moments": [],
            "success": False,
            "error": str(e)
        }

def enhance_moments_with_text(moments_data, segments):
    """Add transcript text to moments"""
    for moment in moments_data.get("key_moments", []):
        # Find the segment closest to the timestamp
        timestamp = moment.get("timestamp", 0)
        closest_segment = min(segments, key=lambda s: abs(s.get("start", 0) - timestamp))
        
        # Add the transcript text from this segment
        moment["transcript_text"] = closest_segment.get("text", "")
    
    return moments_data

def create_fallback_key_moments(segments):
    """Create fallback key moments at regular intervals in the transcript"""
    print("Creating fallback key moments")
    
    if not segments or len(segments) < 5:
        return {"key_moments": []}
        
    # Choose segments at different points in the transcript
    total_segments = len(segments)
    indices = [
        total_segments // 10,                 # 10% into the transcript
        total_segments // 4,                  # 25% into the transcript
        total_segments // 2,                  # 50% into the transcript
        (total_segments * 3) // 4,            # 75% into the transcript
        total_segments - (total_segments // 8) # Near the end
    ]
    
    # Ensure indices are within range
    indices = [min(i, total_segments - 1) for i in indices]
    # Ensure indices are unique
    indices = list(set(indices))
    
    fallback_moments = []
    for i, idx in enumerate(indices):
        segment = segments[idx]
        timestamp = segment.get('start', 0)
        
        # Create a title based on time
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        fallback_moments.append({
            "timestamp": timestamp,
            "title": f"Key Moment at {time_str}",
            "description": f"Important point in the discussion at timestamp {time_str}",
            "transcript_text": segment.get('text', '')
        })
    
    print(f"Created {len(fallback_moments)} fallback moments")
    return {"key_moments": fallback_moments}

def format_timestamp(seconds):
    """Format seconds into MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def extract_key_visual_moments(
    transcript: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    video_path: Optional[str] = None,
    max_moments: int = 5
) -> Dict[str, Any]:
    """
    Extract key visual moments from a transcript where screenshots would be valuable.
    
    Args:
        transcript: Full transcript text
        segments: List of transcript segments with timestamps (optional)
        video_path: Path to the video file (optional)
        max_moments: Maximum number of key moments to identify
        
    Returns:
        Dictionary with key_moments list and metadata
    """
    logger.info(f"Extracting up to {max_moments} key visual moments from transcript")
    
    try:
        # First try rule-based extraction
        key_moments = find_visual_cues_in_transcript(transcript)
        
        # If rule-based extraction didn't find enough moments, create generic ones
        if len(key_moments) < max_moments and segments and len(segments) > 0:
            logger.info(f"Rule-based approach found {len(key_moments)} moments, adding generic ones to reach {max_moments}")
            
            # Get total transcript duration
            total_duration = segments[-1]["end"] if segments[-1].get("end") else 0
            
            if total_duration > 0:
                # How many more moments do we need
                needed = max_moments - len(key_moments)
                
                # Create evenly spaced moments (beginning, 25%, 50%, 75%, near end)
                spacing = [0.05, 0.25, 0.5, 0.75, 0.95]  # Avoid exact beginning/end
                
                for i, pct in enumerate(spacing[:needed]):
                    timestamp_seconds = total_duration * pct
                    
                    # Format timestamp as MM:SS
                    minutes = int(timestamp_seconds // 60)
                    seconds = int(timestamp_seconds % 60)
                    formatted_timestamp = f"{minutes:02d}:{seconds:02d}"
                    
                    moment_type = "Introduction" if i == 0 else \
                                "Conclusion" if i == len(spacing) - 1 else \
                                f"Key point {i}"
                    
                    key_moments.append({
                        "timestamp": formatted_timestamp,
                        "description": f"{moment_type} at {formatted_timestamp}",
                        "importance": "high" if i in [0, len(spacing)-1] else "medium"
                    })
        
        # Process the extracted moments
        valid_moments = []
        for moment in key_moments:
            # Check for required fields
            if "timestamp" not in moment or not moment["timestamp"]:
                continue
                
            if "description" not in moment or not moment["description"]:
                continue
            
            # Ensure importance is one of the valid values
            if "importance" not in moment or moment["importance"] not in ["high", "medium", "low"]:
                moment["importance"] = "medium"
            
            # Add a title field derived from description
            if "title" not in moment:
                # Create a short title from the description
                title = moment["description"]
                if len(title) > 40:
                    title = title[:37] + "..."
                moment["title"] = title
            
            # If segments are provided, find the matching segment text
            if segments:
                # Convert timestamp to seconds for comparison
                try:
                    from modules.video_processing import parse_timestamp
                    timestamp_seconds = parse_timestamp(moment["timestamp"])
                except ImportError:
                    # Fallback if parse_timestamp is unavailable
                    parts = str(moment["timestamp"]).split(":")
                    if len(parts) == 2:
                        timestamp_seconds = int(parts[0]) * 60 + float(parts[1])
                    else:
                        timestamp_seconds = float(moment["timestamp"])
                
                # Find the segment that contains this timestamp
                matching_segment = None
                for segment in segments:
                    if "start" in segment and "end" in segment:
                        if segment["start"] <= timestamp_seconds <= segment["end"]:
                            matching_segment = segment
                            break
                
                # If we found a matching segment, add its text
                if matching_segment and "text" in matching_segment:
                    moment["transcript_text"] = matching_segment["text"]
                else:
                    # Otherwise, try to find the closest segment
                    closest_segment = min(segments, key=lambda s: 
                                          abs((s.get("start", 0) + s.get("end", 0))/2 - timestamp_seconds)
                                          if "start" in s and "end" in s else float('inf'))
                    if closest_segment and "text" in closest_segment:
                        moment["transcript_text"] = closest_segment["text"]
                    else:
                        moment["transcript_text"] = "No matching transcript text found"
            
            valid_moments.append(moment)
        
        # Sort by timestamp
        try:
            from modules.video_processing import parse_timestamp
            valid_moments.sort(key=lambda m: parse_timestamp(m["timestamp"]) if "timestamp" in m else 0)
        except ImportError:
            # Fallback sorting if parse_timestamp is unavailable
            def simple_parse(ts):
                if not ts:
                    return 0
                parts = str(ts).split(":")
                if len(parts) == 2:
                    return int(parts[0]) * 60 + float(parts[1])
                return float(ts)
            valid_moments.sort(key=lambda m: simple_parse(m.get("timestamp", 0)))
        
        # Ensure we don't exceed the max number of moments
        if len(valid_moments) > max_moments:
            valid_moments = valid_moments[:max_moments]
        
        logger.info(f"Successfully extracted {len(valid_moments)} key visual moments")
        return {
            "key_moments": valid_moments,
            "success": True,
            "count": len(valid_moments)
        }
        
    except Exception as e:
        logger.error(f"Error extracting key visual moments: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Even if there's an error, try to return some default moments
        if segments and len(segments) > 0:
            # Create a few default moments at regular intervals
            total_duration = segments[-1]["end"] if segments[-1].get("end") else 0
            default_moments = []
            
            if total_duration > 0:
                intervals = [0.1, 0.5, 0.9]  # Beginning, middle, end
                for i, pct in enumerate(intervals):
                    timestamp_seconds = total_duration * pct
                    minutes = int(timestamp_seconds // 60)
                    seconds = int(timestamp_seconds % 60)
                    desc = "Beginning" if i == 0 else "Middle" if i == 1 else "End"
                    
                    default_moments.append({
                        "timestamp": f"{minutes:02d}:{seconds:02d}",
                        "description": f"{desc} of video",
                        "importance": "medium",
                        "title": f"{desc} of video"
                    })
                
                return {
                    "key_moments": default_moments,
                    "success": False,
                    "error": str(e),
                    "note": "Using default moments due to error"
                }
        
        return {
            "key_moments": [],
            "success": False,
            "error": str(e)
        }

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