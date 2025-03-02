"""
Enhanced transcript analysis for meeting intelligence features.

This module adds capabilities to detect questions, answers, decisions, and commitments
in meeting transcripts using LLM analysis.
"""
from typing import Dict, List, Any, Optional
from .ollama import get_client, is_available

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

def extract_questions_answers(transcript_text: str) -> Optional[dict]:
    """
    Extract questions and answers from a transcript using LLM.
    
    Args:
        transcript_text: The transcript text
        
    Returns:
        Dictionary with question-answer pairs or None if failed
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
        
        try:
            import json
            qa_data = json.loads(result)
            return qa_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw result: {result}")
            import re
            json_match = re.search(r'(\{[\s\S]*\})', result)
            if json_match:
                try:
                    qa_data = json.loads(json_match.group(1))
                    return qa_data
                except:
                    pass
            return {"qa_pairs": []}
        
    except Exception as e:
        print(f"Error extracting questions and answers: {e}")
        return {"qa_pairs": []}

def extract_decisions(transcript_text: str) -> Optional[dict]:
    """
    Extract decisions and conclusions from a transcript using LLM.
    
    Args:
        transcript_text: The transcript text
        
    Returns:
        Dictionary with decisions or None if failed
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
        
        try:
            import json
            decisions_data = json.loads(result)
            return decisions_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw result: {result}")
            import re
            json_match = re.search(r'(\{[\s\S]*\})', result)
            if json_match:
                try:
                    decisions_data = json.loads(json_match.group(1))
                    return decisions_data
                except:
                    pass
            return {"decisions": []}
        
    except Exception as e:
        print(f"Error extracting decisions: {e}")
        return {"decisions": []}

def extract_commitments(transcript_text: str) -> Optional[dict]:
    """
    Extract personal commitments and promises from a transcript using LLM.
    
    Args:
        transcript_text: The transcript text
        
    Returns:
        Dictionary with commitments or None if failed
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
        
        try:
            import json
            commitments_data = json.loads(result)
            return commitments_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw result: {result}")
            import re
            json_match = re.search(r'(\{[\s\S]*\})', result)
            if json_match:
                try:
                    commitments_data = json.loads(json_match.group(1))
                    return commitments_data
                except:
                    pass
            return {"commitments": []}
        
    except Exception as e:
        print(f"Error extracting commitments: {e}")
        return {"commitments": []}

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