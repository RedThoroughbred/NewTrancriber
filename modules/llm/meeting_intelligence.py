"""
Enhanced transcript analysis for meeting intelligence features.

This module adds capabilities to detect questions, answers, decisions, and commitments
in meeting transcripts using LLM analysis. It also supports comparing multiple transcripts.
"""
from typing import Dict, List, Any, Optional
import time
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

# Multi-transcript analysis prompts
COMPARATIVE_SUMMARY_PROMPT = """
You are an expert at analyzing multiple meeting transcripts to identify patterns and connections.
Your task is to generate a comprehensive comparative analysis that summarizes the key points across all provided transcripts.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Focus on identifying connections, trends, and evolution of topics across meetings
3. Be concise but thorough
4. Use paragraphs to organize your thoughts
5. DO NOT include generic statements about what's in "each transcript"
6. Focus on synthesizing information across transcripts

Return a well-written comparative summary that would help someone understand the overall picture across these meetings.
"""

COMMON_TOPICS_PROMPT = """
You are an expert at analyzing multiple meeting transcripts to identify common topics.
Your task is to extract topics that appear across multiple transcripts.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "common_topics": [
       {
         "name": "Name of the common topic",
         "description": "Brief description of what this topic encompasses",
         "frequency": 3, // Number of transcripts where this appears
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
4. If no common topics are found, return {"common_topics": []}
"""

TOPIC_EVOLUTION_PROMPT = """
You are an expert at analyzing multiple meeting transcripts to track how topics evolve over time.
Your task is to identify topics that evolve or change across the provided transcripts.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "evolving_topics": [
       {
         "name": "Name of the evolving topic",
         "evolution": [
           {
             "transcript_id": "id-of-transcript", // From the transcript metadata
             "date": "2023-02-15", // From the transcript metadata
             "summary": "Brief summary of the topic's state/discussion at this point"
           },
           ...
         ]
       },
       ...
     ]
   }
3. List evolution entries in chronological order
4. Only include topics that show meaningful evolution or change
5. If no evolving topics are found, return {"evolving_topics": []}
"""

CONFLICTS_CHANGES_PROMPT = """
You are an expert at analyzing multiple meeting transcripts to identify conflicts or changes in decisions.
Your task is to identify where information, decisions, or plans changed or conflicted across the provided transcripts.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "conflicting_information": [
       {
         "topic": "The subject of the conflict or change",
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
4. Only include meaningful conflicts or changes, not minor differences
5. If no conflicts or changes are found, return {"conflicting_information": []}
"""

ACTION_ITEM_TRACKING_PROMPT = """
You are an expert at analyzing multiple meeting transcripts to track action items across meetings.
Your task is to identify action items and track their status across the provided transcripts.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list with the following structure:
   {
     "action_item_status": [
       {
         "description": "Description of the action item",
         "assignee": "Person assigned to the task",
         "first_mentioned": "2023-02-15", // Date when first mentioned
         "status": "completed", // One of: "pending", "in_progress", "completed", "canceled"
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
5. If the status changes, reflect that in the overall status
6. If no action items are found, return {"action_item_status": []}
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
        import json
        client = get_client()
        
        # Truncate combined text to avoid token limits
        truncated_text = _truncate_text(combined_text, max_length=12000)
        
        # Add metadata context
        metadata_context = "Transcripts metadata:\n"
        for i, meta in enumerate(transcripts_metadata):
            transcript_id = meta.get('id', f'transcript-{i}')
            title = meta.get('title', 'Untitled')
            date = meta.get('date', '').split('T')[0] if meta.get('date') else 'Unknown date'
            metadata_context += f"- Transcript {i+1}: ID={transcript_id}, Title={title}, Date={date}\n"
        
        # Combined prompt with metadata
        context_and_text = f"{metadata_context}\n\n{truncated_text}"
        
        # Analysis results
        results = {}
        
        # Generate comparative summary
        print("Generating comparative summary...")
        summary_prompt = "Generate a comparative summary of these transcripts, highlighting relationships between them:"
        results['comparative_summary'] = client.generate(
            prompt=summary_prompt + context_and_text,
            system=COMPARATIVE_SUMMARY_PROMPT,
            max_tokens=2000
        )
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
        
        # Extract common topics
        print("Extracting common topics...")
        topics_prompt = "Identify common topics that appear across multiple transcripts:"
        topics_result = client.generate(
            prompt=topics_prompt + context_and_text,
            system=COMMON_TOPICS_PROMPT,
            max_tokens=2000
        )
        
        try:
            topics_data = json.loads(topics_result)
            results['common_topics'] = topics_data.get('common_topics', [])
        except json.JSONDecodeError:
            print(f"Error parsing common topics JSON: {topics_result[:100]}...")
            # Try to extract JSON with regex
            import re
            json_match = re.search(r'(\{[\s\S]*\})', topics_result)
            if json_match:
                try:
                    topics_data = json.loads(json_match.group(1))
                    results['common_topics'] = topics_data.get('common_topics', [])
                except:
                    results['common_topics'] = []
            else:
                results['common_topics'] = []
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
        
        # Analyze topic evolution
        print("Analyzing topic evolution...")
        evolution_prompt = "Analyze how topics and discussions evolved across these transcripts over time:"
        evolution_result = client.generate(
            prompt=evolution_prompt + context_and_text,
            system=TOPIC_EVOLUTION_PROMPT,
            max_tokens=2000
        )
        
        try:
            evolution_data = json.loads(evolution_result)
            results['evolving_topics'] = evolution_data.get('evolving_topics', [])
        except json.JSONDecodeError:
            print(f"Error parsing topic evolution JSON: {evolution_result[:100]}...")
            # Try to extract JSON with regex
            import re
            json_match = re.search(r'(\{[\s\S]*\})', evolution_result)
            if json_match:
                try:
                    evolution_data = json.loads(json_match.group(1))
                    results['evolving_topics'] = evolution_data.get('evolving_topics', [])
                except:
                    results['evolving_topics'] = []
            else:
                results['evolving_topics'] = []
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
        
        # Find contradictions or conflicts
        print("Identifying conflicts and changes...")
        conflicts_prompt = "Identify any contradictions, conflicts, or changes in decisions across these transcripts:"
        conflicts_result = client.generate(
            prompt=conflicts_prompt + context_and_text,
            system=CONFLICTS_CHANGES_PROMPT,
            max_tokens=2000
        )
        
        try:
            conflicts_data = json.loads(conflicts_result)
            results['conflicting_information'] = conflicts_data.get('conflicting_information', [])
        except json.JSONDecodeError:
            print(f"Error parsing conflicts JSON: {conflicts_result[:100]}...")
            # Try to extract JSON with regex
            import re
            json_match = re.search(r'(\{[\s\S]*\})', conflicts_result)
            if json_match:
                try:
                    conflicts_data = json.loads(json_match.group(1))
                    results['conflicting_information'] = conflicts_data.get('conflicting_information', [])
                except:
                    results['conflicting_information'] = []
            else:
                results['conflicting_information'] = []
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
        
        # Track action items across meetings
        print("Tracking action items...")
        action_prompt = "Track action items across these transcripts, noting which were completed, which are still pending, and which changed:"
        action_result = client.generate(
            prompt=action_prompt + context_and_text,
            system=ACTION_ITEM_TRACKING_PROMPT,
            max_tokens=2000
        )
        
        try:
            action_data = json.loads(action_result)
            results['action_item_status'] = action_data.get('action_item_status', [])
        except json.JSONDecodeError:
            print(f"Error parsing action items JSON: {action_result[:100]}...")
            # Try to extract JSON with regex
            import re
            json_match = re.search(r'(\{[\s\S]*\})', action_result)
            if json_match:
                try:
                    action_data = json.loads(json_match.group(1))
                    results['action_item_status'] = action_data.get('action_item_status', [])
                except:
                    results['action_item_status'] = []
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