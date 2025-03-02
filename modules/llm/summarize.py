"""
Transcript analysis and summarization using LLMs.

This module provides tools to analyze transcripts using LLMs,
including summarization, topic extraction, and question answering.
"""
import re
from typing import Dict, List, Any, Optional
from .ollama import get_client, is_available

# System prompts for different analysis types
SUMMARY_SYSTEM_PROMPT = """
You are an expert at summarizing video and audio transcripts.
Your task is to create a concise, comprehensive summary that captures the main points
of the transcript. Focus on the key ideas, eliminate redundancy, and maintain the
original meaning without adding new information.The summary should be detailed enough to convey the full context and important 
points of the discussion, typically 3-5 paragraphs or 8-12 bullet points depending 
on the length of the original transcript.

IMPORTANT FORMATTING RULES:
1. Do NOT start with phrases like "Here is a summary", "Sure", or any other introduction
2. Go directly into the summary content with no preamble
3. Organize the summary in bullet points
4. Keep each bullet point brief and factual
5. Use present tense when possible
"""

TOPICS_SYSTEM_PROMPT = """
You are an expert at analyzing video and audio transcripts.
Your task is to identify and extract the main topics or themes discussed in the transcript.

IMPORTANT FORMATTING RULES:
1. Do NOT start with phrases like "Here are the topics", "Sure", or any introduction
2. Go directly into listing the topics with no preamble
3. Do NOT include a heading like "Topics" or "Main Topics"

For each topic:
1. Format the topic name as a numbered list item (e.g., "1. Meeting Introduction")
2. Under each topic, add 2-3 bullet points with key points from the transcript
3. Each bullet point should start with a dash or asterisk
4. Keep bullet points short and focused

Example format:
1. Project Status Update
   - Development team completed backend API endpoints
   - QA reported 3 critical bugs to be fixed
   - Timeline extended by 2 weeks

2. Budget Discussion
   - Current spend is 15% under budget
   - Additional resources needed for marketing
"""

QA_SYSTEM_PROMPT = """
You are an expert at understanding video and audio transcripts.
Answer the user's question based solely on the information provided in the transcript.
If the answer is not in the transcript, state that clearly. Provide direct quotes 
from the transcript where possible to support your answer.
"""

ACTION_ITEMS_SYSTEM_PROMPT = """
You are an expert at analyzing meeting transcripts and identifying action items, tasks, and commitments.
Your task is to extract all action items from the transcript, including who is responsible and when it's due (if mentioned).

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Go directly into listing the action items with no preamble
3. Return a JSON-formatted list of action items with the following structure:
   {
     "action_items": [
       {
         "task": "Brief description of the task",
         "assignee": "Person responsible (or 'Unassigned' if not specified)",
         "due": "Due date or timeframe (or 'Not specified' if not mentioned)",
         "context": "The sentence or phrase containing the action item",
         "priority": "High/Medium/Low based on urgency language in the transcript"
       }
     ]
   }
4. Only include clear action items, tasks, or commitments
5. Look for phrases like "will do", "needs to", "should", "going to", "I'll", "we'll", etc.
6. If no action items are found, return {"action_items": []}

Example action items to detect:
- "John will prepare the report by Friday"
- "We need to update the website next week"
- "I'll send you the documentation tomorrow"
- "Sarah should follow up with the client"
- "Let's schedule a meeting to discuss this further"
"""

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

def summarize_transcript(transcript_text: str, max_length: int = 500) -> Optional[str]:
    """
    Generate a summary of a transcript using LLM.
    
    Args:
        transcript_text: The transcript text
        max_length: Maximum length of the summary
        
    Returns:
        Generated summary or None if failed
    """
    if not is_available():
        return None
        
    try:
        client = get_client()
        
        # Truncate transcript if too long
        truncated_text = _truncate_text(transcript_text)
        
        prompt = f"Please summarize the following transcript concisely:\n\n{truncated_text}"
        
        summary = client.generate(
            prompt=prompt,
            system=SUMMARY_SYSTEM_PROMPT,
            max_tokens=max_length
        )
        
        return summary
        
    except Exception as e:
        print(f"Error summarizing transcript: {e}")
        return None

def extract_topics(transcript_text: str, max_topics: int = 5) -> Optional[List[Dict[str, Any]]]:
    """
    Extract main topics from a transcript using LLM.
    
    Args:
        transcript_text: The transcript text
        max_topics: Maximum number of topics to extract
        
    Returns:
        List of topics with descriptions or None if failed
    """
    if not is_available():
        return None
        
    try:
        client = get_client()
        
        # Truncate transcript if too long
        truncated_text = _truncate_text(transcript_text)
        
        prompt = (
            f"Please identify the {max_topics} main topics discussed in this transcript. "
            f"For each topic, provide a short name and brief description:\n\n{truncated_text}"
        )
        
        result = client.generate(
            prompt=prompt,
            system=TOPICS_SYSTEM_PROMPT,
            max_tokens=1000
        )
        
        # Parse the result into structured format if needed
        topics = []
        if result:
            lines = result.split('\n')
            current_topic = None
            in_numbered_list = False
            number_pattern = r'^\d+\.\s'
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a numbered list item (e.g. "1. Topic name")
                is_numbered = bool(re.match(number_pattern, line))
                
                if is_numbered:
                    # Extract the topic name without the number prefix
                    topic_name = re.sub(number_pattern, '', line)
                    
                    if current_topic:
                        topics.append(current_topic)
                        
                    current_topic = {"name": topic_name, "points": []}
                    in_numbered_list = True
                    
                elif not line.startswith('-') and not line.startswith('*') and not in_numbered_list:
                    # This is a topic header (not in a numbered list and not a bullet point)
                    if current_topic:
                        topics.append(current_topic)
                        
                    current_topic = {"name": line, "points": []}
                    
                elif current_topic:
                    # This is a bullet point or description text for the current topic
                    point = line.lstrip('-*•').strip()
                    current_topic["points"].append(point)
            
            if current_topic:
                topics.append(current_topic)
                
        return topics[:max_topics]
        
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return None

def answer_question(transcript_text: str, question: str) -> Optional[str]:
    """
    Answer a question based on transcript content using LLM.
    
    Args:
        transcript_text: The transcript text
        question: The question to answer
        
    Returns:
        Answer to the question or None if failed
    """
    if not is_available():
        return None
        
    try:
        client = get_client()
        
        # Truncate transcript if too long
        truncated_text = _truncate_text(transcript_text)
        
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the transcript:\n\n{truncated_text}\n\nQuestion: {question}"}
        ]
        
        answer = client.chat(
            messages=messages,
            max_tokens=500
        )
        
        return answer
        
    except Exception as e:
        print(f"Error answering question: {e}")
        return None
        
def extract_action_items(transcript_text: str) -> Optional[dict]:
    if not is_available():
        return {"action_items": []}
    
    try:
        client = get_client()
        truncated_text = _truncate_text(transcript_text)
        prompt = (
            f"Extract all action items, tasks, commitments, and follow-ups from this transcript. "
            f"Include who is responsible and when it's due if mentioned.\n\n{truncated_text}"
        )
        
        result = client.generate(
            prompt=prompt,
            system=ACTION_ITEMS_SYSTEM_PROMPT,
            max_tokens=1000
        )
        print(f"Raw action items result: {result}")  # Add this
        
        try:
            import json
            action_items = json.loads(result)
            return action_items
        except json.JSONDecodeError as e:
            print(f"Error parsing action items JSON: {e}")
            print(f"Raw result: {result}")
            json_match = re.search(r'(\{[\s\S]*\})', result)
            if json_match:
                try:
                    action_items = json.loads(json_match.group(1))
                    return action_items
                except:
                    pass
            return {"action_items": []}
        
    except Exception as e:
        print(f"Error extracting action items: {e}")
        return {"action_items": []}