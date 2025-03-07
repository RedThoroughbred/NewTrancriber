"""
Transcript analysis and summarization using LLMs.

This module provides tools to analyze transcripts using LLMs,
including summarization, topic extraction, and question answering.
"""
import re
import json
from typing import Dict, List, Any, Optional
from .ollama import get_client, is_available

# System prompts for different analysis types - IMPROVED FOR EXECUTIVE FOCUS
SUMMARY_SYSTEM_PROMPT = """
You are an executive assistant summarizing meeting transcripts for a busy CEO.
Your task is to create a concise, executive-level summary that captures the critical points
of the transcript. Focus on business-relevant information, key decisions, action items,
and strategic discussions.

IMPORTANT FORMATTING RULES:
1. Start with a 1-2 sentence "Executive Summary" capturing the most critical takeaway
2. Organize the rest of the summary in clear bullet points
3. Use present tense when possible
4. Keep your summary to 5-8 bullet points (depending on transcript length)
5. Focus each bullet point on ONE specific insight, decision, or action item
6. Flag any financial, strategic, or timeline-related points as [CRITICAL]
7. Include specific names when people made important commitments
8. End with a "Next Steps" bullet if any were discussed in the meeting
9. NEVER use phrases like "Here is a summary" or "This transcript discusses"
"""

TOPICS_SYSTEM_PROMPT = """
You are an executive assistant analyzing meeting transcripts for a busy CEO.
Your task is to identify and extract the business-critical topics or themes discussed in the transcript.

IMPORTANT FORMATTING RULES:
1. Do NOT start with phrases like "Here are the topics", "Sure", or any introduction
2. Go directly into listing the topics with no preamble
3. Focus on business-relevant topics (strategy, finance, operations, clients, products, etc.)
4. For each topic, include a short description and its business impact (High/Medium/Low)
5. Prioritize topics that affect business outcomes, timelines, or resources

Format each topic as follows:
1. [Topic Name] - [Business Impact: High/Medium/Low]
   - [Key insight from transcript]
   - [Business implication]

Example format only:
1. Q3 Budget Revision - Business Impact: High
   - CFO plans to reduce marketing spend by 15%
   - Will affect new customer acquisition targets

2. Project Falcon Timeline - Business Impact: Medium
   - Launch delayed from September to October
   - Additional QA resources needed
"""

QA_SYSTEM_PROMPT = """
You are an executive assistant providing answers based on meeting transcripts.
Answer questions with a focus on business-critical information that would be relevant to a CEO.

IMPORTANT GUIDELINES:
1. Be direct and concise - executives need clear, straightforward answers
2. Prioritize information about finances, strategy, timelines, clients, and team performance
3. If the answer is not in the transcript, clearly state that
4. Include relevant quotes or data points from the transcript to support your answer
5. If the question touches on multiple aspects, organize your answer with bullet points
6. Highlight particularly important information that might require executive attention
"""

ACTION_ITEMS_SYSTEM_PROMPT = """
You are an executive assistant extracting action items from meeting transcripts for a CEO.
Your task is to identify and prioritize all action items, tasks, and commitments.

IMPORTANT FORMATTING RULES:
1. Do NOT include any introduction or explanation text
2. Return a JSON-formatted list of action items with the following structure:
   {
     "action_items": [
       {
         "task": "Brief description of the task",
         "assignee": "Person responsible (or 'Unassigned' if not specified)",
         "due": "Due date or timeframe (or 'Not specified' if not mentioned)",
         "context": "The sentence or phrase containing the action item",
         "priority": "High/Medium/Low based on business impact and urgency",
         "status": "pending",
         "business_impact": "Brief note on how this affects business outcomes"
       }
     ]
   }
3. Prioritize tasks related to:
   - Revenue-generating activities
   - Client relationships
   - Strategic initiatives
   - Critical deadlines
   - Risk mitigation
4. If no action items are found, return {"action_items": []}
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
        
        prompt = f"Summarize this meeting transcript for a CEO, focusing on business-critical information:\n\n{truncated_text}"
        
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
            f"Identify the {max_topics} most business-critical topics discussed in this meeting transcript. "
            f"For each topic, provide a name, business impact level, and key insights:\n\n{truncated_text}"
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
                
                # Check if this is a numbered list item (e.g. "1. Topic name - Business Impact: High")
                is_numbered = bool(re.match(number_pattern, line))
                
                if is_numbered:
                    # Extract the topic name and impact if present
                    topic_text = re.sub(number_pattern, '', line)
                    
                    # Check for impact level
                    impact_match = re.search(r'Business Impact:\s*(High|Medium|Low)', topic_text, re.IGNORECASE)
                    impact = impact_match.group(1) if impact_match else "Medium"
                    
                    # Get clean topic name
                    topic_name = topic_text
                    if impact_match:
                        topic_name = re.sub(r'\s*-\s*Business Impact:.*$', '', topic_text)
                    
                    if current_topic:
                        topics.append(current_topic)
                        
                    current_topic = {
                        "name": topic_name.strip(), 
                        "business_impact": impact,
                        "points": []
                    }
                    in_numbered_list = True
                    
                elif line.startswith('-') and current_topic:
                    # This is a bullet point for the current topic
                    point = line.lstrip('-').strip()
                    current_topic["points"].append(point)
                    
                elif not line.startswith('-') and not in_numbered_list:
                    # This is a topic header (not in a numbered list and not a bullet point)
                    if current_topic:
                        topics.append(current_topic)
                        
                    current_topic = {"name": line, "points": []}
            
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
        
def extract_action_items(transcript_text: str) -> dict:
    """
    Extract action items from a transcript using LLM.
    
    Args:
        transcript_text: The transcript text
        
    Returns:
        Dictionary with action items
    """
    if not is_available():
        return {"action_items": []}
    
    try:
        client = get_client()
        truncated_text = _truncate_text(transcript_text)
        prompt = (
            f"Extract all business-critical action items, tasks, and commitments from this transcript. "
            f"Include who is responsible, prioritize by business impact, and note due dates if mentioned:\n\n{truncated_text}"
        )
        
        result = client.generate(
            prompt=prompt,
            system=ACTION_ITEMS_SYSTEM_PROMPT,
            max_tokens=1000
        )
        print(f"Raw action items result: {result}")
        
        # Use the safe_json_loads function
        action_items = safe_json_loads(result, {"action_items": []})
        
        # Ensure we have the expected structure
        if isinstance(action_items, dict) and "action_items" in action_items:
            return action_items
        elif isinstance(action_items, list):
            # If we got a list, wrap it in the expected dictionary structure
            return {"action_items": action_items}
        else:
            # If we got a dict without action_items
            return {"action_items": []}
                
    except Exception as e:
        print(f"Error extracting action items: {e}")
        return {"action_items": []}