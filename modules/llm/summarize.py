"""
Transcript analysis and summarization using LLMs.

This module provides tools to analyze transcripts using LLMs,
including summarization, topic extraction, and question answering.
"""
from typing import Dict, List, Any, Optional
from .ollama import get_client, is_available

# System prompts for different analysis types
SUMMARY_SYSTEM_PROMPT = """
You are an expert at summarizing video and audio transcripts.
Your task is to create a concise, comprehensive summary that captures the main points
of the transcript. Focus on the key ideas, eliminate redundancy, and maintain the
original meaning without adding new information. Organize the summary in bullet points.
"""

TOPICS_SYSTEM_PROMPT = """
You are an expert at analyzing video and audio transcripts.
Your task is to identify and extract the main topics or themes discussed in the transcript.
For each topic, include a brief description and list key points or quotes.
Structure your response as a list of topics with bullet points for each.
"""

QA_SYSTEM_PROMPT = """
You are an expert at understanding video and audio transcripts.
Answer the user's question based solely on the information provided in the transcript.
If the answer is not in the transcript, state that clearly. Provide direct quotes 
from the transcript where possible to support your answer.
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
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if not line.startswith('-') and not line.startswith('*'):
                    # This is a topic header
                    if current_topic:
                        topics.append(current_topic)
                        
                    current_topic = {"name": line, "points": []}
                elif current_topic:
                    # This is a bullet point
                    point = line.lstrip('-*').strip()
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