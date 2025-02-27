"""
Supabase storage module for transcript data.
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .client import get_client, is_available

def store_transcript(
    transcript_data: Dict[str, Any],
    transcript_id: str
) -> bool:
    """
    Store a transcript in Supabase.
    
    Args:
        transcript_data: The transcript data dictionary
        transcript_id: The unique ID for the transcript
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not is_available():
        return False
        
    try:
        client = get_client()
        
        # Prepare the data for insertion
        data = {
            "id": transcript_id,
            "title": transcript_data.get("title", "Untitled"),
            "topic": transcript_data.get("topic", ""),
            "source_type": "youtube" if "youtube_url" in transcript_data else "upload",
            "source_url": transcript_data.get("youtube_url", ""),
            "original_filename": transcript_data.get("original_filename", ""),
            "content": json.dumps({
                "transcript": transcript_data.get("transcript", ""),
                "segments": transcript_data.get("segments", [])
            }),
            "created_at": datetime.now().isoformat()
        }
        
        # Insert into transcripts table
        result = client.table("transcripts").insert(data).execute()
        
        return len(result.data) > 0
    except Exception as e:
        print(f"Error storing transcript: {e}")
        return False

def get_transcript(transcript_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a transcript from Supabase by ID.
    
    Args:
        transcript_id: The unique ID for the transcript
        
    Returns:
        The transcript data or None if not found
    """
    if not is_available():
        return None
        
    try:
        client = get_client()
        
        result = client.table("transcripts").select("*").eq("id", transcript_id).execute()
        
        if not result.data:
            return None
            
        # Convert from database format to app format
        db_record = result.data[0]
        content = json.loads(db_record["content"])
        
        return {
            "id": db_record["id"],
            "title": db_record["title"],
            "topic": db_record["topic"],
            "original_filename": db_record["original_filename"],
            "date": db_record["created_at"],
            "transcript": content["transcript"],
            "segments": content["segments"],
            "youtube_url": db_record.get("source_url", "")
        }
        
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return None

def list_transcripts() -> List[Dict[str, Any]]:
    """
    List all transcripts from Supabase.
    
    Returns:
        A list of transcript summary data (without full content)
    """
    if not is_available():
        return []
        
    try:
        client = get_client()
        
        result = client.table("transcripts").select(
            "id,title,topic,source_type,created_at,original_filename"
        ).order("created_at", desc=True).execute()
        
        return result.data
        
    except Exception as e:
        print(f"Error listing transcripts: {e}")
        return []

def delete_transcript(transcript_id: str) -> bool:
    """
    Delete a transcript from Supabase.
    
    Args:
        transcript_id: The unique ID for the transcript
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not is_available():
        return False
        
    try:
        client = get_client()
        
        # Delete vectors first (foreign key constraint)
        client.table("transcript_vectors").delete().eq("transcript_id", transcript_id).execute()
        
        # Delete transcript
        result = client.table("transcripts").delete().eq("id", transcript_id).execute()
        
        return len(result.data) > 0
        
    except Exception as e:
        print(f"Error deleting transcript: {e}")
        return False

def search_transcripts(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search transcripts using text matching.
    This is a basic search without vector capabilities.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        
    Returns:
        List of matching transcripts
    """
    if not is_available():
        return []
        
    try:
        client = get_client()
        
        # Use Postgres text search
        result = client.rpc(
            "search_transcripts_text", 
            {"query_text": query, "match_count": limit}
        ).execute()
        
        return result.data
        
    except Exception as e:
        print(f"Error searching transcripts: {e}")
        return []