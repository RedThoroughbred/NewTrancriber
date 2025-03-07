"""
modules/llm/json_parser.py - Utilities for extracting JSON from LLM responses
"""
import re
import json
import logging
from typing import Dict, List, Any, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_json_from_llm_response(response: str) -> Optional[Any]:
    """
    Extract and parse JSON from an LLM's text response.
    
    Handles common issues with LLM JSON outputs:
    - Extracts from markdown code blocks
    - Fixes missing quotes in property names
    - Fixes trailing commas
    - Handles single quotes instead of double quotes
    
    Args:
        response: Raw text response from the LLM
        
    Returns:
        Parsed JSON object or None if extraction fails
    """
    if not response:
        return None
    
    # Try to extract JSON from a code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON content between curly braces
        # This regex tries to find the outermost JSON object
        json_match = re.search(r'(\{[\s\S]*\})', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Last resort: try to find any array
            json_match = re.search(r'(\[[\s\S]*\])', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                logger.warning("No JSON content found in the LLM response")
                return None
    
    # Clean up the extracted JSON string
    json_str = json_str.strip()
    
    # Try to parse the JSON directly
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.info(f"Initial JSON parsing failed: {str(e)}. Attempting to fix common issues.")
        
        # Try to fix common issues with LLM JSON output
        try:
            # Fix trailing commas
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            
            # Fix missing quotes around property names
            json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            
            # Fix single quotes to double quotes, but carefully (avoiding contractions)
            json_str = re.sub(r'(?<![a-zA-Z])\'([^\']*?)\'(?![a-zA-Z])', r'"\1"', json_str)
            
            # Fix unquoted values that should be strings
            # json_str = re.sub(r':[ ]*([a-zA-Z][a-zA-Z0-9_]+)([,}])', r':"\1"\2', json_str)
            
            return json.loads(json_str)
        except json.JSONDecodeError as e2:
            logger.warning(f"Failed to parse JSON after fixing common issues: {str(e2)}")
            
            # Last resort for debugging: print the problematic JSON
            logger.debug(f"Problematic JSON: {json_str}")
            
            # Return None to indicate failure
            return None

def extract_key_moments(llm_response: str) -> List[Dict[str, Any]]:
    """
    Extract key visual moments from an LLM response.
    
    First tries to parse as JSON, then falls back to regex pattern matching
    if JSON parsing fails.
    
    Args:
        llm_response: Raw text response from the LLM
        
    Returns:
        List of key moments (timestamp, description, importance, etc.)
    """
    # First try JSON extraction
    json_data = extract_json_from_llm_response(llm_response)
    
    # If JSON extraction worked, determine the structure
    if json_data:
        # If it's already a list of moments
        if isinstance(json_data, list):
            # Validate each item has required fields
            for item in json_data:
                if not isinstance(item, dict):
                    continue
                    
                # Make sure there's a timestamp field
                if "timestamp" not in item and "time" in item:
                    item["timestamp"] = item["time"]
                elif "timestamp" not in item and "timecode" in item:
                    item["timestamp"] = item["timecode"]
                
                # Make sure there's a description field
                if "description" not in item and "text" in item:
                    item["description"] = item["text"]
                elif "description" not in item and "content" in item:
                    item["description"] = item["content"]
            
            return json_data
        
        # If it's an object with a key_moments field
        elif isinstance(json_data, dict):
            for field in ["key_moments", "moments", "visual_moments", "timestamps"]:
                if field in json_data and isinstance(json_data[field], list):
                    return json_data[field]
            
            # If we can't find a moments array, but it's a dict with timestamp-like fields,
            # it might be a single moment
            if any(key in json_data for key in ["timestamp", "time", "timecode"]):
                return [json_data]
    
    # Fallback to regex pattern matching for timestamps and descriptions
    logger.info("JSON extraction failed, falling back to regex pattern matching")
    
    moments = []
    # Match patterns like "1. At [00:15:30]", "Moment 1: [00:15:30]", etc.
    patterns = [
        # Match format with number, timestamp, then description
        r'(?:(?:\d+)[\.:\)]\s*)?(?:At\s*)?(?:\[|\()?(\d{1,2}:\d{2}(?::\d{2})?|(?:\d+)(?:\.\d+)?s?)(?:\]|\))?\s*(?:\-|:)?\s*(.+?)(?=(?:\d+)[\.:\)]|$)',
        # Match format with "Timestamp: HH:MM:SS - Description"
        r'(?:Timestamp|Time|At):\s*(?:\[|\()?(\d{1,2}:\d{2}(?::\d{2})?|(?:\d+)(?:\.\d+)?s?)(?:\]|\))?\s*(?:\-|:)?\s*(.+?)(?=\n|$)',
        # Match format with "HH:MM:SS - Description"
        r'(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:\-|:)\s*(.+?)(?=\n|$)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, llm_response, re.MULTILINE)
        for match in matches:
            timestamp = match.group(1).strip()
            description = match.group(2).strip()
            
            # Only add if we have both timestamp and description
            if timestamp and description:
                # Extract importance if present in the description
                importance = "medium"  # Default importance
                importance_match = re.search(r'\(importance:?\s*(high|medium|low)\)', description, re.IGNORECASE)
                if importance_match:
                    importance = importance_match.group(1).lower()
                    # Remove the importance annotation from the description
                    description = re.sub(r'\s*\(importance:?\s*(high|medium|low)\)', '', description, flags=re.IGNORECASE)
                
                moments.append({
                    "timestamp": timestamp,
                    "description": description,
                    "importance": importance
                })
    
    return moments