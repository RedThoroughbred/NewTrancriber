"""
modules/video_processing.py - Enhanced video processing for screenshot extraction
"""
import os
import cv2
import re
import logging
from typing import List, Dict, Any, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_timestamp(timestamp: Union[str, float, int]) -> float:
    """
    Parse timestamp in various formats to seconds.
    
    Handles formats:
    - Float/int seconds directly
    - "MM:SS" format
    - "HH:MM:SS" format
    - "SS.ms" format
    
    Args:
        timestamp: Timestamp in various formats
        
    Returns:
        Timestamp in seconds (float)
    """
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    
    if not isinstance(timestamp, str):
        raise ValueError(f"Timestamp must be string, int, or float, got {type(timestamp)}")
    
    # Try to parse HH:MM:SS or MM:SS format
    if ":" in timestamp:
        parts = timestamp.split(":")
        if len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + float(parts[1])
    
    # Try to parse direct seconds (possibly with decimal)
    try:
        # Strip 's' suffix if present (e.g., "10s")
        if timestamp.endswith('s') and timestamp[:-1].replace('.', '', 1).isdigit():
            return float(timestamp[:-1])
        return float(timestamp)
    except ValueError:
        logger.error(f"Could not parse timestamp: {timestamp}")
        return 0.0

def extract_frame(video_path: str, timestamp: Union[str, float, int]) -> Optional[object]:
    """
    Extract a frame from a video at a specific timestamp.
    
    Args:
        video_path: Path to the video file
        timestamp: Timestamp in seconds or formatted string (MM:SS, HH:MM:SS)
        
    Returns:
        Image as a numpy array, or None if failed
    """
    try:
        # Parse timestamp to seconds if it's a string
        timestamp_seconds = parse_timestamp(timestamp)
        
        # Check if video path exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: FPS={fps}, Frames={total_frames}, Duration={duration:.2f}s")
        
        # Validate timestamp
        if timestamp_seconds < 0:
            logger.warning(f"Negative timestamp {timestamp_seconds}, using 0")
            timestamp_seconds = 0
        
        if duration > 0 and timestamp_seconds > duration:
            logger.warning(f"Timestamp {timestamp_seconds}s exceeds video duration {duration:.2f}s, using last frame")
            timestamp_seconds = max(0, duration - 1)  # Use last frame (with 1s buffer)
        
        # Calculate frame number from timestamp
        frame_number = int(timestamp_seconds * fps)
        
        # Make sure frame number is valid
        if frame_number >= total_frames:
            frame_number = total_frames - 1
        
        logger.info(f"Seeking to frame {frame_number} at time {timestamp_seconds:.2f}s")
            
        # Set frame position and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        # Release the video capture object
        cap.release()
        
        if ret:
            return frame
        else:
            logger.error(f"Could not read frame at {timestamp_seconds}s")
            
            # Fallback: try an earlier frame if this one failed
            if frame_number > 10:
                logger.info(f"Trying fallback frame at {frame_number - 10}")
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 10)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    logger.info("Fallback frame extraction succeeded")
                    return frame
            
            return None
            
    except Exception as e:
        logger.error(f"Error extracting frame: {str(e)}")
        return None

def save_frame(frame: object, output_path: str, quality: int = 95) -> bool:
    """
    Save a frame as an image file with error handling.
    
    Args:
        frame: Frame as numpy array
        output_path: Path to save the image
        quality: JPEG quality (0-100, higher is better)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create params for JPEG quality
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        
        # Save the frame as an image
        result = cv2.imwrite(output_path, frame, encode_params)
        
        if result:
            logger.info(f"Successfully saved frame to {output_path}")
            return True
        else:
            logger.error(f"Failed to save frame to {output_path}")
            return False
    except Exception as e:
        logger.error(f"Error saving frame: {str(e)}")
        
        # Try to save with a different name if there's an issue with the path
        try:
            alt_path = os.path.join(
                os.path.dirname(output_path),
                f"backup_{os.path.basename(output_path)}"
            )
            result = cv2.imwrite(alt_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if result:
                logger.info(f"Saved frame to alternate path: {alt_path}")
                return True
        except Exception as backup_error:
            logger.error(f"Backup save also failed: {str(backup_error)}")
        
        return False

def extract_screenshots_for_transcript(
    transcript_id: str,
    video_path: str, 
    timestamps: List[Union[str, float, int]],
    output_folder: str
) -> List[Dict[str, Any]]:
    """
    Extract screenshots at specific timestamps and save them.
    
    Args:
        transcript_id: ID of the transcript
        video_path: Path to the video file
        timestamps: List of timestamps in seconds or formatted strings
        output_folder: Folder to save the screenshots
        
    Returns:
        List of dictionaries with timestamp and screenshot path
    """
    results = []
    
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    logger.info(f"Extracting {len(timestamps)} screenshots for transcript {transcript_id}")
    logger.info(f"Video path: {video_path}")
    logger.info(f"Output folder: {output_folder}")
    
    # Validate video path
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return results
    
    # Check if it's a valid video file
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return results
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: FPS={fps}, Frames={total_frames}, Duration={duration:.2f}s")
        cap.release()
    except Exception as e:
        logger.error(f"Error validating video: {str(e)}")
        return results
    
    for i, timestamp in enumerate(timestamps):
        try:
            # Parse the timestamp
            timestamp_seconds = parse_timestamp(timestamp)
            
            # Generate a descriptive filename with padded index and formatted timestamp
            minutes = int(timestamp_seconds // 60)
            seconds = int(timestamp_seconds % 60)
            filename = f"{transcript_id}_screenshot_{i:02d}_{minutes:02d}m{seconds:02d}s.jpg"
            filepath = os.path.join(output_folder, filename)
            
            logger.info(f"Processing screenshot {i+1}/{len(timestamps)} at {timestamp_seconds:.2f}s")
            
            # Extract the frame
            frame = extract_frame(video_path, timestamp_seconds)
            
            if frame is not None:
                # Save the frame
                success = save_frame(frame, filepath)
                
                if success:
                    # Create web-friendly path for frontend
                    static_folder = os.path.join(os.path.dirname(output_folder), 'static')
                    if static_folder in filepath:
                        web_path = '/' + os.path.relpath(filepath, static_folder)
                    else:
                        web_path = filepath
                    
                    # Add to results
                    results.append({
                        "timestamp": timestamp_seconds,
                        "screenshot_path": filepath,
                        "web_path": web_path,
                        "index": i
                    })
                    logger.info(f"Saved screenshot at {timestamp_seconds:.2f}s to {filepath}")
                else:
                    logger.error(f"Failed to save screenshot at {timestamp_seconds:.2f}s")
            else:
                logger.error(f"Failed to extract frame at {timestamp_seconds:.2f}s")
        except Exception as e:
            logger.error(f"Error processing screenshot at index {i}: {str(e)}")
    
    logger.info(f"Extracted {len(results)}/{len(timestamps)} screenshots successfully")
    return results

def parse_timestamp(timestamp: Union[str, float, int]) -> float:
    """
    Parse timestamp in various formats to seconds.
    
    Handles formats:
    - Float/int seconds directly
    - "MM:SS" format
    - "HH:MM:SS" format
    - "SS.ms" format
    
    Args:
        timestamp: Timestamp in various formats
        
    Returns:
        Timestamp in seconds (float)
    """
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    
    if not isinstance(timestamp, str):
        raise ValueError(f"Timestamp must be string, int, or float, got {type(timestamp)}")
    
    # Try to parse HH:MM:SS or MM:SS format
    if ":" in timestamp:
        parts = timestamp.split(":")
        if len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + float(parts[1])
    
    # Try to parse direct seconds (possibly with decimal)
    try:
        # Strip 's' suffix if present (e.g., "10s")
        if timestamp.endswith('s') and timestamp[:-1].replace('.', '', 1).isdigit():
            return float(timestamp[:-1])
        return float(timestamp)
    except ValueError:
        print(f"Could not parse timestamp: {timestamp}")
        return 0.0