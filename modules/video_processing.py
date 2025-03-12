"""
modules/video_processing.py - Enhanced video processing for screenshot extraction
"""
import os
import cv2
import re
import logging
import numpy as np
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

def enhance_image_for_readability(frame, is_ui_screenshot=True):
    """
    Apply advanced image processing techniques to enhance text readability in screenshots,
    with special handling for UI screenshots.
    
    Args:
        frame: Original frame as numpy array
        is_ui_screenshot: Whether this is a UI screenshot (enables text-specific enhancements)
        
    Returns:
        Enhanced frame
    """
    try:
        import cv2
        import numpy as np
        
        # Make a copy to avoid modifying the original
        enhanced = frame.copy()
        
        # Increase resolution (optional - can help with small text)
        # This is a simple scaling - more advanced upscaling could be implemented
        scale_factor = 1.5  # Increase size by 50%
        if scale_factor > 1:
            enhanced = cv2.resize(enhanced, None, fx=scale_factor, fy=scale_factor, 
                                interpolation=cv2.INTER_CUBIC)
        
        # UI-specific enhancements for text clarity
        if is_ui_screenshot:
            # 1. Reduce noise with a bilateral filter (preserves edges)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 2. Enhance text edges
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to make text more prominent
            kernel = np.ones((2,2), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Create mask to enhance text areas
            mask = dilated_edges.astype(np.float32) / 255.0
            mask = cv2.cvtColor(mask.astype(np.float32), cv2.COLOR_GRAY2BGR)
            
            # Enhance contrast in text areas - with proper type handling
            try:
                # Convert to float32 for consistent mathematical operations
                enhanced_float = enhanced.astype(np.float32)
                # Create the weighted overlay with proper type handling
                enhanced_overlay = cv2.multiply(enhanced_float, mask)
                # Combine using addWeighted with explicit types
                combined = cv2.addWeighted(enhanced_float, 1.2, enhanced_overlay, 0.3, 0)
                # Convert back to uint8 for display/storage
                enhanced = np.clip(combined, 0, 255).astype(np.uint8)
            except Exception as e:
                logger.warning(f"Image enhancement step failed: {str(e)}")

        # 3. Apply adaptive color correction
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel with careful parameters for UI
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L channel back
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 4. Apply sharpening
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # 5. Enhance saturation slightly (makes colors more vivid)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)  # Increase saturation by 20%
        s = np.clip(s, 0, 255).astype(np.uint8)
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        # 6. Final resize back to original size if we scaled up
        if scale_factor > 1:
            original_h, original_w = frame.shape[:2]
            enhanced = cv2.resize(enhanced, (original_w, original_h), 
                                interpolation=cv2.INTER_AREA)
        
        return enhanced
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return frame  # Return original frame if enhancement fails

def extract_frame(video_path: str, timestamp: Union[str, float, int], enhance: bool = True, is_ui_screenshot: bool = True) -> Optional[object]:
    """
    Extract a frame from a video at a specific timestamp with enhanced quality.
    
    Args:
        video_path: Path to the video file
        timestamp: Timestamp in seconds or formatted string (MM:SS, HH:MM:SS)
        enhance: Whether to apply readability enhancements
        is_ui_screenshot: Whether this is a UI screenshot (enables text-specific enhancements)
        
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
            # Extract additional frames and stack them for noise reduction (helps with text clarity)
            if enhance and is_ui_screenshot:
                # Stack multiple frames if they're available (reduces noise)
                additional_frames = []
                if frame_number > 0:
                    cap = cv2.VideoCapture(video_path)
                    # Get a few nearby frames
                    for offset in range(-2, 3, 1):
                        if offset == 0:  # Skip the main frame, we already have it
                            continue
                        target_frame = max(0, min(total_frames-1, frame_number + offset))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        ret, nearby_frame = cap.read()
                        if ret:
                            additional_frames.append(nearby_frame)
                    cap.release()
                
                if additional_frames:
                    # Stack and average frames to reduce noise (improves text clarity)
                    all_frames = [frame] + additional_frames
                    stacked_frame = np.mean(all_frames, axis=0).astype(np.uint8)
                    frame = stacked_frame
            
            # Apply image enhancement if requested
            if enhance:
                frame = enhance_image_for_readability(frame, is_ui_screenshot)
                
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
                
                if ret and enhance:
                    frame = enhance_image_for_readability(frame, is_ui_screenshot)
                    logger.info("Fallback frame extraction succeeded")
                    return frame
            
            return None
            
    except Exception as e:
        logger.error(f"Error extracting frame: {str(e)}")
        return None

def save_frame(frame: object, output_path: str, quality: int = 100, use_png: bool = True) -> dict:
    """
    Save a frame as an image file with high quality settings.
    
    Args:
        frame: Frame as numpy array
        output_path: Path to save the image
        quality: JPEG quality (0-100, higher is better)
        use_png: Whether to save as PNG (lossless) instead of JPEG
        
    Returns:
        Dictionary with success status and actual path used
    """
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        actual_path = output_path  # Default to original path
        
        # Determine output format
        if use_png:
            # Switch to PNG format (lossless, better for text)
            png_path = output_path.replace('.jpg', '.png')
            
            # Use highest compression for PNG (0 = no compression, 9 = max compression)
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # Using 1 for good balance of quality vs size
            
            result = cv2.imwrite(png_path, frame, compression_params)
            actual_path = png_path  # Update output path
        else:
            # Create params for maximum JPEG quality
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            
            # For JPEG, use the best chroma subsampling (more detail)
            encode_params.extend([cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            encode_params.extend([cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
            
            # Save the frame as a high-quality JPEG
            result = cv2.imwrite(output_path, frame, encode_params)
        
        if result:
            logger.info(f"Successfully saved frame to {actual_path}")
            return {"success": True, "path": actual_path}
        else:
            logger.error(f"Failed to save frame to {actual_path}")
            return {"success": False, "path": None}
        
    except Exception as e:
        logger.error(f"Error saving frame: {str(e)}")
        
        # Try to save with a different name if there's an issue with the path
        try:
            alt_path = os.path.join(
                os.path.dirname(output_path),
                f"backup_{os.path.basename(output_path)}"
            )
            
            if use_png:
                alt_path = alt_path.replace('.jpg', '.png')
                
            if use_png:
                result = cv2.imwrite(alt_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                result = cv2.imwrite(alt_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                
            if result:
                logger.info(f"Saved frame to alternate path: {alt_path}")
                return {"success": True, "path": alt_path}
        except Exception as backup_error:
            logger.error(f"Backup save also failed: {str(backup_error)}")
        
        return {"success": False, "path": None}

def extract_screenshots_for_transcript(
    transcript_id: str,
    video_path: str, 
    timestamps: List[Union[str, float, int]],
    output_folder: str
) -> List[Dict[str, Any]]:
    """
    Extract screenshots at specific timestamps and save them with enhanced quality.
    
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
    
    # Whether to use PNG (better quality but larger files)
    use_png = True  # Set to False for JPG
    
    for i, timestamp in enumerate(timestamps):
        try:
            # Parse the timestamp
            timestamp_seconds = parse_timestamp(timestamp)
            
            # Generate a descriptive filename with padded index and formatted timestamp
            minutes = int(timestamp_seconds // 60)
            seconds = int(timestamp_seconds % 60)
            
            # Choose the right extension based on format
            ext = ".png" if use_png else ".jpg"
            filename = f"{transcript_id}_screenshot_{i:02d}_{minutes:02d}m{seconds:02d}s{ext}"
            filepath = os.path.join(output_folder, filename)
            
            logger.info(f"Processing screenshot {i+1}/{len(timestamps)} at {timestamp_seconds:.2f}s")
            
            # Extract the frame with advanced enhancement for UI screenshots
            # Use is_ui_screenshot=True for UI specific optimizations
            frame = extract_frame(video_path, timestamp_seconds, enhance=True, is_ui_screenshot=True)
            
            if frame is not None:
                # Save the frame with maximum quality
                save_result = save_frame(frame, filepath, quality=100, use_png=use_png)
                
                if save_result["success"] and save_result["path"]:
                    # Use the actual path returned by save_frame
                    actual_path = save_result["path"]
                    
                    # Create web-friendly path for frontend
                    static_folder = 'static'
                    if static_folder in actual_path:
                        # Ensure proper path formatting with single slash after static
                        parts = actual_path.split(static_folder)
                        if len(parts) > 1:
                            web_path = '/' + static_folder + parts[1]
                        else:
                            web_path = '/' + actual_path
                    else:
                        web_path = actual_path
                    
                    # Add to results
                    results.append({
                        "timestamp": timestamp_seconds,
                        "screenshot_path": actual_path,
                        "web_path": web_path,
                        "index": i
                    })
                    logger.info(f"Saved screenshot at {timestamp_seconds:.2f}s to {actual_path}")
                else:
                    logger.error(f"Failed to save screenshot at {timestamp_seconds:.2f}s")
            else:
                logger.error(f"Failed to extract frame at {timestamp_seconds:.2f}s")
        except Exception as e:
            logger.error(f"Error processing screenshot at index {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
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