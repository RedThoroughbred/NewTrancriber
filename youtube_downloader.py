import os
import uuid
from yt_dlp import YoutubeDL
import re

def is_valid_youtube_url(url):
    """Check if the URL is a valid YouTube URL."""
    youtube_regex = (
        r'(https?://)?(www\.)?' 
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/' 
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?\s]{11})')
    
    match = re.match(youtube_regex, url)
    return match is not None

def get_youtube_id(url):
    """Extract YouTube video ID from a URL."""
    # Simple method for most YouTube URLs
    if 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0].split('&')[0]
    elif 'youtube.com/watch' in url:
        try:
            from urllib.parse import parse_qs, urlparse
            query = parse_qs(urlparse(url).query)
            return query.get('v', [''])[0]
        except:
            # Fallback to regex if parsing fails
            import re
            match = re.search(r'v=([^&]+)', url)
            if match:
                return match.group(1)
    
    return None

def download_youtube_video(url, output_folder):
    """
    Download a YouTube video and return info about it.
    
    Returns:
        tuple: (local_filepath, video_title, video_id)
    """
    if not is_valid_youtube_url(url):
        raise ValueError("Invalid YouTube URL")
    
    video_id = get_youtube_id(url)
    if not video_id:
        raise ValueError("Could not extract YouTube video ID")
    
    # Create a unique ID for the file
    file_id = str(uuid.uuid4())
    output_path = os.path.join(output_folder, f"{file_id}.mp4")
    
    # Options for yt-dlp
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        # Download the video
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown Title')
            
        return {
            'file_path': output_path,
            'title': title,
            'video_id': video_id,
            'file_id': file_id,
            'duration': info.get('duration', 0),
            'thumbnail': info.get('thumbnail', ''),
            'channel': info.get('channel', 'Unknown Channel'),
            'original_url': url
        }
    except Exception as e:
        raise Exception(f"Error downloading YouTube video: {str(e)}")