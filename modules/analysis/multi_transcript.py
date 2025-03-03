"""
Multi-transcript analysis module.
Provides functionality for analyzing multiple transcripts together.
"""
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

def load_transcript(transcript_id: str, transcripts_folder: str) -> Optional[Dict[str, Any]]:
    """
    Load a transcript from file.
    
    Args:
        transcript_id: The ID of the transcript to load
        transcripts_folder: Folder containing transcript files
        
    Returns:
        The transcript data or None if not found
    """
    try:
        transcript_path = os.path.join(transcripts_folder, f"{transcript_id}.json")
        if not os.path.exists(transcript_path):
            return None
            
        with open(transcript_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading transcript {transcript_id}: {e}")
        return None

def analyze_multiple_transcripts(
    transcript_ids: List[str],
    transcripts_folder: str,
    use_llm: bool = True,
    analysis_id: str = None
) -> Dict[str, Any]:
    """
    Analyze multiple transcripts together to find patterns and insights.
    
    Args:
        transcript_ids: List of transcript IDs to analyze
        transcripts_folder: Folder containing transcript files
        use_llm: Whether to use LLM for enhanced analysis
        analysis_id: Optional ID for the analysis, creates new one if not provided
        
    Returns:
        A dictionary containing the analysis results
    """
    # Initialize the results structure
    analysis_id = analysis_id or str(uuid.uuid4())
    print(f"Analyzing transcripts with ID: {analysis_id}")
    results = {
        'id': analysis_id,
        'date': datetime.now().isoformat(),
        'transcript_ids': transcript_ids,
        'comparative_summary': '',
        'common_topics': [],
        'evolving_topics': [],
        'conflicting_information': [],
        'action_item_status': []
    }
    
    # Load all transcripts
    transcripts = []
    
    for transcript_id in transcript_ids:
        transcript = load_transcript(transcript_id, transcripts_folder)
        if transcript:
            transcripts.append(transcript)
    
    if not transcripts:
        return results
    
    # Add metadata about the transcripts
    results['transcripts_metadata'] = [
        {
            'id': t.get('id', ''),
            'title': t.get('title', 'Untitled'),
            'date': t.get('date', ''),
            'topic': t.get('topic', '')
        }
        for t in transcripts
    ]
    
    # Basic analysis without LLM
    common_topics = find_common_topics(transcripts)
    results['common_topics'] = common_topics
    results['evolving_topics'] = track_topic_evolution(transcripts)
    results['action_item_status'] = track_action_items(transcripts)
    
    # Generate a meaningful comparative summary without LLM
    print("Generating comparative summary...")
    try:
        summary = generate_basic_summary(transcripts, common_topics)
        print(f"Summary generated: {len(summary)} characters")
        if len(summary) < 100:
            print(f"WARNING: Summary is very short: '{summary}'")
        results['comparative_summary'] = summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        results['comparative_summary'] = f"Error generating summary: {str(e)}"
    
    # We'll skip the LLM analysis for now since it's not working well
    # This can be re-enabled later when the LLM integration is more robust
    
    # Store the results
    results_path = os.path.join(transcripts_folder, f"comparison_{analysis_id}.json")
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f)
    except Exception as e:
        print(f"Error saving analysis results: {e}")
        
    return results

def generate_basic_summary(transcripts: List[Dict[str, Any]], common_topics: List[Dict[str, Any]]) -> str:
    """
    Generate a meaningful summary comparing the transcripts without using LLM.
    
    Args:
        transcripts: List of transcript data
        common_topics: List of common topics identified
        
    Returns:
        A comparative summary string
    """
    print(f"Generating summary for {len(transcripts)} transcripts")
    # Debug what we're working with
    for i, t in enumerate(transcripts):
        print(f"Transcript {i+1} for summary: {t.get('title', 'Untitled')}")
        has_text = 'transcript' in t and t['transcript']
        word_count = len(t.get('transcript', '').split()) if has_text else 0
        print(f"  - Has text: {'Yes' if has_text else 'No'}, Word count: {word_count}")
    if len(transcripts) < 2:
        return "Analysis requires at least two transcripts to compare."
    
    # Sort transcripts by date if available
    sorted_transcripts = sorted(
        transcripts, 
        key=lambda x: x.get('date', '0000-00-00')
    )
    
    # Get date range
    start_date = sorted_transcripts[0].get('date', '').split('T')[0] if sorted_transcripts[0].get('date') else 'unknown date'
    end_date = sorted_transcripts[-1].get('date', '').split('T')[0] if sorted_transcripts[-1].get('date') else 'unknown date'
    
    # Start building the summary
    summary = f"## Comparative Analysis of {len(transcripts)} Transcripts\n\n"
    summary += f"This analysis compares transcripts from {start_date} to {end_date}.\n\n"
    
    # Add information about the transcripts
    summary += "### Analyzed Transcripts\n\n"
    for i, t in enumerate(sorted_transcripts):
        title = t.get('title', 'Untitled')
        date = t.get('date', '').split('T')[0] if t.get('date') else 'unknown date'
        topic = t.get('topic', 'No topic specified')
        summary += f"- **Transcript {i+1}**: {title} ({date}) - {topic}\n"
    
    summary += "\n"
    
    # Add information about common topics
    if common_topics:
        summary += "### Key Common Topics\n\n"
        summary += "The following topics appear across multiple transcripts:\n\n"
        
        for i, topic in enumerate(common_topics[:5]):  # Show top 5 topics
            topic_name = topic.get('name', 'Unknown')
            topic_freq = topic.get('frequency', 0)
            summary += f"- **{topic_name}**: Appears in {topic_freq} transcripts\n"
            
            # Add details for each occurrence
            for occurrence in topic.get('transcripts', [])[:3]:  # Show up to 3 occurrences
                transcript_title = occurrence.get('title', 'Untitled')
                transcript_date = occurrence.get('date', 'unknown date')
                summary += f"  - Found in \"{transcript_title}\" ({transcript_date})\n"
        
        if len(common_topics) > 5:
            summary += f"\nPlus {len(common_topics) - 5} more common topics.\n"
        
        summary += "\n"
    else:
        summary += "### Topics\n\nNo common topics were identified across these transcripts.\n\n"
    
    # Find overlapping participants if available
    all_participants = {}
    has_participants = False
    
    for transcript in transcripts:
        # Extract speakers from segments if available
        if 'segments' in transcript and transcript['segments']:
            for segment in transcript['segments']:
                if 'speaker' in segment and segment['speaker']:
                    has_participants = True
                    speaker = segment['speaker']
                    if speaker not in all_participants:
                        all_participants[speaker] = set()
                    all_participants[speaker].add(transcript.get('id', ''))
    
    if has_participants:
        # Find participants in multiple transcripts
        common_participants = {
            speaker: transcript_ids 
            for speaker, transcript_ids in all_participants.items() 
            if len(transcript_ids) > 1
        }
        
        if common_participants:
            summary += "### Common Participants\n\n"
            summary += "These people participated in multiple transcripts:\n\n"
            
            for speaker, transcript_ids in common_participants.items():
                summary += f"- **{speaker}**: Participated in {len(transcript_ids)} transcripts\n"
                
            summary += "\n"
    
    # Add text similarity analysis
    summary += "### Content Comparison\n\n"
    
    if len(transcripts) == 2:
        # For exactly 2 transcripts, do a direct comparison
        t1 = sorted_transcripts[0]
        t2 = sorted_transcripts[1]
        
        t1_text = t1.get('transcript', '').lower()
        t2_text = t2.get('transcript', '').lower()
        
        # Get word counts
        t1_words = len(t1_text.split())
        t2_words = len(t2_text.split())
        
        # Do a very simple similarity check
        common_words = set(t1_text.split()) & set(t2_text.split())
        similarity_percentage = len(common_words) / (len(set(t1_text.split()) | set(t2_text.split()))) * 100
        
        summary += f"Comparing \"{t1.get('title', 'Transcript 1')}\" ({t1_words} words) and \"{t2.get('title', 'Transcript 2')}\" ({t2_words} words):\n\n"
        
        # Compare lengths
        if abs(t1_words - t2_words) > 0.3 * max(t1_words, t2_words):
            summary += f"- The transcripts differ significantly in length ({t1_words} vs {t2_words} words)\n"
        else:
            summary += f"- The transcripts are similar in length ({t1_words} vs {t2_words} words)\n"
        
        # Compare content
        summary += f"- Content similarity: Approximately {similarity_percentage:.1f}% overlap in vocabulary\n"
        
        # Find unique terms in each
        unique_t1 = get_significant_terms(t1_text, t2_text)
        unique_t2 = get_significant_terms(t2_text, t1_text)
        
        if unique_t1:
            summary += f"- Terms uniquely emphasized in \"{t1.get('title', 'Transcript 1')}\": {', '.join(unique_t1[:5])}\n"
            
        if unique_t2:
            summary += f"- Terms uniquely emphasized in \"{t2.get('title', 'Transcript 2')}\": {', '.join(unique_t2[:5])}\n"
    else:
        # For more than 2 transcripts
        summary += "Key statistics for the analyzed transcripts:\n\n"
        
        # Get word counts
        word_counts = []
        for transcript in sorted_transcripts:
            word_count = len(transcript.get('transcript', '').split())
            word_counts.append(word_count)
            title = transcript.get('title', 'Untitled')
            summary += f"- \"{title}\": {word_count} words\n"
        
        # Calculate average and variance
        avg_words = sum(word_counts) / len(word_counts)
        summary += f"- Average transcript length: {avg_words:.0f} words\n"
    
    summary += "\n"
    
    # Add conclusion
    summary += "### Summary\n\n"
    if common_topics:
        topic_names = [topic.get('name', '') for topic in common_topics[:3]]
        summary += f"These transcripts primarily discuss {', '.join(topic_names)}. "
    
    if len(transcripts) > 2:
        summary += f"The analysis covers {len(transcripts)} transcripts spanning " + \
                  f"from {start_date} to {end_date}."
    else:
        summary += f"The comparison reveals both similarities and differences between these two transcripts."
    
    return summary

def get_significant_terms(text1: str, text2: str, min_occurrences: int = 3) -> List[str]:
    """
    Find terms that appear significantly more in text1 than text2.
    
    Args:
        text1: The primary text to analyze
        text2: The comparison text
        min_occurrences: Minimum occurrences to be considered significant
        
    Returns:
        List of significant terms unique to text1
    """
    # Get word frequencies
    text1_words = text1.lower().split()
    text2_words = text2.lower().split()
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'a', 'in', 'to', 'of', 'that', 'is', 'it', 'for',
        'i', 'you', 'we', 'they', 'this', 'that', 'these', 'those',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'but', 'or', 'as',
        'if', 'then', 'else', 'when', 'up', 'down', 'in', 'out', 'on', 'off'
    }
    
    # Count words in text1
    text1_counts = {}
    for word in text1_words:
        if len(word) > 3 and word not in stop_words:  # Only consider meaningful words
            text1_counts[word] = text1_counts.get(word, 0) + 1
    
    # Count words in text2
    text2_counts = {}
    for word in text2_words:
        if len(word) > 3 and word not in stop_words:
            text2_counts[word] = text2_counts.get(word, 0) + 1
    
    # Find terms unique to or much more frequent in text1
    significant_terms = []
    
    for word, count in text1_counts.items():
        if count >= min_occurrences:
            text2_count = text2_counts.get(word, 0)
            # Word is either unique to text1 or at least 3x more frequent
            if text2_count == 0 or count / text2_count >= 3:
                significant_terms.append(word)
    
    # Sort by frequency in text1
    significant_terms.sort(key=lambda w: text1_counts[w], reverse=True)
    
    return [w.capitalize() for w in significant_terms]

def find_common_topics(transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find common topics across transcripts.
    
    Args:
        transcripts: List of transcript data
        
    Returns:
        List of common topics with metadata
    """
    all_topics = {}
    
    # Extract all topics from each transcript
    for transcript in transcripts:
        transcript_id = transcript.get('id', '')
        transcript_date = transcript.get('date', '').split('T')[0]
        transcript_title = transcript.get('title', 'Untitled')
        
        # Check for topics in different formats
        if 'topics' in transcript and isinstance(transcript['topics'], list):
            # New format with topics array
            for topic in transcript['topics']:
                if isinstance(topic, dict) and 'name' in topic:
                    topic_name = topic['name'].lower()
                    if topic_name not in all_topics:
                        all_topics[topic_name] = {
                            'name': topic['name'],
                            'description': topic.get('description', ''),
                            'frequency': 0,
                            'transcripts': []
                        }
                    all_topics[topic_name]['frequency'] += 1
                    all_topics[topic_name]['transcripts'].append({
                        'id': transcript_id,
                        'date': transcript_date,
                        'title': transcript_title
                    })
        
        # Check for main topic
        if 'topic' in transcript and transcript['topic']:
            topic_name = transcript['topic'].lower()
            if topic_name not in all_topics:
                all_topics[topic_name] = {
                    'name': transcript['topic'],
                    'description': '',
                    'frequency': 0,
                    'transcripts': []
                }
            all_topics[topic_name]['frequency'] += 1
            all_topics[topic_name]['transcripts'].append({
                'id': transcript_id,
                'date': transcript_date,
                'title': transcript_title
            })
            
        # Also extract topics from transcript text using simple keyword extraction
        if 'transcript' in transcript and transcript['transcript']:
            transcript_text = transcript['transcript'].lower()
            # List of common business/meeting keywords to look for
            keywords = [
                "project", "budget", "deadline", "timeline", "milestone", 
                "decision", "approval", "strategy", "marketing", "sales", 
                "customer", "client", "product", "feature", "release",
                "report", "analysis", "performance", "metrics", "goal",
                "objective", "task", "issue", "problem", "solution",
                "development", "design", "implementation", "testing", "launch"
            ]
            
            # Find occurrences of keywords
            for keyword in keywords:
                # Only consider meaningful occurrences (not just passing mentions)
                if keyword in transcript_text and transcript_text.count(keyword) >= 3:
                    # Make the first letter uppercase
                    display_name = keyword.capitalize()
                    
                    if keyword not in all_topics:
                        all_topics[keyword] = {
                            'name': display_name,
                            'description': f"Discussions about {display_name.lower()}",
                            'frequency': 0,
                            'transcripts': [],
                            'auto_detected': True
                        }
                    all_topics[keyword]['frequency'] += 1
                    
                    # Only add if not already added from this transcript
                    if not any(t['id'] == transcript_id for t in all_topics[keyword]['transcripts']):
                        all_topics[keyword]['transcripts'].append({
                            'id': transcript_id,
                            'date': transcript_date,
                            'title': transcript_title,
                            'occurrences': transcript_text.count(keyword)
                        })
    
    # Find topics that appear in multiple transcripts
    common_topics = [
        topic for topic_name, topic in all_topics.items()
        if topic['frequency'] > 1
    ]
    
    # Sort by frequency (descending)
    common_topics.sort(key=lambda x: x['frequency'], reverse=True)
    
    return common_topics

def track_topic_evolution(transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Track how topics evolved across transcripts.
    
    Args:
        transcripts: List of transcript data
        
    Returns:
        List of topic evolution data
    """
    # Sort transcripts by date
    sorted_transcripts = sorted(
        transcripts, 
        key=lambda x: x.get('date', '0000-00-00')
    )
    
    # Get common topics
    common_topics = find_common_topics(transcripts)
    topic_names = [t['name'].lower() for t in common_topics]
    
    evolution_data = []
    
    for topic in common_topics:
        topic_data = {
            'name': topic['name'],
            'evolution': []
        }
        
        for transcript in sorted_transcripts:
            transcript_id = transcript.get('id', '')
            transcript_date = transcript.get('date', '').split('T')[0]
            
            # Check if this transcript mentions the topic
            mentioned = False
            
            # Check in main topic
            if 'topic' in transcript and transcript['topic'] and transcript['topic'].lower() == topic['name'].lower():
                mentioned = True
            
            # Check in topics list
            if 'topics' in transcript and isinstance(transcript['topics'], list):
                for t in transcript['topics']:
                    if isinstance(t, dict) and 'name' in t and t['name'].lower() == topic['name'].lower():
                        mentioned = True
                        break
            
            # Check in summary
            if 'summary' in transcript and transcript['summary']:
                if topic['name'].lower() in transcript['summary'].lower():
                    mentioned = True
            
            if mentioned:
                # This transcript mentions the topic - add to evolution
                topic_data['evolution'].append({
                    'transcript_id': transcript_id,
                    'date': transcript_date,
                    'summary': f"Mentioned in {transcript.get('title', 'Untitled')}"
                })
        
        if topic_data['evolution']:
            evolution_data.append(topic_data)
    
    return evolution_data

def track_action_items(transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Track action items across transcripts.
    
    Args:
        transcripts: List of transcript data
        
    Returns:
        List of action items with status tracking
    """
    # Sort transcripts by date
    sorted_transcripts = sorted(
        transcripts, 
        key=lambda x: x.get('date', '0000-00-00')
    )
    
    all_action_items = {}
    
    # Process each transcript for action items
    for transcript in sorted_transcripts:
        transcript_id = transcript.get('id', '')
        transcript_date = transcript.get('date', '').split('T')[0]
        transcript_title = transcript.get('title', 'Untitled')
        
        # Extract action items from this transcript
        if 'action_items' in transcript and isinstance(transcript['action_items'], list):
            for item in transcript['action_items']:
                if isinstance(item, dict):
                    item_desc = item.get('description', '').lower()
                    if not item_desc:
                        continue
                    
                    # Create a key for this action item
                    item_key = item_desc.strip()
                    
                    if item_key not in all_action_items:
                        # First time seeing this action item
                        all_action_items[item_key] = {
                            'description': item.get('description', ''),
                            'assignee': item.get('assignee', ''),
                            'first_mentioned': transcript_date,
                            'first_transcript': transcript_title,
                            'status': 'pending',
                            'mentions': [],
                            'notes': ''
                        }
                    
                    # Add this mention
                    all_action_items[item_key]['mentions'].append({
                        'transcript_id': transcript_id,
                        'date': transcript_date,
                        'transcript_title': transcript_title,
                        'status': item.get('status', 'pending'),
                        'assignee': item.get('assignee', '')
                    })
                    
                    # Update status based on this mention
                    if item.get('status') == 'completed':
                        all_action_items[item_key]['status'] = 'completed'
                        all_action_items[item_key]['notes'] = f"Completed as mentioned in {transcript_title}"
                    elif item.get('status') == 'in_progress' and all_action_items[item_key]['status'] != 'completed':
                        all_action_items[item_key]['status'] = 'in_progress'
                        all_action_items[item_key]['notes'] = f"In progress as of {transcript_title}"
    
    # Convert to list and sort by first_mentioned
    action_items = list(all_action_items.values())
    action_items.sort(key=lambda x: x.get('first_mentioned', '0000-00-00'))
    
    return action_items

def save_analysis_results(
    analysis_results: Dict[str, Any],
    transcripts_folder: str
) -> str:
    """
    Save analysis results to a file.
    
    Args:
        analysis_results: The analysis results to save
        transcripts_folder: Folder to save the results
        
    Returns:
        The ID of the saved analysis
    """
    try:
        analysis_id = analysis_results.get('id', str(uuid.uuid4()))
        analysis_path = os.path.join(transcripts_folder, f"comparison_{analysis_id}.json")
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f)
            
        return analysis_id
    except Exception as e:
        print(f"Error saving analysis results: {e}")
        return ""

def load_analysis_results(
    analysis_id: str,
    transcripts_folder: str
) -> Optional[Dict[str, Any]]:
    """
    Load analysis results from a file.
    
    Args:
        analysis_id: The ID of the analysis to load
        transcripts_folder: Folder containing analysis files
        
    Returns:
        The analysis results or None if not found
    """
    try:
        analysis_path = os.path.join(transcripts_folder, f"comparison_{analysis_id}.json")
        if not os.path.exists(analysis_path):
            return None
            
        with open(analysis_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading analysis results {analysis_id}: {e}")
        return None