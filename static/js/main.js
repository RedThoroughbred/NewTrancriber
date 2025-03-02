document.addEventListener('DOMContentLoaded', function() {
    // Dark mode toggle
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
        });
    }
    
    // Initialize tabs
    const tabs = document.querySelectorAll('.sidebar-item');
    const panels = document.querySelectorAll('.panel');
    
    function setActiveTab(tabId) {
        // Hide all panels
        panels.forEach(panel => panel.classList.add('hidden'));
        
        // Remove active class from all tabs
        tabs.forEach(tab => tab.classList.remove('active'));
        
        // Add active class to selected tab
        document.getElementById(tabId).classList.add('active');
        
        // Show selected panel
        document.getElementById(tabId.replace('-tab', '-panel')).classList.remove('hidden');
    }
    
    // Add click listeners to tabs
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            if (!this.classList.contains('disabled')) {
                setActiveTab(this.id);
            }
        });
    });
});

// YouTube URL validation
function validateYoutubeUrl(url) {
    if (!url) return null;
    
    // Handle youtu.be URLs
    if (url.includes('youtu.be')) {
        const parts = url.split('/');
        return parts[parts.length - 1].split('?')[0];
    }
    // Handle youtube.com URLs
    else if (url.includes('youtube.com')) {
        try {
            // Try to extract v parameter using URLSearchParams
            const searchParams = new URL(url).searchParams;
            return searchParams.get('v');
        } catch (e) {
            // Fallback to regex extraction
            const match = /[?&]v=([^&#]*)/.exec(url);
            return match ? match[1] : null;
        }
    }
    return null;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Function to jump to a specific segment in the transcript
function jumpToSegment(index) {
    // Update current segment
    if (currentSegmentIndex >= 0) {
        document.getElementById(`segment-${currentSegmentIndex}`).classList.remove('bg-dark-900');
        const timelineSegments = document.querySelectorAll('.timeline-segment');
        if (timelineSegments.length > currentSegmentIndex) {
            timelineSegments[currentSegmentIndex].classList.remove('active');
        }
    }
    
    currentSegmentIndex = index;
    const segment = segments[index];
    
    // Update UI
    document.getElementById(`segment-${index}`).classList.add('bg-dark-900');
    const timelineSegments = document.querySelectorAll('.timeline-segment');
    if (timelineSegments.length > index) {
        timelineSegments[index].classList.add('active');
    }
    document.getElementById('current-time').textContent = formatTime(segment.start);
    
    // Scroll to segment
    document.getElementById(`segment-${index}`).scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Update timeline progress
    document.getElementById('timeline-progress').style.width = `${(segment.start / segments[segments.length - 1].end * 100)}%`;
    
    // In a full implementation, this would control video playback
    console.log(`Jumping to ${formatTime(segment.start)} - ${segment.text}`);
}

// Function to format time from seconds to minutes:seconds
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Function to open the video location
function openVideoLocation() {
    // In a real app, this would either open the file or show a modal with the path
    const filepath = document.querySelector('[data-filepath]')?.dataset.filepath;
    alert('Video file location: ' + (filepath || 'Not available'));
}