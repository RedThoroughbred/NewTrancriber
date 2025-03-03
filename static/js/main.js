// Stripped-down theme toggle function
function toggleTheme() {
    if (document.body.classList.contains('light-mode')) {
        // Switch to dark mode
        document.body.classList.remove('light-mode');
        document.body.classList.add('dark-mode');
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    } else {
        // Switch to light mode
        document.body.classList.remove('dark-mode');
        document.body.classList.add('light-mode');
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Load theme preference
    const savedTheme = localStorage.getItem('theme') || 'dark';
    
    // Apply theme
    if (savedTheme === 'light') {
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
        document.documentElement.setAttribute('data-theme', 'light');
    } else {
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
        document.documentElement.setAttribute('data-theme', 'dark');
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
    
    // Multiple transcript selection handlers
    initMultiTranscriptSelectionHandlers();
});

// Multiple transcript selection functionality
function initMultiTranscriptSelectionHandlers() {
    const selectAllCheckbox = document.getElementById('select-all');
    const transcriptCheckboxes = document.querySelectorAll('.transcript-select');
    const multiTranscriptActions = document.getElementById('multi-transcript-actions');
    const selectedCountElement = document.getElementById('selected-count');
    const compareTranscriptsBtn = document.getElementById('compare-transcripts-btn');
    
    // Skip if we're not on the dashboard page
    if (!selectAllCheckbox || !multiTranscriptActions) return;
    
    // Track selected transcripts
    let selectedTranscripts = [];
    
    // Function to update UI based on selection
    function updateSelectionUI() {
        const count = selectedTranscripts.length;
        
        // Update counter
        if (selectedCountElement) {
            selectedCountElement.textContent = count;
        }
        
        // Show/hide multi-transcript actions
        if (count >= 2) {
            multiTranscriptActions.classList.remove('hidden');
        } else {
            multiTranscriptActions.classList.add('hidden');
        }
        
        // Update select all checkbox
        if (selectAllCheckbox) {
            selectAllCheckbox.checked = count > 0 && count === transcriptCheckboxes.length;
            selectAllCheckbox.indeterminate = count > 0 && count < transcriptCheckboxes.length;
        }
    }
    
    // Handle individual transcript selection
    transcriptCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const transcriptId = this.dataset.id;
            const transcriptTitle = this.dataset.title;
            
            if (this.checked) {
                // Add to selected list if not already there
                if (!selectedTranscripts.some(t => t.id === transcriptId)) {
                    selectedTranscripts.push({
                        id: transcriptId,
                        title: transcriptTitle
                    });
                }
            } else {
                // Remove from selected list
                selectedTranscripts = selectedTranscripts.filter(t => t.id !== transcriptId);
            }
            
            // Sync checkboxes with same data-id (table and card views)
            document.querySelectorAll(`.transcript-select[data-id="${transcriptId}"]`).forEach(cb => {
                cb.checked = this.checked;
            });
            
            updateSelectionUI();
        });
    });
    
    // Handle select all checkbox
    if (selectAllCheckbox) {
        selectAllCheckbox.addEventListener('change', function() {
            const isChecked = this.checked;
            
            transcriptCheckboxes.forEach(checkbox => {
                checkbox.checked = isChecked;
                
                const transcriptId = checkbox.dataset.id;
                const transcriptTitle = checkbox.dataset.title;
                
                if (isChecked) {
                    // Add to selected list if not already there
                    if (!selectedTranscripts.some(t => t.id === transcriptId)) {
                        selectedTranscripts.push({
                            id: transcriptId,
                            title: transcriptTitle
                        });
                    }
                }
            });
            
            if (!isChecked) {
                // Clear selection if unchecked
                selectedTranscripts = [];
            }
            
            updateSelectionUI();
        });
    }
    
    // Compare transcripts button handler
    if (compareTranscriptsBtn) {
        compareTranscriptsBtn.addEventListener('click', function() {
            if (selectedTranscripts.length < 2) {
                alert('Please select at least 2 transcripts to compare');
                return;
            }
            
            // Get selected transcript IDs
            const transcriptIds = selectedTranscripts.map(t => t.id).join(',');
            
            // Navigate to compare page
            window.location.href = `/compare-transcripts?ids=${transcriptIds}`;
        });
    }
}

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