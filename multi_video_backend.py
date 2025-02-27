@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Enhanced transcribe endpoint that handles multiple files with individual metadata.
    
    For each file, it expects:
    - file_0, file_1, file_2, etc.
    - title_0, title_1, title_2, etc. (optional)
    - topic_0, topic_1, topic_2, etc. (optional)
    - file_count: total number of files
    
    It also handles YouTube URL and file path inputs as before.
    """
    # Clean upload folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    results = []
    
    # Check if we're dealing with YouTube URL
    if 'youtube_url' in request.form:
        from youtube_downloader import download_youtube_video
        
        youtube_url = request.form.get('youtube_url')
        
        # Get metadata
        title = request.form.get('title', '')
        topic = request.form.get('topic', '')
        
        try:
            # Download the YouTube video
            video_info = download_youtube_video(youtube_url, app.config['UPLOAD_FOLDER'])
            
            file_id = video_info['file_id']
            filepath = video_info['file_path']
            
            # Use video title if no title provided
            if not title:
                title = video_info['title']
            
            # Process with Whisper
            model = get_model()
            result = model.transcribe(filepath)
            
            # Save transcript as JSON with metadata
            transcript_data = {
                'id': file_id,
                'title': title,
                'topic': topic,
                'original_filename': f"{video_info['title']}.mp4",
                'date': datetime.now().isoformat(),
                'filepath': filepath,
                'transcript': result['text'],
                'segments': result['segments'],
                'youtube_id': video_info['video_id'],
                'youtube_url': youtube_url,
                'youtube_channel': video_info['channel']
            }
            
            transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{file_id}.json")
            with open(transcript_path, 'w') as f:
                json.dump(transcript_data, f)
            
            results.append({
                'id': file_id,
                'title': title,
                'success': True
            })
            
        except Exception as e:
            results.append({
                'filename': 'YouTube Video',
                'success': False,
                'error': str(e)
            })
            
    # Check if we're dealing with a file path
    elif 'file_path' in request.form:
        import shutil
        
        file_path = request.form.get('file_path')
        
        # Get metadata
        title = request.form.get('title', os.path.basename(file_path))
        topic = request.form.get('topic', '')
        
        try:
            # Verify the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Generate a unique ID
            file_id = str(uuid.uuid4())
            filename = os.path.basename(file_path)
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Copy the file to our uploads directory
            shutil.copy2(file_path, dest_path)
            
            # Process with Whisper
            model = get_model()
            result = model.transcribe(dest_path)
            
            # Save transcript as JSON with metadata
            transcript_data = {
                'id': file_id,
                'title': title,
                'topic': topic,
                'original_filename': filename,
                'date': datetime.now().isoformat(),
                'filepath': file_path,  # Store the original path
                'transcript': result['text'],
                'segments': result['segments']
            }
            
            transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{file_id}.json")
            with open(transcript_path, 'w') as f:
                json.dump(transcript_data, f)
            
            results.append({
                'id': file_id,
                'title': title,
                'success': True
            })
            
        except Exception as e:
            results.append({
                'filename': os.path.basename(file_path),
                'success': False,
                'error': str(e)
            })
            
    # Check if the post request has multiple files with per-file metadata
    elif 'file_count' in request.form:
        file_count = int(request.form.get('file_count', 0))
        
        for i in range(file_count):
            file_key = f'file_{i}'
            title_key = f'title_{i}'
            topic_key = f'topic_{i}'
            
            if file_key not in request.files:
                continue
                
            file = request.files[file_key]
            if file.filename == '':
                continue
                
            # Get metadata for this specific file
            title = request.form.get(title_key, file.filename)
            topic = request.form.get(topic_key, '')
            
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            extension = os.path.splitext(original_filename)[1]
            filename = f"{file_id}{extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            
            # Process with Whisper
            try:
                model = get_model()
                result = model.transcribe(filepath)
                
                # Save transcript as JSON with metadata
                transcript_data = {
                    'id': file_id,
                    'title': title,
                    'topic': topic,
                    'original_filename': original_filename,
                    'date': datetime.now().isoformat(),
                    'filepath': filepath,
                    'transcript': result['text'],
                    'segments': result['segments']
                }
                
                transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{file_id}.json")
                with open(transcript_path, 'w') as f:
                    json.dump(transcript_data, f)
                
                results.append({
                    'id': file_id,
                    'title': title,
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'filename': original_filename,
                    'success': False,
                    'error': str(e)
                })
    
    # Fallback to the standard file upload format
    elif 'file' in request.files:
        files = request.files.getlist('file')
        
        for file in files:
            if file.filename == '':
                continue
                
            if file:
                # Generate unique filename
                original_filename = secure_filename(file.filename)
                file_id = str(uuid.uuid4())
                extension = os.path.splitext(original_filename)[1]
                filename = f"{file_id}{extension}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the file
                file.save(filepath)
                
                # Get metadata
                title = request.form.get('title', original_filename)
                topic = request.form.get('topic', '')
                
                # Process with Whisper
                try:
                    model = get_model()
                    result = model.transcribe(filepath)
                    
                    # Save transcript as JSON with metadata
                    transcript_data = {
                        'id': file_id,
                        'title': title,
                        'topic': topic,
                        'original_filename': original_filename,
                        'date': datetime.now().isoformat(),
                        'filepath': filepath,
                        'transcript': result['text'],
                        'segments': result['segments']
                    }
                    
                    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{file_id}.json")
                    with open(transcript_path, 'w') as f:
                        json.dump(transcript_data, f)
                    
                    results.append({
                        'id': file_id,
                        'title': title,
                        'success': True
                    })
                    
                except Exception as e:
                    results.append({
                        'filename': original_filename,
                        'success': False,
                        'error': str(e)
                    })
    else:
        return jsonify({'error': 'No file or YouTube URL provided'}), 400
    
    return jsonify(results)