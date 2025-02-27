"""
Integration module for connecting the modules to the Flask app.

This module provides utilities to:
1. Check which advanced features are available
2. Extend the Flask app with new routes for these features
3. Provide a consistent interface between file storage and database
"""
import os
import importlib.util
from typing import Dict, List, Any, Optional, Callable
from flask import Flask, request, jsonify

def feature_is_available(feature_name: str) -> bool:
    """
    Check if a particular feature module is available and configured.
    
    Args:
        feature_name: Name of the feature ('supabase', 'vectors', 'llm')
        
    Returns:
        bool: True if the feature is available
    """
    if feature_name == 'supabase':
        try:
            from .supabase.client import is_available
            return is_available()
        except (ImportError, Exception):
            return False
            
    elif feature_name == 'vectors':
        try:
            # Check if either OpenAI API key or sentence-transformers is available
            if os.environ.get("OPENAI_API_KEY"):
                return True
                
            # Check if sentence-transformers is installed
            return importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            return False
            
    elif feature_name == 'llm':
        try:
            from .llm.ollama import is_available
            return is_available()
        except (ImportError, Exception):
            return False
    
    return False

def register_supabase_routes(app: Flask) -> None:
    """
    Register routes for Supabase integration.
    
    Args:
        app: Flask application instance
    """
    if not feature_is_available('supabase'):
        return
        
    from .supabase import storage
    
    @app.route('/api/transcripts', methods=['GET'])
    def list_transcripts_api():
        """List all transcripts from Supabase"""
        transcripts = storage.list_transcripts()
        return jsonify(transcripts)
    
    @app.route('/api/transcripts/<transcript_id>', methods=['GET'])
    def get_transcript_api(transcript_id):
        """Get a transcript by ID from Supabase"""
        transcript = storage.get_transcript(transcript_id)
        if not transcript:
            return jsonify({"error": "Transcript not found"}), 404
        return jsonify(transcript)
    
    @app.route('/api/transcripts/<transcript_id>', methods=['DELETE'])
    def delete_transcript_api(transcript_id):
        """Delete a transcript by ID from Supabase"""
        success = storage.delete_transcript(transcript_id)
        if not success:
            return jsonify({"error": "Failed to delete transcript"}), 500
        return jsonify({"success": True})

def register_vector_routes(app: Flask) -> None:
    """
    Register routes for vector search.
    
    Args:
        app: Flask application instance
    """
    if not feature_is_available('vectors'):
        return
        
    from .vectors import search, embeddings
    
    @app.route('/api/search/semantic', methods=['POST'])
    def semantic_search_api():
        """Perform semantic search on transcripts"""
        data = request.json
        query = data.get('query', '')
        limit = data.get('limit', 5)
        
        if not query:
            return jsonify([])
            
        results = search.semantic_search(query, limit)
        return jsonify(results)
    
    @app.route('/api/search/hybrid', methods=['POST'])
    def hybrid_search_api():
        """Perform hybrid semantic+keyword search on transcripts"""
        data = request.json
        query = data.get('query', '')
        limit = data.get('limit', 5)
        
        if not query:
            return jsonify([])
            
        results = search.hybrid_search(query, limit)
        return jsonify(results)
    
    # Add hook to create embeddings after transcription
    @app.after_request
    def process_transcription_response(response):
        """Create embeddings after successful transcription"""
        if request.endpoint == 'transcribe' and response.status_code == 200:
            try:
                # Get the response data
                data = response.get_json()
                
                # For each successful transcript
                for result in data:
                    if result.get('success'):
                        transcript_id = result.get('id')
                        
                        # Get the transcript text from file
                        import json
                        import os
                        transcript_path = os.path.join(
                            app.config['TRANSCRIPT_FOLDER'], 
                            f"{transcript_id}.json"
                        )
                        
                        with open(transcript_path, 'r') as f:
                            transcript_data = json.load(f)
                            
                        # Create embeddings in background
                        # In a production app, this would use a task queue
                        # For now, we'll just do it synchronously
                        embeddings.create_embeddings_for_transcript(
                            transcript_id, 
                            transcript_data['transcript']
                        )
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                
        return response

def register_llm_routes(app: Flask) -> None:
    """
    Register routes for LLM integration.
    
    Args:
        app: Flask application instance
    """
    if not feature_is_available('llm'):
        return
        
    from .llm import summarize, ollama
    
    @app.route('/api/transcripts/<transcript_id>/summary', methods=['GET'])
    def get_transcript_summary(transcript_id):
        """Get a summary of a transcript using LLM"""
        import json
        import os
        
        # Get transcript data
        transcript_path = os.path.join(
            app.config['TRANSCRIPT_FOLDER'], 
            f"{transcript_id}.json"
        )
        
        if not os.path.exists(transcript_path):
            return jsonify({"error": "Transcript not found"}), 404
            
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
            
        # Generate summary
        summary = summarize.summarize_transcript(transcript_data['transcript'])
        
        if not summary:
            return jsonify({"error": "Failed to generate summary"}), 500
            
        return jsonify({"summary": summary})
    
    @app.route('/api/transcripts/<transcript_id>/topics', methods=['GET'])
    def get_transcript_topics(transcript_id):
        """Extract topics from a transcript using LLM"""
        import json
        import os
        
        # Get transcript data
        transcript_path = os.path.join(
            app.config['TRANSCRIPT_FOLDER'], 
            f"{transcript_id}.json"
        )
        
        if not os.path.exists(transcript_path):
            return jsonify({"error": "Transcript not found"}), 404
            
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
            
        # Extract topics
        topics = summarize.extract_topics(transcript_data['transcript'])
        
        if topics is None:
            return jsonify({"error": "Failed to extract topics"}), 500
            
        return jsonify({"topics": topics})
    
    @app.route('/api/transcripts/<transcript_id>/ask', methods=['POST'])
    def ask_transcript_question(transcript_id):
        """Ask a question about a transcript using LLM"""
        import json
        import os
        
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Get transcript data
        transcript_path = os.path.join(
            app.config['TRANSCRIPT_FOLDER'], 
            f"{transcript_id}.json"
        )
        
        if not os.path.exists(transcript_path):
            return jsonify({"error": "Transcript not found"}), 404
            
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
            
        # Answer the question
        answer = summarize.answer_question(transcript_data['transcript'], question)
        
        if answer is None:
            return jsonify({"error": "Failed to answer question"}), 500
            
        return jsonify({"answer": answer})
    
    @app.route('/api/llm/status', methods=['GET'])
    def get_llm_status():
        """Get status of the LLM service and available models"""
        client = ollama.get_client()
        
        if not client.is_available():
            return jsonify({
                "available": False,
                "message": "Ollama service is not available"
            })
            
        # Get available models
        models = client.list_models()
        
        return jsonify({
            "available": True,
            "models": models,
            "default_model": client.default_model
        })

def integrate_with_app(app: Flask) -> None:
    """
    Integrate all available modules with the Flask app.
    
    Args:
        app: Flask application instance
    """
    # Register routes for each feature if available
    register_supabase_routes(app)
    register_vector_routes(app)
    register_llm_routes(app)
    
    # Add a status endpoint to check which features are available
    @app.route('/api/status', methods=['GET'])
    def get_status():
        """Get status of all advanced features"""
        return jsonify({
            "supabase": feature_is_available('supabase'),
            "vectors": feature_is_available('vectors'),
            "llm": feature_is_available('llm')
        })