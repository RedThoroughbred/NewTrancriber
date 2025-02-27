# Transcription App Modules

This directory contains optional modules for the transcription app that extend its functionality with advanced features:

## Features

1. **Supabase Integration** - Store transcripts in a Supabase database
   - PostgreSQL database for transcript storage
   - Vector support for semantic search
   - SQL migrations for setting up the database

2. **Vector Search** - Search transcripts using natural language
   - Convert transcripts to vector embeddings
   - Support for OpenAI and local sentence-transformer embeddings
   - Semantic and hybrid search capabilities

3. **Ollama Integration** - Use local LLMs for transcript analysis
   - Auto-detection of Ollama running locally
   - Transcript summarization
   - Topic extraction
   - Question answering

## Usage

To enable these modules, update the `app.py` file to include:

```python
from modules.integration import integrate_with_app

# After creating your Flask app
app = Flask(__name__)
# ... your existing setup code ...

# Integrate advanced modules
integrate_with_app(app)
```

## Dependencies

Each module has its own set of dependencies:

1. **Supabase**:
   - `supabase-py`

2. **Vector Search**:
   - OpenAI API access (`openai`) OR
   - `sentence-transformers` for local embeddings

3. **Ollama**:
   - Ollama running locally
   - No additional Python dependencies beyond `requests`

## Configuration

Set the following environment variables:

```
# Supabase
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key

# OpenAI (if using OpenAI embeddings)
OPENAI_API_KEY=your-openai-key

# Ollama
OLLAMA_URL=http://localhost:11434  # Default Ollama URL
```

## API Endpoints

The modules add the following API endpoints:

### Supabase

- `GET /api/transcripts` - List all transcripts
- `GET /api/transcripts/<id>` - Get a specific transcript
- `DELETE /api/transcripts/<id>` - Delete a transcript

### Vector Search

- `POST /api/search/semantic` - Semantic search (natural language)
- `POST /api/search/hybrid` - Hybrid search (semantic + keyword)

### Ollama LLM

- `GET /api/transcripts/<id>/summary` - Get transcript summary
- `GET /api/transcripts/<id>/topics` - Extract topics from transcript
- `POST /api/transcripts/<id>/ask` - Ask questions about the transcript
- `GET /api/llm/status` - Check Ollama availability and models

### Status

- `GET /api/status` - Check which features are available