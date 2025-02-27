"""
Vector embedding generation and storage module.

This module supports two ways of generating embeddings:
1. Using OpenAI's embedding API
2. Using a local sentence-transformers model
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional, Union, Generator
import nltk
from ..supabase.client import get_client, is_available

# Ensure NLTK is available for text splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Default max tokens for each chunk
DEFAULT_CHUNK_SIZE = 512

def split_text_into_chunks(text: str, max_tokens: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Split text into chunks of approximately max_tokens.
    Uses NLTK to split by sentences to maintain coherence.
    
    Args:
        text: The text to split
        max_tokens: Approximate max tokens per chunk
        
    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # Rough approximation of tokens (words)
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size > max_tokens and current_chunk:
            # Save current chunk and start a new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

class EmbeddingProvider:
    """Base class for embedding providers"""
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        raise NotImplementedError()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        return [self.embed_text(text) for text in texts]


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI API-based embeddings"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
            self.model = model
        except (ImportError, Exception) as e:
            raise RuntimeError(f"Failed to initialize OpenAI embeddings: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in one API call"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating batch OpenAI embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 1536] * len(texts)


class LocalEmbeddings(EmbeddingProvider):
    """Local sentence-transformers embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except (ImportError, Exception) as e:
            raise RuntimeError(f"Failed to initialize local embeddings: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using local model"""
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating local embedding: {e}")
            # Get the model's embedding dimension
            dim = self.model.get_sentence_embedding_dimension()
            # Return a zero vector as fallback
            return [0.0] * dim
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using local model"""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating batch local embeddings: {e}")
            # Get the model's embedding dimension
            dim = self.model.get_sentence_embedding_dimension()
            # Return zero vectors as fallback
            return [[0.0] * dim] * len(texts)


def get_embedding_provider() -> EmbeddingProvider:
    """
    Get the appropriate embedding provider based on environment.
    
    Returns:
        An EmbeddingProvider instance
    """
    # Check for OpenAI API key first
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddings()
        except Exception as e:
            print(f"Failed to use OpenAI embeddings: {e}")
    
    # Fall back to local embeddings
    try:
        return LocalEmbeddings()
    except Exception as e:
        raise RuntimeError(
            f"No embedding provider available. Please install sentence-transformers "
            f"or configure OPENAI_API_KEY. Error: {e}"
        )

def create_embeddings_for_transcript(transcript_id: str, transcript_text: str) -> bool:
    """
    Create and store embeddings for a transcript.
    
    Args:
        transcript_id: The transcript ID
        transcript_text: The full transcript text
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not is_available():
        return False
    
    try:
        # Split text into chunks
        chunks = split_text_into_chunks(transcript_text)
        
        if not chunks:
            return False
            
        # Get embeddings
        provider = get_embedding_provider()
        embeddings = provider.embed_batch(chunks)
        
        # Store in Supabase
        client = get_client()
        data = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data.append({
                "chunk_text": chunk,
                "content_vector": embedding,
                "chunk_index": i,
                "transcript_id": transcript_id
            })
        
        # Insert in batches of 10 to avoid request size limits
        batch_size = 10
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            client.table("transcript_vectors").insert(batch).execute()
        
        return True
        
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return False