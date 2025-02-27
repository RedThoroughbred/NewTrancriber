"""
Vector-based semantic search for transcripts.
"""
from typing import List, Dict, Any, Optional
from .embeddings import get_embedding_provider
from ..supabase.client import get_client, is_available

def semantic_search(
    query: str, 
    limit: int = 5, 
    min_similarity: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on transcript vectors.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        min_similarity: Minimum similarity threshold (0-1)
        
    Returns:
        List of search results with transcript info and relevance score
    """
    if not is_available():
        return []
        
    try:
        # Generate embedding for query
        provider = get_embedding_provider()
        query_embedding = provider.embed_text(query)
        
        # Search using vector similarity in Supabase
        client = get_client()
        
        # Using RPC function for vector search
        result = client.rpc(
            "match_transcripts", 
            {
                "query_embedding": query_embedding,
                "match_threshold": min_similarity,
                "match_count": limit
            }
        ).execute()
        
        return result.data
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []

def hybrid_search(
    query: str,
    limit: int = 5,
    semantic_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining vector similarity and text matching.
    
    Args:
        query: The search query
        limit: Maximum number of results
        semantic_weight: Weight for semantic results vs keyword (0-1)
        
    Returns:
        List of search results with transcript info and relevance score
    """
    if not is_available():
        return []
        
    try:
        # Generate embedding for query
        provider = get_embedding_provider()
        query_embedding = provider.embed_text(query)
        
        # Search using RPC function
        client = get_client()
        result = client.rpc(
            "hybrid_search_transcripts", 
            {
                "query_text": query,
                "query_embedding": query_embedding,
                "match_count": limit,
                "similarity_threshold": 0.5,
                "semantic_weight": semantic_weight
            }
        ).execute()
        
        return result.data
        
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        return []