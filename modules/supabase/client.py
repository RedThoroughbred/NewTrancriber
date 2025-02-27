"""
Supabase client connection module.
This handles connecting to Supabase and provides database access.
"""
import os
from supabase import create_client, Client

# Singleton pattern for Supabase client
_supabase_client = None

def get_client() -> Client:
    """Get or create a Supabase client instance"""
    global _supabase_client
    
    if _supabase_client is None:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
            )
        
        _supabase_client = create_client(supabase_url, supabase_key)
    
    return _supabase_client

def is_available() -> bool:
    """Check if Supabase is configured and available"""
    try:
        # Try to get environment variables
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            return False
            
        # Try to connect and make a simple query
        client = get_client()
        client.table("transcripts").select("count", count="exact").execute()
        return True
    except Exception:
        return False