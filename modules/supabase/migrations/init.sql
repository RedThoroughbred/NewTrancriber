-- Supabase SQL migrations
-- Run this in your Supabase project SQL editor

-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Transcripts table for basic metadata and content
CREATE TABLE IF NOT EXISTS transcripts (
  id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  topic TEXT,
  source_type TEXT NOT NULL, -- youtube, upload, etc.
  source_url TEXT,
  original_filename TEXT,
  content JSONB NOT NULL, -- Full transcript data
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Vector table for semantic search
CREATE TABLE IF NOT EXISTS transcript_vectors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content_vector VECTOR(1536), -- OpenAI embedding dimension
  chunk_text TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  transcript_id UUID REFERENCES transcripts(id) ON DELETE CASCADE
);

-- Create index for transcript vectors
CREATE INDEX IF NOT EXISTS transcript_vectors_transcript_id_idx ON transcript_vectors(transcript_id);

-- Create vector index
CREATE INDEX IF NOT EXISTS transcript_vectors_vector_idx ON transcript_vectors USING ivfflat (content_vector vector_l2_ops);

-- Simple text search function
CREATE OR REPLACE FUNCTION search_transcripts_text(query_text TEXT, match_count INT)
RETURNS TABLE (
  transcript_id UUID,
  title TEXT,
  snippet TEXT,
  created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    t.id AS transcript_id,
    t.title,
    ts_headline(
      (t.content->>'transcript'), 
      to_tsquery(query_text),
      'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=20'
    ) AS snippet,
    t.created_at
  FROM
    transcripts t
  WHERE
    to_tsvector(t.content->>'transcript') @@ to_tsquery(query_text) OR
    t.title ILIKE '%' || query_text || '%' OR
    t.topic ILIKE '%' || query_text || '%'
  ORDER BY
    t.created_at DESC
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql;