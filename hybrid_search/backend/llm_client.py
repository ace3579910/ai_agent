import os
import requests
import json
from typing import Dict, Any, Optional

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:3b-cloud")

def query_llm(messages: list, format: str = None) -> Optional[str]:
    """
    Sends a chat request to the Ollama API.
    Falls back to a simple response if Ollama is unavailable.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    if format:
        payload["format"] = format

    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        print(f"Warning: Could not connect to Ollama at {OLLAMA_URL}")
        print("Make sure Ollama is running: ollama serve")
        return None
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def summarize_text(text: str) -> str:
    """
    Generates a concise summary of the provided text.
    """
    prompt = f"""
    You are an expert financial research assistant. 
    Summarize the following text efficiently, preserving key numerical data, dates, and factual claims.
    Do not add any external information.
    
    Text:
    {text}
    
    Summary:
    """
    messages = [{"role": "user", "content": prompt}]
    result = query_llm(messages)
    return result if result else "Failed to generate summary. Make sure Ollama is running."

def analyze_query(query: str, context: list) -> Dict[str, Any]:
    """
    Analyzes a user query based on retrieved document chunks.
    Returns a structured JSON response.
    """
    if not context:
        return {
            "answer": "No documents found to analyze.",
            "key_points": [],
            "confidence": "Low"
        }
    
    context_str = "\n\n".join([f"[{c.get('source', 'Unknown')}]: {c.get('content', '')[:500]}" for c in context])
    
    prompt = f"""
    You are a precise financial analyst. Answer the query using ONLY the provided context.
    
    Context:
    {context_str}
    
    Query: {query}
    
    Provide your answer based on the context. If the answer is not in the context, state clearly.
    """
    
    messages = [{"role": "user", "content": prompt}]
    response = query_llm(messages)
    
    if not response:
        return {
            "answer": "Error: Could not connect to Ollama. Make sure it's running with: ollama serve",
            "key_points": [],
            "confidence": "Low"
        }
    
    # Try to extract key points from the response
    lines = response.split('\n')
    key_points = [line.strip() for line in lines if line.strip().startswith('-') or line.strip().startswith('•')][:3]
    
    return {
        "answer": response,
        "key_points": key_points or ["See analysis above"],
        "confidence": "Medium"
    }
