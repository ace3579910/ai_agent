import uvicorn
import os

if __name__ == "__main__":
    # Ensure we are in the hybrid_search directory
    # (Though this script is expected to be run from there)
    
    print("Starting AI Financial Research Assistant...")
    print("Dashboard will be available at: http://127.0.0.1:8000/frontend/index.html")
    
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
