from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import uvicorn
from src.models.academic_summarizer import create_summarizer
import json
import os
from datetime import datetime

class SummarizationRequest(BaseModel):
    text: str
    citations: Optional[List[Dict[str, str]]] = None
    max_length: Optional[int] = 1024
    min_length: Optional[int] = 128
    summary_type: Optional[str] = "default"  # default, technical, or brief

class SummarizationResponse(BaseModel):
    summary: str
    metadata: Dict[str, any]

app = FastAPI(
    title="Academic Paper Summarizer API",
    description="API for generating summaries of academic papers",
    version="1.0.0"
)

# Global variables
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
request_log = []

@app.on_event("startup")
async def load_model():
    """Load the model on server startup."""
    global model
    try:
        model = create_summarizer()
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model initialization failed")

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    """
    Generate a summary for the provided text.
    
    Args:
        request: SummarizationRequest containing text and optional parameters
    
    Returns:
        SummarizationResponse containing the generated summary and metadata
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Record request
        request_time = datetime.now()
        
        # Generate summary
        with torch.no_grad():
            summary = model.summarize(
                text=request.text,
                citations=request.citations
            )
        
        # Calculate metadata
        metadata = {
            "timestamp": request_time.isoformat(),
            "input_length": len(request.text.split()),
            "summary_length": len(summary.split()),
            "processing_time": (datetime.now() - request_time).total_seconds(),
            "model_version": "1.0.0",
            "summary_type": request.summary_type
        }
        
        # Log request
        log_request(request, metadata, success=True)
        
        return SummarizationResponse(
            summary=summary,
            metadata=metadata
        )
    
    except Exception as e:
        # Log failed request
        log_request(request, {"error": str(e)}, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check the health status of the server."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get usage statistics."""
    total_requests = len(request_log)
    successful_requests = sum(1 for req in request_log if req["success"])
    
    if total_requests > 0:
        success_rate = successful_requests / total_requests
        avg_processing_time = sum(
            req["metadata"].get("processing_time", 0)
            for req in request_log if req["success"]
        ) / successful_requests
    else:
        success_rate = 0
        avg_processing_time = 0
    
    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": success_rate,
        "average_processing_time": avg_processing_time
    }

def log_request(request: SummarizationRequest, metadata: Dict, success: bool):
    """Log API request details."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request": {
            "text_length": len(request.text),
            "has_citations": request.citations is not None,
            "summary_type": request.summary_type
        },
        "metadata": metadata,
        "success": success
    }
    
    request_log.append(log_entry)
    
    # Keep only last 1000 requests
    if len(request_log) > 1000:
        request_log.pop(0)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    ) 