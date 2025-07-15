"""
FastAPI Application for Domain Name Suggestion LLM

This module provides a REST API endpoint for domain generation requests with:
- Safety filtering and validation
- Comprehensive error handling
- Request logging and monitoring
- Rate limiting and authentication (optional)
"""

import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from ..models.domain_generator import DomainGenerator, GenerationConfig
from ..evaluation.safety_checker import SafetyChecker
from ..utils.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Domain Name Suggestion LLM API",
    description="AI-powered domain name generation with safety filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and safety checker
domain_generator = None
safety_checker = None
config = None


class DomainRequest(BaseModel):
    """Request model for domain generation."""
    business_description: str = Field(..., min_length=10, max_length=1000, 
                                    description="Description of the business")
    num_suggestions: int = Field(default=3, ge=1, le=10, 
                               description="Number of domain suggestions to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0,
                                       description="Generation temperature")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0,
                                 description="Top-p sampling parameter")
    
    @validator('business_description')
    def validate_business_description(cls, v):
        """Validate business description."""
        if not v.strip():
            raise ValueError("Business description cannot be empty")
        return v.strip()


class DomainSuggestion(BaseModel):
    """Response model for domain suggestions."""
    domain: str
    confidence: float
    reasoning: str
    tld: str
    metadata: Dict[str, Any]


class DomainResponse(BaseModel):
    """Response model for domain generation."""
    suggestions: List[DomainSuggestion]
    status: str
    message: Optional[str] = None
    metadata: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model."""
    status: str = "error"
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    safety_checker_loaded: bool
    version: str
    timestamp: float


@app.on_event("startup")
async def startup_event():
    """Initialize models and safety checker on startup."""
    global domain_generator, safety_checker, config
    
    try:
        logger.info("Initializing Domain Name Suggestion LLM API...")
        
        # Load configuration
        config = load_config("config/model_config.yaml")
        
        # Initialize safety checker
        safety_checker = SafetyChecker("config/model_config.yaml")
        logger.info("Safety checker initialized")
        
        # Initialize domain generator
        model_version = config.get('default_model_version', 'v2_qlora')
        domain_generator = DomainGenerator("config/model_config.yaml", model_version)
        logger.info(f"Domain generator initialized with model version: {model_version}")
        
        logger.info("API initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    # Add processing time to response headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Domain Name Suggestion LLM API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=domain_generator is not None,
        safety_checker_loaded=safety_checker is not None,
        version="1.0.0",
        timestamp=time.time()
    )


@app.post("/suggest", response_model=DomainResponse)
async def suggest_domains(request: DomainRequest):
    """
    Generate domain name suggestions for a business description.
    
    This endpoint:
    1. Validates the business description for safety
    2. Generates domain suggestions using the AI model
    3. Filters suggestions for appropriateness
    4. Returns ranked suggestions with confidence scores
    """
    
    # Check if models are loaded
    if domain_generator is None or safety_checker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please try again later."
        )
    
    try:
        # Step 1: Safety check
        safety_result = safety_checker.check_safety(request.business_description)
        
        if not safety_result.is_safe:
            return DomainResponse(
                suggestions=[],
                status="blocked",
                message=f"Request contains inappropriate content: {', '.join(safety_result.blocked_reasons)}",
                metadata={
                    "safety_check": {
                        "is_safe": False,
                        "risk_level": safety_result.risk_level,
                        "blocked_reasons": safety_result.blocked_reasons
                    }
                }
            )
        
        # Step 2: Generate domain suggestions
        generation_config = GenerationConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.num_suggestions
        )
        
        suggestions = domain_generator.generate_domains(
            request.business_description,
            num_suggestions=request.num_suggestions,
            generation_config=generation_config
        )
        
        # Step 3: Filter suggestions for safety
        filtered_suggestions = safety_checker.filter_domain_suggestions(
            [{"domain": s.domain, "confidence": s.confidence, "reasoning": s.reasoning, "tld": s.tld} 
             for s in suggestions],
            request.business_description
        )
        
        # Convert to response format
        domain_suggestions = []
        for suggestion in filtered_suggestions:
            domain_suggestions.append(DomainSuggestion(
                domain=suggestion["domain"],
                confidence=suggestion["confidence"],
                reasoning=suggestion["reasoning"],
                tld=suggestion["tld"],
                metadata={"generation_method": domain_generator.model_version}
            ))
        
        # Step 4: Return response
        return DomainResponse(
            suggestions=domain_suggestions,
            status="success",
            metadata={
                "safety_check": {
                    "is_safe": True,
                    "risk_level": safety_result.risk_level,
                    "confidence": safety_result.confidence
                },
                "generation": {
                    "model_version": domain_generator.model_version,
                    "num_requested": request.num_suggestions,
                    "num_generated": len(domain_suggestions),
                    "temperature": request.temperature,
                    "top_p": request.top_p
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating domain suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate domain suggestions: {str(e)}"
        )


@app.post("/suggest/batch", response_model=List[DomainResponse])
async def suggest_domains_batch(requests: List[DomainRequest]):
    """
    Generate domain suggestions for multiple business descriptions.
    
    This endpoint processes multiple requests in batch for efficiency.
    """
    
    if len(requests) > 10:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 10 requests"
        )
    
    results = []
    for request in requests:
        try:
            result = await suggest_domains(request)
            results.append(result)
        except Exception as e:
            # Add error result for failed requests
            results.append(DomainResponse(
                suggestions=[],
                status="error",
                message=str(e),
                metadata={"error": True}
            ))
    
    return results


@app.get("/safety/check")
async def check_safety(business_description: str):
    """
    Check if a business description is safe for domain generation.
    
    This endpoint provides detailed safety analysis without generating domains.
    """
    
    if safety_checker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Safety checker not loaded"
        )
    
    try:
        report = safety_checker.get_safety_report(business_description)
        return report
        
    except Exception as e:
        logger.error(f"Error in safety check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Safety check failed: {str(e)}"
        )


@app.get("/models/info")
async def get_model_info():
    """Get information about the loaded models."""
    
    if domain_generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_version": domain_generator.model_version,
        "base_model": config['base_model']['name'],
        "training_method": config['versions'][domain_generator.model_version]['training_method'],
        "generation_config": {
            "max_length": config['generation']['max_length'],
            "temperature": config['generation']['temperature'],
            "top_p": config['generation']['top_p'],
            "top_k": config['generation']['top_k']
        },
        "safety_config": {
            "blocked_keywords_count": len(config['safety']['blocked_keywords']),
            "blocked_domains_count": len(config['safety']['blocked_domains']),
            "min_confidence_threshold": config['safety']['min_confidence_threshold']
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error",
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            details={"path": str(request.url)}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            status="error",
            message="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"path": str(request.url)}
        ).dict()
    )


def main():
    """Run the FastAPI application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain Name Suggestion LLM API")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Domain Name Suggestion LLM API on {args.host}:{args.port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main() 