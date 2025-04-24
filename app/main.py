# backend/app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import logging
import json
from typing import List, Optional, Dict, Any

load_dotenv()

from . import models
from . import services

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

app = FastAPI(title="Supply Chain AI Agent Backend", version="0.1.2") # Version bump

origins = ["http://localhost", "http://localhost:8501"] # Add others if needed
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- API Endpoints ---

@app.post("/chat", response_model=models.ChatResponse)
async def chat_endpoint(message: models.ChatMessage): # Use the updated model
    """Receives user chat message (potentially with context) and returns AI response."""
    # Log using the correct fields from the updated ChatMessage model
    logging.info(f"Received chat message: Tab='{message.active_tab}', NewsContextIncluded={message.news_summary is not None}, Query='{message.text[:50]}...'")
    try:
        # Pass context directly to service function
        response_text = await services.get_gemini_chat_response(
            prompt=message.text,
            news_summary=message.news_summary, # Pass summary from request
            active_tab=message.active_tab
        )
        logging.info(f"Sending chat response: {response_text[:50]}...")
        return models.ChatResponse(response=response_text)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Unhandled error in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing chat request.")

@app.post("/fetch-news")
async def fetch_news_endpoint(request: models.NewsQuery):
    """Fetches news and returns an AI summary."""
    logging.info(f"Received news fetch request for query: {request.query}")
    try:
        summary = await services.fetch_and_summarize_news(request.query)
        logging.info(f"Returning news summary: {summary[:100]}...")
        return {"summary": summary} # Return simple dictionary
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Unhandled error in /fetch-news: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching/summarizing news.")

@app.post("/run_simulation", response_model=models.SimulationResultResponse)
async def run_simulation_endpoint(request: models.SimulationRequest):
    """Triggers a baseline simulation run and returns status and timeseries data."""
    logging.info(f"Received request to run simulation: {request}")
    try:
        result_dict = await services.trigger_simulation_run()
        logging.info(f"Simulation run result status: {result_dict.get('status')}")
        return result_dict # FastAPI validates against response_model
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Unhandled error in /run_simulation endpoint: {e}", exc_info=True)
        # Ensure error response matches the model for consistency
        # Return the SimulationResultResponse model instance on error
        return models.SimulationResultResponse(
            status="error",
            message=f"Internal Server Error: {str(e)}",
            timeseries_data=None # Explicitly None
        )

# --- Recommendation Endpoint ---
@app.post("/get_simulation_recommendations", response_model=models.RecommendationResponse)
async def get_recommendations_endpoint(request: models.RecommendationRequest):
    """Receives simulation timeseries data and returns structured AI recommendations."""
    logging.info(f"Received request for simulation recommendations with {len(request.simulation_data)} data points.")
    if not request.simulation_data:
         # Return empty list instead of 400 maybe? Or keep 400. Let's keep 400.
         raise HTTPException(status_code=400, detail="Simulation data cannot be empty.")
    try:
        # Service function returns list of dicts matching Recommendation model structure
        recommendations_list = await services.generate_simulation_recommendations(request.simulation_data)
        logging.info(f"Returning {len(recommendations_list)} recommendations.")
        # FastAPI validates the structure against the response model
        return models.RecommendationResponse(recommendations=recommendations_list)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Unhandled error generating recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error generating recommendations: {str(e)}")


# --- REMOVED Duplicate Recommendation Endpoint ---


@app.get("/health", status_code=200)
async def health_check():
    logging.debug("Health check endpoint called.")
    return {"status": "ok"}

# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting Uvicorn server for backend...")
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)