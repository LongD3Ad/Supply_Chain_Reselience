# backend/app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Request Models ---

class ChatMessage(BaseModel):
    """Model for incoming chat messages, potentially including context."""
    text: str
    active_tab: Optional[str] = None # Context sent from frontend
    news_summary: Optional[str] = None # Context sent from frontend

class NewsQuery(BaseModel):
    """Model for news fetching requests."""
    query: str = Field(
        default="supply chain disruption OR logistics OR semiconductor shortage",
        examples=["port congestion delays", "semiconductor manufacturing news"]
    )

class SimulationRequest(BaseModel):
    """Model for simulation trigger requests (basic for MVP baseline)."""
    pass # No parameters needed for baseline trigger

class RecommendationRequest(BaseModel):
    """Model for recommendation requests, expecting simulation results."""
    # Expecting the *full* data structure now
    simulation_data: List[Dict[str, Any]] = Field(..., min_length=0) # Allow empty list

# --- Response Models ---

class ChatResponse(BaseModel):
    """Standard response for chat endpoint."""
    response: str

# News endpoint returns {"summary": str}, no specific model needed

class SimulationResultResponse(BaseModel):
    """Response model for the simulation endpoint."""
    status: str # e.g., "completed", "error"
    message: Optional[str] = None
    # Includes subset data needed for plotting in frontend dialog
    timeseries_data: Optional[List[Dict[str, Any]]] = None # Format: [{'date': 'YYYY-MM-DD', 'inventory_level_units': N, 'shortage_units': N}]
    # --- NEW: Includes full data needed for recommendations ---
    full_simulation_data: Optional[List[Dict[str, Any]]] = None # All columns, dates as strings, NaNs handled

class Recommendation(BaseModel):
    """Structure for a single recommendation item."""
    priority: int
    category: str
    text: str

class RecommendationResponse(BaseModel):
    """Response model for the recommendation endpoint."""
    recommendations: List[Recommendation] # List of structured recommendations