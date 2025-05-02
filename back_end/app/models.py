# backend/app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Model for Page-Specific Context (Keep if used by /chat) ---
class PageContext(BaseModel):
    """Model for page-specific context sent from the frontend."""
    selected_factories: Optional[List[str]] = []
    date_range: Optional[List[str]] = []  # Expecting list of two ISO date strings [start, end] or similar

# --- NEW: Market Data Models (Defined BEFORE they are used) ---
class MarketDataSymbol(BaseModel):
    """Model for market data symbols and their time series."""
    symbol: str
    values: List[float]
    dates: List[str] # Store dates as strings

class MarketDataResponse(BaseModel):
    """Response model for market data endpoint."""
    # Return dict where key is symbol, value contains symbol data
    market_data: Dict[str, MarketDataSymbol]


# --- Request Models ---

class ChatMessage(BaseModel):
    """Model for incoming chat messages, potentially including context."""
    text: str
    active_tab: Optional[str] = None # Context sent from frontend (tab name)
    news_summary: Optional[str] = None # Context sent from frontend
    page_context: Optional[PageContext] = None # NEW: Page-specific context

class NewsQuery(BaseModel):
    """Model for news fetching requests."""
    query: str = Field(
        default="supply chain disruption OR logistics OR semiconductor shortage",
        examples=["port congestion delays", "semiconductor manufacturing news"]
    )

# --- UPDATED: SimulationRequest (Add market_data) ---
# Now uses MarketDataSymbol which is defined above
class SimulationRequest(BaseModel):
    """Enhanced model for simulation trigger requests."""
    risk_events: Optional[List[str]] = Field(default=None, description="List of risk event types selected by user")
    risk_severity: Optional[Dict[str, float]] = Field(default=None, description="Severity scores (0.0-1.0) for each selected risk event")
    cost_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Optional user-provided cost parameters") # Keep as dict for flexibility
    # --- Add market_data, using the model defined above ---
    market_data: Optional[Dict[str, MarketDataSymbol]] = Field(default=None, description="Market data time series by symbol")


class RecommendationRequest(BaseModel):
    simulation_data: List[Dict[str, Any]] = Field(..., min_length=0)
    risk_summary: Optional[Dict[str, Any]] = None
    # --- Use the MarketDataResponse model or similar structure ---
    # The backend service returns Dict[str, MarketDataSymbol] for market_data
    market_data: Optional[Dict[str, MarketDataSymbol]] = None # <--- Use the model defined above
    financial_summary: Optional[Dict[str, Any]] = None


# --- Response Models ---

class ChatResponse(BaseModel):
    """Standard response for chat endpoint."""
    response: str

# --- SimulationResultResponse (Uses SimulationFinancialSummary which is defined below) ---
class SimulationResultResponse(BaseModel):
    status: str # e.g., "completed", "error"
    message: Optional[str] = None
    timeseries_data: Optional[List[Dict[str, Any]]] = None # For plotting
    full_simulation_data: Optional[List[Dict[str, Any]]] = None # For recommendations
    risk_summary: Optional[Dict[str, Any]] = None # Info about applied risks
    financial_summary: Optional["SimulationFinancialSummary"] = None # Forward reference if defined later


# --- SimulationFinancialSummary (Defined AFTER it's used in SimulationResultResponse via forward ref) ---
# This is fine because SimulationResultResponse uses a string forward reference "SimulationFinancialSummary"
class SimulationFinancialSummary(BaseModel):
    """Calculated financial KPIs from simulation results."""
    total_holding_cost: float = Field(default=0.0)
    total_stockout_cost: float = Field(default=0.0)
    estimated_total_logistics_cost: float = Field(default=0.0)
    estimated_total_production_cost: float = Field(default=0.0)
    estimated_total_supply_chain_cost: float = Field(default=0.0)
    calculation_error: Optional[str] = None

# Update forward reference in SimulationResultResponse after SimulationFinancialSummary is defined
SimulationResultResponse.update_forward_refs()


class Recommendation(BaseModel):
    """Structure for a single recommendation item."""
    priority: int
    category: str
    text: str

class RecommendationResponse(BaseModel):
    """Response model for the recommendation endpoint."""
    recommendations: List[Recommendation]


# --- News Response (Enhanced) ---
class ProcessedNewsResponse(BaseModel):
    summary: str
    risk_events: List[str]
    risk_severity: Dict[str, float]