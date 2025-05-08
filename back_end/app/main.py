# backend/app/main.py
from fastapi import FastAPI, HTTPException, Query # <-- Import Query
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any # <-- Import typing helpers
logging.basicConfig(level=logging.DEBUG)

load_dotenv()
from . import models
from . import services

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
app = FastAPI(title="Supply Chain AI Agent Backend", version="0.2.0") # Version bump
origins = ["http://localhost", "http://localhost:8501"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API Endpoints ---

@app.get("/test", status_code=200)
async def test_endpoint():
    return {"status": "test ok"}

@app.post("/chat", response_model=models.ChatResponse)
async def chat_endpoint(message: models.ChatMessage):
    logging.info(f"Chat Request: Tab='{message.active_tab}', News={message.news_summary is not None}, PageCtx={message.page_context is not None}, Query='{message.text[:50]}...'")
    try:
        # Pass page_context as dict
        page_ctx_dict = message.page_context.dict() if message.page_context else None
        response_text = await services.get_gemini_chat_response(
            prompt=message.text, news_summary=message.news_summary,
            active_tab=message.active_tab, page_context=page_ctx_dict # Pass dict
        )
        return models.ChatResponse(response=response_text)
    except Exception as e: logging.error(f"/chat error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Chat Error: {e}")

# --- UPDATED fetch-news endpoint to return richer structure ---
@app.post("/fetch-news", response_model=models.ProcessedNewsResponse) # Use new response model
async def fetch_news_endpoint(request: models.NewsQuery):
    logging.info(f"News Request: Query='{request.query}'")
    try:
        # Service function now returns dict {"summary", "risk_events", "risk_severity"}
        processed_news = await services.fetch_and_summarize_news(request.query)
        logging.info(f"Returning news summary & {len(processed_news.get('risk_events',[]))} potential risks.")
        # FastAPI validates the returned dict against the response_model
        return processed_news
    except Exception as e: logging.error(f"/fetch-news error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"News Fetch Error: {e}")

# --- NEW Market Data Endpoint ---
@app.get("/fetch-market-data", response_model=models.MarketDataResponse)
async def fetch_market_data_endpoint(symbols: str = Query(..., description="Comma-separated market symbols (e.g., 'OIL,FX')")):
    """Fetches SYNTHETIC market data for requested symbols."""
    logging.info(f"Market Data Request: Symbols='{symbols}'")
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        if not symbols_list: raise HTTPException(status_code=400, detail="No valid symbols provided.")
        
        # Call the service function
        raw_market_data = await services.fetch_market_data(symbols_list)
        
        # Transform raw dictionaries into MarketDataSymbol objects
        market_data_dict = {}
        for symbol, data in raw_market_data.items():
            market_data_dict[symbol] = models.MarketDataSymbol(
                symbol=data["symbol"],
                values=data["values"],
                dates=data["dates"]
            )
        
        # Return the properly structured response
        return models.MarketDataResponse(market_data=market_data_dict)
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"/fetch-market-data error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Market Data Error: {e}")
logging.info("Market data endpoint registered")

# Use the updated SimulationRequest model that includes cost_overrides
@app.post("/run_simulation", response_model=models.SimulationResultResponse)
async def run_simulation_endpoint(request: models.SimulationRequest): # Accepts SimulationRequest
    logging.info(f"Simulation Request: Risks={request.risk_events}, Costs={'Yes' if request.cost_overrides else 'No'}")
    try:
        # Pass risk parameters AND cost overrides from request to service function
        result_dict = await services.trigger_simulation_run(
            risk_events=request.risk_events,
            risk_severity=request.risk_severity,
            cost_overrides=request.cost_overrides.dict() if request.cost_overrides else None # Pass as dict
            # market_series=request.market_series # Not accepting market_series in request yet
        )
        # FastAPI validates the returned dict against SimulationResultResponse model
        return result_dict
    except Exception as e:
        logging.error(f"/run_simulation error: {e}", exc_info=True)
        # Ensure the error response structure matches the model
        return models.SimulationResultResponse(
            status="error",
            message=f"Sim Error: {str(e)}",
            timeseries_data=None,
            full_simulation_data=None,
            risk_summary=None,
            financial_summary={"calculation_error": f"Sim failed: {str(e)}"}
        )


@app.post("/get_simulation_recommendations", response_model=models.RecommendationResponse)
async def get_recommendations_endpoint(request: models.RecommendationRequest): # Accepts RecommendationRequest
    logging.info(f"Recommendation Request: Data points={len(request.simulation_data)}, RiskSummaryIncluded={request.risk_summary is not None}")
    if not request.simulation_data: raise HTTPException(status_code=400, detail="Simulation data required.")
    try:
        # Pass simulation data and context to service
        recommendations_list = await services.generate_simulation_recommendations(
            simulation_data=request.simulation_data,
            risk_summary=request.risk_summary, # Pass context
            market_data=request.market_data, # Pass context
            financial_summary=request.financial_summary
        )
        return models.RecommendationResponse(recommendations=recommendations_list)
    except Exception as e: logging.error(f"/get_recs error: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Rec Gen Error: {e}")

@app.get("/health", status_code=200)
async def health_check(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting Uvicorn server...")
    uvicorn.run("back_end.app.main:app", host="0.0.0.0", port=8000, reload=True)