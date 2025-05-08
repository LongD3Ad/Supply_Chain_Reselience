# backend/app/services.py
import google.generativeai as genai
import requests
import os
import sys
import logging
from fastapi import HTTPException
import pandas as pd
import numpy as np # <--- ADD for market data generation
import json
from typing import List, Optional, Dict, Any
import asyncio
from . import financials

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

try:
    from . import financials
    FINANCIALS_AVAILABLE = True
    logging.info("Successfully imported backend financials module.")
except ImportError as e:
    logging.error(f"Could not import backend financials module: {e}. Financial calculations disabled.")
    FINANCIALS_AVAILABLE = False
    # Define dummy functions/defaults if import fails
    class DummyFinancials:
        DEFAULT_COST_PARAMS = {}
        def calculate_financial_kpis(self, *args, **kwargs):
            logging.warning("Financial calculations attempted but module not loaded.")
            return {"calculation_error": "Financials module not available."}
    financials = DummyFinancials()


# --- Attempt Imports from src (Allow failure) ---
try:
    # Ensure these are the correct imports and path is set correctly above
    from simulation.engine import run_factory_simulation
    from config import get_config # Use config getter from src
    SIMULATION_AVAILABLE = True
    logging.info("Successfully imported simulation engine and config from src.")
except ImportError as e:
    logging.warning(f"Could not import simulation engine or config from src: {e}. Simulation endpoint disabled.")
    SIMULATION_AVAILABLE = False
    # Define dummy functions if import fails
    def run_factory_simulation(*args, **kwargs):
        logging.warning("Simulation engine not loaded, cannot run simulation.")
        # Return empty DataFrame and log matching expected types
        return pd.DataFrame(columns=['date', 'factory_name']), pd.DataFrame(columns=['risk_type', 'start_day'])
    def get_config():
        # Provide minimal structure including financial_defaults for robustness
        return {"factories": [], "financial_defaults": financials.DEFAULT_COST_PARAMS}


try:
    # Ensure these are the correct imports from your src/recommendations
    # get_gemini_recommendations must accept _gemini_model_instance
    from recommendations.gemini import summarize_data_for_gemini, get_gemini_recommendations
    RECOMMENDATIONS_AVAILABLE = True
    logging.info("Successfully imported recommendation components from src.")
except ImportError as e:
    logging.warning(f"Could not import recommendation components from src: {e}. Recommendations endpoint disabled.")
    RECOMMENDATIONS_AVAILABLE = False
    # Define dummy functions if import fails
    def summarize_data_for_gemini(*args, **kwargs):
        logging.warning("Recommendation summarizer not loaded.")
        return "Summary engine not available."
    # Ensure get_gemini_recommendations returns structure compatible with Recommendation model
    def get_gemini_recommendations(*args, **kwargs):
        logging.warning("Recommendation generator not loaded.")
        return [{"priority": 0, "category": "Error", "text": "Recommendation engine not loaded."}]

# --- Import News Processor from src ---
# Requires src/recommendations/news_processor.py and spacy model
try:
    from recommendations.news_processor import analyze_news_summary
    NEWS_PROCESSOR_AVAILABLE = True
    logging.info("Successfully imported news processor components.")
except ImportError as e:
    logging.warning(f"Could not import news processor: {e}. NER features disabled.")
    NEWS_PROCESSOR_AVAILABLE = False
    def analyze_news_summary(*args, **kwargs):
        logging.warning("News processor not loaded.")
        return [], {} # Return empty structure

# --- Initialize Gemini Model ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY") # Used by fetch_and_summarize_news

if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY environment variable is required")
if not NEWS_API_KEY: logging.warning("NEWS_API_KEY environment variable missing. News fetching disabled.")

gemini_model = None # Initialize as None
try:
    # Ensure this model name is correct and available
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Use appropriate model name
    logging.info(f"Gemini Model ('{gemini_model.model_name}') initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Gemini Model: {e}", exc_info=True)
    gemini_model = None # Explicitly set to None on failure


# --- Service Functions ---

# Modified to include page_context parameter as a Dict
async def get_gemini_chat_response(
    prompt: str,
    news_summary: Optional[str] = None,
    active_tab: Optional[str] = None,
    page_context: Optional[Dict[str, Any]] = None
) -> str:
    """Gets a response from the Gemini model, using provided context."""
    if not gemini_model:
        logging.error("Gemini model not available for chat.")
        raise HTTPException(status_code=503, detail="AI model not initialized")

    # --- Construct Prompt ---
    context_lines = []
    if active_tab: context_lines.append(f"User is viewing: '{active_tab}' tab.")
    # Add page context information if available
    if page_context:
        factories = page_context.get("selected_factories")
        date_range = page_context.get("date_range")
        if factories: context_lines.append(f"Selected factories: {', '.join(factories)}")
        if dates and len(dates)==2: context_lines.append(f"Selected date range: {dates[0]} to {dates[1]}")
        # Add any additional page-specific context that might be useful
        for key, value in page_context.items():
            # Avoid duplicating factory/date range if already included
            if key not in ["selected_factories", "date_range"] and value:
                context_lines.append(f"Context - {key}: {value}")

    if news_summary: context_lines.append(f"\nRecent News:\n---\n{news_summary}\n---")

    # Build the full prompt based on whether context exists
    if context_lines:
        context_str = '\n'.join(context_lines)
        full_prompt = f"Context:\n{context_str}\n\nUser query: {prompt}\n\nAssistant response:"
        logging.debug("Including context in chat prompt")
    else:
        full_prompt = f"User query: {prompt}\n\nAssistant response:"
        logging.debug("No extra context for this chat prompt")

    logging.debug(f"Chat Prompt Context (Tab='{active_tab}', News={news_summary is not None}, PageCtx={page_context is not None})")

    try:
        # Call Gemini API
        response = await gemini_model.generate_content_async(full_prompt)

        # Robust response handling
        if not response.parts and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             logging.warning(f"Gemini response blocked: {response.prompt_feedback.block_reason}")
             return f"My response was blocked by the safety filter: {response.prompt_feedback.block_reason}"

        response_text = ''.join(part.text for part in response.parts) if response.parts else getattr(response, 'text', "")
        if not response_text:
             logging.warning("Gemini returned an empty response text.")
             return "I couldn't generate a response for that."

        return response_text
    except Exception as e:
        logging.error(f"Gemini Chat Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI Error: {e}")


# fetch_and_summarize_news remains largely the same, returns dict
async def fetch_and_summarize_news(query: str) -> Dict[str, Any]:
    """Fetches news, summarizes, extracts risks, returns structured dict."""
    if not NEWS_API_KEY: return {"summary": "News disabled (key missing).", "risk_events": [], "risk_severity": {}}
    if not gemini_model: raise HTTPException(status_code=503, detail="AI model needed for summary")

    articles_text, summary_text = "", "Failed to generate summary."
    risk_events, risk_severity = [], {} # Initialize risk results

    try:
        # Fetch news articles (same as before)
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&pageSize=10&language=en&sortBy=relevancy"
        logging.info(f"Fetching news from: {url}")
        response = requests.get(url, timeout=15); response.raise_for_status()
        articles = response.json().get("articles", []); logging.info(f"Fetched {len(articles)} articles.")
        if not articles: return {"summary": "No news found.", "risk_events": [], "risk_severity": {}}
        articles_text = "\n\n".join([f"Title: {a.get('title','N/A')}\nDesc: {a.get('description','N/A')}" for a in articles[:5]])
    except requests.exceptions.RequestException as e: raise HTTPException(status_code=502, detail=f"News API Error: {e}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"News processing error: {e}")

    try:
        # Generate summary (same as before)
        summary_prompt = f"Summarize key supply chain insights/disruptions from news:\n{articles_text}\n\nConcise Summary:"
        summary_response = await gemini_model.generate_content_async(summary_prompt)
        # (Same summary text extraction as before)
        if not summary_response.parts and hasattr(summary_response, 'prompt_feedback') and summary_response.prompt_feedback.block_reason:
             summary_text = f"AI summary blocked: {summary_response.prompt_feedback.block_reason}"
        else:
            summary_text = ''.join(part.text for part in summary_response.parts) if summary_response.parts else getattr(summary_response,'text',"")
            if not summary_text: summary_text = "AI returned empty summary."
    except Exception as e: logging.error(f"Gemini summary error: {e}", exc_info=True); summary_text = f"Could not summarize. Headlines:\n{articles_text}"

    # --- Analyze summary text for risks ---
    if NEWS_PROCESSOR_AVAILABLE and summary_text and not summary_text.startswith("AI summary blocked"):
        try:
            # Run potentially CPU-bound analysis in threadpool
            risk_events, risk_severity = await asyncio.to_thread(analyze_news_summary, summary_text)
            logging.info(f"Extracted risks: {risk_events}, Severity: {risk_severity}")
        except Exception as nlp_e:
             logging.error(f"News NLP analysis failed: {nlp_e}", exc_info=True)
             # Continue without NLP results

    # Return enhanced structure with risk information
    return {"summary": summary_text, "risk_events": risk_events, "risk_severity": risk_severity}


# fetch_market_data generates synthetic data, returns structured dict
async def fetch_market_data(symbols: List[str]) -> Dict[str, Any]:
    """Fetches SYNTHETIC market data for requested symbols."""
    if not symbols: return {}
    logging.info(f"Generating synthetic market data for: {', '.join(symbols)}")
    market_data_response = {}
    end_date = pd.Timestamp.now(tz=None).normalize()
    start_date = end_date - pd.Timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D') # Daily data
    dates_str = [d.strftime('%Y-%m-%d') for d in date_range]

    for symbol in symbols:
        try:
            symbol_upper = symbol.upper()
            seed = hash(symbol_upper + end_date.strftime('%Y%m%d')) % (2**32) # Seed varies slightly daily
            rng = np.random.default_rng(seed=seed)
            if symbol_upper == 'OIL': base, vol = 80.0, 1.5
            elif symbol_upper == 'FX': base, vol = 100.0, 0.3
            elif symbol_upper == 'SHIPPING': base, vol = 1500.0, 15.0
            else: base, vol = 50.0, 1.0

            returns = rng.normal(0, vol / 100, size=len(date_range))
            values = [base]
            for r in returns: values.append(max(0, values[-1] * (1 + r)))
            values = np.round(values[1:], 2).tolist()

            # Return structure matching MarketDataSymbol model
            market_data_response[symbol] = {"symbol": symbol, "values": values, "dates": dates_str}
            logging.info(f"Generated {len(values)} points for {symbol}")
        except Exception as e:
            logging.error(f"Error generating market data for {symbol}: {e}", exc_info=True)
            market_data_response[symbol] = {"symbol": symbol, "values": [], "dates": []}
    return market_data_response


# --- UPDATED trigger_simulation_run (Accepts cost_overrides, market_data) ---
# Accepts SimulationRequest fields as individual parameters
async def trigger_simulation_run(
    risk_events: Optional[List[str]] = None,
    risk_severity: Optional[Dict[str, float]] = None,
    cost_overrides: Optional[Dict[str, Any]] = None,
    market_data: Optional[Dict[str, Any]] = None # <--- Use Dict[str, Any] for easier handling
) -> Dict[str, Any]: # Returns dict matching SimulationResultResponse
    """Enhanced simulation trigger incorporating risks, costs, market data, and calculating financials."""
    if not SIMULATION_AVAILABLE: raise HTTPException(status_code=501, detail="Sim engine not available.")

    try:
        logging.info(f"Triggering simulation run with Risks={risk_events}, Costs={'Yes' if cost_overrides else 'No'}, MarketData={'Yes' if market_data else 'No'}")
        sim_config = get_config()
        if not sim_config or not sim_config.get("factories"): raise ValueError("Invalid sim config.")

        # --- Get Effective Cost Parameters ---
        effective_costs = sim_config.get("financial_defaults", financials.DEFAULT_COST_PARAMS).copy()
        if cost_overrides:
            for key, value in cost_overrides.items():
                 if key in effective_costs and value is not None:
                      effective_costs[key] = value
        logging.info(f"Effective simulation costs: {effective_costs}")

        # --- Apply Risk Event Modifications ---
        first_factory_config = sim_config["factories"][0]; factory_name = first_factory_config.get("name", "Unknown")
        default_risks_params = sim_config.get("risk_factors", {})
        default_duration = sim_config.get("simulation_defaults", {}).get("sim_duration_weeks", 52)
        start_date = pd.Timestamp.now(tz=None).normalize()
        effective_risk_factors = default_risks_params.copy()
        applied_risk_summary = {"applied_risk_events": risk_events or [], "risk_severity": risk_severity or {}}

        if risk_events and risk_severity:
            logging.info(f"Processing {len(risk_events)} custom risk events...")
            for risk_name in risk_events:
                severity = risk_severity.get(risk_name, 0.5)
                logging.info(f" - Applying risk '{risk_name}' with severity {severity:.2f}")
                # Ensure target risk key exists in effective_risk_factors before modifying
                if ("congestion" in risk_name.lower() or "bottleneck" in risk_name.lower()):
                    target_risk_key = "logistics_bottleneck"
                    if target_risk_key in effective_risk_factors:
                        target_params = effective_risk_factors[target_risk_key]
                        lt_range = target_params.get("lead_time_multiplier_range", [1.0, 1.5])
                        target_params["lead_time_multiplier_range"] = [round(1.0 + (lt_range[0]-1.0) + severity*1.5, 2), round(1.0 + (lt_range[1]-1.0) + severity*1.5, 2)]
                        target_params["prob_per_day"] = min(0.1, target_params.get("prob_per_day", 0.025) * (1 + severity))
                elif "strike" in risk_name.lower():
                    target_risk_key = "pandemic_closure" # Example mapping
                    if target_risk_key in effective_risk_factors:
                        target_params = effective_risk_factors[target_risk_key]
                        impact_range = target_params.get("impact_factor_range", [0.5, 0.8])
                        target_params["impact_factor_range"] = [round(max(0.0, impact_range[0]*(1-severity*0.8)), 2), round(max(0.1, impact_range[1]*(1-severity*0.5)), 2)]
                        target_params["prob_per_day"] = min(0.1, target_params.get("prob_per_day", 0.01) * (1 + severity))
                # Add more mappings...

            applied_risk_summary["effective_risk_factors_applied"] = {k:v for k,v in effective_risk_factors.items() if k in default_risks_params and v != default_risks_params[k]}


        # --- Apply Market Data Modifications (FIXED Logic) ---
        # Check if market_data was provided and contains 'OIL' with values
        if market_data and "OIL" in market_data and market_data["OIL"].get("values") and len(market_data["OIL"]["values"]) >= 2: # <--- ACCESS 'values' CORRECTLY
             oil_values = market_data["OIL"]["values"]
             if oil_values[0] != 0:
                  oil_change_pct = (oil_values[-1] / oil_values[0]) - 1.0
                  if oil_change_pct > 0.1: # Only apply if oil increased > 10%
                       target_risk_key = "logistics_bottleneck"
                       if target_risk_key in effective_risk_factors:
                           target_params = effective_risk_factors[target_risk_key]
                           lt_range = target_params.get("lead_time_multiplier_range", [1.0, 1.5])
                           oil_factor = oil_change_pct * 0.2
                           target_params["lead_time_multiplier_range"] = [round(lt_range[0] + oil_factor, 2), round(lt_range[1] + oil_factor, 2)]
                           logging.info(f"Applied oil price effect ({oil_change_pct:.1%}) to logistics lead time.")
                           applied_risk_summary["market_factor_applied"] = f"Oil Change {oil_change_pct:.1%}"
             else: logging.warning("Oil values invalid or insufficient length in market_data.")
        elif market_data:
             logging.info("Market data provided but 'OIL' symbol missing or data invalid/insufficient.")
        else:
             logging.info("No market data provided to simulation trigger.")


        logging.info(f"Running simulation for {factory_name} with effective risk factors.")
        # --- Run Simulation (Use asyncio.to_thread) ---
        df_sim_results, df_risk_log = await asyncio.to_thread(
            run_factory_simulation,
            factory_config=first_factory_config,
            risk_factors_config=effective_risk_factors, # Pass potentially modified factors
            sim_duration_weeks=default_duration,
            sim_start_date=start_date
        )
        logging.info(f"Simulation completed. Results shape: {df_sim_results.shape}")

        # --- Calculate Financial KPIs ---
        financial_kpis = financials.calculate_financial_kpis(df_sim_results, effective_costs)
        logging.info(f"Financial KPIs calculated: {financial_kpis}")


        # --- Process Results for Response ---
        data_for_plotting = []
        data_for_recommendations = []
        if not df_sim_results.empty:
            # Convert date to string YYYY-MM-DD for JSON
            if 'date' in df_sim_results.columns:
                 df_sim_results['date_str'] = pd.to_datetime(df_sim_results['date']).dt.strftime('%Y-%m-%d')
                 # Rename date_str to 'date' in copy for plotting
                 df_sim_results_processed = df_sim_results.copy().rename(columns={'date_str': 'date'})
            else:
                 logging.warning("Date column missing in simulation results.")
                 df_sim_results_processed = df_sim_results.copy() # Use original if date missing
                 df_sim_results_processed['date'] = 'N/A' # Add dummy date col

            # Handle NaNs and types for ALL columns needed by summarizer/recs
            # Use the full processed DataFrame for recommendations data
            data_for_recommendations = df_sim_results_processed.to_dict('records')

            # Select and process plotting subset (date, inventory, shortage)
            plot_cols_subset = ['date', 'inventory_level_units', 'shortage_units'] # Use the 'date' column in processed df
            existing_plot_cols = [c for c in plot_cols_subset if c in df_sim_results_processed.columns]
            if 'date' in existing_plot_cols:
                df_plot_subset = df_sim_results_processed[existing_plot_cols].copy()
                 # Fill numeric NaNs with 0 for plotting
                for col in ['inventory_level_units', 'shortage_units']:
                    if col in df_plot_subset.columns:
                        df_plot_subset[col] = pd.to_numeric(df_plot_subset[col], errors='coerce').fillna(0).round(2)
                data_for_plotting = df_plot_subset.to_dict('records')


        logging.info(f"Prepared {len(data_for_plotting)} plot points and {len(data_for_recommendations)} recommendation points.")

        # --- Return Response ---
        return {
                "status": "completed",
                "message": f"Simulation run complete for {factory_name} with {len(risk_events or [])} risks applied.",
                "timeseries_data": data_for_plotting, # For plotting
                "full_simulation_data": data_for_recommendations, # For recommendations
                "risk_summary": applied_risk_summary, # Info on applied risks
                "financial_summary": financial_kpis # Include financial KPIs
            }

    except Exception as e:
        logging.error(f"Simulation run failed: {e}", exc_info=True)
        # Ensure the error response structure matches the model
        # Return the SimulationResultResponse model instance on error
        return {
            "status": "error",
            "message": f"Sim Error: {str(e)}",
            "timeseries_data": None,
            "full_simulation_data": None,
            "risk_summary": None,
            "financial_summary": {"calculation_error": f"Sim failed: {str(e)}"}
        }

# --- UPDATED generate_simulation_recommendations ---
async def generate_simulation_recommendations(
    simulation_data: List[Dict[str, Any]],
    risk_summary: Optional[Dict[str, Any]] = None, # Accept risk context
    market_data: Optional[Dict[str, Any]] = None, # Accept market context
    financial_summary: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Enhanced recommendations including risk/market context."""
    if not RECOMMENDATIONS_AVAILABLE: raise HTTPException(status_code=501, detail="Rec engine not available.")
    if not gemini_model: raise HTTPException(status_code=503, detail="AI model not initialized")
    if not simulation_data:
        logging.warning("generate_simulation_recommendations called with empty simulation_data.")
        return []

    try:
        df_sim_results = pd.DataFrame(simulation_data)
        if df_sim_results.empty:
             logging.warning("DataFrame created from simulation_data is empty.")
             return []

        # Data validation: date conversion/dropna, fillna for safety
        if 'date' in df_sim_results.columns:
            df_sim_results['date'] = pd.to_datetime(df_sim_results['date'], errors='coerce')
            # Drop rows where date couldn't be parsed
            original_rows = len(df_sim_results)
            df_sim_results.dropna(subset=['date'], inplace=True)
            if len(df_sim_results) < original_rows:
                 logging.warning(f"Dropped {original_rows - len(df_sim_results)} rows due to invalid dates in recommendation input.")
        else:
             logging.warning("No 'date' column found in simulation data for recommendations.")
             # Decide how to proceed: error out or try to continue without dates?
             # return [] # Option: Stop if dates are crucial
             # Or fill potentially missing numeric/object columns just in case
             numeric_cols = df_sim_results.select_dtypes(include=np.number).columns
             string_cols = df_sim_results.select_dtypes(include=['object', 'string']).columns
             df_sim_results[numeric_cols] = df_sim_results[numeric_cols].fillna(0)
             df_sim_results[string_cols] = df_sim_results[string_cols].fillna('N/A')


        if df_sim_results.empty:
            logging.warning("DataFrame is empty after date processing/validation.")
            return []

        logging.info(f"Generating recommendations from DataFrame shape: {df_sim_results.shape}")

        sim_start = df_sim_results['date'].min().date() if 'date' in df_sim_results and not df_sim_results['date'].isnull().all() else "Unknown Start"
        sim_end = df_sim_results['date'].max().date() if 'date' in df_sim_results and not df_sim_results['date'].isnull().all() else "Unknown End"
        factory = df_sim_results['factory_name'].iloc[0] if 'factory_name' in df_sim_results and not df_sim_results.empty and 'factory_name' in df_sim_results.columns else "Unknown Factory"


        # Summarize Data - Run in threadpool as it might involve computation
        # Pass None for df_risks for now, unless you derive it from risk_summary
        data_summary = await asyncio.to_thread(
             summarize_data_for_gemini,
             df_results=df_sim_results,
             df_risks=None,
             factory_name=factory,
             start_date=sim_start, # Pass Python date objects or formatted strings
             end_date=sim_end
         )

        # --- Build Enhanced Context ---
        context_parts = [f"Simulation analysis for factory: {factory}"]
        if risk_summary: # ... (Add risk context - keep existing) ...
             context_parts.append("\nApplied Risk Scenario:")
             applied = risk_summary.get('applied_risk_events', [])
             context_parts.append(f"- Events: {', '.join(applied) if applied else 'None'}")
             if risk_summary.get('risk_severity'):
                 context_parts.append("- Severity:")
                 for r, s in risk_summary['risk_severity'].items(): context_parts.append(f"  - {r}: {s:.1f}")
        if market_data: 
             context_parts.append("\nMarket Context:")
             for symbol, market_data_symbol_obj in market_data.items(): # Renamed 'data' to 'market_data_symbol_obj' for clarity
                  # Access attributes directly using dot notation
                  if hasattr(market_data_symbol_obj, 'values') and market_data_symbol_obj.values and len(market_data_symbol_obj.values) >= 2:
                       # Ensure values are numbers before division
                       try:
                           start_val = float(market_data_symbol_obj.values[0])
                           end_val = float(market_data_symbol_obj.values[-1])
                           if start_val != 0:
                               change = ((end_val / start_val) - 1.0) * 100
                           else:
                               change = float('inf') if end_val > 0 else (float('-inf') if end_val < 0 else 0) # Handle division by zero more gracefully

                           trend = "increased" if change > 0.1 else ("decreased" if change < -0.1 else "stable") # Added a small threshold for "stable"
                           context_parts.append(f"- {symbol} {trend} by {abs(change):.1f}% over the period.")
                       except (ValueError, TypeError) as e:
                           logging.warning(f"Could not process market data for {symbol} due to value error: {e}. Values: {market_data_symbol_obj.values}")
                           context_parts.append(f"- {symbol}: Data format error.")
                  else:
                       context_parts.append(f"- {symbol}: Insufficient data.")
        # --- NEW: Add Financial Context ---
        if financial_summary:
            context_parts.append("\nSimulation Financial Impact:")
            if financial_summary.get("calculation_error"):
                 context_parts.append(f"- Financial calculation failed: {financial_summary['calculation_error']}")
            else:
                 # Include key financial figures
                 total_cost = financial_summary.get("estimated_total_supply_chain_cost")
                 stockout_cost = financial_summary.get("total_stockout_cost")
                 holding_cost = financial_summary.get("total_holding_cost")

                 if total_cost is not None: context_parts.append(f"- Estimated Total Cost: ${total_cost:,.2f}")
                 if stockout_cost is not None: context_parts.append(f"- Total Stockout Cost: ${stockout_cost:,.2f}")
                 if holding_cost is not None: context_parts.append(f"- Total Holding Cost: ${holding_cost:,.2f}")
                 # Add others as desired

        enhanced_context = "\n".join(context_parts)
        logging.info(f"Context for recommendations: {enhanced_context}")

        # Get Recommendations using enhanced context
        recommendations_raw = await asyncio.to_thread(
             get_gemini_recommendations,
             _data_summary=data_summary, # Still include operational summary
             _context_text=enhanced_context, # Pass the combined context
             _gemini_model_instance=gemini_model
         )
        logging.info(f"Generated {len(recommendations_raw)} raw recommendations.")

        # (Keep validation logic)
        validated_recommendations = [] # ...
        for rec in recommendations_raw:
             if isinstance(rec, dict) and all(k in rec for k in ['priority', 'category', 'text']):
                  validated_recommendations.append({
                       "priority": int(rec.get('priority', 0)), "category": str(rec.get('category', 'Unknown')),
                       "text": str(rec.get('text', 'N/A')) })
             else: logging.warning(f"Skipping malformed recommendation: {rec}")
        return validated_recommendations
    except Exception as e:
        logging.error(f"Failed to generate recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rec generation failed: {str(e)}")