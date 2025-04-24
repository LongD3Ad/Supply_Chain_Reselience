# backend/app/services.py
import google.generativeai as genai
import requests
import os
import sys
import logging
from fastapi import HTTPException
import pandas as pd
import json
from typing import List, Optional, Dict, Any
import asyncio

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)

# --- Attempt Imports ---
try:
    from simulation.engine import run_factory_simulation
    from config import get_config
    SIMULATION_AVAILABLE = True
    logging.info("Successfully imported simulation engine and config from src.")
except ImportError as e:
    logging.warning(f"Could not import simulation engine or config from src: {e}. Simulation endpoint disabled.")
    SIMULATION_AVAILABLE = False
    def run_factory_simulation(*args, **kwargs): raise NotImplementedError("Sim engine not loaded.")
    def get_config(): return {"factories": []}

try:
    from recommendations.gemini import summarize_data_for_gemini, get_gemini_recommendations
    RECOMMENDATIONS_AVAILABLE = True
    logging.info("Successfully imported recommendation components from src.")
except ImportError as e:
    logging.warning(f"Could not import recommendation components from src: {e}. Recommendations endpoint disabled.")
    RECOMMENDATIONS_AVAILABLE = False
    def summarize_data_for_gemini(*args, **kwargs): raise NotImplementedError("Rec engine not loaded.")
    def get_gemini_recommendations(*args, **kwargs): return [{"priority": 0, "category": "Error", "text": "Rec engine not loaded."}]

# --- Initialize Gemini Model ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY environment variable is required")
if not NEWS_API_KEY: logging.warning("NEWS_API_KEY environment variable missing. News fetching disabled.")
gemini_model = None
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    logging.info(f"Gemini Model ('{gemini_model.model_name}') initialized successfully.")
except Exception as e: logging.error(f"Failed to initialize Gemini Model: {e}", exc_info=True)

# --- Service Functions ---

# (get_gemini_chat_response remains the same as previous version)
async def get_gemini_chat_response(prompt: str, news_summary: Optional[str] = None, active_tab: Optional[str] = None) -> str:
    """Gets a response from the Gemini model, using provided context."""
    if not gemini_model:
        logging.error("Gemini model not available for chat.")
        raise HTTPException(status_code=503, detail="AI model not initialized")

    # --- CORRECTED PROMPT CONSTRUCTION ---
    context_lines = []
    if active_tab:
        context_lines.append(f"The user is viewing the '{active_tab}' tab.")
    if news_summary:
        # Add clear separation for the news summary
        context_lines.append("\nConsider this recent news summary:\n---\n" + news_summary + "\n---")

    # Start building the prompt piece by piece
    prompt_parts = []
    if context_lines:
        context_str = "\n".join(context_lines)
        prompt_parts.append("Context:")
        prompt_parts.append(context_str)
        prompt_parts.append("\n") # Add clear separation after context
        logging.info("Including context in chat prompt")
    else:
        logging.info("No extra context for this chat prompt")

    # Add the user query and the final instruction
    prompt_parts.append(f"User query: {prompt}")
    prompt_parts.append("\nAssistant response:") # Ensure newline before assistant response instruction

    # Join all parts into the final prompt string
    full_prompt = "\n".join(prompt_parts)
    # --- END OF CORRECTION ---

    logging.debug(f"Sending prompt to Gemini (Context: Tab='{active_tab}', NewsProvided={news_summary is not None}) Query='{prompt[:50]}...'")
    # Log the full prompt ONLY if needed for deep debugging and ensure no sensitive info is logged raw
    # logging.debug(f"Full prompt:\n{full_prompt}")

    try:
        response = await gemini_model.generate_content_async(full_prompt)
        # (Rest of the try...except block remains the same)
        if not response.parts and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             logging.warning(f"Gemini response blocked: {response.prompt_feedback.block_reason}")
             return f"My response was blocked by the safety filter: {response.prompt_feedback.block_reason}"

        response_text = ''.join(part.text for part in response.parts) if response.parts else ""
        if not response_text and hasattr(response, 'text'): response_text = response.text
        if not response_text:
             logging.warning("Gemini returned an empty response text.")
             return "I couldn't generate a response for that."

        return response_text
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")

# (fetch_and_summarize_news remains the same as previous version)
async def fetch_and_summarize_news(query: str) -> str:
    if not NEWS_API_KEY: return "News fetching disabled (API key missing)."
    if not gemini_model: raise HTTPException(status_code=503, detail="AI model not initialized")
    articles_text, summary_text = "", "Failed to generate news summary."
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&pageSize=10&language=en&sortBy=relevancy"
        response = requests.get(url, timeout=15); response.raise_for_status()
        articles = response.json().get("articles", []); logging.info(f"Fetched {len(articles)} articles.")
        if not articles: return "No relevant news articles found."
        articles_text = "\n\n".join([f"Title: {a.get('title','N/A')}\nDesc: {a.get('description','N/A')}" for a in articles[:5]])
    except requests.exceptions.RequestException as e: raise HTTPException(status_code=502, detail=f"News API Error: {str(e)}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error processing news: {str(e)}")
    try:
        summary_prompt = f"Summarize key supply chain insights/disruptions from these news excerpts:\n{articles_text}\n\nConcise Summary:"
        summary_response = await gemini_model.generate_content_async(summary_prompt)
        if not summary_response.parts and hasattr(summary_response, 'prompt_feedback') and summary_response.prompt_feedback.block_reason:
             summary_text = f"AI summary blocked: {summary_response.prompt_feedback.block_reason}"
        else:
            summary_text = ''.join(part.text for part in summary_response.parts) if summary_response.parts else ""
            if not summary_text and hasattr(summary_response, 'text'): summary_text = summary_response.text
            if not summary_text: summary_text = "AI returned an empty summary."
        return summary_text
    except Exception as e: logging.error(f"Gemini summarization failed: {e}", exc_info=True); return f"Could not summarize. Headlines:\n{articles_text}"


# --- UPDATED trigger_simulation_run ---
async def trigger_simulation_run() -> dict:
    """Triggers baseline SimPy simulation, returns status, message, plotting data, and full data."""
    if not SIMULATION_AVAILABLE: raise HTTPException(status_code=501, detail="Sim engine not available.")

    try:
        logging.info("Triggering baseline simulation run...")
        sim_config = get_config()
        if not sim_config or not sim_config.get("factories"): raise ValueError("Invalid sim config.")

        first_factory_config = sim_config["factories"][0]
        factory_name = first_factory_config.get("name", "Unknown")
        default_risks = sim_config.get("risk_factors", {})
        default_duration = sim_config.get("simulation_defaults", {}).get("sim_duration_weeks", 52)
        start_date = pd.Timestamp.now(tz=None).normalize()

        logging.info(f"Running simulation for: {factory_name}, Duration: {default_duration} weeks")
        df_sim_results, df_risk_log = run_factory_simulation(
            factory_config=first_factory_config, risk_factors_config=default_risks,
            sim_duration_weeks=default_duration, sim_start_date=start_date
        )
        logging.info(f"Simulation completed. Results shape: {df_sim_results.shape}")

        # --- Prepare Data for Response ---
        data_for_plotting = []
        data_for_recommendations = []

        if not df_sim_results.empty:
            # Ensure date is string for JSON
            if 'date' in df_sim_results.columns and pd.api.types.is_datetime64_any_dtype(df_sim_results['date']):
                 df_sim_results['date_str'] = df_sim_results['date'].dt.strftime('%Y-%m-%d')
            elif 'date' in df_sim_results.columns:
                 df_sim_results['date_str'] = df_sim_results['date'].astype(str)
            else:
                 df_sim_results['date_str'] = None # Handle missing date case

            df_sim_results_processed = df_sim_results.copy()

            # Handle NaNs and types for ALL potential columns needed by summarizer/recs
            # Define numeric columns expected by summarizer (adjust list as needed)
            numeric_cols = [
                'base_production_capacity', 'adjusted_production_units', 'production_impact_factor',
                'lead_time_days', 'simulated_demand_units', 'shipped_units',
                'inventory_level_units', 'shortage_units', 'safety_stock_level',
                'lead_time_multiplier', 'active_risks_count' # Add others if used
            ]
            for col in numeric_cols:
                if col in df_sim_results_processed.columns:
                    # Fill NaN with 0 for counts/units, maybe keep NaN or use mean/median for factors/multipliers if appropriate
                    fill_val = 0
                    df_sim_results_processed[col] = pd.to_numeric(df_sim_results_processed[col], errors='coerce').fillna(fill_val)
                else:
                    # Add missing numeric columns expected by summarizer? Or let it handle KeyError?
                    # Let's assume summarizer checks existence. If not, add: df_sim_results_processed[col] = fill_val
                    pass

            # Handle string columns
            str_cols = ['factory_name', 'factory_location', 'factory_type', 'active_risks']
            for col in str_cols:
                 if col in df_sim_results_processed.columns:
                      df_sim_results_processed[col] = df_sim_results_processed[col].fillna('N/A').astype(str)
                 else:
                      # Add missing string columns if summarizer requires them
                      # df_sim_results_processed[col] = 'N/A'
                      pass

            # Prepare plotting subset (date_str, inventory, shortage)
            plot_cols_subset = ['date_str', 'inventory_level_units', 'shortage_units']
            existing_plot_cols = [c for c in plot_cols_subset if c in df_sim_results_processed.columns]
            if 'date_str' in existing_plot_cols:
                data_for_plotting = df_sim_results_processed[existing_plot_cols].rename(columns={'date_str': 'date'}).to_dict('records')

            # Prepare full data for recommendations (use date_str)
            # Rename date_str back to 'date' for consistency if needed by summarizer
            if 'date_str' in df_sim_results_processed.columns:
                 df_sim_results_processed.rename(columns={'date_str': 'date'}, inplace=True)
                 data_for_recommendations = df_sim_results_processed.to_dict('records')
            else: # Should not happen if date exists
                 data_for_recommendations = df_sim_results_processed.to_dict('records')


        logging.info(f"Prepared {len(data_for_plotting)} plot points and {len(data_for_recommendations)} recommendation points.")
        return {
                "status": "completed",
                "message": f"Baseline simulation run complete for {factory_name}.",
                "timeseries_data": data_for_plotting, # For plotting
                "full_simulation_data": data_for_recommendations # For recommendations
            }

    except Exception as e:
        logging.error(f"Simulation run failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation execution failed: {str(e)}")


# --- UPDATED generate_simulation_recommendations ---
async def generate_simulation_recommendations(simulation_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates structured recommendations using imported functions."""
    if not RECOMMENDATIONS_AVAILABLE: raise HTTPException(status_code=501, detail="Rec engine not available.")
    if not gemini_model: raise HTTPException(status_code=503, detail="AI model not initialized")
    if not simulation_data: return []

    try:
        df_sim_results = pd.DataFrame(simulation_data)
        if df_sim_results.empty: return []

        # --- Optional Safety Check ---
        # Add missing columns with default values if summarizer requires them strictly
        required_cols_for_summary = ['adjusted_production_units', 'lead_time_days', 'shortage_units', 'inventory_level_units'] # Example list
        for col in required_cols_for_summary:
            if col not in df_sim_results.columns:
                logging.warning(f"Column '{col}' missing for recommendations, adding fallback NA/0.")
                # Decide appropriate fallback (NA or 0)
                if col in ['shortage_units']: df_sim_results[col] = 0
                else: df_sim_results[col] = pd.NA # Use pandas NA

        # Convert 'date' string back to datetime for summarizer
        if 'date' in df_sim_results.columns:
             df_sim_results['date'] = pd.to_datetime(df_sim_results['date'], errors='coerce')
             df_sim_results.dropna(subset=['date'], inplace=True)
             if df_sim_results.empty: return []
        else: logging.warning("Date missing for recommendations.")

        logging.info(f"Generating recommendations from DataFrame shape: {df_sim_results.shape}")

        sim_start = df_sim_results['date'].min().date() if 'date' in df_sim_results and df_sim_results['date'].notna().any() else None
        sim_end = df_sim_results['date'].max().date() if 'date' in df_sim_results and df_sim_results['date'].notna().any() else None
        factory = df_sim_results['factory_name'].iloc[0] if 'factory_name' in df_sim_results and not df_sim_results.empty else "Unknown"

        # Summarize (Run potentially blocking sync function in thread)
        data_summary = await asyncio.to_thread(
             summarize_data_for_gemini,
             df_results=df_sim_results, df_risks=None, factory_name=factory,
             start_date=sim_start, end_date=sim_end
         )
        logging.info(f"Data summary generated (length {len(data_summary)} chars)")

        recommendations_raw = await asyncio.to_thread(
         get_gemini_recommendations,      # Target function
         _data_summary=data_summary,      # Args for the function
         _gemini_model_instance=gemini_model # Pass the model using the underscore argument name
        )
        logging.info(f"Generated {len(recommendations_raw)} raw recommendations.")

        # Validate structure
        validated_recommendations = []
        for rec in recommendations_raw:
             if isinstance(rec, dict) and all(k in rec for k in ['priority', 'category', 'text']):
                  validated_recommendations.append({
                       "priority": int(rec.get('priority', 0)), "category": str(rec.get('category', 'Unknown')),
                       "text": str(rec.get('text', 'N/A'))
                  })
             else: logging.warning(f"Skipping malformed recommendation: {rec}")
        return validated_recommendations
    except KeyError as e:
         logging.error(f"Missing column error during rec generation (likely summarizer): {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Internal error generating recs (missing data: {e}).")
    except Exception as e:
        logging.error(f"Failed to generate recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rec generation failed: {str(e)}")