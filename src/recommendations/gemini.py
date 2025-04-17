# src/recommendations/gemini.py
import streamlit as st
import google.generativeai as genai
import logging
import re
import pandas as pd # Needed for summarizer
import numpy as np # Needed for summarizer
from datetime import datetime, timedelta # Needed for summarizer

# --- Gemini API Setup ---
GEMINI_ENABLED = False
gemini_model = None

try:
    # Prioritize Streamlit secrets if available
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        logging.info("Using Gemini API Key from st.secrets.")
    else:
        # Fallback to environment variable (optional, remove if not needed)
        # GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        # if GEMINI_API_KEY:
        #     logging.info("Using Gemini API Key from environment variable.")
        # else:
             # Fallback to hardcoded key (FOR DEBUGGING ONLY - REMOVE FOR PRODUCTION)
             TEMP_HARDCODED_API_KEY = "YOUR_GEMINI_API_KEY_HERE" # <-- **DANGER: REPLACE OR REMOVE!**
             if TEMP_HARDCODED_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
                  GEMINI_API_KEY = TEMP_HARDCODED_API_KEY
                  logging.warning("Using HARDCODED Gemini API Key (TEMPORARY - REMOVE FOR PRODUCTION).")
             else:
                  GEMINI_API_KEY = None
                  logging.warning("Gemini API Key not found in st.secrets or hardcoded. Recommendations disabled.")

    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        GEMINI_ENABLED = True
        logging.info("Gemini API configured successfully.")
    else:
         # Ensure GEMINI_ENABLED is False if no key was found/configured
         GEMINI_ENABLED = False

except ImportError:
    logging.error("google.generativeai library not installed. pip install google-generativeai")
    GEMINI_ENABLED = False
except Exception as e:
    logging.error(f"Failed to configure Gemini API: {e}", exc_info=True)
    GEMINI_ENABLED = False
    gemini_model = None


# --- Recommendation Engine Functions ---

# (Copy the summarize_data_for_gemini function definition from your original script here)
def summarize_data_for_gemini(df_results, df_risks, factory_name="Overall", start_date=None, end_date=None):
    # (Content of summarize_data_for_gemini function - NO CHANGES NEEDED YET)
    if df_results is None or df_results.empty: return "No simulation data available for analysis."
    summary = f"## Supply Chain Performance Summary\n"
    if factory_name != "Overall": summary += f"### Factory: {factory_name}\n"
    else: summary += f"### Aggregated across {len(df_results['factory_name'].unique())} factories\n"
    if start_date and end_date: summary += f"### Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
    summary += "\n**Key Performance Indicators (KPIs):**\n"
    total_prod = df_results['adjusted_production_units'].sum()
    avg_lead_time = df_results['lead_time_days'].mean()
    max_lead_time = df_results['lead_time_days'].max()
    total_shortage = df_results['shortage_units'].sum()
    avg_inventory = df_results['inventory_level_units'].mean()
    min_inventory = df_results['inventory_level_units'].min()
    final_inventory = df_results['inventory_level_units'].iloc[-1] if not df_results.empty else 'N/A'
    summary += f"- Total Production: {total_prod:,.0f} units\n"
    summary += f"- Average Lead Time: {avg_lead_time:.1f} days (Max: {max_lead_time:.1f} days)\n"
    summary += f"- Total Shortage: {total_shortage:,.0f} units\n"
    summary += f"- Average Inventory: {avg_inventory:,.0f} units (Min: {min_inventory:,.0f})\n"
    summary += f"- Final Inventory: {final_inventory:,.0f} units\n"
    summary += "\n**Notable Trends:**\n"
    if not df_results.empty and len(df_results) > 1:
        lt_start, lt_end = df_results['lead_time_days'].iloc[0], df_results['lead_time_days'].iloc[-1]
        lead_time_trend = "Stable"
        if pd.notna(lt_start) and pd.notna(lt_end):
            if lt_end > lt_start * 1.2: lead_time_trend = "Increasing Significantly"
            elif lt_end > lt_start * 1.05: lead_time_trend = "Increasing Slightly"
            elif lt_end < lt_start * 0.8: lead_time_trend = "Decreasing Significantly"
            elif lt_end < lt_start * 0.95: lead_time_trend = "Decreasing Slightly"
        summary += f"- Lead Time Trend: {lead_time_trend}\n"
        inv_start, inv_end = df_results['inventory_level_units'].iloc[0], df_results['inventory_level_units'].iloc[-1]
        inventory_trend = "Stable"
        if pd.notna(inv_start) and pd.notna(inv_end):
            if inv_end < inv_start * 0.7: inventory_trend = "Decreasing Significantly"
            elif inv_end < inv_start * 0.95: inventory_trend = "Decreasing Slightly"
            elif inv_end > inv_start * 1.3: inventory_trend = "Increasing Significantly"
            elif inv_end > inv_start * 1.05: inventory_trend = "Increasing Slightly"
        summary += f"- Inventory Trend: {inventory_trend}\n"
        if total_shortage > 0:
            shortage_weeks = df_results[df_results['shortage_units'] > 0]['week'].nunique()
            max_shortage = df_results['shortage_units'].max()
            summary += f"- Shortages occurred in {shortage_weeks} weeks (Max weekly: {max_shortage:,.0f} units).\n"
        else: summary += "- No shortages occurred.\n"
    else: summary += "- Insufficient data for trend analysis.\n"
    summary += "\n**Risk Event Analysis:**\n"
    if df_risks is not None and not df_risks.empty:
        summary += f"- Total Risk Events Logged: {len(df_risks)}\n"
        risk_counts = df_risks['risk_type'].value_counts()
        if not risk_counts.empty:
            summary += "- Event Frequency & Avg Impact (from log):\n"
            for risk_type, count in risk_counts.items():
                risk_data = df_risks[df_risks['risk_type'] == risk_type]
                avg_duration, avg_impact, avg_lt_mult = risk_data['duration_days'].mean(), risk_data['impact_factor'].mean(), risk_data['lead_time_multiplier'].mean()
                summary += f"  - {risk_type.replace('_', ' ').title()}: {count} times (Avg Duration: {avg_duration:.1f}d, Avg Prod Impact: {avg_impact:.2f}, Avg LT Mult: {avg_lt_mult:.2f})\n"
        else: summary += "- Risk log provided but no events found.\n"
    else:
        summary += "- No specific risk events logged. Analyzing weekly impact data:\n"
        risky_weeks_prod = df_results[df_results['production_impact_factor'].fillna(1.0) < 0.95]
        risky_weeks_lt = df_results[df_results['lead_time_multiplier'].fillna(1.0) > 1.05]
        if not risky_weeks_prod.empty or not risky_weeks_lt.empty:
             summary += f"  - Weeks with production impact (<0.95): {len(risky_weeks_prod)}\n"
             summary += f"  - Weeks with lead time impact (>1.05): {len(risky_weeks_lt)}\n"
             active_risks_col = df_results.get('active_risks')
             if active_risks_col is not None and (active_risks_col != 'none').any():
                  dominant_risks = active_risks_col[active_risks_col != 'none'].str.split(', ').explode().mode()
                  if not dominant_risks.empty: summary += f"  - Dominant risk types mentioned: {', '.join(dominant_risks.tolist())}\n"
             else: summary += "  - No specific risk types recorded in weekly data.\n"
        else: summary += "  - No significant production or lead time disruptions detected in weekly data.\n"
    return summary

# (Copy the parse_recommendations function definition from your original script here)
def parse_recommendations(gemini_response_text):
    # (Content of parse_recommendations function - NO CHANGES NEEDED YET)
    recommendations = []
    priority_map = {"critical": 5, "high priority": 4, "high": 4, "medium priority": 3, "medium": 3, "low priority": 2, "low": 2, "informational": 1, "info": 1}
    priority_categories = {5: "Critical", 4: "High", 3: "Medium", 2: "Low", 1: "Informational", 0: "Unknown"}
    sections = re.split(r'\n(?= *(?:Critical|High|Medium|Low|Info(?:rmational)?)(?: Priority)? *:?)', gemini_response_text, flags=re.IGNORECASE | re.MULTILINE)
    for section in sections:
        section = section.strip()
        if not section: continue
        current_priority, current_category, matched_keyword = 0, "Unknown", False
        first_line = section.split('\n', 1)[0].lower()
        for keyword, weight in priority_map.items():
            if first_line.startswith(keyword + ":"):
                current_priority, current_category, matched_keyword = weight, priority_categories[weight], True
                section = section[len(first_line):].strip(); break
        if not matched_keyword and len(recommendations) > 0 and sections.index(section) > 0: continue
        items = re.findall(r'(?:^|\n) *[\*\-\+] *(.*)', section)
        if not items: items = re.findall(r'(?:^|\n) *\d+[\.\)] *(.*)', section)
        if not items and matched_keyword: items = [section]
        elif not items and not matched_keyword: items = [line.strip() for line in section.split('\n') if len(line.strip()) > 10]
        for item_text in items:
            item_text = item_text.strip()
            if item_text:
                item_text = re.sub(r'^\s*Reason:\s*', '', item_text, flags=re.IGNORECASE)
                recommendations.append({"priority": current_priority, "category": current_category, "text": item_text})
    if not recommendations and gemini_response_text:
        recommendations.append({"priority": 0, "category": "Unknown", "text": gemini_response_text.strip()})
    recommendations.sort(key=lambda x: x['priority'], reverse=True)
    return recommendations

# (Copy the get_gemini_recommendations function definition from your original script here)
# Add caching here as it's specific to this function's inputs/outputs
@st.cache_data(show_spinner="Generating AI recommendations...")
def get_gemini_recommendations(_data_summary, _context_text=""):
    # (Content of get_gemini_recommendations function - NO CHANGES NEEDED YET)
    # Includes the check for GEMINI_ENABLED and gemini_model
    if not GEMINI_ENABLED or gemini_model is None:
        logging.warning("Gemini is not enabled/initialized. Skipping recommendation generation.")
        return [{"priority": 0, "category": "Error", "text": "Gemini API not configured or failed to initialize."}]
    prompt = f"""
    **Role:** You are an expert Supply Chain Analyst.**Task:** Analyze the provided supply chain simulation data summary and generate specific, actionable recommendations.**Context:** {_context_text}**Simulation Data Summary:**
    ```
    {_data_summary}
    ```
    **Instructions:**
    1. Identify key issues based *only* on the provided summary data.
    2. Generate concrete, actionable recommendations.
    3. Provide a *brief* reasoning for each recommendation, linking it to the data points.
    4. Categorize each recommendation by priority using *exactly* these labels followed by a colon: `Critical:`, `High Priority:`, `Medium Priority:`, `Low Priority:`, `Informational:`.
    5. Structure your response clearly, grouping recommendations under their priority category. Use bullet points (*, -, +) or numbered lists.
    **Output Format Example:**
    Critical:
    *   [Recommendation 1 text]. Reason: [Brief reasoning based on data]
    High Priority:
    *   [Recommendation 2 text]. Reason: [Brief reasoning based on data]
    **Begin Analysis and Recommendations:**
    """
    try:
        logging.info("Sending request to Gemini API.")
        if gemini_model is None: raise ValueError("Gemini model not initialized.")
        response = gemini_model.generate_content(prompt)
        response_text = ""
        if response and response.parts: response_text = ''.join(part.text for part in response.parts)
        elif hasattr(response, 'text'): response_text = response.text
        else:
             safety_feedback = getattr(response, 'prompt_feedback', None); block_reason = getattr(safety_feedback, 'block_reason', None)
             if block_reason: raise ValueError(f"Gemini response blocked. Reason: {block_reason}")
             else: raise ValueError("Received unexpected or empty response from Gemini.")
        logging.info("Received response from Gemini. Parsing recommendations.")
        parsed_recs = parse_recommendations(response_text)
        if not parsed_recs:
             logging.warning("Failed to parse recommendations from Gemini response.")
             return [{"priority": 0, "category": "Parsing Error", "text": "Could not parse recommendations. Raw response:\n" + response_text}]
        return parsed_recs
    except Exception as e:
        logging.error(f"Error calling Gemini API or processing response: {e}", exc_info=True)
        return [{"priority": 0, "category": "API Error", "text": f"Failed to generate recommendations: {e}"}]