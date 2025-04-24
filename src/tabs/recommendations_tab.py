# src/tabs/recommendations_tab.py
import streamlit as st
import logging
import requests # <-- ADD
import pandas as pd # <-- ADD if not already present, for df conversion
import json # <-- ADD for potential formatting

# Import backend URL
from config import BACKEND_URL # <-- ADD

def render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter, selected_factories):
    """Renders the Recommendations tab using the backend API."""
    st.header("🧠 AI-Powered Recommendations")

    # --- REMOVE check for GEMINI_ENABLED ---
    # The backend handles API availability. We check for errors from the API call instead.

    st.markdown("Analyze simulation results using AI. Recommendations based on *currently selected historical data* or the *most recent live simulation run*.")
    st.markdown("---")

    # --- Display Function (Internal - keep as is) ---
    def display_recommendations(recommendations, source_info):
        st.info(source_info, icon="💡")
        # Define mapping from priority level (int) to CSS class and display name
        priority_map = {
            5: ("Critical", "recommendation-critical"),
            4: ("High", "recommendation-high"),
            3: ("Medium", "recommendation-medium"),
            2: ("Low", "recommendation-low"),
            1: ("Informational", "recommendation-informational"),
            0: ("ALERTS!!", "recommendation-unknown") # Default/Error case
        }

        # Group recommendations by priority level
        grouped_recs = {}
        if isinstance(recommendations, list): # Ensure input is a list
            for rec in recommendations:
                if isinstance(rec, dict): # Ensure each item is a dict
                    # Get priority, default to 0 if missing or invalid
                    try:
                        prio = int(rec.get('priority', 0))
                    except (ValueError, TypeError):
                        prio = 0
                    if prio not in grouped_recs: grouped_recs[prio] = []
                    grouped_recs[prio].append(rec)
                else:
                     logging.warning(f"Skipping non-dictionary item in recommendations: {rec}")
        else:
            logging.error(f"display_recommendations received non-list input: {type(recommendations)}")
            st.error("Failed to display recommendations due to invalid data format.")
            st.json(recommendations) # Show raw data if possible
            return

        if not grouped_recs:
            # Handle case where input list was valid but empty or contained only invalid items
            st.info("No valid recommendations to display.")
            return

        # Display recommendations grouped by priority
        displayed_any = False
        # Iterate through sorted priorities (highest first)
        for priority_level in sorted(grouped_recs.keys(), reverse=True):
             # Get display name and CSS class from map, using level 0 as fallback
             cat_name, cat_class = priority_map.get(priority_level, priority_map[0])
             recs_in_group = grouped_recs[priority_level]

             if recs_in_group: # Ensure there are recommendations for this level
                 # Make Critical/High expanded by default
                 is_expanded = priority_level >= 4
                 with st.expander(f"{cat_name} Priority ({len(recs_in_group)})", expanded=is_expanded):
                     for rec in recs_in_group:
                         # Safely get text, default to 'N/A'
                         rec_text = rec.get('text', 'N/A')
                         # Use markdown with the specific CSS class for styling
                         st.markdown(f'<div class="{cat_class}">{rec_text}</div>', unsafe_allow_html=True)
                         displayed_any = True

        if not displayed_any:
             # This case should ideally not be reached if grouped_recs was not empty
             st.info("No recommendations were displayed.")

    # --- Option 1: Live run recommendations (No Change Needed Here) ---
    # This section still uses session_state populated by the Simulation Tab or Dialog
    st.subheader("Recommendations from Last Live Simulation")
    if 'recommendations_live' in st.session_state and st.session_state['recommendations_live']:
        sim_factory_name = st.session_state.get('sim_factory_name', 'Unknown Factory')
        # Assume 'recommendations_live' holds the correct structured list of dicts
        display_recommendations(
            st.session_state['recommendations_live'],
            f"Showing recommendations for the live simulation run of **{sim_factory_name}**."
        )
    else:
        st.info("No live simulation run yet, or recommendations not generated/available. Run a simulation first.")

    st.markdown("---")

    # --- Option 2: Historical data recommendations (MODIFIED TO USE API) ---
    st.subheader("Recommendations for Selected Historical Data")
    if df_filtered is not None and not df_filtered.empty:
        # Button to trigger generation via API
        if st.button("Analyze Selected Historical Data & Get Recommendations", key="gen_hist_rec_api"):
             st.session_state.pop('recommendations_hist', None) # Clear previous results

             rec_endpoint = f"{BACKEND_URL}/get_simulation_recommendations"
             hist_recommendations = [] # Default empty list
             error_message = None

             try:
                 with st.spinner("Analyzing historical data via backend..."):
                     # --- Prepare Data Payload ---
                     # Ensure dates are strings, handle NaNs appropriately for JSON
                     df_payload = df_filtered.copy()
                     if 'date' in df_payload.columns:
                          df_payload['date'] = pd.to_datetime(df_payload['date']).dt.strftime('%Y-%m-%d')
                     # Fill NaNs for relevant columns before sending
                     # Adjust columns and fill values as needed by the backend summarizer
                     numeric_cols = [ 'adjusted_production_units', 'lead_time_days', 'shortage_units', 'inventory_level_units', 'production_impact_factor', 'lead_time_multiplier', 'simulated_demand_units', 'shipped_units', 'base_production_capacity', 'safety_stock_level', 'active_risks_count']
                     for col in numeric_cols:
                          if col in df_payload.columns:
                               df_payload[col] = pd.to_numeric(df_payload[col], errors='coerce').fillna(0) # Example: fill numeric with 0
                     str_cols = ['active_risks', 'factory_name', 'factory_location', 'factory_type']
                     for col in str_cols:
                          if col in df_payload.columns:
                               df_payload[col] = df_payload[col].fillna('N/A')

                     # Convert DataFrame to list of dicts
                     simulation_data_payload = df_payload.to_dict('records')

                     # --- Call Backend API ---
                     response = requests.post(
                         rec_endpoint,
                         json={"simulation_data": simulation_data_payload},
                         timeout=60 # Adjust timeout as needed
                     )
                     response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                     # --- Process Response ---
                     response_data = response.json() # Expects RecommendationResponse model
                     hist_recommendations = response_data.get("recommendations", []) # Get list of dicts
                     logging.info(f"Received {len(hist_recommendations)} historical recommendations from backend.")

             except requests.exceptions.RequestException as e:
                 error_message = f"Error connecting to backend: {e}. Is it running at {BACKEND_URL}?"
                 logging.error(f"Error calling {rec_endpoint}: {e}", exc_info=True)
             except Exception as e:
                 error_message = f"An unexpected error occurred: {e}"
                 logging.error(f"Error processing recommendations from {rec_endpoint}: {e}", exc_info=True)

             # --- Store or Display Results/Errors ---
             if error_message:
                  # Store error as a recommendation for display
                  st.session_state['recommendations_hist'] = [{"priority": 0, "category": "Error", "text": error_message}]
             else:
                  st.session_state['recommendations_hist'] = hist_recommendations

             # Update tracker (no change needed here)
             st.session_state['last_hist_rec_filters'] = (selected_run_file, tuple(sorted(selected_factories)), start_date_filter, end_date_filter)
             st.rerun() # Rerun to display the newly fetched recommendations or error


        # Display historical recommendations if they exist in session state
        if 'recommendations_hist' in st.session_state and st.session_state['recommendations_hist'] is not None:
             hist_info = f"Showing recommendations for **{selected_run_file or 'Selected Data'}** (Filtered: {len(selected_factories)} factories, {start_date_filter} to {end_date_filter})."
             # The display_recommendations function handles showing the list of dicts
             display_recommendations(st.session_state['recommendations_hist'], hist_info)

    else:
        st.warning("Please select a valid simulation run and filters in the sidebar to enable historical data analysis.")

    st.markdown("---")
    st.caption("Disclaimer: AI-generated recommendations require expert review before implementation.")