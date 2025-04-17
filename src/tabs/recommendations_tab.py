# src/tabs/recommendations_tab.py
import streamlit as st
import logging

# Import recommendation functions and Gemini status
from recommendations.gemini import (
    GEMINI_ENABLED,
    summarize_data_for_gemini,
    get_gemini_recommendations
)

def render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter, selected_factories):
    """Renders the Recommendations tab."""
    st.header("🧠 AI-Powered Recommendations")

    if not GEMINI_ENABLED:
        st.warning("Gemini API Key not configured or failed to initialize. Please check setup (`.streamlit/secrets.toml` or environment variables) and API key to enable this feature.", icon="⚠️")
        return # Stop rendering if Gemini is not available

    st.markdown("Analyze simulation results using AI. Recommendations based on *currently selected historical data* or the *most recent live simulation run*.")
    st.markdown("---")

    # --- Display Function (Internal to this tab) ---
    def display_recommendations(recommendations, source_info):
        # (Copy the display_recommendations function from original Tab 6 here)
        st.info(source_info, icon="💡")
        priority_categories_display = {
            5: ("Critical", "recommendation-critical"), 4: ("High", "recommendation-high"),
            3: ("Medium", "recommendation-medium"), 2: ("Low", "recommendation-low"),
            1: ("Informational", "recommendation-informational"), 0: ("Recommendations", "recommendation-unknown")
        }
        grouped_recs = {}
        for rec in recommendations:
            prio = rec.get('priority', 0)
            if prio not in grouped_recs: grouped_recs[prio] = []
            grouped_recs[prio].append(rec)

        displayed_any = False
        for priority_level in sorted(grouped_recs.keys(), reverse=True):
             if priority_level in priority_categories_display and grouped_recs[priority_level]:
                 cat_name, cat_class = priority_categories_display[priority_level]
                 # Make Critical/High expanded by default, others collapsed
                 is_expanded = priority_level >= 4
                 with st.expander(f"{cat_name} Priority Recommendations ({len(grouped_recs[priority_level])})", expanded=is_expanded):
                     for rec in grouped_recs[priority_level]:
                         # Use markdown with the custom CSS class div
                         st.markdown(f'<div class="{cat_class}">{rec["text"]}</div>', unsafe_allow_html=True)
                         displayed_any = True
        if not displayed_any:
             is_error = any(rec.get("category") in ["API Error", "Parsing Error", "Error"] for rec in recommendations)
             if is_error:
                  st.error("Could not generate or display recommendations due to an error.")
                  for rec in recommendations:
                       if rec.get("category") in ["API Error", "Parsing Error", "Error"]:
                            st.markdown(f'<div class="recommendation-critical">{rec["text"]}</div>', unsafe_allow_html=True)
             else:
                  st.info("No specific recommendations were generated or parsed for this data.")


    # --- Option 1: Live run recommendations ---
    st.subheader("Recommendations from Last Live Simulation")
    # (Copy live recommendations display logic from original Tab 6 here)
    if 'recommendations_live' in st.session_state and st.session_state['recommendations_live']:
        sim_factory_name = st.session_state.get('sim_factory_name', 'Unknown Factory')
        display_recommendations(
            st.session_state['recommendations_live'],
            f"Showing recommendations for the live simulation run of **{sim_factory_name}**."
        )
    else:
        st.info("No live simulation run yet, or recommendations not generated/available. Run a simulation in the 'Live Simulation' tab.")


    st.markdown("---")

    # --- Option 2: Historical data recommendations ---
    st.subheader("Recommendations for Selected Historical Data")
    # (Copy historical recommendations display logic from original Tab 6 here)
    if df_filtered is not None and not df_filtered.empty:
        # Button to trigger generation
        if st.button("Analyze Selected Historical Data & Generate Recommendations", key="gen_hist_rec"):
             st.session_state.pop('recommendations_hist', None) # Clear previous
             with st.spinner("Analyzing historical data and generating recommendations..."):
                 # Prepare context and summary
                 factory_context = f"{len(selected_factories)} selected factories" if len(selected_factories) != 1 else f"factory '{selected_factories[0]}'"
                 hist_context = f"Analysis of historical simulation data for {factory_context} from {start_date_filter.strftime('%Y-%m-%d')} to {end_date_filter.strftime('%Y-%m-%d')}."
                 # Pass df_filtered for both results and risk inference
                 hist_summary = summarize_data_for_gemini(
                     df_filtered,
                     None, # Pass None for df_risks, summarizer handles it
                     factory_name="Overall Selected" if len(selected_factories) != 1 else selected_factories[0],
                     start_date=start_date_filter,
                     end_date=end_date_filter
                 )
                 # Call Gemini function
                 recommendations_hist = get_gemini_recommendations(hist_summary, hist_context)
                 # Store results in session state
                 st.session_state['recommendations_hist'] = recommendations_hist
                 # Update tracker for cache invalidation logic in app.py
                 st.session_state.last_hist_rec_filters = (selected_run_file, tuple(sorted(selected_factories)), start_date_filter, end_date_filter)
                 # No explicit success message needed here, display logic handles it


        # Display historical recommendations if they exist
        if 'recommendations_hist' in st.session_state and st.session_state['recommendations_hist']:
             hist_info = f"Showing recommendations for **{selected_run_file}** (Filtered: {len(selected_factories)} factories, {start_date_filter} to {end_date_filter})."
             display_recommendations(st.session_state['recommendations_hist'], hist_info)
        # Removed the elif condition here, as it's redundant if button not pressed

    else:
        st.warning("Please select a valid simulation run and filters in the sidebar to enable historical data recommendations.")


    st.markdown("---")
    st.caption("Disclaimer: AI-generated recommendations require expert review before implementation.")