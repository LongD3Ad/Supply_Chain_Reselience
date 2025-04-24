# src/tabs/simulation_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import logging
from datetime import datetime, timedelta
import google.generativeai as genai
import os
from dotenv import load_dotenv # <--- ADD

# --- Load Environment Variables ---
# Load variables from .env file in the project root
load_dotenv() # <--- ADD THIS CALL

# Import simulation function and config getter
from simulation.engine import run_factory_simulation
from config import get_config
# Import recommendation functions (requires model instance)
from recommendations.gemini import summarize_data_for_gemini, get_gemini_recommendations

def render_tab():
    """Renders the Live Simulation tab."""
    st.header("🔬 Live Factory Simulation")
    st.markdown("Configure and run a live simulation. Results appear below and recommendations in the next tab or AI Assistant.")

    sim_config = get_config()
    sim_col1, sim_col2 = st.columns([1, 2])
    factory_sim_config = None
    selected_factory_sim = None
    sim_start_datetime = datetime.now()

    with sim_col1:
        # (Your existing simulation setup widgets: factory select, duration, date)
        st.subheader("Simulation Setup")
        factory_options = [f['name'] for f in sim_config.get('factories', [])]
        if not factory_options: st.error("No factories defined in config."); return
        selected_factory_sim = st.selectbox("Select Factory Profile:", factory_options, key="sim_factory_select")
        factory_sim_config = next((f for f in sim_config['factories'] if f['name'] == selected_factory_sim), None)
        sim_duration_weeks = st.number_input("Duration (Weeks):", 4, 156, sim_config.get('simulation_defaults',{}).get('sim_duration_weeks', 52), 4, key="sim_duration")
        default_start_dt = datetime.now() - timedelta(weeks=sim_duration_weeks)
        sim_start_date_input = st.date_input("Start Date:", default_start_dt.date(), key="sim_start_date")
        sim_start_datetime = datetime.combine(sim_start_date_input, datetime.min.time()) if sim_start_date_input else datetime.combine(default_start_dt.date(), datetime.min.time())
        run_button = st.button("🚀 Run Live Simulation", key="run_sim_button", disabled=(factory_sim_config is None))

    with sim_col2:
        # (Your existing risk parameter adjustment widgets)
        st.subheader("Adjust Risk Parameters (Optional)")
        interactive_risk_params = {} # Populate this based on sliders as before...
        default_risk_params = sim_config.get('risk_factors', {})
        # ... (Loop through risks, add sliders, populate interactive_risk_params) ...
        for risk, params in default_risk_params.items():
             with st.expander(f"Configure: {risk.replace('_', ' ').title()}", expanded=False):
                  # ... your sliders using params.get() ...
                  prob = st.slider(f"Prob/Day", 0.0, 0.1, params.get("prob_per_day", 0.01), 0.001, format="%.3f", key=f"{risk}_prob")
                  # ... other sliders ...
                  interactive_risk_params[risk] = { # Store adjusted values
                       "prob_per_day": prob,
                       # ... other adjusted params ...
                       "impact_factor_range": params.get("impact_factor_range"), # Example: Get ranges from sliders
                       "duration_range_days": params.get("duration_range_days"),
                       "lead_time_multiplier_range": params.get("lead_time_multiplier_range"),
                       "mean_days_between_attempts": params.get("mean_days_between_attempts")
                  }


    # --- Simulation Execution ---
    st.markdown("---")
    if run_button and factory_sim_config:
        st.session_state.pop('sim_results', None); st.session_state.pop('sim_risk_log', None); st.session_state.pop('recommendations_live', None)

        with st.spinner(f"Running {sim_duration_weeks}-week simulation for {selected_factory_sim}..."):
            try:
                df_sim_results, df_risk_log = run_factory_simulation(
                    factory_config=factory_sim_config,
                    risk_factors_config=interactive_risk_params or default_risk_params, # Use interactive if adjusted
                    sim_duration_weeks=sim_duration_weeks,
                    sim_start_date=sim_start_datetime
                )
                st.session_state['sim_results'] = df_sim_results
                st.session_state['sim_risk_log'] = df_risk_log
                st.session_state['sim_factory_name'] = selected_factory_sim
                st.success(f"Live simulation for '{selected_factory_sim}' complete!")

                # --- Trigger Recommendation Generation (using .env primarily) ---
                if not df_sim_results.empty:
                    frontend_gemini_model = None
                    api_key = None
                    try:
                        # *** MODIFIED Logic: Prioritize .env file ***
                        api_key = os.environ.get("GOOGLE_API_KEY") # Check env var loaded by dotenv
                        if api_key:
                            logging.info("Using Gemini key from environment variables for live recs.")
                        elif "GEMINI_API_KEY" in st.secrets: # Fallback to secrets.toml
                            api_key = st.secrets["GEMINI_API_KEY"]
                            logging.info("Using Gemini key from st.secrets for live recs.")

                        if api_key:
                            genai.configure(api_key=api_key)
                            frontend_gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Use desired model
                            logging.info("Frontend Gemini model initialized for live recommendations.")
                        else:
                            st.warning("Gemini API Key not found in environment variables or secrets. Cannot generate live recommendations.")

                    except Exception as config_e:
                        st.error(f"Failed to configure Gemini API for live recommendations: {config_e}")
                        logging.error(f"Frontend Gemini config error: {config_e}", exc_info=True)

                    # --- Call Recommendation Function IF Model Initialized ---
                    if frontend_gemini_model:
                         st.info("Generating AI recommendations for live run...")
                         # Prep data for summarizer
                         sim_start = df_sim_results['date'].min().date() if 'date' in df_sim_results and not df_sim_results.empty else None
                         sim_end = df_sim_results['date'].max().date() if 'date' in df_sim_results and not df_sim_results.empty else None
                         sim_summary = summarize_data_for_gemini(
                              df_sim_results, df_risk_log, factory_sim_config['name'], sim_start, sim_end
                         )
                         # Call function, passing the initialized model
                         recommendations = get_gemini_recommendations(
                              _data_summary=sim_summary,
                              _context_text=f"Live sim for {selected_factory_sim}",
                              _gemini_model_instance=frontend_gemini_model # Pass the instance
                         )
                         st.session_state['recommendations_live'] = recommendations
                         st.success("AI recommendations generated and stored.")
                    else:
                         st.session_state['recommendations_live'] = [{"priority":0, "category":"Config Error", "text":"Gemini not configured for frontend."}]

                elif df_sim_results.empty: st.warning("Simulation produced no data.")
            except Exception as e: st.error(f"Simulation run failed: {e}"); logging.error("Sim run error", exc_info=True)
            
    # Display results if they exist in session state
    # (Copy results display logic - KPIs, Charts, Risk Log - from original Tab 5 here)
    if 'sim_results' in st.session_state and st.session_state['sim_results'] is not None and not st.session_state['sim_results'].empty:
        st.subheader(f"Live Simulation Results for: {st.session_state['sim_factory_name']}")
        df_display_sim = st.session_state['sim_results']
        df_display_risk = st.session_state.get('sim_risk_log')

        # KPIs
        scol1, scol2, scol3, scol4 = st.columns(4)
        sim_total_prod = df_display_sim['adjusted_production_units'].sum()
        sim_avg_lead = df_display_sim['lead_time_days'].mean()
        sim_total_shortage = df_display_sim['shortage_units'].sum()
        sim_avg_inv = df_display_sim['inventory_level_units'].mean()
        scol1.metric("Total Production", f"{sim_total_prod:,.0f} units")
        scol2.metric("Avg Lead Time", f"{sim_avg_lead:.1f} days" if pd.notna(sim_avg_lead) else "N/A")
        scol3.metric("Total Shortage", f"{sim_total_shortage:,.0f} units")
        scol4.metric("Avg Inventory", f"{sim_avg_inv:,.0f} units" if pd.notna(sim_avg_inv) else "N/A")

        # Charts
        st.plotly_chart(px.line(df_display_sim, x='date', y='adjusted_production_units', title='Simulated Production'), use_container_width=True)
        st.plotly_chart(px.line(df_display_sim, x='date', y='inventory_level_units', title='Simulated Inventory Level'), use_container_width=True)
        st.plotly_chart(px.line(df_display_sim, x='date', y='lead_time_days', title='Simulated Lead Time'), use_container_width=True)
        st.plotly_chart(px.bar(df_display_sim, x='date', y='shortage_units', title='Simulated Shortage'), use_container_width=True)

        # Risk Timeline Chart & Log
        st.subheader("Live Simulation Risk Event Timeline & Log")
        if df_display_risk is not None and not df_display_risk.empty and 'start_date' in df_display_risk.columns and 'end_date' in df_display_risk.columns:
             df_display_risk['Start'] = pd.to_datetime(df_display_risk['start_date'])
             df_display_risk['Finish'] = pd.to_datetime(df_display_risk['end_date'])
             df_display_risk['Task'] = df_display_risk['risk_type'].str.replace('_', ' ').str.title()
             fig_timeline = px.timeline(df_display_risk, x_start="Start", x_end="Finish", y="Task", title="Simulated Risk Events Timeline", color="Task", hover_data=["duration_days", "impact_factor", "lead_time_multiplier"])
             fig_timeline.update_yaxes(autorange="reversed"); st.plotly_chart(fig_timeline, use_container_width=True)
             st.dataframe(df_display_risk[['risk_type', 'start_date', 'end_date', 'duration_days', 'impact_factor', 'lead_time_multiplier']].rename(columns={'risk_type':'Risk Type', 'start_date':'Start', 'end_date':'End', 'duration_days':'Duration (d)', 'impact_factor':'Prod. Impact', 'lead_time_multiplier':'LT Multiplier'}), use_container_width=True)
        else:
             st.info("No specific risk events were logged during this simulation run.")

    elif run_button: # Button pressed but no results
         st.warning("Simulation did not produce results. Please check logs or parameters.")