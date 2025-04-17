# src/tabs/simulation_tab.py
import streamlit as st
import pandas as pd
import plotly.express as px
import logging
from datetime import datetime, timedelta

# Import simulation function and config getter
from simulation.engine import run_factory_simulation
from config import get_config
# Import recommendation functions if triggering from here
from recommendations.gemini import GEMINI_ENABLED, summarize_data_for_gemini, get_gemini_recommendations

def render_tab():
    """Renders the Live Simulation tab."""
    st.header("🔬 Live Factory Simulation")
    st.markdown("Configure and run a live simulation. Results appear below and recommendations in the next tab.")

    sim_config = get_config() # Get the loaded config

    sim_col1, sim_col2 = st.columns([1, 2])

    factory_sim_config = None
    selected_factory_sim = None
    sim_start_datetime = datetime.now() # Default

    with sim_col1:
        # --- Simulation Setup ---
        # (Copy simulation setup controls from original Tab 5 here)
        st.subheader("Simulation Setup")
        factory_options = [f['name'] for f in sim_config.get('factories', [])]
        if not factory_options:
            st.error("Error: No factories defined in simulation configuration.")
        else:
            selected_factory_sim = st.selectbox(
                "Select Factory Profile:",
                options=factory_options,
                key="sim_factory_select"
            )
            factory_sim_config = next((f for f in sim_config['factories'] if f['name'] == selected_factory_sim), None)

        sim_duration_weeks = st.number_input(
            "Simulation Duration (Weeks):",
            min_value=4, max_value=156,
            value=sim_config.get('simulation_defaults', {}).get('sim_duration_weeks', 52),
            step=4,
            key="sim_duration"
        )

        default_start_dt = datetime.now() - timedelta(weeks=sim_duration_weeks)
        sim_start_date_input = st.date_input(
            "Simulation Start Date:",
            value=default_start_dt.date(),
            key="sim_start_date"
        )
        # Ensure conversion to datetime at the beginning of the day
        sim_start_datetime = datetime.combine(sim_start_date_input, datetime.min.time()) if sim_start_date_input else datetime.combine(default_start_dt.date(), datetime.min.time())


        run_button = st.button("🚀 Run Live Simulation", key="run_sim_button", disabled=(factory_sim_config is None))


    with sim_col2:
        # --- Risk Parameter Adjustment ---
        # (Copy risk parameter sliders from original Tab 5 here)
        st.subheader("Adjust Risk Parameters (Overrides Defaults)")
        interactive_risk_params = {}
        default_risk_params = sim_config.get('risk_factors', {})
        for risk, params in default_risk_params.items():
            with st.expander(f"Configure: {risk.replace('_', ' ').title()}", expanded=False):
                prob = st.slider(f"Prob/Day", 0.0, 0.1, params.get("prob_per_day", 0.01), 0.001, format="%.3f", key=f"{risk}_prob", help="Avg daily probability.")
                impact_range = st.slider(f"Impact Factor Range", 0.0, 1.0, tuple(params.get("impact_factor_range", [0.5, 0.8])), 0.01, key=f"{risk}_impact_range", help="Production capacity multiplier range (0=stop, 1=no effect).")
                duration_range = st.slider(f"Duration Range (Days)", 1, 180, tuple(params.get("duration_range_days", [7, 28])), 1, key=f"{risk}_duration_range")
                lt_mult_range = st.slider(f"Lead Time Multiplier Range", 1.0, 5.0, tuple(params.get("lead_time_multiplier_range", [1.0, 1.5])), 0.1, key=f"{risk}_lt_range", help="Factor applied to base lead time.")
                # Store interactive values
                interactive_risk_params[risk] = {
                    "prob_per_day": prob,
                    "impact_factor_range": list(impact_range),
                    "duration_range_days": list(duration_range),
                    "lead_time_multiplier_range": list(lt_mult_range),
                    "mean_days_between_attempts": params.get("mean_days_between_attempts", 90) # Keep from config
                }
        st.caption("Adjust sliders to override default risk parameters for this run.")


    # --- Simulation Execution and Results Display ---
    st.markdown("---")
    if run_button and factory_sim_config:
        # Clear previous results before new run
        st.session_state.pop('sim_results', None)
        st.session_state.pop('sim_risk_log', None)
        st.session_state.pop('recommendations_live', None)

        with st.spinner(f"Running {sim_duration_weeks}-week simulation for {selected_factory_sim}... Please wait."):
            try:
                # Call the simulation function
                df_sim_results, df_risk_log = run_factory_simulation(
                    factory_config=factory_sim_config,
                    risk_factors_config=interactive_risk_params, # Use interactive params
                    sim_duration_weeks=sim_duration_weeks,
                    sim_start_date=sim_start_datetime # Pass datetime
                )
                # Store results in session state
                st.session_state['sim_results'] = df_sim_results
                st.session_state['sim_risk_log'] = df_risk_log
                st.session_state['sim_factory_name'] = selected_factory_sim
                st.success(f"Live simulation for '{selected_factory_sim}' complete!")

                # --- Trigger Recommendation Generation ---
                if GEMINI_ENABLED and not df_sim_results.empty:
                     st.info("Generating AI recommendations...")
                     rec_context = f"Live simulation run for factory '{selected_factory_sim}' over {sim_duration_weeks} weeks, starting {sim_start_datetime.strftime('%Y-%m-%d')}."
                     sim_end_date = sim_start_datetime + timedelta(weeks=sim_duration_weeks)
                     sim_summary = summarize_data_for_gemini(df_sim_results, df_risk_log, factory_sim_config['name'], sim_start_datetime.date(), sim_end_date.date())
                     recommendations = get_gemini_recommendations(sim_summary, rec_context)
                     st.session_state['recommendations_live'] = recommendations
                     st.success("AI recommendations generated. View them in the 'Recommendations' tab.")
                elif not GEMINI_ENABLED:
                     st.warning("Gemini API not configured/enabled. Skipping AI recommendations.")
                elif df_sim_results.empty:
                     st.warning("Simulation ran but produced no data. Skipping recommendations.")

            except Exception as e:
                st.error(f"Simulation run failed: {e}")
                logging.error(f"SimPy simulation error: {e}", exc_info=True)
                # Clear potentially incomplete results
                st.session_state.pop('sim_results', None)
                st.session_state.pop('sim_risk_log', None)
                st.session_state.pop('recommendations_live', None)


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