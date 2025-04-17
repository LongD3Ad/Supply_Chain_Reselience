# src/app.py
import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta
import os # Needed for path joining

# --- Configuration and Constants ---
from config import get_config, DATA_DIR, FORECAST_PERIODS # Import necessary items
from data_handler import load_data, get_available_runs, get_filtered_data
# Import tab rendering functions
from tabs import summary_tab, factory_tab, risk_tab, forecast_tab, simulation_tab, recommendations_tab

# --- Page Config and Styling ---
st.set_page_config(layout="wide", page_title="Supply Chain Simulation & Analysis")
# (Copy the CSS st.markdown block from your original script here)
st.markdown("""
    <style>
        .css-18e3th9 {padding-top: 2rem; padding-bottom: 2rem;}
        .css-1d391kg {font-size: 1.1rem;}
        .stSelectbox label {font-weight: bold;}
        .recommendation-critical { border-left: 5px solid #dc3545; padding: 10px; margin-bottom: 10px; background-color: #f8d7da; color: #58151a; }
        .recommendation-high { border-left: 5px solid #fd7e14; padding: 10px; margin-bottom: 10px; background-color: #fff3cd; color: #664d03; }
        .recommendation-medium { border-left: 5px solid #ffc107; padding: 10px; margin-bottom: 10px; background-color: #fff9e0; color: #665100; }
        .recommendation-low { border-left: 5px solid #0dcaf0; padding: 10px; margin-bottom: 10px; background-color: #cff4fc; color: #055160; }
        .recommendation-informational { border-left: 5px solid #6c757d; padding: 10px; margin-bottom: 10px; background-color: #e2e3e5; color: #41464b; }
        .recommendation-unknown { border-left: 5px solid #adb5bd; padding: 10px; margin-bottom: 10px; background-color: #f8f9fa; color: #495057; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Base Config ---
# get_config is cached, so this is efficient
sim_config_data = get_config()

# --- Main Title ---
st.title("Supply Chain Risk Simulation & Analysis Dashboard")
st.markdown("""
Explore simulation runs capturing supply chain performance affected by various risks.
Use the sidebar to select a run and filter data. The tabs below cover overall performance,
factory details, risk analysis, predictive forecasting, live simulation, and AI-powered recommendations.
""")

# --- Sidebar ---
st.sidebar.header("Scenario Selection & Controls")

# --- Example Disruption Scenarios (Informational) ---
# (Copy the Example Disruption Scenarios sidebar section from your original script here)
st.sidebar.subheader("Disruption Scenarios")

# Links to implemented pages
#st.sidebar.page_link("pages/0_Baseline_Operations.py", label="Baseline Operations", icon="📊")
st.sidebar.page_link("pages/1_Suez_Canal_Blockage.py", label="Suez Canal Blockage (2021)", icon="🚢")
st.sidebar.page_link("pages/2_COVID_Vaccine_Inequities.py", label="COVID Vaccine Inequities", icon="💉")
st.sidebar.page_link("pages/3_Russia_Ukraine_War.py", label="Russia-Ukraine War Impacts", icon="⚔️")
st.sidebar.page_link("pages/4_LA_LB_Port_Congestion.py", label="LA/LB Port Congestion (2024)", icon="⚓")

# Display other scenarios as captions (or remove if not needed)
st.sidebar.markdown("---") # Separator if keeping other examples
st.sidebar.caption("- Texas Winter Storm Uri (Infrastructure Failure)")
st.sidebar.caption("- Renesas Factory Fire (Production Outage)")
st.sidebar.caption("- China COVID Zero Policy (Regional Shutdown)")
st.sidebar.caption("- Colonial Pipeline Attack (Cyber, Logistics)")
st.sidebar.caption("- Panama Canal Drought (Logistics Constraint)")
st.sidebar.caption("- Global Semiconductor Shortage (Sustained Imbalance)")

st.sidebar.markdown("---") # Separator before next section

# --- Controls for Historical Data (Generated CSVs) ---
st.sidebar.subheader("Load Generated Simulation Run")
available_files = get_available_runs(DATA_DIR) # Use function from data_handler

df_main = None # Initialize df_main
selected_run_file = None

if not available_files:
    st.sidebar.warning(f"No simulation CSV files found in '{DATA_DIR}'. Run a live simulation or place CSVs there.")
else:
    selected_run_file = st.sidebar.selectbox(
        "Select Run File:",
        available_files,
        index=0,
        key="historical_run_select",
        help="Select a pre-generated CSV file representing a simulation run."
    )
    if selected_run_file:
        file_path = os.path.join(DATA_DIR, selected_run_file)
        # Load data using the function from data_handler
        df_main = load_data(file_path)
    else:
        df_main = None # Handle case where selectbox might somehow return None

# --- Filtering Controls ---
df_filtered = pd.DataFrame() # Initialize df_filtered to be safe
selected_factories = []
start_date_filter = datetime.now().date() - timedelta(days=365) # Default start
end_date_filter = datetime.now().date()   # Default end

if df_main is not None and not df_main.empty:
    available_factories = sorted(df_main['factory_name'].unique().tolist())
    try:
        min_date = df_main['date'].min().date()
        max_date = df_main['date'].max().date()
    except Exception as e:
        st.sidebar.error(f"Error processing dates in the selected file: {e}")
        min_date = start_date_filter # Use default if error
        max_date = end_date_filter
        df_main = None # Invalidate df_main on date error

    if df_main is not None: # Check again after potential invalidation
        st.sidebar.header("Filters for Selected Run")
        selected_factories = st.sidebar.multiselect(
            "Select Factories:",
            options=available_factories,
            default=available_factories,
            key="hist_factory_filter"
            )

        # Initialize dates in session state if not present or invalid
        if 'start_date_filter' not in st.session_state or not isinstance(st.session_state.start_date_filter, type(min_date)) or st.session_state.start_date_filter < min_date or st.session_state.start_date_filter > max_date :
             st.session_state.start_date_filter = min_date
        if 'end_date_filter' not in st.session_state or not isinstance(st.session_state.end_date_filter, type(min_date)) or st.session_state.end_date_filter < min_date or st.session_state.end_date_filter > max_date:
             st.session_state.end_date_filter = max_date


        # Date range selection using session state
        try:
            start_date_filter = st.sidebar.date_input(
                "Start Date:",
                value=st.session_state.start_date_filter,
                min_value=min_date,
                max_value=max_date,
                key='start_date_filter_widget' # Use a different key for widget if needed
            )
            # Update session state if widget changes
            st.session_state.start_date_filter = start_date_filter

            end_date_filter = st.sidebar.date_input(
                "End Date:",
                value=st.session_state.end_date_filter,
                min_value=start_date_filter, # Dynamic min based on start date
                max_value=max_date,
                key='end_date_filter_widget'
            )
            st.session_state.end_date_filter = end_date_filter

        except Exception as e:
             st.sidebar.error(f"Date selection error: {e}")
             # Use last known good dates from session state or defaults
             start_date_filter = st.session_state.start_date_filter
             end_date_filter = st.session_state.end_date_filter


        # Ensure start_date is not after end_date
        if start_date_filter > end_date_filter:
            st.sidebar.error("Error: Start date must be on or before end date.")
            df_filtered = pd.DataFrame() # Use empty dataframe
            st.session_state['recommendations_hist'] = None # Clear recommendations on invalid date
        else:
            # Apply filters using the function from data_handler
            df_filtered = get_filtered_data(df_main, selected_factories, start_date_filter, end_date_filter)
            if df_filtered.empty and selected_factories: # Only warn if factories were selected but yielded no data
                 st.sidebar.warning("No data matches the current filter selection.")
                 # Clear recommendations if filters result in no data
                 st.session_state.pop('recommendations_hist', None)

# Handle cases where df_main is None initially or after errors
elif df_main is None:
     if selected_run_file: # Selection exists but loading failed
         st.sidebar.error(f"Failed to load or process data from {selected_run_file}.")
     # Keep df_filtered empty
     st.session_state.pop('recommendations_hist', None) # Clear any stale recommendations


# --- Clear historical recommendations if selection changed ---
# Moved this logic here after df_filtered is finalized
current_filters_key = (selected_run_file, tuple(sorted(selected_factories)), start_date_filter, end_date_filter)
if 'last_hist_rec_filters' not in st.session_state or current_filters_key != st.session_state.get('last_hist_rec_filters'):
     if 'recommendations_hist' in st.session_state:
          logging.info("Historical data selection changed, clearing cached recommendations.")
          st.session_state.pop('recommendations_hist', None)
     # Update the tracker regardless of whether recommendations existed
     st.session_state['last_hist_rec_filters'] = current_filters_key


# ---------------- Define Tabs ----------------
tab_titles = ["📈 Overall Summary", "🏭 Factory Deep Dive", "🚨 Risk Analysis", "🔮 Predictive Forecast", "🔬 Live Simulation", "🧠 Recommendations"]
tabs = st.tabs(tab_titles)

# ---------- Render Tabs by Calling Functions ----------
with tabs[0]:
    summary_tab.render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter, selected_factories)

with tabs[1]:
    factory_tab.render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter)

with tabs[2]:
    risk_tab.render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter)

with tabs[3]:
    forecast_tab.render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter)

with tabs[4]:
    simulation_tab.render_tab() # This tab manages its own state via session_state

with tabs[5]:
    recommendations_tab.render_tab(df_filtered, selected_run_file, start_date_filter, end_date_filter, selected_factories)


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Supply Chain Digital Twin Dashboard V2.3 (Refactored)")