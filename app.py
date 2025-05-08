# src/app.py
import streamlit as st
import pandas as pd
import logging
from datetime import datetime, date # Import date explicitly
import os
import requests
from dotenv import load_dotenv # To load .env file for backend URL if needed
import json # For formatting simulation details
import plotly.express as px # For plotting sim results in dialog

# --- Environment Loading ---
# Load variables from .env file in the root directory. This should be at the very top.
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# --- Configuration and Constants ---
# Ensure BACKEND_URL is correctly imported from config
from config import get_config, DATA_DIR, FORECAST_PERIODS, BACKEND_URL
from data_handler import load_data, get_available_runs, get_filtered_data
# Import tab rendering functions
from tabs import summary_tab, factory_tab, risk_tab, forecast_tab, simulation_tab, recommendations_tab
# --- Import Modal Display Function ---
from tabs.sim_results_modal import display_simulation_results_modal


# --- Page Config and Styling ---
st.set_page_config(
    layout="wide",
    page_title="Advanced Supply Chain Intelligence",
    page_icon="🤖"
)

# Apply CSS (Combined styles for app and chat)
st.markdown("""
    <style>
        /* Main app styles */
        .main .block-container {padding-top: 1rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem;}
        h1 {color: #1E3A8A; margin-bottom: 0.5rem !important;}
        h2 {color: #1E3A8A; margin-top: 1rem !important;}
        h3 {color: #2563EB; margin-top: 0.75rem !important;}

        /* Sidebar styling for dark mode visibility */
        /* Check Streamlit theme for background color */
        /* [data-testid="stSidebar"] {
            background-color: #333333;
            color: #FFFFFF;
        } */
        /* Style elements within the sidebar for better visibility */
        /* [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] a, [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] .st-ba, .st-bb, .st-br {
            color: #FFFFFF;
        } */


        /* Button styling */
        .stButton>button { background-color: #2563EB; color: white; font-weight: 600; border: none; padding: 0.5rem 1rem; border-radius: 0.375rem; }
        .stButton>button:hover { background-color: #1D4ED8; }
        .stButton>button:disabled { background-color: #9CA3AF; }

        /* Chat Message Styling */
        .stChatMessage { border-radius: 10px; margin-bottom: 10px; padding: 10px; }
        .stChatMessage[data-testid="chatAvatarIcon-assistant"] + div { background-color: #f0f2f6; color: #333; border: 1px solid #e0e0e0; }
        .stChatMessage[data-testid="chatAvatarIcon-user"] + div { background-color: #e7f0fe; border: 1px solid #d0e0fd; }

        /* Recommendation styling */
        .recommendation-critical { border-left: 5px solid #dc3545; padding: 10px; margin-bottom: 10px; background-color: #f8d7da; color: #58151a; border-radius: 0.375rem; }
        .recommendation-high { border-left: 5px solid #fd7e14; padding: 10px; margin-bottom: 10px; background-color: #fff3cd; color: #664d03; border-radius: 0.375rem; }
        .recommendation-medium { border-left: 5px solid #ffc107; padding: 10px; margin-bottom: 10px; background-color: #fff9e0; color: #665100; border-radius: 0.375rem; }
        .recommendation-low { border-left: 5px solid #0dcaf0; padding: 10px; margin-bottom: 10px; background-color: #cff4fc; color: #055160; border-radius: 0.375rem; }
        .recommendation-informational { border-left: 5px solid #6c757d; padding: 10px; margin-bottom: 10px; background-color: #e2e3e5; color: #41464b; border-radius: 0.375rem; }
        .recommendation-unknown { border-left: 5px solid #adb5bd; padding: 10px; margin-bottom: 10px; background-color: #f8f9fa; color: #495057; border-radius: 0.375rem; }

        /* Adjust chat column width */
         div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
             padding-left: 1rem;
         }
    </style>
""", unsafe_allow_html=True)

# --- Load Base Config ---
# get_config loads DEFAULT_SIM_CONFIG which now includes financial_defaults
sim_config_data = get_config()


# --- Initialize Session State (Consolidated & Updated) ---
st.session_state.setdefault('active_tab', '📈 Overall Summary') # Track active tab
st.session_state.setdefault('latest_news_summary', None)      # Cache fetched news summary text
st.session_state.setdefault('latest_extracted_risks', [])      # Cache risks from news
st.session_state.setdefault('latest_risk_severity', {})     # Cache severity from news

# Simulation Results Data (for modal and recommendations)
st.session_state.setdefault('simulation_plot_data', None) # Plotting subset
st.session_state.setdefault('simulation_results_for_rec', None) # Full data for recs
st.session_state.setdefault('simulation_risk_summary', None) # Applied risk info
st.session_state.setdefault('financial_summary_from_sim', None) # Calculated financial summary
st.session_state.setdefault('recommendations_from_sim', None) # Generated recommendations list

# Chat Pane Visibility & History
st.session_state.setdefault('chat_open', False) # Controls if the chat pane is visible
st.session_state.setdefault('show_sim_modal', False) # Controls if the simulation results modal is open
st.session_state.setdefault('messages', [{"role": "assistant", "content": "👋 Hello! Ask me a question, fetch news, or run a simulation."}]) # Chat messages

# Dashboard Filter State (used by sidebar)
st.session_state.setdefault('start_date_filter', None)
st.session_state.setdefault('end_date_filter', None)
st.session_state.setdefault('hist_factory_filter', None)
st.session_state.setdefault('recommendations_hist', None) # Historical recs (if needed)
st.session_state.setdefault('last_hist_rec_filters', None)
st.session_state.setdefault('prev_hist_rec_filters_key', None)

# Cost Parameter Overrides State (used by AI assistant UI)
st.session_state.setdefault('override_holding_cost_unit_weekly', None)
st.session_state.setdefault('override_stockout_cost_unit', None)
st.session_state.setdefault('override_base_logistics_cost_unit', None)
st.session_state.setdefault('override_production_cost_unit', None)

st.session_state.setdefault('fetched_market_data', None) # Cache fetched market data
st.session_state.setdefault('pending_simulation_payload', None)

# --- Helper Function to Toggle Chat Visibility ---
def toggle_chat():
    st.session_state.chat_open = not st.session_state.chat_open
    # Rerun to update layout based on chat_open state
    # st.rerun() # No explicit rerun needed if button is in header

# --- Main Layout ---
# Use columns to create the main content area and the chat side pane
main_col, chat_col = st.columns([4, 1])

# --- Main Content Area (Dashboard) ---
with main_col:
    # --- App Header ---
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("🔗 Advanced Supply Chain Intelligence")
        st.markdown("""
            <div style="margin-bottom: 1.5rem;">
                <span style="background-color: #DBEAFE; color: #1E40AF; font-weight: 600; padding: 0.25rem 0.5rem; border-radius: 0.25rem; margin-right: 0.5rem;">MVP Demo v0.4</span>
                <span style="color: #4B5563; font-size: 0.9rem;">AI-powered analytics with persistent chat and financial impact</span>
            </div>
        """, unsafe_allow_html=True)
    with header_col2:
        # Button to toggle the visibility of the chat side pane
        toggle_button_label = "Hide AI Assistant" if st.session_state.chat_open else "Show AI Assistant"
        if st.button(f"🤖 {toggle_button_label}", use_container_width=True, on_click=toggle_chat, key="toggle_chat_hdr"):
             pass # The on_click handler manages the state and rerun

    # --- Sidebar (Moved the actual sidebar content here) ---
    # The st.sidebar commands render content in the sidebar regardless of where they are called
    # This structure helps keep the app.py main flow cleaner
    st.sidebar.header("Dashboard Controls")

    with st.sidebar.expander("📚 Case Studies", expanded=False):
        st.markdown("Explore historical disruptions:")
        # Page links... ensure your pages folder exists
        st.page_link("pages/1_Suez_Canal_Blockage.py", label="Suez Canal Blockage (2021)", icon="🚢")
        st.page_link("pages/2_COVID_Vaccine_Inequities.py", label="COVID Vaccine Inequities", icon="💉")
        st.page_link("pages/3_Russia_Ukraine_War.py", label="Russia-Ukraine War Impacts", icon="⚔️")
        st.page_link("pages/4_LA_LB_Port_Congestion.py", label="LA/LB Port Congestion (2024)", icon="⚓")

    # Data Selection & Filtering (Generated Runs Only for MVP Demo)
    # This section populates df_filtered, selected_factories, start/end_date_filter
    df_main = None
    df_filtered = pd.DataFrame()
    selected_run_file = None
    selected_factories = st.session_state.get('hist_factory_filter',[]) # Default to state value
    start_date_filter = st.session_state.get('start_date_filter')
    end_date_filter = st.session_state.get('end_date_filter')


    with st.sidebar.expander("📊 Data Selection & Filters", expanded=True):
        available_files = get_available_runs(DATA_DIR)
        if available_files:
            selected_run_file = st.selectbox(
                "Select Simulation Dataset:", available_files, index=0, key="historical_run_select",
                help="Choose a saved simulation dataset to analyze"
            )
            if selected_run_file:
                try:
                    file_path = os.path.join(DATA_DIR, selected_run_file)
                    df_main = load_data(file_path)
                except Exception as e:
                    st.error(f"Failed to load dataset: {str(e)}")
                    logging.error(f"Error loading {selected_run_file}: {e}", exc_info=True)
                    df_main = None
        else: st.warning("No simulation data found in simulation_data directory.")

        # --- Filtering Logic (Only if df_main is successfully loaded) ---
        if df_main is not None and not df_main.empty:
            try:
               min_date_dt, max_date_dt = df_main['date'].min(), df_main['date'].max()
               min_date, max_date = min_date_dt.date(), max_date_dt.date() # Get date objects
               available_factories = sorted(df_main['factory_name'].unique().tolist())

               # --- Factory Filter Widget Setup ---
               # Use existing state for default if available and valid, else use all available
               current_selection_from_state = st.session_state.get("hist_factory_filter")
               default_factories = available_factories # Start with all as default
               if isinstance(current_selection_from_state, list):
                   # Validate factories from state against available ones
                   validated_selection = [f for f in current_selection_from_state if f in available_factories]
                   if validated_selection: default_factories = validated_selection
                   elif current_selection_from_state: logging.warning("Factories in state not in available data.") # Log if state had values but they were invalid

               selected_factories_widget = st.multiselect(
                   "Select Factories:", available_factories, default=default_factories,
                   key="hist_factory_filter_widget" # Widget key
               )
               # Update state immediately when widget changes
               if selected_factories_widget != st.session_state.get('hist_factory_filter'):
                   st.session_state.hist_factory_filter = selected_factories_widget
                   # No need for explicit rerun here, filter below uses the widget value

               selected_factories = selected_factories_widget # Use widget value for filtering


               # --- Date Filter Widget Setup ---
               state_start = st.session_state.get('start_date_filter')
               state_end = st.session_state.get('end_date_filter')

               # Robustly determine default values for date widgets
               widget_start_default = min_date
               if isinstance(state_start, date): widget_start_default = state_start
               try: widget_start_default = max(min_date, widget_start_default) # Ensure bounds
               except TypeError: widget_start_default = min_date # Handle if min_date is not date type

               widget_end_default = max_date
               if isinstance(state_end, date): widget_end_default = state_end
               try: widget_end_default = min(max_date, widget_end_default) # Ensure bounds
               except TypeError: widget_end_default = max_date # Handle if max_date is not date type

               # Ensure start is not after end default
               if widget_start_default > widget_end_default: widget_start_default = widget_end_default

               st.markdown("**Date Range**")
               start_date_widget = st.date_input(
                   "From:", value=widget_start_default, min_value=min_date, max_value=max_date,
                   key='start_date_filter_widget' # Widget key
               )
               end_date_widget = st.date_input(
                   "To:", value=widget_end_default, min_value=start_date_widget, max_value=max_date, # Min depends on start widget
                   key='end_date_filter_widget' # Widget key
               )

               # Update state immediately when widgets change
               if start_date_widget != st.session_state.get('start_date_filter'):
                   st.session_state.start_date_filter = start_date_widget
                   # No explicit rerun
               if end_date_widget != st.session_state.get('end_date_filter'):
                   st.session_state.end_date_filter = end_date_widget
                   # No explicit rerun

               # Use the latest widget values for filtering
               start_date_filter = start_date_widget
               end_date_filter = end_date_widget


               # --- Apply Filters ---
               if start_date_filter > end_date_filter:
                   st.sidebar.error("Error: Start date must be on or before end date.")
                   df_filtered = pd.DataFrame() # Ensure empty
               else:
                    df_filtered = get_filtered_data(df_main, selected_factories, start_date_filter, end_date_filter)
                    if df_filtered.empty and selected_factories:
                         st.sidebar.warning("No data matches the current filter selection.")

            except Exception as e:
               st.error(f"Filter Setup Error: {e}")
               logging.error(f"Error setting filters: {e}", exc_info=True)
               df_filtered = pd.DataFrame() # Ensure empty on error
               selected_factories = [] # Ensure empty
               start_date_filter = None # Ensure None
               end_date_filter = None # Ensure None
        else:
             # If df_main is None (loading failed or no files), ensure filtered is empty and filters are None
             if selected_run_file: logging.warning(f"df_main is None for {selected_run_file}")
             df_filtered = pd.DataFrame()
             selected_factories = []
             start_date_filter = None
             end_date_filter = None


    # --- Sidebar Footer ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """<div style="text-align: center; color: #6B7280; font-size: 0.8rem;"><p>© 2025 SupplyChain AI</p><p>Version 0.4.0</p></div>""", # Version bump
        unsafe_allow_html=True
    )

    # --- Recommendation Cache Clearing Logic (Keep this) ---
    # Clears historical recs if dataset or filter changes
    current_run_display = selected_run_file if selected_run_file else "N/A"
    # Convert date objects to strings or None for cache key consistency
    start_date_key = start_date_filter.isoformat() if isinstance(start_date_filter, date) else None
    end_date_key = end_date_filter.isoformat() if isinstance(end_date_filter, date) else None
    current_filters_key = (current_run_display, tuple(sorted(selected_factories)), start_date_key, end_date_key)
    if current_filters_key != st.session_state.get('prev_hist_rec_filters_key'):
        if 'recommendations_hist' in st.session_state:
            logging.info("Filters changed, clearing historical recommendations.")
            st.session_state.pop('recommendations_hist', None)
        st.session_state['prev_hist_rec_filters_key'] = current_filters_key


    # --- Tab Definitions & Active Tab Tracking ---
    # Define main tabs for the dashboard
    tab_titles = ["📈Overall Summary", "🏭 Factory DeepDive", "🚨 Risk Analysis", "🔮Predictive Forecast", "🔬Live Simulation", "🧠 Recommendation Engine"]
    tabs = st.tabs(tab_titles)

    # --- Render Tabs & Update Active Tab State ---
    # Pass necessary data (df_filtered, filter values) to each tab
    with tabs[0]: st.session_state.active_tab = tab_titles[0]; summary_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter, selected_factories)
    with tabs[1]: st.session_state.active_tab = tab_titles[1]; factory_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter)
    with tabs[2]: st.session_state.active_tab = tab_titles[2]; risk_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter)
    with tabs[3]: st.session_state.active_tab = tab_titles[3]; forecast_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter)
    with tabs[4]: st.session_state.active_tab = tab_titles[4]; simulation_tab.render_tab()
    with tabs[5]: st.session_state.active_tab = tab_titles[5]; recommendations_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter, selected_factories)


# --- Chat Side Pane ---
# This column only renders its content if st.session_state.chat_open is True
with chat_col:
    if st.session_state.chat_open:
        st.markdown("##### 🤖 AI Assistant")
        action_triggered = False # Flag used ONLY to trigger rerun AFTER chat input or button click

        # --- V V V INSERT THE NEW PROCESSING BLOCK HERE V V V ---
        # --- LOGIC TO EXECUTE PENDING SIMULATION ---
        # This block will run on reruns if a simulation was triggered.
        if st.session_state.get('pending_simulation_payload') is not None:
            sim_payload_to_process = st.session_state.pending_simulation_payload
            # Consume the trigger immediately to prevent re-execution if something fails before reset
            st.session_state.pending_simulation_payload = None
            action_triggered = True # Ensure UI updates after processing

            sim_endpoint = f"{BACKEND_URL}/run_simulation"
            # Retrieve market_data from the payload if it was stored there
            market_data_from_payload = sim_payload_to_process.get("market_data") 

            # Display spinner while processing
            with st.spinner("Simulation in progress... Fetching recommendations... Please wait."):
                try:
                    logging.info(f"Processing pending simulation. Sending sim request: { {k:v for k,v in sim_payload_to_process.items() if k != 'market_data' or v is not None} }") # Log carefully
                    response = requests.post(sim_endpoint, json=sim_payload_to_process, timeout=240)
                    response.raise_for_status()
                    sim_data = response.json()

                    st.session_state.simulation_plot_data = sim_data.get('timeseries_data')
                    st.session_state.simulation_results_for_rec = sim_data.get('full_simulation_data')
                    st.session_state.simulation_risk_summary = sim_data.get('risk_summary')
                    st.session_state.financial_summary_from_sim = sim_data.get('financial_summary')

                    if st.session_state.simulation_results_for_rec:
                        rec_endpoint = f"{BACKEND_URL}/get_simulation_recommendations"
                        rec_request_payload = {
                            "simulation_data": st.session_state.simulation_results_for_rec,
                            "risk_summary": st.session_state.simulation_risk_summary,
                            "market_data": market_data_from_payload, # Use market data passed with sim payload
                            "financial_summary": st.session_state.financial_summary_from_sim
                        }
                        logging.info(f"Sending rec request. Payload keys: {rec_request_payload.keys()}")
                        rec_response = requests.post(rec_endpoint, json=rec_request_payload, timeout=120)
                        rec_response.raise_for_status()
                        recs = rec_response.json().get("recommendations", [])
                        st.session_state.recommendations_from_sim = recs

                        # Update chat and prepare modal
                        st.session_state.messages.append({"role":"assistant", "content": f"✅ Simulation complete! Found {len(recs)} recommendations. Results ready."})
                        st.session_state.show_sim_modal = True
                    else:
                        st.session_state.messages.append({"role":"assistant", "content": f"⚠️ Simulation run incomplete or produced no data for analysis."})
                        if st.session_state.simulation_plot_data or st.session_state.financial_summary_from_sim or st.session_state.simulation_risk_summary:
                            st.session_state.show_sim_modal = True
                        else:
                            st.warning("Simulation did not return any displayable data.")

                except Exception as e:
                    error_msg = f"**Simulation/Recommendation Error:** {str(e)}"
                    logging.error(f"Sim or Rec API call error during pending execution: {e}", exc_info=True)
                    st.session_state.messages.append({"role":"assistant", "content": error_msg})
                    # Clear potentially incomplete results
                    st.session_state.simulation_plot_data = None
                    st.session_state.simulation_results_for_rec = None
                    st.session_state.simulation_risk_summary = None
                    st.session_state.financial_summary_from_sim = None
                    st.session_state.recommendations_from_sim = None
                    st.session_state.show_sim_modal = False
            
            # If modal needs to show, or any other major UI update from this block, a rerun is essential.
            # The action_triggered flag handled by the end of chat_col will manage this.
            # Or, if modal needs to pop up immediately without waiting for chat input or other button:
            if st.session_state.get("show_sim_modal"):
                 st.rerun() # This ensures the modal dialog logic is triggered.

        # --- Simulation Setup & Actions ---
        st.markdown("---")
        st.markdown("**Simulation Setup**")

        # Fetch News & Analyze Risks (UI)
        news_query = st.text_input("News Query:", value="supply chain disruption", key="side_news_query", label_visibility="collapsed", placeholder="Topic for news briefing...")
        if st.button("📰 Fetch News & Analyze Risks", key="side_fetch_news", use_container_width=True):
            if news_query:
                 action_triggered = True # Trigger rerun
                 brief_endpoint = f"{BACKEND_URL}/fetch-news"; st.session_state.latest_news_summary = None; st.session_state.latest_extracted_risks = []; st.session_state.latest_risk_severity = {}
                 try:
                     with st.spinner("Fetching & analyzing news..."):
                          response = requests.post(brief_endpoint, json={"query": news_query}, timeout=45); response.raise_for_status()
                          news_data = response.json() # Expects {"summary":..., "risk_events":..., "risk_severity":...}
                          summary = news_data.get('summary','N/A'); risks = news_data.get('risk_events',[]); severity = news_data.get('risk_severity',{})
                          st.session_state.latest_news_summary = summary; st.session_state.latest_extracted_risks = risks; st.session_state.latest_risk_severity = severity
                          # Add result to chat history
                          st.session_state.messages.append({"role":"assistant", "content":f"**News Briefing ({news_query}):**\n{summary}\n\n**Detected Risks:** {', '.join(risks) if risks else 'None'}"})
                 except Exception as e:
                     error_msg = f"**News Error:** {str(e)}"
                     st.session_state.messages.append({"role":"assistant", "content": error_msg})
                     logging.error(f"News Fetch/Analyze Error: {e}", exc_info=True)
                     st.session_state.latest_news_summary = None; st.session_state.latest_extracted_risks = []; st.session_state.latest_risk_severity = {} # Clear state on error
            else: st.toast("Enter news query.")

        # Risk Event Selection (UI)
        available_risks = ["port congestion", "strike", "weather disruption", "oil price spike", "chip shortage", "transportation issues", "material shortage", "cyber attack", "geopolitical tension"]
        # Use extracted risks as default selection if available
        default_selected_risks = st.session_state.get('latest_extracted_risks', [])
        selected_risks_for_sim = st.multiselect("Select Risk Events to Simulate:", available_risks, default=default_selected_risks, key="sim_risk_select_v2")

        # Severity Input (UI)
        risk_severity_for_sim = {} # Dict to hold risk_name -> severity_value for selected risks
        if selected_risks_for_sim:
             st.caption("Adjust Severity (0.1-1.0):")
             # Display sliders in columns if more than a few risks selected
             num_cols_severity = 2 if len(selected_risks_for_sim) > 2 else len(selected_risks_for_sim)
             severity_cols = st.columns(num_cols_severity) if num_cols_severity > 0 else [] # Handle 0 risks case

             for i, risk in enumerate(selected_risks_for_sim):
                  default_sev = st.session_state.get('latest_risk_severity', {}).get(risk, 0.5) # Use extracted or default 0.5
                  col_index = i % num_cols_severity if num_cols_severity > 0 else 0

                  with (severity_cols[col_index] if num_cols_severity > 0 else st): # Use column context if available
                       risk_severity_for_sim[risk] = st.slider(
                           f"{risk[:10]}...", 0.1, 1.0, default_sev, 0.1,
                           key=f"sim_sev_{risk}_v2", label_visibility="collapsed", help=f"Severity for {risk}"
                       )
        st.markdown("---")

        # Cost Parameter Overrides UI
        st.markdown("**Cost Parameter Overrides**")
        st.caption("Leave blank to use default config.")
        cost_cols = st.columns(2)
        with cost_cols[0]:
             holding_cost_input = st.text_input("Holding Cost / Unit / Week:", value=st.session_state.get('override_holding_cost_unit_weekly'), key='input_holding_cost_v2', placeholder='e.50')
             stockout_cost_input = st.text_input("Stockout Cost / Unit:", value=st.session_state.get('override_stockout_cost_unit'), key='input_stockout_cost_v2', placeholder='e.g., 10.00')
        with cost_cols[1]:
             logistics_cost_input = st.text_input("Base Logistics Cost / Unit:", value=st.session_state.get('override_base_logistics_cost_unit'), key='input_logistics_cost_v2', placeholder='e.g., 2.00')
             production_cost_input = st.text_input("Production Cost / Unit:", value=st.session_state.get('override_production_cost_unit'), key='input_production_cost_v2', placeholder='e.g., 5.00')

        # Update Cost Override State from Inputs
        def parse_float_input(input_str):
            if not input_str or str(input_str).strip() == "": return None # Treat empty/whitespace as None
            try: return float(input_str)
            except (ValueError, TypeError): return input_str # Return invalid string for potential feedback

        st.session_state.override_holding_cost_unit_weekly = parse_float_input(holding_cost_input)
        st.session_state.override_stockout_cost_unit = parse_float_input(stockout_cost_input)
        st.session_state.override_base_logistics_cost_unit = parse_float_input(logistics_cost_input)
        st.session_state.override_production_cost_unit = parse_float_input(production_cost_input)

        # Optional: Add validation feedback here if inputs are invalid strings
        # if isinstance(st.session_state.override_holding_cost_unit_weekly, str): st.warning("Invalid input for Holding Cost.")

        st.markdown("---")

        # Run Simulation Button
        if st.button("▶️ Run Parameterized Sim", key="side_run_param_sim_v3_fixed", use_container_width=True, help="Run simulation with selected risks and costs"):
            action_triggered = True 
            
            # Clear previous results from session state
            st.session_state.simulation_plot_data = None
            st.session_state.simulation_results_for_rec = None
            st.session_state.simulation_risk_summary = None
            st.session_state.financial_summary_from_sim = None
            st.session_state.recommendations_from_sim = None
            st.session_state.show_sim_modal = False # Ensure modal is reset

            # --- Gather Cost Overrides for Payload ---
            cost_overrides_payload = {}
            if isinstance(st.session_state.override_holding_cost_unit_weekly, float):
                 cost_overrides_payload["holding_cost_unit_weekly"] = st.session_state.override_holding_cost_unit_weekly
            if isinstance(st.session_state.override_stockout_cost_unit, float):
                 cost_overrides_payload["stockout_cost_unit"] = st.session_state.override_stockout_cost_unit
            if isinstance(st.session_state.override_base_logistics_cost_unit, float):
                 cost_overrides_payload["base_logistics_cost_unit"] = st.session_state.override_base_logistics_cost_unit
            if isinstance(st.session_state.override_production_cost_unit, float):
                 cost_overrides_payload["production_cost_unit"] = st.session_state.override_production_cost_unit
            cost_overrides_for_request = cost_overrides_payload if cost_overrides_payload else None

            # Get market data (which should have been fetched by its own button and stored in session state)
            market_data_for_sim = st.session_state.get("fetched_market_data")

            # Prepare the simulation request payload
            sim_request_payload = {
                "risk_events": selected_risks_for_sim,
                "risk_severity": risk_severity_for_sim, 
                "cost_overrides": cost_overrides_for_request,
                "market_data": market_data_for_sim 
            }

            # Set the payload in session_state to be processed on the next rerun
            st.session_state.pending_simulation_payload = sim_request_payload
            
            # Add a message to the chat history indicating sim is queued
            st.session_state.messages.append({"role":"assistant", "content":f"⏳ Queued simulation with selected parameters. Processing shortly..."})
            
            # Rerun to show the "Queued..." message and to trigger the pending simulation logic block
            st.rerun()

            # --- AFTER RERUN: Make the Backend Calls ---
            # This code runs on the rerun triggered above.
            # It should only execute ONCE per button click sequence.
            # We need to avoid calling the API multiple times if the state changes again during the calls.
            # Streamlit usually handles this correctly IF action_triggered is True and we rerun once.

            try:
                 # Make the backend call to run simulation
                 logging.info(f"Sending sim request payload (costs, risks, severity, market included): {sim_request_payload.keys()}")
                 response = requests.post(sim_endpoint, json=sim_request_payload, timeout=240) # Increase timeout
                 response.raise_for_status() # Check for HTTP errors
                 sim_data = response.json() # Expects SimulationResultResponse

                 # Store ALL results from the simulation response
                 st.session_state.simulation_plot_data = sim_data.get('timeseries_data')
                 st.session_state.simulation_results_for_rec = sim_data.get('full_simulation_data')
                 st.session_state.simulation_risk_summary = sim_data.get('risk_summary')
                 st.session_state.financial_summary_from_sim = sim_data.get('financial_summary')


                 # Fetch recommendations automatically after simulation
                 if st.session_state.simulation_results_for_rec:
                     rec_endpoint = f"{BACKEND_URL}/get_simulation_recommendations"
                     # Pass ALL relevant context to recommendation endpoint
                     rec_request_payload = {
                         "simulation_data": st.session_state.simulation_results_for_rec,
                         "risk_summary": st.session_state.simulation_risk_summary,
                         "market_data": market_data_for_sim, # Pass market data fetched earlier
                         "financial_summary": st.session_state.financial_summary_from_sim
                     }
                     with st.spinner("Generating recommendations..."):
                         logging.info(f"Sending rec request payload (context included): {rec_request_payload.keys()}")
                         rec_response = requests.post(rec_endpoint, json=rec_request_payload, timeout=120) # Increase timeout
                         rec_response.raise_for_status()
                         recs = rec_response.json().get("recommendations", [])
                         st.session_state.recommendations_from_sim = recs

                     # Add success message to chat
                     st.session_state.messages.append({"role":"assistant", "content": f"✅ Simulation complete! Found {len(recs)} recommendations. Results ready."})
                     # Trigger modal display AFTER recommendations are fetched
                     st.session_state.show_sim_modal = True

                 else: # Simulation did not produce data
                     st.session_state.messages.append({"role":"assistant", "content": f"⚠️ Simulation run incomplete or produced no data for analysis."})
                     # Still trigger modal display if plot data or financial summary exists
                     if st.session_state.simulation_plot_data or st.session_state.financial_summary_from_sim or st.session_state.simulation_risk_summary:
                         st.session_state.show_sim_modal = True
                     else: # No data at all
                         st.warning("Simulation did not return any displayable data.")

            except Exception as e:
                # This catches errors from sim run API call or rec API call
                error_msg = f"**Simulation/Recommendation Error:** {str(e)}"
                logging.error(f"Sim or Rec API call error: {e}", exc_info=True)
                # Append error message to chat
                st.session_state.messages.append({"role":"assistant", "content": error_msg})
                # Clear potentially incomplete results
                st.session_state.simulation_plot_data = None
                st.session_state.simulation_results_for_rec = None
                st.session_state.simulation_risk_summary = None
                st.session_state.financial_summary_from_sim = None
                st.session_state.recommendations_from_sim = None
                st.session_state.show_sim_modal = False # Ensure modal doesn't open if error

            # Trigger final rerun after all API calls and state updates are done
            # This rerun will display the final chat messages and trigger modal if show_sim_modal is True
            st.rerun()


        # Fetch Market Data Button (Keep existing logic, UPDATE KEY)
        # Add a separate button specifically for fetching market data for context
        if st.button("📈 Fetch Market Data", key="side_fetch_market_data", use_container_width=True):
            action_triggered = True # Trigger rerun
            market_endpoint = f"{BACKEND_URL}/fetch-market-data"
            market_symbols = "OIL,FX,SHIPPING" # Example symbols - could be made selectable later

            st.session_state.fetched_market_data = None # Clear previous market data
            try:
                with st.spinner(f"Fetching market data for {market_symbols}..."):
                    # Use GET for market data as per backend endpoint definition @app.get
                    response = requests.get(f"{market_endpoint}?symbols={market_symbols}", timeout=30) # Pass symbols as query param
                    response.raise_for_status() # Check for HTTP errors (like 404!)
                    market_data = response.json().get("market_data") # Expect {"market_data": {...}}
                    st.session_state.fetched_market_data = market_data # Store fetched data

                    msg_content = f"📈 **Market Data Briefing ({market_symbols}):**\n"
                    if market_data:
                        for symbol, data in market_data.items():
                             if data.get("values") and len(data["values"]) >= 2:
                                  start_val = data["values"][0]
                                  end_val = data["values"][-1]
                                  change_pct = ((end_val / start_val) - 1) * 100 if start_val != 0 else float('inf')
                                  trend = "⬆️ Increased" if change_pct > 0.1 else ("⬇️ Decreased" if change_pct < -0.1 else "➡️ Stable")
                                  msg_content += f"- **{symbol}:** {trend} ({change_pct:.1f}%) over last 30 days.\n"
                             else: msg_content += f"- **{symbol}:** No data available.\n"
                    else: msg_content += "No market data received."

                    st.session_state.messages.append({"role":"assistant", "content": msg_content})

            except requests.exceptions.RequestException as e:
                error_msg = f"**Market Data Error:** Failed to fetch ({str(e)})."
                st.session_state.messages.append({"role":"assistant", "content": error_msg})
                logging.error(f"Market Data Fetch Error: {e}", exc_info=True)
                st.session_state.fetched_market_data = None # Ensure clear state on error
            except Exception as e: # Catch JSON decode or other unexpected errors
                 error_msg = f"**Market Data Error:** Unexpected error ({str(e)})."
                 st.session_state.messages.append({"role":"assistant", "content": error_msg})
                 logging.error(f"Market Data Fetch Unexpected Error: {e}", exc_info=True)
                 st.session_state.fetched_market_data = None

            st.rerun() # Rerun after fetching to update chat and potentially enable Sim button

        # Get Recommendations Button (Keep existing logic, UPDATE KEY)
        # The button is now only enabled AFTER a sim has produced full data
        rec_button_disabled = st.session_state.get('simulation_results_for_rec') is None
        if st.button("💡 Get Recs", key="side_get_recs_v3", use_container_width=True, disabled=rec_button_disabled):
             action_triggered = True # Trigger rerun
             rec_endpoint = f"{BACKEND_URL}/get_simulation_recommendations"
             sim_payload_for_recs = st.session_state.get('simulation_results_for_rec') # Get full data

             if sim_payload_for_recs:
                try:
                    with st.spinner("Generating recommendations..."):
                         # Send the full simulation data and ALL context
                         rec_request_payload = {
                             "simulation_data": sim_payload_for_recs,
                             "risk_summary": st.session_state.get("simulation_risk_summary"), # Pass risk context
                             "market_data": st.session_state.get("fetched_market_data"), # Pass market context
                             "financial_summary": st.session_state.get("financial_summary_from_sim") # Pass financial context
                         }
                         logging.info(f"Sending rec request payload (context included): {rec_request_payload.keys()}")
                         response = requests.post(rec_endpoint, json=rec_request_payload, timeout=90)
                         response.raise_for_status()
                         recs = response.json().get("recommendations", []) # Expecting list of Recommendation dicts
                         st.session_state.recommendations_from_sim = recs # Store recs
                         st.session_state.messages.append({"role":"assistant", "content":f"Generated {len(recs)} recommendations:", "type":"recommendations", "data":recs})
                except Exception as e:
                     error_msg = f"**Recommendation Error:** {str(e)}"
                     st.session_state.messages.append({"role":"assistant", "content": error_msg})
                     logging.error(f"Rec Gen Error: {e}", exc_info=True)
                     st.session_state.recommendations_from_sim = None # Clear on error
                else: st.session_state.messages.append({"role":"assistant", "content":"Error: No simulation data found for recommendations."})

             st.rerun() # Rerun after attempting recommendations


        # Chat History Container (Keep existing logic)
        st.markdown("---")
        st.markdown("##### Conversation")
        # Set fixed height for scrollable chat history
        chat_container = st.container(height=250, border=True) # Adjusted height and added border
        with chat_container:
            for message in st.session_state.messages:
                 with st.chat_message(message["role"]):
                      st.markdown(message["content"], unsafe_allow_html=True)
                      # Note: Plotting/Recs display is now in the modal (sim_results_modal.py)
                      # Remove or comment out the inline plotting/recs display logic here if it exists


        # Chat Input (Keep existing logic)
        if prompt := st.chat_input("Ask the assistant...", key="side_pane_chat_input_v4"):
            action_triggered = True # Trigger rerun
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Prepare payload with ALL relevant context for the chat API
            start_iso = start_date_filter.isoformat() if isinstance(start_date_filter, date) else None
            end_iso = end_date_filter.isoformat() if isinstance(end_date_filter, date) else None
            page_ctx_payload = {"selected_factories": selected_factories, "date_range": [start_iso, end_iso]}

            payload = {
                "text": prompt,
                "active_tab": st.session_state.get('active_tab', 'Unknown'),
                "news_summary": st.session_state.get('latest_news_summary'), # Include news summary
                "page_context": page_ctx_payload, # Include dashboard context
                # Could also include summaries of sim results, risks, financials if chat needs that detail
                # "sim_summary": st.session_state.get('financial_summary_from_sim'), # Example
            }
            api_endpoint = f"{BACKEND_URL}/chat"; ai_response_content = "Error contacting AI backend."
            try:
                 with st.spinner("AI thinking..."):
                      logging.info(f"Sending payload to /chat: { {k:v[:50]+'...' if isinstance(v,str) and v is not None and len(v)>50 else v for k,v in payload.items()} }")
                      response = requests.post(api_endpoint, json=payload, timeout=60); response.raise_for_status()
                      ai_response_content = response.json().get("response", "Error: Invalid response from AI.")
            except Exception as e:
                 ai_response_content = f"**AI Communication Error:** {str(e)}"
                 logging.error(f"Chat API Error: {e}", exc_info=True)

            st.session_state.messages.append({"role": "assistant", "content": ai_response_content})
            # No explicit rerun needed here, chat_input handles it

        # --- Final Rerun ---
        # This single rerun is triggered by any button click.
        # It ensures the state updates are processed and the UI reflects them.
        if action_triggered:
             st.rerun()


# --- Simulation Results Modal Display (Handles the modal window) ---
# Define the function that builds the dialog's UI
@st.dialog("Simulation Analysis Report")
def show_simulation_report_modal():
    """Function decorated by st.dialog to display the simulation results modal."""
    logging.info("Rendering Simulation Analysis Report Modal.")
    # Call the helper function that contains the actual UI elements (tabs, charts, etc.)
    # Pass the data stored in session state
    display_simulation_results_modal(
        st.session_state.get("simulation_plot_data"),
        st.session_state.get("recommendations_from_sim"),
        st.session_state.get("simulation_risk_summary"),
        st.session_state.get("financial_summary_from_sim")
    )
    # Add the close button inside the function decorated by st.dialog
    if st.button("Close Report", key="close_sim_modal_btn_v4"): # Unique key for this widget
        st.session_state.show_sim_modal = False # Update state to close modal
        st.rerun() # Rerun to remove the dialog from the screen

# --- Trigger Modal Display ---
# This block checks the flag and CALLS the decorated function to show the dialog
# This block should be placed after the function definition
if st.session_state.get("show_sim_modal", False):
    # Call the function defined above. DO NOT use 'with' here.
    show_simulation_report_modal()

# --- END OF app.py Script ---