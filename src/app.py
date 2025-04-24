# src/app.py
import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import io      # Keep for potential future use
import requests # To talk to backend
from dotenv import load_dotenv # To load .env file for backend URL if needed
import sys
import json # For formatting simulation details
import plotly.express as px # For plotting sim results in dialog
from config import BACKEND_URL
# --- Path Setup (If needed, usually not required with standard structure) ---
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

# --- Environment Loading ---
load_dotenv() # Loads variables from .env file in the root directory

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# --- Configuration and Constants ---
# Ensure BACKEND_URL is correctly imported or retrieved from env
from config import get_config, DATA_DIR, FORECAST_PERIODS, BACKEND_URL
from data_handler import load_data, get_available_runs, get_filtered_data
# Import tab rendering functions
from tabs import summary_tab, factory_tab, risk_tab, forecast_tab, simulation_tab, recommendations_tab

# --- Page Config and Styling ---
st.set_page_config(
    layout="wide",
    page_title="Advanced Supply Chain Intelligence", # Updated Title for Demo
    page_icon="🤖" # Updated Icon
)

# Apply CSS (Keep your preferred styles)
st.markdown("""
    <style>
        /* Main app styles */
        .main .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
        h1 {color: #1E3A8A; margin-bottom: 0.5rem !important;}
        h2 {color: #1E3A8A; margin-top: 1rem !important;}
        h3 {color: #2563EB; margin-top: 0.75rem !important;}
        /* Sidebar styling for dark mode visibility */
        [data-testid="stSidebar"] {
            background-color: #333333; /* Dark background color */
            color: #FFFFFF; /* Light text color */
        }
        /* Style elements within the sidebar for better visibility */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] .st-ba, /* Streamlit elements like select boxes */
        [data-testid="stSidebar"] .st-bb,
        [data-testid="stSidebar"] .st-br {
            color: #FFFFFF;
        }
        /* Button styling */
        .stButton>button { background-color: #2563EB; color: white; font-weight: 600; border: none; padding: 0.5rem 1rem; border-radius: 0.375rem; }
        .stButton>button:hover { background-color: #1D4ED8; }
        .stButton>button:disabled { background-color: #9CA3AF; }
        /* Dialog styling */
        [data-testid="stDialog"] {border-radius: 0.75rem;}
        [data-testid="stDialog"] h1 {font-size: 1.5rem; color: #111827;} /* Dialog Title */
        .stChatMessage { border-radius: 10px; margin-bottom: 10px; padding: 10px; }
        .stChatMessage[data-testid="chatAvatarIcon-assistant"] + div { background-color: #f0f2f6; color: #333; }
        .stChatMessage[data-testid="chatAvatarIcon-user"] + div { background-color: #e7f0fe; }
        /* Recommendation styling */
        .recommendation-critical { border-left: 5px solid #dc3545; padding: 10px; margin-bottom: 10px; background-color: #f8d7da; color: #58151a; border-radius: 0.375rem; }
        .recommendation-high { border-left: 5px solid #fd7e14; padding: 10px; margin-bottom: 10px; background-color: #fff3cd; color: #664d03; border-radius: 0.375rem; }
        .recommendation-medium { border-left: 5px solid #ffc107; padding: 10px; margin-bottom: 10px; background-color: #fff9e0; color: #665100; border-radius: 0.375rem; }
        .recommendation-low { border-left: 5px solid #0dcaf0; padding: 10px; margin-bottom: 10px; background-color: #cff4fc; color: #055160; border-radius: 0.375rem; }
        .recommendation-informational { border-left: 5px solid #6c757d; padding: 10px; margin-bottom: 10px; background-color: #e2e3e5; color: #41464b; border-radius: 0.375rem; }
        .recommendation-unknown { border-left: 5px solid #adb5bd; padding: 10px; margin-bottom: 10px; background-color: #f8f9fa; color: #495057; border-radius: 0.375rem; }
    </style>
""", unsafe_allow_html=True)

# --- Load Base Config ---
sim_config_data = get_config()

# --- App Header Section ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🔗 Advanced Supply Chain Intelligence") # Updated title
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <span style="background-color: #DBEAFE; color: #1E40AF; font-weight: 600; padding: 0.25rem 0.5rem; border-radius: 0.25rem; margin-right: 0.5rem;">MVP Demo</span>
            <span style="color: #4B5563; font-size: 0.9rem;">Integrated AI-powered analytics and simulation</span>
        </div>
    """, unsafe_allow_html=True)

# --- Initialize Session State (Ensure all needed keys are here) ---
st.session_state.setdefault('active_tab', '📈 Overall Summary')
st.session_state.setdefault('latest_news_summary', None)
st.session_state.setdefault('simulation_results_for_rec', None) # Stores timeseries_data dict list
st.session_state.setdefault('ai_assistant_open', False)
st.session_state.setdefault('chat_history', [{"role": "assistant", "content": "👋 Hello! Ask me a question, fetch news, or run a simulation."}])
# Remove trigger states, handle directly in button logic now
# st.session_state.setdefault('trigger_simulation', False)
# st.session_state.setdefault('trigger_news', False)
st.session_state.setdefault('start_date_filter', None)
st.session_state.setdefault('end_date_filter', None)
st.session_state.setdefault('hist_factory_filter', None)
# Keep recommendation cache logic for historical tab if desired
st.session_state.setdefault('recommendations_hist', None)
st.session_state.setdefault('last_hist_rec_filters', None)
st.session_state.setdefault('prev_hist_rec_filters_key', None)

with header_col2:
    # AI Assistant button in header
    if st.button("🤖 Launch AI Assistant", use_container_width=True):
        st.session_state.ai_assistant_open = not st.session_state.ai_assistant_open
        # No rerun here, dialog logic handles visibility


# --- Sidebar Setup ---
st.sidebar.header("Dashboard Controls")

# --- REMOVED Quick Actions Expander (Actions are now in Dialog) ---

# Case Studies Expander
with st.sidebar.expander("📚 Case Studies", expanded=False):
    st.markdown("Explore historical disruptions:")
    st.page_link("pages/1_Suez_Canal_Blockage.py", label="Suez Canal Blockage (2021)", icon="🚢")
    st.page_link("pages/2_COVID_Vaccine_Inequities.py", label="COVID Vaccine Inequities", icon="💉")
    st.page_link("pages/3_Russia_Ukraine_War.py", label="Russia-Ukraine War Impacts", icon="⚔️")
    st.page_link("pages/4_LA_LB_Port_Congestion.py", label="LA/LB Port Congestion (2024)", icon="⚓")

# Data Selection & Filtering (Generated Runs Only for MVP Demo)
df_main = None
selected_run_file = None
df_filtered = pd.DataFrame()
selected_factories = []

with st.sidebar.expander("📊 Data Selection & Filters", expanded=True):
    available_files = get_available_runs(DATA_DIR)
    if not available_files:
        st.warning("No simulation data found.")
    else:
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

    # Filtering logic (only if df_main is loaded)
    if df_main is not None and not df_main.empty:
        try: # Your existing robust filtering logic here...
           min_date, max_date = df_main['date'].min().date(), df_main['date'].max().date()
           # ... date validation/retrieval from state ...
           state_start = st.session_state.start_date_filter or min_date; state_end = st.session_state.end_date_filter or max_date
           # ... bounds checking ...
           start_date_filter = state_start; end_date_filter = state_end # Use determined dates
           available_factories = sorted(df_main['factory_name'].unique().tolist())
           current_selection = st.session_state.get("hist_factory_filter", available_factories); # ... factory validation ...
           selected_factories = st.multiselect("Select Factories:", available_factories, default=current_selection, key="hist_factory_filter_widget")
           start_date_widget = st.date_input("From:", value=start_date_filter, min_value=min_date, max_value=max_date, key='start_date_filter_widget')
           end_date_widget = st.date_input("To:", value=end_date_filter, min_value=start_date_widget, max_value=max_date, key='end_date_filter_widget')
           # Update state if widgets change (optional rerun logic)
           st.session_state.start_date_filter = start_date_widget; st.session_state.end_date_filter = end_date_widget; st.session_state.hist_factory_filter = selected_factories
           start_date_filter, end_date_filter = start_date_widget, end_date_widget # Use widget values
           df_filtered = get_filtered_data(df_main, selected_factories, start_date_filter, end_date_filter)
        except Exception as e: st.error(f"Filter Error: {e}"); df_filtered = pd.DataFrame()

# --- Footer ---
# (Your sidebar footer remains)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """<div style="text-align: center; color: #6B7280; font-size: 0.8rem;"><p>© 2025 SupplyChain AI</p><p>Version 0.2.0</p></div>""",
    unsafe_allow_html=True
)

# --- Recommendation Cache Clearing Logic ---
current_run_display = selected_run_file if selected_run_file else "N/A"
current_filters_key = (current_run_display, tuple(sorted(selected_factories)), start_date_filter, end_date_filter)
if current_filters_key != st.session_state.get('prev_hist_rec_filters_key'):
    if 'recommendations_hist' in st.session_state:
        logging.info("Filters changed, clearing historical recommendations.")
        st.session_state.pop('recommendations_hist', None)
    st.session_state['prev_hist_rec_filters_key'] = current_filters_key

# --- Tab Definitions & Active Tab Tracking ---
tab_titles = ["📈 Overall Summary", "🏭 Factory Deep Dive", "🚨 Risk Analysis", "🔮 Predictive Forecast", "🔬 Live Simulation", "🧠 Recommendations"]
tabs = st.tabs(tab_titles)

# --- Render Tabs & Update Active Tab State ---
with tabs[0]:
    st.session_state.active_tab = tab_titles[0]
    summary_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter, selected_factories)
with tabs[1]:
    st.session_state.active_tab = tab_titles[1]
    factory_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter)
with tabs[2]:
    st.session_state.active_tab = tab_titles[2]
    risk_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter)
with tabs[3]:
    st.session_state.active_tab = tab_titles[3]
    forecast_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter)
with tabs[4]:
    st.session_state.active_tab = tab_titles[4]
    simulation_tab.render_tab()
with tabs[5]:
    st.session_state.active_tab = tab_titles[5]
    recommendations_tab.render_tab(df_filtered, current_run_display, start_date_filter, end_date_filter, selected_factories)


# --- AI Assistant Dialog ---

if st.session_state.get('ai_assistant_open', False):

    @st.dialog("🤖 AI Assistant")
    def ai_assistant_dialog():
        st.info("Demo: Chat | News Summary | Simulation w/ Charts | Recommendations")
        st.markdown("---")

        # --- Display Chat History & Results ---
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)

                    # --- Plotting Simulation Results ---
                    # Uses 'timeseries_data' which is the plotting subset
                    if message.get("type") == "simulation_result" and message.get("plot_data"):
                        sim_plot_data = message["plot_data"]
                        try:
                            df_plot = pd.DataFrame(sim_plot_data)
                            if not df_plot.empty and 'date' in df_plot.columns:
                                df_plot['date'] = pd.to_datetime(df_plot['date'], errors='coerce').dt.date # Use date only for plot axis
                                df_plot.dropna(subset=['date'], inplace=True)
                                if not df_plot.empty:
                                    with st.expander("Simulation Charts", expanded=True):
                                        # (Your plotting logic using df_plot['inventory_level_units'] etc.)
                                        plot_made = False
                                        if 'inventory_level_units' in df_plot.columns and df_plot['inventory_level_units'].notna().any():
                                             fig_inv = px.line(df_plot, x='date', y='inventory_level_units', title='Inventory')
                                             fig_inv.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=250)
                                             st.plotly_chart(fig_inv, use_container_width=True); plot_made=True
                                        if 'shortage_units' in df_plot.columns and df_plot['shortage_units'].notna().any():
                                             if pd.to_numeric(df_plot['shortage_units'], errors='coerce').sum() > 0:
                                                  fig_short = px.bar(df_plot, x='date', y='shortage_units', title='Shortage')
                                                  fig_short.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=200)
                                                  st.plotly_chart(fig_short, use_container_width=True); plot_made=True
                                             else: st.caption("No shortage occurred."); plot_made=True
                                        if not plot_made: st.caption("Plot data missing.")
                                else: st.caption("No valid plot data.")
                            else: st.caption("No plot data found.")
                        except Exception as e: st.error(f"Plot Error: {e}"); logging.error("Plotting error", exc_info=True)

                    # --- Displaying Recommendations ---
                    elif message.get("type") == "recommendations" and message.get("data"):
                         with st.expander("Recommendations", expanded=True):
                              # (Your recommendation display logic)
                              recommendations = message["data"]
                              if recommendations:
                                   recommendations.sort(key=lambda x: x.get('priority', 0), reverse=True)
                                   for rec in recommendations:
                                        prio=rec.get("priority",0); cat=rec.get("category","Info"); txt=rec.get("text","N/A")
                                        p_map={5:"critical", 4:"high", 3:"medium", 2:"low", 1:"informational"}; css=f"recommendation-{p_map.get(prio, 'unknown')}"
                                        st.markdown(f'<div class="{css}"><strong>{cat} (P{prio}):</strong> {txt}</div>', unsafe_allow_html=True)
                              else: st.info("No recommendations generated.")

        # --- Action Buttons & Chat Input ---
        st.markdown("---")
        action_triggered = False

        btn_cols = st.columns([1, 1.5, 1])
        with btn_cols[0]:
             if st.button("▶️ Run Sim", key="diag_run_sim_final_v2", help="Trigger simulation & show charts"):
                action_triggered = True
                sim_endpoint = f"{BACKEND_URL}/run_simulation"
                st.session_state.simulation_results_for_rec = None # Clear before new run
                try:
                     with st.spinner("Running simulation..."):
                          response = requests.post(sim_endpoint, json={}, timeout=120)
                          response.raise_for_status()
                          sim_data = response.json() # Expects SimulationResultResponse model
                          status = sim_data.get('status','?'); msg = sim_data.get('message','?');
                          plot_data = sim_data.get('timeseries_data') # Subset for plotting
                          full_data = sim_data.get('full_simulation_data') # Full data for recs

                          # Store full data for rec button BEFORE adding message to history
                          if status == "completed" and full_data:
                               st.session_state.simulation_results_for_rec = full_data

                          # Add message with plot data attached
                          st.session_state.chat_history.append({
                              "role":"assistant", "content":f"**Sim:** {status} - {msg}",
                              "type":"simulation_result", "plot_data": plot_data # Attach plot data only
                          })
                except Exception as e: st.session_state.chat_history.append({"role":"assistant", "content":f"**Sim Error:** {e}"}); logging.error(f"Sim Error: {e}", exc_info=True)

        with btn_cols[1]:
             news_query = st.text_input("News Query:", value="supply chain disruption", key="diag_news_query_final_v2", label_visibility="collapsed")
             if st.button("📰 Fetch Briefing", key="diag_fetch_brief_final_v2"):
                 action_triggered = True
                 brief_endpoint = f"{BACKEND_URL}/fetch-news"
                 try:
                     with st.spinner("Fetching news..."):
                          response = requests.post(brief_endpoint, json={"query": news_query}, timeout=45)
                          response.raise_for_status(); summary = response.json().get('summary','N/A')
                          st.session_state.latest_news_summary = summary # Cache summary
                          st.session_state.chat_history.append({"role":"assistant", "content":f"**News ({news_query}):**\n{summary}"})
                 except Exception as e: st.session_state.chat_history.append({"role":"assistant", "content":f"**News Error:** {e}"}); logging.error(f"News Error: {e}", exc_info=True); st.session_state.latest_news_summary=None

        with btn_cols[2]:
             # Check if the FULL data for recommendations exists
             rec_button_disabled = st.session_state.get('simulation_results_for_rec') is None
             if st.button("💡 Get Recs", key="diag_get_recs_final_v2", disabled=rec_button_disabled):
                  action_triggered = True
                  rec_endpoint = f"{BACKEND_URL}/get_simulation_recommendations"
                  # Send the FULL data stored in session state
                  sim_payload = st.session_state.simulation_results_for_rec
                  if sim_payload:
                       try:
                            with st.spinner("Generating recommendations..."):
                                 response = requests.post(rec_endpoint, json={"simulation_data": sim_payload}, timeout=60)
                                 response.raise_for_status()
                                 recs = response.json().get("recommendations", [])
                                 st.session_state.chat_history.append({"role":"assistant", "content":f"Found {len(recs)} recommendations:", "type":"recommendations", "data":recs})
                       except Exception as e: st.session_state.chat_history.append({"role":"assistant", "content":f"**Rec Error:** {e}"}); logging.error(f"Rec Error: {e}", exc_info=True)
                  else: st.session_state.chat_history.append({"role":"assistant", "content":"No simulation data for recs."})

        # --- Chat Input ---
        if prompt := st.chat_input("Ask the assistant...", key="ai_chat_input_final"):
            action_triggered = True
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            payload = { "text": prompt, "active_tab": st.session_state.get('active_tab'), "news_summary": st.session_state.get('latest_news_summary') }
            api_endpoint = f"{BACKEND_URL}/chat"; ai_response = "Error contacting AI backend."
            try:
                with st.spinner("AI thinking..."):
                    response = requests.post(api_endpoint, json=payload, timeout=30)
                    response.raise_for_status(); ai_response = response.json().get("response", "Error: Invalid response.")
            except Exception as e: ai_response = f"AI Error: {e}"; logging.error(f"Chat Error: {e}", exc_info=True)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

        # --- Rerun if action occurred ---
        if action_triggered: st.rerun()

    # Render the dialog
    if st.session_state.get('ai_assistant_open', False): ai_assistant_dialog()
