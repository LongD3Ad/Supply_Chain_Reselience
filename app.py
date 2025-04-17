# src/app.py
import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta
import os # Needed for path joining
import io 

# Define the columns the dashboard expects/needs
# Essential for basic filtering and display
ESSENTIAL_COLS = ['date', 'factory_name']
# Core metrics used in most visualizations
CORE_METRIC_COLS = ['adjusted_production_units', 'inventory_level_units', 'shipped_units', 'simulated_demand_units', 'shortage_units']
# Optional columns for specific features/context
OPTIONAL_COLS = ['lead_time_days', 'production_impact_factor', 'active_risks', 'factory_location', 'factory_type', 'base_production_capacity', 'safety_stock_level']
# Combine all expected columns (order matters for display perhaps)
ALL_EXPECTED_COLS = ESSENTIAL_COLS + CORE_METRIC_COLS + OPTIONAL_COLS

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

st.sidebar.markdown("---") # Separator

# --- User Data Upload Section ---
st.sidebar.subheader("Upload Your Own Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    key='user_uploaded_file' # Use a key to access the widget's state
)

# Initialize session state variables for upload workflow
if 'raw_user_df' not in st.session_state:
    st.session_state.raw_user_df = None
if 'user_column_mapping' not in st.session_state:
    st.session_state.user_column_mapping = {}
if 'user_data_processed' not in st.session_state:
    st.session_state.user_data_processed = None # This will hold the final df
if 'user_upload_error' not in st.session_state:
    st.session_state.user_upload_error = None

# Trigger reading the file when a new file is uploaded
if uploaded_file is not None:
    # Check if it's a new file compared to the last processed one
    # This simple check assumes file name/size might differ, not perfect but okay
    if st.session_state.raw_user_df is None or uploaded_file.name != st.session_state.get('last_uploaded_filename') or uploaded_file.size != st.session_state.get('last_uploaded_filesize'):
        try:
            logging.info(f"Reading uploaded file: {uploaded_file.name}")
            if uploaded_file.name.lower().endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.session_state.raw_user_df = df_upload
            st.session_state.user_column_mapping = {} # Reset mapping for new file
            st.session_state.user_data_processed = None # Clear old processed data
            st.session_state.user_upload_error = None
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.last_uploaded_filesize = uploaded_file.size
            logging.info(f"Successfully read {len(df_upload)} rows from uploaded file.")
            # Rerun to display mapping options immediately
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            logging.error(f"Error reading uploaded file {uploaded_file.name}: {e}", exc_info=True)
            st.session_state.raw_user_df = None
            st.session_state.user_upload_error = f"Error reading file: {e}"


# --- Column Mapping UI (Displayed only if a raw DataFrame exists) ---
if st.session_state.raw_user_df is not None:
    with st.sidebar.expander("Map Data Columns", expanded=True):
        st.write("Map your columns to the dashboard's expected fields:")
        raw_columns = ['<Select Column>'] + list(st.session_state.raw_user_df.columns)
        mapping = {}
        has_essential_mapping = True

        for target_col in ALL_EXPECTED_COLS:
            # Try to pre-select if column name matches exactly (case-insensitive)
            default_selection = '<Select Column>'
            for raw_col in st.session_state.raw_user_df.columns:
                if raw_col.lower().replace(" ", "_") == target_col.lower():
                    default_selection = raw_col
                    break

            # Use previously selected value if available in session state
            current_selection = st.session_state.user_column_mapping.get(target_col, default_selection)
            # Ensure the current selection is actually valid given the raw columns
            if current_selection not in raw_columns:
                current_selection = '<Select Column>'


            # Add '*' for essential columns
            label = f"{target_col}{' *' if target_col in ESSENTIAL_COLS else ''}"
            selected_col = st.selectbox(
                label,
                options=raw_columns,
                index=raw_columns.index(current_selection), # Set current selection
                key=f"map_{target_col}" # Unique key for each selectbox
            )
            mapping[target_col] = selected_col if selected_col != '<Select Column>' else None
            # Update session state immediately (optional, but good for persistence)
            st.session_state.user_column_mapping[target_col] = selected_col

            # Check if essential columns are mapped
            if target_col in ESSENTIAL_COLS and not mapping[target_col]:
                has_essential_mapping = False

        st.caption("* Essential columns required for basic functionality.")

        if st.button("Process Uploaded Data", key="process_upload_button", disabled=not has_essential_mapping):
            st.session_state.user_upload_error = None # Clear previous errors
            try:
                processed_df = pd.DataFrame()
                final_mapping = st.session_state.user_column_mapping # Use the latest mapping state
                valid_map_count = 0

                for target_col, source_col in final_mapping.items():
                    if source_col and source_col != '<Select Column>':
                        if source_col in st.session_state.raw_user_df.columns:
                            processed_df[target_col] = st.session_state.raw_user_df[source_col]
                            valid_map_count += 1
                        else:
                            logging.warning(f"Source column '{source_col}' for target '{target_col}' not found in raw data during processing. Skipping.")

                if valid_map_count == 0:
                     raise ValueError("No columns were successfully mapped.")

                logging.info(f"Processing uploaded data with mapping: {final_mapping}")

                # --- Type Conversions & Validation ---
                # Date Conversion (Essential)
                if 'date' in processed_df.columns:
                    processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
                    if processed_df['date'].isnull().all():
                        raise ValueError("The mapped 'date' column could not be parsed as dates.")
                    processed_df.dropna(subset=['date'], inplace=True) # Remove rows where date is invalid
                else:
                    # This should have been caught by the button's disabled state
                    raise ValueError("Essential column 'date' was not mapped.")

                # Factory Name (Essential)
                if 'factory_name' not in processed_df.columns:
                     raise ValueError("Essential column 'factory_name' was not mapped.")
                processed_df['factory_name'] = processed_df['factory_name'].astype(str)

                # Numeric Conversions (Core & Optional)
                numeric_cols_to_convert = CORE_METRIC_COLS + [c for c in OPTIONAL_COLS if c in processed_df.columns and c not in ['active_risks', 'factory_location', 'factory_type']] # Exclude string cols
                for col in numeric_cols_to_convert:
                    if col in processed_df.columns:
                        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                        # Optional: Fill NaNs created by coercion if needed, e.g., with 0 for units/shortage
                        # if col in ['shortage_units', 'adjusted_production_units']:
                        #     processed_df[col].fillna(0, inplace=True)

                # String Conversions (Optional)
                string_cols = ['active_risks', 'factory_location', 'factory_type']
                for col in string_cols:
                    if col in processed_df.columns:
                         processed_df[col] = processed_df[col].fillna('N/A').astype(str)
                    # Ensure these optional columns exist even if not mapped, fill with default
                    elif col not in processed_df.columns:
                         processed_df[col] = 'N/A'


                # Sort and store
                processed_df = processed_df.sort_values(by=['factory_name', 'date'])
                st.session_state.user_data_processed = processed_df
                st.success("Data processed successfully!")
                logging.info(f"Processed user data shape: {processed_df.shape}")
                # Maybe clear raw_df and mapping here if desired, or keep for re-processing
                # st.session_state.raw_user_df = None
                # st.session_state.user_column_mapping = {}


            except Exception as e:
                st.error(f"Error processing data: {e}")
                logging.error(f"Error processing uploaded data: {e}", exc_info=True)
                st.session_state.user_data_processed = None # Ensure no stale data on error
                st.session_state.user_upload_error = f"Error processing data: {e}"

        # Display error message if processing failed
        if st.session_state.user_upload_error:
             st.error(st.session_state.user_upload_error)


st.sidebar.markdown("---") # Separator before next section

# --- Data Source Selection ---
# Determine default source: Use Uploaded if successfully processed, otherwise Generated
default_source_index = 1 if st.session_state.get('user_data_processed') is not None else 0
data_source = st.sidebar.radio(
    "Select Data Source:",
    ("Load Generated Simulation Run", "Use Uploaded File"),
    index=default_source_index,
    key="data_source_select",
    horizontal=True,
)

# --- Set df_main based on selection ---
df_main = None # Reset df_main
selected_run_file_display = None # For display in tabs

if data_source == "Use Uploaded File":
    if st.session_state.get('user_data_processed') is not None:
        df_main = st.session_state.user_data_processed
        selected_run_file_display = f"Uploaded: {st.session_state.get('last_uploaded_filename', 'User File')}"
        st.sidebar.success("Using processed uploaded data.")
    else:
        st.sidebar.warning("No processed user data available. Please upload and process a file.")
        # Fallback to generated if upload selected but not ready? Or show error?
        # For now, df_main remains None, tabs will show 'No data'.
else: # Load Generated Simulation Run
    # (Keep the existing logic for loading generated runs here)
    available_files = get_available_runs(DATA_DIR)
    if not available_files:
        # Handled above, just ensure df_main is None
        pass
    else:
        # Use the selectbox value stored by the widget
        selected_run_file = st.session_state.get('historical_run_select', available_files[0]) # Get current value
        if selected_run_file:
            file_path = os.path.join(DATA_DIR, selected_run_file)
            df_main = load_data(file_path) # Load the selected generated run
            selected_run_file_display = selected_run_file # Use the actual filename for display

# --- Filtering Controls ---
# The rest of the filtering logic now operates on df_main, regardless of its source
df_filtered = pd.DataFrame() # Initialize df_filtered
# ... (rest of the filtering logic: if df_main is not None...)
# Important: Make sure selected_factories, start/end_date_filter are defined BEFORE this block
selected_factories = []
start_date_filter = datetime.now().date() - timedelta(days=365) # Default start
end_date_filter = datetime.now().date()   # Default end

if df_main is not None and not df_main.empty:
    # Check if essential columns exist AFTER potential upload/processing
    if not all(col in df_main.columns for col in ESSENTIAL_COLS):
         st.sidebar.error(f"Loaded data is missing essential columns ({', '.join(ESSENTIAL_COLS)}). Cannot proceed with filtering.")
         df_main = None # Invalidate if essential cols missing

    else:
        # --- Filtering Controls UI --- (This is the block you already have)
        available_factories = sorted(df_main['factory_name'].unique().tolist())
        try:
            min_date = df_main['date'].min().date()
            max_date = df_main['date'].max().date()
             # Ensure date range is valid
            if min_date > max_date:
                 st.sidebar.error("Error in data: Minimum date is after maximum date.")
                 raise ValueError("Invalid date range in data")

        except Exception as e:
            st.sidebar.error(f"Error processing dates in the {data_source} data: {e}")
            min_date = start_date_filter # Use default if error
            max_date = end_date_filter
            df_main = None # Invalidate df_main on date error

        if df_main is not None: # Check again after date processing
            st.sidebar.header("Filters for Selected Data") # Changed header
            # ... (multiselect for selected_factories) ...
            selected_factories = st.sidebar.multiselect(
                "Select Factories:", options=available_factories,
                # Reset default if source changes or available factories change
                default=available_factories if st.session_state.get('last_data_source') == data_source else available_factories,
                key="hist_factory_filter"
            )

            # ... (date input logic, ensuring session state keys are initialized safely) ...
            # Initialize date session state if needed
            if 'start_date_filter' not in st.session_state or not isinstance(st.session_state.start_date_filter, type(min_date)) or st.session_state.start_date_filter < min_date or st.session_state.start_date_filter > max_date : st.session_state.start_date_filter = min_date
            if 'end_date_filter' not in st.session_state or not isinstance(st.session_state.end_date_filter, type(min_date)) or st.session_state.end_date_filter < min_date or st.session_state.end_date_filter > max_date: st.session_state.end_date_filter = max_date

            try:
                start_date_filter = st.sidebar.date_input("Start Date:", value=st.session_state.start_date_filter, min_value=min_date, max_value=max_date, key='start_date_filter_widget')
                st.session_state.start_date_filter = start_date_filter # Update state from widget
                end_date_filter = st.sidebar.date_input("End Date:", value=st.session_state.end_date_filter, min_value=start_date_filter, max_value=max_date, key='end_date_filter_widget')
                st.session_state.end_date_filter = end_date_filter # Update state from widget
            except Exception as e:
                 st.sidebar.error(f"Date selection error: {e}")
                 start_date_filter = st.session_state.start_date_filter # Fallback to state/default
                 end_date_filter = st.session_state.end_date_filter

            # Apply filters only if dates are valid
            if start_date_filter > end_date_filter:
                st.sidebar.error("Error: Start date must be on or before end date.")
                df_filtered = pd.DataFrame() # Ensure filtered is empty
                st.session_state.pop('recommendations_hist', None)
            else:
                df_filtered = get_filtered_data(df_main, selected_factories, start_date_filter, end_date_filter)
                # Optionally update cache invalidation key based on source
                st.session_state['last_hist_rec_filters'] = (selected_run_file_display, tuple(sorted(selected_factories)), start_date_filter, end_date_filter)
                if df_filtered.empty and selected_factories:
                    st.sidebar.warning("No data matches the current filter selection.")
                    st.session_state.pop('recommendations_hist', None)


# Store current data source to detect changes
st.session_state['last_data_source'] = data_source

# --- Clear historical recommendations if selection changed ---
# (This logic can likely remain as is, using the updated last_hist_rec_filters key)
current_filters_key = st.session_state.get('last_hist_rec_filters') # Get the key set above
if 'prev_hist_rec_filters_key' not in st.session_state or current_filters_key != st.session_state.get('prev_hist_rec_filters_key'):
    if 'recommendations_hist' in st.session_state:
        logging.info("Historical data selection or source changed, clearing cached recommendations.")
        st.session_state.pop('recommendations_hist', None)
    st.session_state['prev_hist_rec_filters_key'] = current_filters_key # Update previous key tracker

# --- Filtering Controls ---
# df_filtered = pd.DataFrame() # Initialize df_filtered to be safe
# selected_factories = []
# start_date_filter = datetime.now().date() - timedelta(days=365) # Default start
# end_date_filter = datetime.now().date()   # Default end

# if df_main is not None and not df_main.empty:
#     available_factories = sorted(df_main['factory_name'].unique().tolist())
#     try:
#         min_date = df_main['date'].min().date()
#         max_date = df_main['date'].max().date()
#     except Exception as e:
#         st.sidebar.error(f"Error processing dates in the selected file: {e}")
#         min_date = start_date_filter # Use default if error
#         max_date = end_date_filter
#         df_main = None # Invalidate df_main on date error

#     if df_main is not None: # Check again after potential invalidation
#         st.sidebar.header("Filters for Selected Run")
#         selected_factories = st.sidebar.multiselect(
#             "Select Factories:",
#             options=available_factories,
#             default=available_factories,
#             key="hist_factory_filter"
#             )

#         # Initialize dates in session state if not present or invalid
#         if 'start_date_filter' not in st.session_state or not isinstance(st.session_state.start_date_filter, type(min_date)) or st.session_state.start_date_filter < min_date or st.session_state.start_date_filter > max_date :
#              st.session_state.start_date_filter = min_date
#         if 'end_date_filter' not in st.session_state or not isinstance(st.session_state.end_date_filter, type(min_date)) or st.session_state.end_date_filter < min_date or st.session_state.end_date_filter > max_date:
#              st.session_state.end_date_filter = max_date


#         # Date range selection using session state
#         try:
#             start_date_filter = st.sidebar.date_input(
#                 "Start Date:",
#                 value=st.session_state.start_date_filter,
#                 min_value=min_date,
#                 max_value=max_date,
#                 key='start_date_filter_widget' # Use a different key for widget if needed
#             )
#             # Update session state if widget changes
#             st.session_state.start_date_filter = start_date_filter

#             end_date_filter = st.sidebar.date_input(
#                 "End Date:",
#                 value=st.session_state.end_date_filter,
#                 min_value=start_date_filter, # Dynamic min based on start date
#                 max_value=max_date,
#                 key='end_date_filter_widget'
#             )
#             st.session_state.end_date_filter = end_date_filter

#         except Exception as e:
#              st.sidebar.error(f"Date selection error: {e}")
#              # Use last known good dates from session state or defaults
#              start_date_filter = st.session_state.start_date_filter
#              end_date_filter = st.session_state.end_date_filter


#         # Ensure start_date is not after end_date
#         if start_date_filter > end_date_filter:
#             st.sidebar.error("Error: Start date must be on or before end date.")
#             df_filtered = pd.DataFrame() # Use empty dataframe
#             st.session_state['recommendations_hist'] = None # Clear recommendations on invalid date
#         else:
#             # Apply filters using the function from data_handler
#             df_filtered = get_filtered_data(df_main, selected_factories, start_date_filter, end_date_filter)
#             if df_filtered.empty and selected_factories: # Only warn if factories were selected but yielded no data
#                  st.sidebar.warning("No data matches the current filter selection.")
#                  # Clear recommendations if filters result in no data
#                  st.session_state.pop('recommendations_hist', None)

# # Handle cases where df_main is None initially or after errors
# elif df_main is None:
#      if selected_run_file: # Selection exists but loading failed
#          st.sidebar.error(f"Failed to load or process data from {selected_run_file}.")
#      # Keep df_filtered empty
#      st.session_state.pop('recommendations_hist', None) # Clear any stale recommendations


# # --- Clear historical recommendations if selection changed ---
# # Moved this logic here after df_filtered is finalized
# current_filters_key = (selected_run_file, tuple(sorted(selected_factories)), start_date_filter, end_date_filter)
# if 'last_hist_rec_filters' not in st.session_state or current_filters_key != st.session_state.get('last_hist_rec_filters'):
#      if 'recommendations_hist' in st.session_state:
#           logging.info("Historical data selection changed, clearing cached recommendations.")
#           st.session_state.pop('recommendations_hist', None)
#      # Update the tracker regardless of whether recommendations existed
#      st.session_state['last_hist_rec_filters'] = current_filters_key


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