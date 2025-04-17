# src/data_handler.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime, timedelta

# Import DATA_DIR from config
from config import DATA_DIR

@st.cache_data(ttl=3600) # Cache loaded data for an hour
def load_data(file_path):
    """Loads simulation data from a CSV file with error handling and type conversion."""
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        # Define expected numeric columns
        numeric_cols = ['base_production_capacity', 'adjusted_production_units',
                        'production_impact_factor', 'lead_time_days', 'lead_time_multiplier',
                        'simulated_demand_units', 'shipped_units',
                        'inventory_level_units', 'shortage_units', 'safety_stock_level',
                        'active_risks_count']

        for col in numeric_cols:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN
             else:
                logging.warning(f"Column '{col}' not found in {os.path.basename(file_path)}. Skipping conversion.")
                df[col] = np.nan # Add missing numeric columns as NaN

        # Ensure 'active_risks' column exists and is string
        if 'active_risks' not in df.columns:
             logging.warning(f"Column 'active_risks' not found in {os.path.basename(file_path)}. Adding default 'none'.")
             df['active_risks'] = 'none'
        else:
            df['active_risks'] = df['active_risks'].fillna('none').astype(str)

        df = df.sort_values(by=['factory_name', 'date'])
        logging.info(f"Successfully loaded and processed data from {os.path.basename(file_path)}")
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        logging.error(f"Data file not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading or processing data from {os.path.basename(file_path)}: {e}")
        logging.error(f"Failed to load data from {file_path}: {e}", exc_info=True)
        return None


def get_available_runs(data_dir=DATA_DIR):
    """Finds available simulation run CSV files in the specified directory."""
    # Ensure the data directory exists
    if not os.path.isdir(data_dir):
         logging.warning(f"Data directory '{data_dir}' not found. Attempting to create.")
         try:
              os.makedirs(data_dir, exist_ok=True)
              logging.info(f"Created data directory: {data_dir}")
              return [] # Return empty list as no files can exist yet
         except OSError as e:
              logging.error(f"Could not create data directory {data_dir}: {e}")
              st.error(f"Fatal Error: Could not access or create data directory '{data_dir}'. Please check permissions.")
              return [] # Return empty list on error

    # Search for CSV files
    search_path = os.path.join(data_dir, "semiconductor_supplychain_simulated_run_*.csv")
    files = glob.glob(search_path)
    csv_files = [f for f in files if os.path.isfile(f) and f.lower().endswith('.csv')]

    if not csv_files:
         logging.warning(f"No simulation CSV files found matching pattern in {data_dir}")

    # Return sorted list of basenames
    return sorted([os.path.basename(f) for f in csv_files], reverse=True)


@st.cache_data # Cache filtered results
def get_filtered_data(_df, selected_factories, start_date, end_date):
    """Filters the main DataFrame based on factories and date range."""
    if _df is None or _df.empty:
        return pd.DataFrame() # Return empty DataFrame if input is invalid

    if not selected_factories:
         logging.warning("No factories selected for filtering.")
         return pd.DataFrame() # Return empty if no factories are selected

    try:
        # Ensure dates are compatible (datetime objects)
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        # Perform filtering
        filtered = _df[
            (_df['factory_name'].isin(selected_factories)) &
            (_df['date'] >= start_date_dt) &
            (_df['date'] <= end_date_dt)
        ].copy() # Use copy to avoid SettingWithCopyWarning

        # Minimal check for essential columns post-filtering
        essential_cols = ['date', 'factory_name']
        if not all(col in filtered.columns for col in essential_cols) and not filtered.empty:
             logging.error("Filtered DataFrame is missing essential columns (date, factory_name).")
             # Decide how to handle: return empty, raise error, etc.
             return pd.DataFrame() # Safest option

        return filtered

    except Exception as e:
        logging.error(f"Error during data filtering: {e}", exc_info=True)
        st.warning(f"An error occurred during data filtering: {e}")
        return pd.DataFrame() # Return empty DataFrame on error