# backend/app/financials.py
import pandas as pd
import logging
from typing import Dict, Any, Optional

# Define expected cost parameters and their default fill values if missing
# These keys should match what's in your config/user overrides
DEFAULT_COST_PARAMS = {
    "holding_cost_unit_weekly": 0.0,
    "stockout_cost_unit": 0.0,
    "base_logistics_cost_unit": 0.0,
    "production_cost_unit": 0.0
}

def calculate_financial_kpis(
    df_sim_results: pd.DataFrame,
    cost_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculates financial KPIs based on simulation results and cost parameters.

    Args:
        df_sim_results: DataFrame containing simulation results (e.g., inventory, shortage).
                        Expected columns: 'date', 'inventory_level_units',
                        'shortage_units', 'shipped_units', 'adjusted_production_units'.
        cost_params: Dictionary of cost rates (holding, stockout, etc.). Overrides defaults.

    Returns:
        Dictionary containing calculated financial KPIs.
    """
    financial_summary: Dict[str, Any] = {}
    if df_sim_results is None or df_sim_results.empty:
        logging.warning("No simulation results provided for financial calculation.")
        return financial_summary

    # --- Get Effective Cost Parameters ---
    # Start with defaults and update with provided parameters
    effective_costs = DEFAULT_COST_PARAMS.copy()
    if cost_params:
        for key, value in cost_params.items():
            if key in effective_costs and value is not None:
                # Attempt to convert value to float, default to existing effective cost if fails
                try:
                    effective_costs[key] = float(value)
                except (ValueError, TypeError):
                    logging.warning(f"Invalid value for cost parameter '{key}': {value}. Using default.")
                    # Cost remains at DEFAULT_COST_PARAMS value or previously updated value

    logging.info(f"Calculating financial KPIs with effective costs: {effective_costs}")

    # --- Ensure Required Columns Exist and are Numeric ---
    required_cols = ['date', 'inventory_level_units', 'shortage_units', 'shipped_units', 'adjusted_production_units']
    for col in required_cols:
        if col not in df_sim_results.columns:
            logging.error(f"Financial calculation requires column '{col}', which is missing.")
            financial_summary[f"error_{col}_missing"] = True
            # Can we calculate partially? For MVP, maybe just return error if essential missing
            # For now, the calculations below will likely fail or produce NaNs if columns are missing

    # Attempt to convert relevant columns to numeric, coercing errors
    numeric_cols = ['inventory_level_units', 'shortage_units', 'shipped_units', 'adjusted_production_units']
    for col in numeric_cols:
        if col in df_sim_results.columns:
            df_sim_results[col] = pd.to_numeric(df_sim_results[col], errors='coerce')

    # --- Perform Calculations ---
    try:
        # Ensure date column is datetime for calculating duration if needed
        if 'date' in df_sim_results.columns and not pd.api.types.is_datetime64_any_dtype(df_sim_results['date']):
             df_sim_results['date'] = pd.to_datetime(df_sim_results['date'], errors='coerce')
             # Drop rows where date is NaN if date is critical for calc (like duration)
             # df_sim_results.dropna(subset=['date'], inplace=True) # Be careful, might remove all data

        # Total Holding Cost: Sum of daily/weekly inventory levels * cost per unit per period
        # Assuming simulation results are weekly summaries, so Inventory Level is EOW inventory
        # Total cost = Sum(EOW Inventory * Weekly Holding Cost per Unit)
        total_holding_cost = (
            df_sim_results['inventory_level_units'].fillna(0).sum() *
            effective_costs["holding_cost_unit_weekly"]
        ) if 'inventory_level_units' in df_sim_results.columns else 0

        # Total Stockout Cost: Sum of shortage units * cost per unit
        total_stockout_cost = (
            df_sim_results['shortage_units'].fillna(0).sum() *
            effective_costs["stockout_cost_unit"]
        ) if 'shortage_units' in df_sim_results.columns else 0

        # Estimated Total Logistics Cost: Sum of shipped units * base logistics cost per unit
        # Simplification: Does not factor in lead time impacts or variable routing
        total_logistics_cost_est = (
            df_sim_results['shipped_units'].fillna(0).sum() *
            effective_costs["base_logistics_cost_unit"]
        ) if 'shipped_units' in df_sim_results.columns else 0

        # Estimated Total Production Cost: Sum of adjusted production units * production cost per unit
        total_production_cost_est = (
            df_sim_results['adjusted_production_units'].fillna(0).sum() *
            effective_costs["production_cost_unit"]
        ) if 'adjusted_production_units' in df_sim_results.columns else 0


        # Estimated Total Cost (Simple Sum)
        estimated_total_cost = total_holding_cost + total_stockout_cost + total_logistics_cost_est + total_production_cost_est

        # --- Add KPIs to Summary ---
        financial_summary["total_holding_cost"] = round(total_holding_cost, 2)
        financial_summary["total_stockout_cost"] = round(total_stockout_cost, 2)
        financial_summary["estimated_total_logistics_cost"] = round(total_logistics_cost_est, 2)
        financial_summary["estimated_total_production_cost"] = round(total_production_cost_est, 2)
        financial_summary["estimated_total_supply_chain_cost"] = round(estimated_total_cost, 2)

        # Add other potential KPIs (e.g., total revenue estimate if demand is met, profit, ROI - more complex)
        # Estimated Revenue (Simple: Shipped Units * Avg Selling Price) - Need Avg Selling Price parameter
        # Estimated Profit (Simple: Estimated Revenue - Estimated Total Cost)
        # Need to decide how to get selling price if needed.

        logging.info("Financial KPIs calculated successfully.")

    except Exception as e:
        logging.error(f"Error during financial KPI calculation: {e}", exc_info=True)
        financial_summary["calculation_error"] = str(e)

    return financial_summary