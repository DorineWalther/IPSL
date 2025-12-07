import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob

# --- 1. Configuration des chemins et des données ---

# **!!! IMPORTANT: CHANGE THIS PATH TO YOUR DATA DIRECTORY !!!**
# Base path for the original, non-cleaned files (This path must contain all the CM61-LR-scen files)
ORIGINAL_DATA_BASE_PATH = "C:\\Users\\clari\\Documents\\MA 2\\atm processes\\project\\"

# Total analysis period
PERIOD_START = 2025
PERIOD_END = 2094

# Reference period for anomaly calculation (first decade)
REF_START = 2025 
REF_END = 2034

# Scenario and region definitions
REGIONS = ['arctic', 'europe']
SCENARIOS = ['ssp245', 'ssp585']
SCENARIO_COLORS = {'ssp245': 'blue', 'ssp585': 'red'}
TIME_COORD_NAME = 'time_counter' 

# --- Functions for Data Loading (Unified RAW loading) ---

def load_raw_variable(scenario, region, variable_name, base_path):
    """
    Loads a specific variable from the ORIGINAL (non-cleaned) file path.
    Returns the time series of the spatial mean of that variable across lat/lon.
    """
    print(f"Loading RAW variable '{variable_name}' for {scenario.upper()} - {region.upper()}")
    
    # Pattern: CM61-LR-scen-sspXXXX_YYYYMMDD_YYYYMMDD_1M_histmth_REGION.nc
    file_pattern = os.path.join(base_path, f'CM61-LR-scen-{scenario}_*_{region}.nc')
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"Error: No RAW files found for {scenario}-{region} at {base_path}")
        return None
        
    datasets = []
    try:
        for file_path in file_list:
            # Load only the specified variable to save memory
            ds = xr.open_dataset(file_path, engine='netcdf4')
            
            if variable_name not in ds.data_vars:
                 print(f"Warning: Variable '{variable_name}' not found in {os.path.basename(file_path)}. Skipping.")
                 ds.close()
                 continue
                 
            # Extract the variable, convert to Dataset (to keep metadata) and load into memory
            ds_var = ds[variable_name].squeeze().to_dataset().load()
            datasets.append(ds_var)
            
        if not datasets:
            return None
            
        # Concat and calculate the regional mean of the variable across lat/lon
        ds_full = xr.concat(datasets, dim=TIME_COORD_NAME)
        
        # Determine dimensions to average over (lat/lon, but preserve vertical level if present)
        mean_dims = [dim for dim in ds_full[variable_name].dims if dim in ('lat', 'lon')]
        
        # Return the DataArray. If it has vertical levels, the mean over lat/lon will preserve them.
        return ds_full[variable_name].mean(dim=mean_dims)
        
    except Exception as e:
        print(f"Error loading RAW variable '{variable_name}' for {scenario}-{region}: {e}")
        return None


def calculate_regional_anomalies(ds_mean_var):
    """Calculates the regional anomaly based on the reference period."""
    # ds_mean_var is already the regional mean (2D: time or time, vertical_level)
    
    # Calculate the mean over time if it has vertical levels, for the reference period mean.
    # This ensures a single value is used for the reference, similar to T2m.
    if len(ds_mean_var.dims) > 1 and 'vertical' in [d.lower() for d in ds_mean_var.dims]:
        # Calculate the mean over all remaining dimensions (e.g., vertical level)
        ds_mean_time_series = ds_mean_var.mean(dim=[dim for dim in ds_mean_var.dims if dim != TIME_COORD_NAME])
    else:
        ds_mean_time_series = ds_mean_var # Already a 1D time series

    regional_ref = ds_mean_time_series.sel({TIME_COORD_NAME: slice(f'{REF_START}', f'{REF_END}')}).mean()
    regional_anomaly = ds_mean_time_series - regional_ref
    return regional_anomaly

# Define the exponential function for fitting
def exponential_func(t, a, b):
    # a: scaling factor, b: growth rate
    return a * np.exp(b * t)

def plot_t2m_anomalies(analysis_results):
    """
    Plots the T2m temperature anomaly (Moving Average + Exponential Fit).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    MA_WINDOW_MONTHS = 12 # 1-year moving average
    EXCLUDE_PERIOD_MONTHS = 6 # Exclude first/last 6 months for fitting
    
    print("\n--- Exponential Fit Results (Annual Growth Rate) ---")

    for region, results in analysis_results.items():
        for scenario, data in results.items():
            
            # --- 1. SMOOTHED DATA (12-MONTH MOVING AVERAGE) ---
            anom_raw = data['regional_anom']
            # Apply 12-month rolling mean
            anom_smoothed_1yr = anom_raw.rolling(
                {TIME_COORD_NAME: MA_WINDOW_MONTHS}, 
                center=True, 
                min_periods=1
            ).mean() 
            
            base_color = data['color']
            
            # Plot 12-month smoothed data (thin line, high transparency)
            ax.plot(
                anom_raw[TIME_COORD_NAME], anom_smoothed_1yr.values,
                label=f'{region.capitalize()} ({scenario.upper()}) - 1-Year MA', 
                color=base_color, 
                linewidth=1.0,
                alpha=0.6,
                linestyle=':'
            )
            
            # --- 2. EXPONENTIAL REGRESSION (TREND) ---
            
            # 2a. Define the safe data subset for fitting
            N_total = len(anom_raw)
            start_index = EXCLUDE_PERIOD_MONTHS
            end_index = N_total - EXCLUDE_PERIOD_MONTHS
            
            # Data used for fitting
            anom_fit_data = anom_smoothed_1yr.values[start_index:end_index]
            
            # Time array used for fitting (starts from 0, scaled to years)
            time_numeric = np.arange(N_total) / 12.0
            time_fit_data = time_numeric[start_index:end_index]
            
            try:
                # Initial guess (p0)
                popt, pcov = curve_fit(exponential_func, time_fit_data, anom_fit_data, p0=[anom_fit_data[0] if anom_fit_data[0] != 0 else 0.01, 0.01], maxfev=5000)
                b = popt[1] # Extract growth rate 'b'
                
                # Trend line values calculated over the entire period
                trend_line = exponential_func(time_numeric, *popt)
                
                # Plot the exponential trend line (Thick line)
                ax.plot(
                    anom_raw[TIME_COORD_NAME], trend_line,
                    label=f'{region.capitalize()} ({scenario.upper()}) - Exp. Trend (Growth Rate: {b*100:.3f}%/yr)', 
                    color=base_color, 
                    linewidth=3.0,
                    linestyle='-' if region == 'Arctic' else '--'
                )
                
                # Print regression results
                print(f"  {region.capitalize()} ({scenario.upper()}) - Exp. Growth Rate (b): {b:.6f} (Annual)")
                
            except RuntimeError:
                print(f"  {region.capitalize()} ({scenario.upper()}): WARNING - Exponential fit failed to converge.")
            
    ax.set_title(f"T2m Temperature Anomaly (2025-2094) - Smoothed Data and Exponential Trend", fontsize=14)
    ax.set_ylabel(r"T2m Anomaly ($\Delta T$ in K)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.axhline(0.0, color='black', linestyle=':', alpha=0.7)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_arctic_vs_europe_ratio(analysis_results):
    """
    Plots the Europe Amplification Ratio (EAR): Delta T_Europe / Delta T_Arctic.
    A value < 1.0 means the Arctic warms faster than Europe.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    arctic_results = analysis_results['Arctic']
    europe_results = analysis_results['Europe']
    
    # Smoothing window: 10 years (120 months) for the ratio plot
    SMOOTHING_WINDOW_MONTHS = 120 
    
    for scenario in SCENARIOS:
        # Raw regional anomalies
        anom_arctic = arctic_results[scenario.upper()]['regional_anom']
        anom_europe = europe_results[scenario.upper()]['regional_anom']
        
        # Apply 12-month rolling mean to the anomalies before calculating the ratio
        anom_arctic_smoothed = anom_arctic.rolling({TIME_COORD_NAME: 12}, center=True, min_periods=1).mean()
        anom_europe_smoothed = anom_europe.rolling({TIME_COORD_NAME: 12}, center=True, min_periods=1).mean()
        
        # Calculate Ratio on smoothed data
        # Only calculate ratio when the Arctic anomaly magnitude is significant (e.g., > 0.01 K)
        ratio_ear = np.where(np.abs(anom_arctic_smoothed) > 0.01, anom_europe_smoothed / anom_arctic_smoothed, np.nan)
        ratio_ear = xr.DataArray(ratio_ear, coords=anom_arctic.coords)
        
        # Further smoothing for the ratio plot (10-year rolling mean = 120 months)
        ratio_smoothed = ratio_ear.rolling(
            {TIME_COORD_NAME: SMOOTHING_WINDOW_MONTHS}, 
            center=True, 
            min_periods=1
        ).mean() 
        
        # Plot
        ratio_smoothed.plot(
            ax=ax, 
            label=f'Europe / Arctic Ratio ({scenario.upper()}) - 10-Year Rolling Mean', 
            color=SCENARIO_COLORS[scenario], 
            linewidth=2
        )

    ax.set_title(f"Europe Amplification Ratio (EAR): $\Delta T_{{Europe}} / \Delta T_{{Arctic}}$ ({PERIOD_START}-{PERIOD_END})", fontsize=14)
    ax.set_ylabel(r"Europe Amplification Ratio", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.8, label='Ratio = 1 (Equal Warming)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Ratio = 0.5 (Europe warms half as fast)')
    ax.set_ylim(-1.0, 1.5)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_sicf_evolution(sicf_analysis_results):
    """
    Plots the evolution of the Mean Sea Ice Fraction (SICF) in the Arctic.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 5))
    
    for scenario, data in sicf_analysis_results['Arctic'].items():
        # Plot Annual Mean
        data['regional_anom'].plot(
            label=f'Mean SICF ({scenario}) - Annual Mean', 
            color=data['color'], 
            linewidth=2
        )
    
    plt.title(f"Evolution of Mean Sea Ice Fraction (SICF) in the Arctic ({PERIOD_START}-{PERIOD_END})", fontsize=14)
    plt.ylabel("Mean Sea Ice Fraction (0 to 1)", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# --- 3. Main Execution Block ---

if __name__ == '__main__':
    
    # Initialization of results dictionaries
    t2m_analysis_results = {'Arctic': {}, 'Europe': {}}
    sicf_analysis_results = {'Arctic': {}, 'Europe': {}}
    
    # --- Step 1: T2M Analysis (Primary Quantification) ---
    print("\n--- T2M TEMPERATURE ANALYSIS (Primary Quantification) ---")

    for scenario in SCENARIOS:
        for region in REGIONS:
            
            # Load mean T2m time series
            ds_t2m_mean = load_raw_variable(scenario, region, 't2m', ORIGINAL_DATA_BASE_PATH)

            if ds_t2m_mean is None:
                continue

            # Calculate regional T2m anomalies
            regional_anom = calculate_regional_anomalies(ds_t2m_mean)

            # Store T2m results
            t2m_analysis_results[region.capitalize()][scenario.upper()] = {
                'regional_anom': regional_anom,
                'color': SCENARIO_COLORS[scenario]
            }
            
            # --- DEBUGGING T2M (End anomalies) ---
            regional_end = regional_anom.isel({TIME_COORD_NAME: -1}).item()
            print(f"  Final Anomaly ({scenario.upper()}-{region.upper()}): {regional_end:.4f} K")

    # Check if we have enough T2m data to proceed
    if 'Arctic' not in t2m_analysis_results or 'Europe' not in t2m_analysis_results or not t2m_analysis_results['Arctic'] or not t2m_analysis_results['Europe']:
         print("\nFATAL ERROR: Insufficient T2m data loaded for Arctic and/or Europe. Check paths and files.")
    else:
        # --- Step 2: Plot T2m Anomaly + Exponential Trend (Focus on Magnitude) ---
        plot_t2m_anomalies(t2m_analysis_results)
        
        # --- Step 3: Plot Arctic vs Europe Ratio (Focus on Relative Amplification) ---
        plot_arctic_vs_europe_ratio(t2m_analysis_results)


    # --- Step 4: Sea Ice Evolution (sicf) ---

    print("\n--- Sea Ice Fraction (sicf) Analysis ---")
    
    for scenario in SCENARIOS:
        # SICF is only relevant for the Arctic
        ds_sicf_mean = load_raw_variable(scenario, 'arctic', 'sicf', ORIGINAL_DATA_BASE_PATH)
        
        if ds_sicf_mean is None:
            continue
        
        # Calculate annual mean SICF for a clearer trend plot
        # Note: SICF is plotted as its absolute value (0-1), not an anomaly relative to the start
        sicf_annual = ds_sicf_mean.resample({TIME_COORD_NAME: 'YE'}).mean()
        
        sicf_analysis_results['Arctic'][scenario.upper()] = {
            'regional_anom': sicf_annual,
            'color': SCENARIO_COLORS[scenario]
        }

    if any(sicf_analysis_results['Arctic'].values()):
        plot_sicf_evolution(sicf_analysis_results)
    else:
        print("Skipping SICF analysis due to loading errors.")

    print("\nFinal analysis script executed, focusing on T2m, Amplification Ratio, and Sea Ice Fraction.")