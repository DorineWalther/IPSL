import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob

# --- 1. Configuration Paths and Data ---

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

# --- 2. Utility Functions ---

def load_raw_variable(scenario, region, variable_name, base_path):
    """
    Loads a specific variable from the ORIGINAL (non-cleaned) file path,
    calculates the time series of the spatial mean across lat/lon, and loads into memory.
    """
    print(f"Loading RAW variable '{variable_name}' for {scenario.upper()} - {region.upper()}")
    
    file_pattern = os.path.join(base_path, f'CM61-LR-scen-{scenario}_*_{region}.nc')
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"Error: No RAW files found for {scenario}-{region} at {base_path}")
        return None
        
    datasets = []
    try:
        for file_path in file_list:
            ds = xr.open_dataset(file_path, engine='netcdf4')
            
            if variable_name not in ds.data_vars:
                 print(f"Warning: Variable '{variable_name}' not found in {os.path.basename(file_path)}. Skipping.")
                 ds.close()
                 continue
                 
            ds_var = ds[variable_name].squeeze().to_dataset().load()
            datasets.append(ds_var)
            
        if not datasets:
            return None
            
        ds_full = xr.concat(datasets, dim=TIME_COORD_NAME)
        mean_dims = [dim for dim in ds_full[variable_name].dims if dim in ('lat', 'lon')]
        
        return ds_full[variable_name].mean(dim=mean_dims)
        
    except Exception as e:
        print(f"Error loading RAW variable '{variable_name}' for {scenario}-{region}: {e}")
        return None


def calculate_regional_anomalies(ds_mean_var):
    """Calculates the regional anomaly based on the reference period (2025-2034)."""
    
    if len(ds_mean_var.dims) > 1 and 'vertical' in [d.lower() for d in ds_mean_var.dims]:
        ds_mean_time_series = ds_mean_var.mean(dim=[dim for dim in ds_mean_var.dims if dim != TIME_COORD_NAME])
    else:
        ds_mean_time_series = ds_mean_var

    regional_ref = ds_mean_time_series.sel({TIME_COORD_NAME: slice(f'{REF_START}', f'{REF_END}')}).mean()
    regional_anomaly = ds_mean_time_series - regional_ref
    return regional_anomaly

# Define the exponential function for fitting
def exponential_func(t, a, b):
    return a * np.exp(b * t)

# --- 3. Plotting Functions ---

def plot_t2m_anomalies(analysis_results):
    """
    Plots the T2m temperature anomaly (1-Year MA and Exponential Fit).
    Legends in English. Variability band removed.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    MA_WINDOW_MONTHS = 12 # 1-year moving average (Signal)
    
    # Months to exclude at the start (reference period + half of the MA window)
    EXCLUDE_MONTHS_START = (REF_END - REF_START + 1) * 12 + int(MA_WINDOW_MONTHS / 2) 
    EXCLUDE_MONTHS_END = int(MA_WINDOW_MONTHS / 2)
    
    start_plot_year = REF_END + 1 # 2035 (for the title)
    
    print("\n--- Exponential Fit Results (Annual Growth Rate) ---")

    for region, results in analysis_results.items():
        for scenario, data in results.items():
            
            anom_raw = data['regional_anom']
            
            # --- 1. MOVING AVERAGE (MA) ---
            min_periods_ma = max(1, int(MA_WINDOW_MONTHS/2))

            anom_smoothed_1yr = anom_raw.rolling(
                {TIME_COORD_NAME: MA_WINDOW_MONTHS}, 
                center=True, 
                min_periods=min_periods_ma
            ).mean() 
            
            base_color = data['color']
            
            # Numerical time for regression (starts at t=0)
            N_total = len(anom_raw)
            time_numeric = np.arange(N_total) / 12.0 
            
            # --- 2. EXPONENTIAL REGRESSION ---
            anom_fit_data = anom_smoothed_1yr.values[min_periods_ma:-min_periods_ma]
            time_fit_data = time_numeric[min_periods_ma:-min_periods_ma]
            
            try:
                popt, pcov = curve_fit(exponential_func, time_fit_data, anom_fit_data, p0=[anom_fit_data[0] if anom_fit_data[0] != 0 else 0.01, 0.01], maxfev=5000)
                b = popt[1] 
                trend_line = exponential_func(time_numeric, *popt)
                print(f"  {region.capitalize()} ({scenario.upper()}) - Exp. Growth Rate (b): {b:.6f} (Annual)")

                # --- 3. FILTERING FOR PLOTTING (Excluding edges) ---
                
                idx_start = EXCLUDE_MONTHS_START
                idx_end = N_total - EXCLUDE_MONTHS_END
                
                # Slicing data for plotting
                anom_smoothed_plot = anom_smoothed_1yr.isel({TIME_COORD_NAME: slice(idx_start, idx_end)})
                trend_line_plot = trend_line[idx_start:idx_end]
                
                # --- 4. PLOTTING ---
                
                # Main line (12-month smoothed data)
                ax.plot(
                    anom_smoothed_plot[TIME_COORD_NAME], anom_smoothed_plot.values,
                    label=f'{region.capitalize()} ({scenario.upper()}) - 1-Year MA', 
                    color=base_color, 
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle='-'
                )
                
                # Exponential trend line (Long-term signal)
                ax.plot(
                    anom_smoothed_plot[TIME_COORD_NAME], trend_line_plot, 
                    label=f'{region.capitalize()} ({scenario.upper()}) - Exp. Trend (Growth Rate: {b*100:.3f}%/yr)', 
                    color=base_color, 
                    linewidth=3.0,
                    linestyle='--'
                )
                
            except RuntimeError:
                print(f"  {region.capitalize()} ({scenario.upper()}): WARNING - Exponential fit failed to converge.")
            
    ax.set_title(f"T2m Anomaly ($\Delta T$ vs. 2025-2034) - Starting {start_plot_year}", fontsize=14)
    ax.set_ylabel(r"T2m Anomaly ($\Delta T$ in K)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.axhline(0.0, color='black', linestyle=':', alpha=0.7)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_arctic_vs_europe_ratio(analysis_results):
    """
    Plots the Europe Amplification Ratio (EAR) with 10-Year MA Trend (solid line) 
    and the uncertainty band (1-year Std Dev). Legends are in English.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    arctic_results = analysis_results['Arctic']
    europe_results = analysis_results['Europe']
    
    # Smoothing Windows
    MA_WINDOW_MONTHS_ANOM = 12 # 1 year for smoothing anomalies
    MA_WINDOW_MONTHS_ROBUST = 120 # 10 years for the central tendency (Trend line)
    STDDEV_WINDOW_MONTHS = 12 # 1 year for the standard deviation (Uncertainty)

    # Months to exclude (Ref + half of the Robust MA window)
    EXCLUDE_MONTHS_START = (REF_END - REF_START + 1) * 12 + int(MA_WINDOW_MONTHS_ROBUST / 2)
    EXCLUDE_MONTHS_END = int(MA_WINDOW_MONTHS_ROBUST / 2)

    start_plot_year = REF_END + 1 # 2035

    # Correction: Ensure min_periods is at least 1
    min_periods_anom = max(1, int(MA_WINDOW_MONTHS_ANOM/2))
    min_periods_robust = max(1, int(MA_WINDOW_MONTHS_ROBUST/2))
    min_periods_std = max(1, int(STDDEV_WINDOW_MONTHS/2))


    for scenario in SCENARIOS:
        anom_arctic = arctic_results[scenario.upper()]['regional_anom']
        anom_europe = europe_results[scenario.upper()]['regional_anom']
        
        # 1. Pre-smoothing of anomalies (1 year)
        anom_arctic_smoothed = anom_arctic.rolling({TIME_COORD_NAME: MA_WINDOW_MONTHS_ANOM}, center=True, min_periods=min_periods_anom).mean()
        anom_europe_smoothed = anom_europe.rolling({TIME_COORD_NAME: MA_WINDOW_MONTHS_ANOM}, center=True, min_periods=min_periods_anom).mean()
        
        # 2. Calculate Variability Ratio (based on 1-year smoothing)
        ratio_ear_variability = np.where(np.abs(anom_arctic_smoothed) > 0.01, anom_europe_smoothed / anom_arctic_smoothed, np.nan)
        ratio_ear_variability = xr.DataArray(ratio_ear_variability, coords=anom_arctic.coords)
        
        # 3. Robust Ratio Trend (10-Year MA - Center for the Uncertainty and Trend line)
        ratio_smoothed_robust = ratio_ear_variability.rolling(
            {TIME_COORD_NAME: MA_WINDOW_MONTHS_ROBUST}, 
            center=True, 
            min_periods=min_periods_robust
        ).mean() 
        
        # 4. Calculate Uncertainty (1-year Moving Standard Deviation on the 1-year smoothed ratio)
        ratio_std_dev = ratio_ear_variability.rolling(
             {TIME_COORD_NAME: STDDEV_WINDOW_MONTHS}, 
             center=True, 
             min_periods=min_periods_std
        ).std()
        
        # --- Filtering for Plotting (Excluding edges) ---
        
        N_total = len(anom_arctic)
        idx_start = EXCLUDE_MONTHS_START
        idx_end = N_total - EXCLUDE_MONTHS_END

        # Filter Robust Ratio and Standard Deviation
        ratio_robust_plot = ratio_smoothed_robust.isel({TIME_COORD_NAME: slice(idx_start, idx_end)})
        std_dev_plot = ratio_std_dev.isel({TIME_COORD_NAME: slice(idx_start, idx_end)})
        
        # Calculate Confidence Limits (± 1 Std Dev)
        ratio_upper_bound = ratio_robust_plot + std_dev_plot
        ratio_lower_bound = ratio_robust_plot - std_dev_plot

        # 5. PLOTTING
        
        base_color = SCENARIO_COLORS[scenario]

        # Uncertainty Band (± 1 Std Dev)
        ax.fill_between(
            ratio_robust_plot[TIME_COORD_NAME].values,
            ratio_lower_bound.values,
            ratio_upper_bound.values,
            color=base_color,
            alpha=0.3,
            label=f'EAR ({scenario.upper()}) - $\pm 1 \sigma$ (1 yr Variability)'
        )
        
        # Central Trend Line (10-Year MA - Solid, thinner line)
        ax.plot(
            ratio_robust_plot[TIME_COORD_NAME].values,
            ratio_robust_plot.values,
            color=base_color,
            linewidth=2.5, # Reduced linewidth as requested
            alpha=1.0,
            linestyle='-',
            label=f'EAR ({scenario.upper()}) - 10-Year MA Trend' 
        )

    ax.set_title(f"Europe Amplification Ratio (EAR): $\Delta T_{{Europe}} / \Delta T_{{Arctic}}$ - Starting {start_plot_year}", fontsize=14)
    ax.set_ylabel(r"Europe Amplification Ratio", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.8, label='Ratio = 1 (Equal Warming)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Ratio = 0.5')
    ax.set_ylim(-0.5, 1.5) 
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_sicf_evolution(sicf_analysis_results):
    """
    Plots the evolution of the Mean Sea Ice Fraction (SICF) in the Arctic.
    Uncertainty band removed as requested. Legends are in English.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 5))
    
    start_plot_year = REF_END + 1 # 2035
    
    # We still need these variables for filtering purposes, but they won't be plotted as a band.
    STDDEV_WINDOW_YEARS = 1 
    min_periods_std_year = max(1, int(STDDEV_WINDOW_YEARS/2))


    for scenario, data in sicf_analysis_results['Arctic'].items():
        sicf_annual = data['regional_anom']
        
        # 1. Calculate 1-year mobile standard deviation (used here only for determining valid indices)
        sicf_std_dev = sicf_annual.rolling(
            {TIME_COORD_NAME: STDDEV_WINDOW_YEARS}, 
            center=True, 
            min_periods=min_periods_std_year
        ).std()
        
        # 2. Filtering (Excluding reference period + edges)
        sicf_annual_plot = sicf_annual.sel({TIME_COORD_NAME: slice(f'{start_plot_year}', None)})
        sicf_std_dev_plot = sicf_std_dev.sel({TIME_COORD_NAME: slice(f'{start_plot_year}', None)})
        
        # Exclude points where standard deviation is NaN (i.e., edge effects)
        valid_indices = ~np.isnan(sicf_std_dev_plot.values)
        
        sicf_plot = sicf_annual_plot.values[valid_indices]
        sicf_time = sicf_annual_plot[TIME_COORD_NAME].values[valid_indices]

        # 3. PLOTTING
        base_color = data['color']
        
        # Main line (Annual Mean) - ONLY the line is plotted
        plt.plot(
            sicf_time, sicf_plot,
            label=f'Mean SICF ({scenario.upper()}) - Annual Mean', 
            color=base_color, 
            linewidth=2
        )
        
    
    plt.title(f"Evolution of Mean Sea Ice Fraction (SICF) in the Arctic - Starting {start_plot_year}", fontsize=14)
    plt.ylabel("Mean Sea Ice Fraction (0 to 1)", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def extract_key_data_to_csv(t2m_analysis_results):
    """
    Extracts final anomaly and EAR values (instantaneous and trend) for debugging.
    """
    data_list = []
    
    for scenario in SCENARIOS:
        scenario_upper = scenario.upper()
        
        if scenario_upper not in t2m_analysis_results['Arctic'] or scenario_upper not in t2m_analysis_results['Europe']:
            continue
            
        anom_arctic = t2m_analysis_results['Arctic'][scenario_upper]['regional_anom']
        anom_europe = t2m_analysis_results['Europe'][scenario_upper]['regional_anom']

        # Use the last valid data point
        final_arctic_anom = anom_arctic.isel({TIME_COORD_NAME: -1}).item()
        final_europe_anom = anom_europe.isel({TIME_COORD_NAME: -1}).item()
        
        # Calculate final EAR
        final_ear = final_europe_anom / final_arctic_anom if final_arctic_anom != 0 else np.nan
        
        # Calculate 10-Year MA EAR (Robust Trend) at the end for context
        MA_WINDOW_MONTHS_ANOM = 12
        MA_WINDOW_MONTHS_ROBUST = 120
        min_periods_anom = max(1, int(MA_WINDOW_MONTHS_ANOM/2))
        min_periods_robust = max(1, int(MA_WINDOW_MONTHS_ROBUST/2))
        
        anom_arctic_smoothed = anom_arctic.rolling({TIME_COORD_NAME: MA_WINDOW_MONTHS_ANOM}, center=True, min_periods=min_periods_anom).mean()
        anom_europe_smoothed = anom_europe.rolling({TIME_COORD_NAME: MA_WINDOW_MONTHS_ANOM}, center=True, min_periods=min_periods_anom).mean()
        
        ratio_ear_variability = np.where(np.abs(anom_arctic_smoothed) > 0.01, anom_europe_smoothed / anom_arctic_smoothed, np.nan)
        ratio_ear_variability = xr.DataArray(ratio_ear_variability, coords=anom_arctic.coords)
        
        ratio_smoothed_robust = ratio_ear_variability.rolling(
            {TIME_COORD_NAME: MA_WINDOW_MONTHS_ROBUST}, 
            center=True, 
            min_periods=min_periods_robust
        ).mean() 
        
        # Final point of the robust trend (using a point before the end to avoid MA filter lag)
        final_robust_ear = ratio_smoothed_robust.isel({TIME_COORD_NAME: -int(MA_WINDOW_MONTHS_ROBUST/2)}).item()
        
        data_list.append({
            'Scenario': scenario_upper,
            'Final_Arctic_Anomaly_K': final_arctic_anom,
            'Final_Europe_Anomaly_K': final_europe_anom,
            'Final_EAR_Instantaneous': final_ear,
            'Final_EAR_10Y_MA_Trend': final_robust_ear
        })
        
    df = pd.DataFrame(data_list)
    output_filename = 'Arctic_Europe_Amplification_Summary.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nExtracted key data to {output_filename}")
    
    return output_filename

# --- 4. Main Execution Block ---

if __name__ == '__main__':
    
    # Initialization of results dictionaries
    t2m_analysis_results = {'Arctic': {}, 'Europe': {}}
    sicf_analysis_results = {'Arctic': {}, 'Europe': {}}
    
    # --- Step 1: T2m Analysis (Primary Quantification) ---
    print("\n--- T2m Temperature Analysis (Primary Quantification) ---")

    for scenario in SCENARIOS:
        for region in REGIONS:
            
            ds_t2m_mean = load_raw_variable(scenario, region, 't2m', ORIGINAL_DATA_BASE_PATH)

            if ds_t2m_mean is None:
                continue

            regional_anom = calculate_regional_anomalies(ds_t2m_mean)

            t2m_analysis_results[region.capitalize()][scenario.upper()] = {
                'regional_anom': regional_anom,
                'color': SCENARIO_COLORS[scenario]
            }
            
            regional_end = regional_anom.isel({TIME_COORD_NAME: -1}).item()
            print(f"  Final Anomaly ({scenario.upper()}-{region.upper()}): {regional_end:.4f} K")

    if 'Arctic' not in t2m_analysis_results or 'Europe' not in t2m_analysis_results or not t2m_analysis_results['Arctic'] or not t2m_analysis_results['Europe']:
         print("\nFATAL ERROR: Insufficient T2m data loaded for Arctic and/or Europe. Check paths and files.")
    else:
        # --- Step 2: Extract key data for debugging ---
        csv_filename = extract_key_data_to_csv(t2m_analysis_results)
        
        # --- Step 3: Plot T2m Anomaly + Exponential Trend (Focus on Magnitude) ---
        plot_t2m_anomalies(t2m_analysis_results)
        
        # --- Step 4: Plot Arctic vs Europe Ratio (Focus on Relative Amplification) ---
        plot_arctic_vs_europe_ratio(t2m_analysis_results)


        # --- Step 5: Sea Ice Evolution (sicf) ---

        print("\n--- Sea Ice Fraction (SICF) Analysis ---")
        
        for scenario in SCENARIOS:
            ds_sicf_mean = load_raw_variable(scenario, 'arctic', 'sicf', ORIGINAL_DATA_BASE_PATH)
            
            if ds_sicf_mean is None:
                continue
            
            sicf_annual = ds_sicf_mean.resample({TIME_COORD_NAME: 'YE'}).mean()
            
            sicf_analysis_results['Arctic'][scenario.upper()] = {
                'regional_anom': sicf_annual,
                'color': SCENARIO_COLORS[scenario]
            }

        if any(sicf_analysis_results['Arctic'].values()):
            plot_sicf_evolution(sicf_analysis_results)
        else:
            print("Skipping SICF analysis due to loading errors.")

        # --- Step 6: Debugging the Anomalous Growth Rates ---
        print("\n*** CRITICAL DEBUGGING NOTE ON GROWTH RATES ***")
        print("The exponential growth rates observed (e.g., Europe rate > Arctic rate in SSP245) suggest the assumption of exponential acceleration might not fully capture the regional physics (e.g., stabilization of Arctic warming post-sea-ice loss).")
        print("Rely more heavily on the 10-Year MA Trend and the instantaneous values in the CSV for conclusions about the Europe Amplification Ratio (EAR).")