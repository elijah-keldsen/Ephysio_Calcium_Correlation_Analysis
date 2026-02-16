import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xrscipy.signal.spectral import coherogram
from scipy.signal import correlate, correlation_lags
from scipy.signal import coherence
import pandas as pd
from src import misc_functions
import os
from src.multitaper_spectrogram_python import multitaper_spectrogram
from src2.miniscope.miniscope_data_manager import MiniscopeDataManager
from src2.ephys.ephys_api import EphysAPI
from src2.ephys.ephys_data_manager import EphysDataManager
from src2.multimodal.miniscope_ephys_alignment_utils import sync_neuralynx_miniscope_timestamps, find_ephys_idx_of_TTL_events
from scipy.stats import wilcoxon


#%% metadata
#35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 64, 83, 85, 86, 87, 88, 90, 92, 93, 94, 96, 97, 99, 101, 103, 104, 105, 107, 108, 112
 
line_nums = [46, 47, 64, 88, 97, 101]

# Define the two ephys channels to compare
channel_1 = 'PFCLFPvsCBEEG'  # First ephys channel from ['PFCLFPvsCBEEG', 'PFCEEGvsCBEEG', 'CBvsPCEEG']
channel_2 = 'PFCEEGvsCBEEG'  # Second ephys channel

CUT = [0.5, 4] #frequency range to analyze
data = {
    "dexmedetomidine: 0.00045": [46, 47, 64, 88, 97, 101],
    "dexmedetomidine: 0.0003": [40, 41, 48, 87, 93, 94],
    "propofol": [36, 43, 44, 86, 99, 103],
    "sleep": [35, 37, 38, 83, 90, 92],
    "ketamine": [39, 42, 45, 85, 96, 112]
}

# Time range mapping for each line number as pairs of numbers
#time period in minutes
selections = { 
    46:  [[1,20], [28.24, 75]], 
     47:  [[1,20], [28.24, 75]],  
    64:  [[4,17], [28.24, 75]], 
    88:  [[1,8], [28.24, 83.24]],
    97:  [[1,13], [28.24, 83.24]], 
    101:  [[1,25], [37, 90]],
    
    40:  [[8,20], [55, 75]], 
    41:  [[10,19], [60, 68]],
    48:  [[1,20], [25, 35]],
    87:  [[5,13], [73, 85]],
    93:  [[18,28], [75, 95]],
    94:  [[1,20], [75, 90]] , 
    
    36:  [[1,11], [55, 67]],
    43:  [[15, 20], [40, 60]], 
    44:  [[0,21], [40, 65]],
    86:  [[5,16], [33, 65]],
    99:  [[0,19], [33, 45]],
    103:  [[1,17], [38, 65]], 
    
    35:  [[1,20], [28.24, 83.24]], 
    37:  [[1,20], [28.24, 83.24]], 
    38:  [[1,20], [28.24, 83.24]],
    83:  [[1,20], [28.24, 83.24]],
    90:  [[1,20], [28.24, 83.24]], 
    92:  [[1,20], [28.24, 83.24]], 
    
    39:  [[10,20], [38, 50]], 
    42:  [[1,20], [40, 51]], 
    45:  [[1,15], [40, 60]], 
    85:  [[14,24], [30, 50]], 
    96:  [[1,12], [38, 55]],
    112:  [[1,10], [40, 60]]
}

# Create a reverse mapping from numbers to drug types
number_to_drug = {num: drug for drug, numbers in data.items() for num in numbers}

#These times in minutes are filled in automatically in load_experiment
drug_infusion_start = {}


#%% Load experiment 
def load_experiment(line_num, channel_1_name, channel_2_name):
    """
    Load two ephys signals for coherence analysis.
    
    Input:
    line_num : int
        Experiment line number
    channel_1_name : str
        Name of the first ephys channel
    channel_2_name : str
        Name of the second ephys channel
        
    Returns:
    channel_1_object : Channel object for first ephys signal
    channel_2_object : Channel object for second ephys signal
    fr : float
        Sampling rate (frame rate)
    """
    print(f'Loading experiment {line_num} with channels: {channel_1_name} and {channel_2_name}...')
    
    # Load miniscope data manager (needed for timestamp syncing and frame rate)
    miniscope_data_manager = MiniscopeDataManager(line_num=line_num, filenames=[], auto_import_data=False)
    metadata = miniscope_data_manager._get_miniscope_metadata()
    if metadata:
        miniscope_data_manager.metadata.update(metadata)
        fr = miniscope_data_manager.metadata['frameRate']
    else:
        fr = 30
    
    # Load first ephys channel
    print(f"Loading channel 1: {channel_1_name}")
    ephys_api_1 = EphysAPI()
    ephys_api_1.run(line_num, channel_name=channel_1_name, remove_artifacts=False, filter_type=None, 
                    filter_range=[0.5, 4], plot_channel=False, plot_spectrogram=False, plot_phases=False, 
                    logging_level="CRITICAL")
    channel_1_object = ephys_api_1.ephys_data_manager.get_channel(channel_name=channel_1_name)
    
    # Load second ephys channel - IMPORTANT: We need to use the same ephys_data_manager to ensure both channels
    # come from the same recording and have synchronized timestamps
    print(f"Loading channel 2: {channel_2_name}")
    ephys_api_2 = EphysAPI()
    ephys_api_2.run(line_num, channel_name=channel_2_name, remove_artifacts=False, filter_type=None, 
                    filter_range=[0.5, 4], plot_channel=False, plot_spectrogram=False, plot_phases=False, 
                    logging_level="CRITICAL")
    channel_2_object = ephys_api_2.ephys_data_manager.get_channel(channel_name=channel_2_name)
    
    # Update drug_infusion_start with when the anesthesia starts to be administered
    if number_to_drug[line_num] != 'sleep':
        for idx, label in enumerate(channel_1_object.events['labels']):
            if 'heat' in label:
                print(f"Drug infusion start time label: {label}")
                start_time = channel_1_object.events['timestamps'][idx]
                drug_infusion_start[line_num] = start_time
                break
    
    # Sync timestamps using channel 1 to get the TTL events
    # This ensures both ephys channels are aligned to the same miniscope timestamps
    tCaIm, low_confidence_periods, channel_1_object, miniscope_data_manager = sync_neuralynx_miniscope_timestamps(
        channel_1_object, miniscope_data_manager, delete_TTLs=True, 
        fix_TTL_gaps=True, only_experiment_events=True)
    ephys_idx_all_TTL_events, ephys_idx_ca_events = find_ephys_idx_of_TTL_events(
        tCaIm, channel=channel_1_object, frame_rate=fr, ca_events_idx=None, all_TTL_events=True)
    
    # Downsample both channels using the same TTL event indices
    channel_1_object.signal = channel_1_object.signal[ephys_idx_all_TTL_events]
    channel_2_object.signal = channel_2_object.signal[ephys_idx_all_TTL_events]
    
    # Update sampling rates to match the downsampled rate
    channel_1_object.sampling_rate = np.array(fr)
    channel_2_object.sampling_rate = np.array(fr)
    
    # Replace NaNs with zeros if present (important for ketamine experiments)
    if np.any(np.isnan(channel_1_object.signal)):
        print(f"Line {line_num}: Replacing NaNs in channel 1 ({channel_1_name}) with zeros...")
        channel_1_object.signal = np.nan_to_num(channel_1_object.signal, nan=0.0)
    
    if np.any(np.isnan(channel_2_object.signal)):
        print(f"Line {line_num}: Replacing NaNs in channel 2 ({channel_2_name}) with zeros...")
        channel_2_object.signal = np.nan_to_num(channel_2_object.signal, nan=0.0)
    
    return channel_1_object, channel_2_object, fr
    

#%% Plot lines
WIDTH = 0.5

def plot_lines_ax(line_num, ax):
    "The two functions in this cell help to draw lines on graphs to mark different events"
    
    a = ax.axvline(x=selections[line_num][0][0], color='red', linestyle='--', linewidth=WIDTH)
    ax.axvline(x=selections[line_num][0][0] + 0.33, color='red', linestyle='--', linewidth=WIDTH)
    x = ax.axvline(x=selections[line_num][0][1], color='red', linestyle='--', linewidth=WIDTH)
    ax.axvline(x=selections[line_num][0][1] + 0.33, color='red', linestyle='--', linewidth=WIDTH)
    
    b = ax.axvline(x=selections[line_num][1][0], color='orange', linestyle='--', linewidth=WIDTH)
    ax.axvline(x=selections[line_num][1][0] + 0.33, color='orange', linestyle='--', linewidth=WIDTH)
    y = ax.axvline(x=selections[line_num][1][1], color='orange', linestyle='--', linewidth=WIDTH)
    ax.axvline(x=selections[line_num][1][1] + 0.33, color='orange', linestyle='--', linewidth=WIDTH)
    
    if number_to_drug[line_num] != 'sleep':
        c =  ax.axvline(x=drug_infusion_start[line_num]/60, color='red', linestyle='-', linewidth=1)
        ax.axvline(x=drug_infusion_start[line_num]/60, color='red', linestyle='-', linewidth=1)
        ax.legend([a, b, c], ['Control', 'Treatment', 'Start of Drug Infusion'])
    else:
        ax.legend([a, b], ['Control', 'Treatment'])

def plot_lines_plt(line_num, plt):
    a = plt.axvline(x=selections[line_num][0][0], color='red', linestyle='--', linewidth=WIDTH)
    plt.axvline(x=selections[line_num][0][0] + 0.33, color='red', linestyle='--', linewidth=WIDTH)
    x = plt.axvline(x=selections[line_num][0][1], color='red', linestyle='--', linewidth=WIDTH)
    plt.axvline(x=selections[line_num][0][1] + 0.33, color='red', linestyle='--', linewidth=WIDTH)
    
    b = plt.axvline(x=selections[line_num][1][0], color='orange', linestyle='--', linewidth=WIDTH)
    plt.axvline(x=selections[line_num][1][0] + 0.33, color='orange', linestyle='--', linewidth=WIDTH)
    b = plt.axvline(x=selections[line_num][1][1], color='orange', linestyle='--', linewidth=WIDTH)
    plt.axvline(x=selections[line_num][1][1] + 0.33, color='orange', linestyle='--', linewidth=WIDTH)
    
    if number_to_drug[line_num] != 'sleep':
        c =  plt.axvline(x=drug_infusion_start[line_num]/60, color='red', linestyle='-', linewidth=1)
        plt.axvline(x=drug_infusion_start[line_num]/60, color='red', linestyle='-', linewidth=1)
        plt.legend([a, b, c], ['Control', 'Treatment', 'Start of Drug Infusion'])
    else:
        plt.legend([a, b], ['Control', 'Treatment'])
 

    
    
#%% Plot Coherogram
    
def plot_coherogram(line_num, drug, signal_1, signal_2, fr, channel_1_name, channel_2_name): 
    """Plots the coherogram of two signals"""
    
    # Define the spectrogram parameters
    windowLength = 5  # in seconds
    windowStep = 2.5  # in seconds
    
    # Compute the overlap ratio
    overlap_ratio = 1 - (windowStep / windowLength)
    
    # Convert time coordinates to minutes
    signal_1_data = xr.DataArray(
        signal_1,
        dims=["time"],
        coords={"time": np.arange(len(signal_1)) / (fr * 60)}  # Convert to minutes
    )
    
    signal_2_data = xr.DataArray(
        signal_2,
        dims=["time"],
        coords={"time": np.arange(len(signal_2)) / (fr * 60)}  # Convert to minutes
    )
    
    # Compute coherogram
    coh = coherogram(
        signal_1_data,
        signal_2_data,
        fs=fr,  # Sampling frequency
        seglen=windowLength,  # Match spectrogram's window length
        overlap_ratio=overlap_ratio,  # Calculated overlap ratio
        nrolling=8,  # Rolling average over 8 FFT windows
        window="hann"  # Hann window
    )
    
    coh["time"] = coh["time"] / 60
    
    # Use imshow for plotting the coherogram
    coh_magnitude = abs(coh) ** 2  # Coherence magnitude squared
    im = coh_magnitude.plot.imshow(
        cmap="viridis", 
        robust=False,  # Disable automatic color scaling
        vmin=0, vmax=0.7,  # Fix the color range to [0, 0.7]
        figsize=(10, 6)  # Set figure size here instead
    )
    
    # Get the current axis and figure
    fig = plt.gcf()
    ax = plt.gca()
    
    # Set frequency range for y-axis (0 to 15 Hz to match original)
    ax.set_ylim([0, 15])
    ax.set_xlabel("Time (minutes)", fontsize=14)
    ax.set_ylabel("Frequency (Hz)", fontsize=14)
    ax.set_title(f"Coherogram: {channel_1_name} vs {channel_2_name} | Line {line_num} | {drug}", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plot_lines_ax(line_num, ax)
    plt.tight_layout()
    
    # SANITIZE THE FILENAME
    safe_drug_name = drug.replace(":", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_channel_1 = channel_1_name.replace("/", "_").replace("\\", "_")
    safe_channel_2 = channel_2_name.replace("/", "_").replace("\\", "_")
    
    # Save the figure
    save_path = r"C:\Path\To\Results\Directory"
    os.makedirs(save_path, exist_ok=True)
    svg_path = os.path.join(save_path, f"coherogram_{safe_channel_1}_vs_{safe_channel_2}_{safe_drug_name}_line_{line_num}.svg")
    
    try:
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
        if os.path.exists(svg_path):
            actual_size = os.path.getsize(svg_path)
            print(f"Successfully saved coherogram: {svg_path}")
            print(f"File size: {actual_size:,} bytes")
        else:
            print(f"ERROR: File was not created at {svg_path}")
    except Exception as e:
        print(f"Error saving coherogram: {e}")
    
    plt.show()
    plt.close(fig)


#%% Plot signal with slices

def plot_signal_with_slices(signal, line_num, title="Signal"):
    """Plot the entire signal with highlighted control and treatment periods"""
    
    # Convert signal indices to minutes
    time_minutes = np.arange(len(signal)) / (fr * 60)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(time_minutes, signal, linewidth=0.5, color='black', alpha=0.7)
    
    # Highlight control period
    control_start, control_end = selections[line_num][0]
    ax.axvspan(control_start, control_end, alpha=0.2, color='red', label='Control')
    
    # Highlight treatment period
    treatment_start, treatment_end = selections[line_num][1]
    ax.axvspan(treatment_start, treatment_end, alpha=0.2, color='orange', label='Treatment')
    
    # Add drug infusion line if applicable
    if number_to_drug[line_num] != 'sleep' and line_num in drug_infusion_start:
        ax.axvline(x=drug_infusion_start[line_num]/60, color='red', linestyle='-', 
                   linewidth=2, label='Drug Infusion Start')
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Signal amplitude')
    ax.set_title(f'{title} - Line {line_num} | {number_to_drug[line_num]}')
    ax.legend()
    plt.tight_layout()
    
    # Save the figure
    safe_title = title.replace(":", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")
    save_path = r"C:\Path\To\Results\Directory"
    os.makedirs(save_path, exist_ok=True)
    svg_path = os.path.join(save_path, f"signal_{safe_title}_line_{line_num}.svg")
    
    try:
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
        if os.path.exists(svg_path):
            print(f"Successfully saved signal plot: {svg_path}")
    except Exception as e:
        print(f"Error saving signal plot: {e}")
    
    plt.show()
    plt.close(fig)


#%% Compute power

def compute_power(line_num, fr, signal, plot_mean_power=True, title="Signal"):
    """Compute and plot power spectrogram"""
    
    # Compute multitaper spectrogram
    tapers = [5, 9]  # [time_bandwidth, number_of_tapers]
    nw = tapers[0]
    k = tapers[1]
    
    spect, stimes, sfreqs = multitaper_spectrogram(
        signal, 
        fs=fr,
        frequency_range=CUT,
        time_bandwidth=nw, 
        num_tapers=k, 
        window_params=[5, 2.5]
    )
    
    # Convert times to minutes
    stimes_minutes = stimes / 60
    
    # Plot spectrogram
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(stimes_minutes, sfreqs, 10 * np.log10(spect), 
                       shading='auto', cmap='viridis')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'{title} Power Spectrogram - Line {line_num}')
    plot_lines_ax(line_num, ax)
    plt.colorbar(im, ax=ax, label='Power (dB)')
    plt.tight_layout()
    
    # Save the spectrogram
    safe_title = title.replace(":", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")
    save_path = r"C:\Path\To\Results\Directory"
    os.makedirs(save_path, exist_ok=True)
    svg_path = os.path.join(save_path, f"spectrogram_{safe_title}_line_{line_num}.svg")
    
    try:
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
        if os.path.exists(svg_path):
            print(f"Successfully saved spectrogram: {svg_path}")
    except Exception as e:
        print(f"Error saving spectrogram: {e}")
    
    plt.show()
    plt.close(fig)
    
    if plot_mean_power:
        # Compute mean power across frequencies
        mean_power = np.mean(spect, axis=0)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(stimes_minutes, mean_power, linewidth=1, color='blue')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Mean Power')
        ax.set_title(f'{title} Mean Power - Line {line_num}')
        plot_lines_ax(line_num, ax)
        plt.tight_layout()
        
        # Save the mean power plot
        svg_path = os.path.join(save_path, f"mean_power_{safe_title}_line_{line_num}.svg")
        
        try:
            fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
            if os.path.exists(svg_path):
                print(f"Successfully saved mean power plot: {svg_path}")
        except Exception as e:
            print(f"Error saving mean power plot: {e}")
        
        plt.show()
        plt.close(fig)


#%% Slice signal

def slice_signal(signal, line_num, fr):
    """Slice signal into control and treatment periods"""
    
    # Get time ranges in minutes
    control_range = selections[line_num][0]
    treatment_range = selections[line_num][1]
    
    # Convert to sample indices
    control_start_idx = int(control_range[0] * 60 * fr)
    control_end_idx = int(control_range[1] * 60 * fr)
    treatment_start_idx = int(treatment_range[0] * 60 * fr)
    treatment_end_idx = int(treatment_range[1] * 60 * fr)
    
    # Slice
    control_signal = signal[control_start_idx:control_end_idx]
    treatment_signal = signal[treatment_start_idx:treatment_end_idx]
    
    return control_signal, treatment_signal


#%% Filter frequency

def filter_frequency(signal_1, signal_2, fr, cut=CUT):
    """Apply bandpass filter to both signals"""
    from scipy.signal import butter, filtfilt
    
    # Design Butterworth bandpass filter
    nyquist = fr / 2
    low = cut[0] / nyquist
    high = cut[1] / nyquist
    b, a = butter(4, [low, high], btype='band')
    
    # Apply filter
    filtered_signal_1 = filtfilt(b, a, signal_1)
    filtered_signal_2 = filtfilt(b, a, signal_2)
    
    return filtered_signal_1, filtered_signal_2


#%% Compute stats

def compute_stats(signal_1, signal_2, line_num, period_name, fr):
    """
    Compute statistics for a pair of signals.
    
    Returns:
    --------
    list : [Signal 1 Power, Signal 2 Power, Coherence, Cross-correlation, Lag]
    """
    
    # Compute power (variance)
    power_1 = np.var(signal_1)
    power_2 = np.var(signal_2)
    
    # Compute coherence
    freqs, coh = coherence(signal_1, signal_2, fs=fr, nperseg=int(fr*5))
    freq_mask = (freqs >= CUT[0]) & (freqs <= CUT[1])
    mean_coherence = np.mean(coh[freq_mask])
    
    # Compute cross-correlation
    xcorr = correlate(signal_1, signal_2, mode='full')
    xcorr_normalized = xcorr / (np.std(signal_1) * np.std(signal_2) * len(signal_1))
    max_xcorr = np.max(np.abs(xcorr_normalized))
    
    # Compute lag
    lags = correlation_lags(len(signal_1), len(signal_2), mode='full')
    lag_at_max = lags[np.argmax(np.abs(xcorr_normalized))] / fr  # Convert to seconds
    
    print(f"  {period_name} - Signal 1 Power: {power_1:.4f}, Signal 2 Power: {power_2:.4f}, "
          f"Coherence: {mean_coherence:.4f}, XC: {max_xcorr:.4f}, Lag: {lag_at_max:.4f}s")
    
    return [power_1, power_2, mean_coherence, max_xcorr, lag_at_max]


#%% Compute mean and std

def compute_mean_std(df):
    """Compute mean and standard deviation for each measurement"""
    
    if df is None or df.empty:
        return None
    
    grouped = df.groupby("Measurement").agg(
        Control_Mean=("Control", "mean"),
        Control_Std=("Control", "std"),
        Treatment_Mean=("Treatment", "mean"),
        Treatment_Std=("Treatment", "std"),
        Ratio_Mean=("Ratio", "mean"),
        Ratio_Std=("Ratio", "std")
    ).reset_index()
    
    return grouped


#%% Add p-values

def add_p_values(original_df, averaged_df):
    """Add p-values from Wilcoxon signed-rank test"""
    
    if original_df is None or averaged_df is None:
        return averaged_df
    
    p_value_map = {}
    
    for measurement in original_df["Measurement"].unique():
        subset = original_df[original_df["Measurement"] == measurement]
        control = subset["Control"].values
        treatment = subset["Treatment"].values
        
        # Perform Wilcoxon signed-rank test
        if len(control) >= 3 and len(treatment) >= 3:
            try:
                stat, p_value = wilcoxon(control, treatment)
            except ValueError:
                p_value = np.nan
        else:
            p_value = np.nan
        p_value_map[measurement] = p_value
    
    # Map the computed p-values back to the averaged_df
    averaged_df["P-value"] = averaged_df["Measurement"].map(p_value_map)
    return averaged_df




    


#%% main

# Define the results directory
results_dir = r"C:\Path\To\Results\Directory"
os.makedirs(results_dir, exist_ok=True)


# Initialize data lists
sleep_data, dex_30_data, dex_45_data, iso_data, prop_data, ketamine_data = [], [], [], [], [], []
rows = ["Signal 1 Power", "Signal 2 Power", "Coherence", "XC", "Lag"]

for line in line_nums:
    print(f'\n{"-"*60}')
    print(f'Running line number: {line}')
    print(f'{"-"*60}')
    drug = number_to_drug.get(line)
    
    # Load both ephys channels
    channel_1_object, channel_2_object, fr = load_experiment(line, channel_1, channel_2)
    
    # Grab the signals
    signal_1 = channel_1_object.signal
    signal_2 = channel_2_object.signal
    
    # Plot both entire signals with the slices that are stored in selections, overlapped
    plot_signal_with_slices(signal_1, line, title=f"Ephys Channel 1: {channel_1}")
    plot_signal_with_slices(signal_2, line, title=f"Ephys Channel 2: {channel_2}")
    
    # Plot ephys spectrogram and mean power signal for both channels
    compute_power(line, fr, signal_1, plot_mean_power=True, title=f"Channel 1: {channel_1}")
    compute_power(line, fr, signal_2, plot_mean_power=True, title=f"Channel 2: {channel_2}")
    
    # Plot coherence of both signals
    plot_coherogram(line, drug, signal_1, signal_2, fr, channel_1, channel_2)
    
    # Slice signals into small segments to use for data analysis
    control_signal_1, treatment_signal_1 = slice_signal(signal_1, line, fr)
    control_signal_2, treatment_signal_2 = slice_signal(signal_2, line, fr)
    
    # Filter the control/treatment signals
    filtered_control_1, filtered_control_2 = filter_frequency(control_signal_1, control_signal_2, fr, cut=CUT)
    filtered_treatment_1, filtered_treatment_2 = filter_frequency(treatment_signal_1, treatment_signal_2, fr, cut=CUT)
 
    
    # Compute stats for control and treatment
    control_list = compute_stats(filtered_control_1, filtered_control_2, line, "Control", fr)
    treatment_list = compute_stats(filtered_treatment_1, filtered_treatment_2, line, "Treatment", fr)
    
    # Compute the ratio for each pair of control and treatment values    
    ratios = []
    
    for control, treatment in zip(control_list, treatment_list):
        if abs(control) < 1e-10:  # Use a small threshold instead of int(control)
            ratios.append(np.nan)  # Use NaN for invalid ratios
        else:
            ratios.append(treatment / control)
    
    # Combine into a row with columns ["lineNum", "Control", "Treatment", "Ratio"]
    combined_data = {
        "line_num": [line] * len(rows),
        "Measurement": rows,
        "Control": control_list,
        "Treatment": treatment_list,
        "Ratio": ratios
    }
    
    # Convert to DataFrame row format and append to appropriate list
    row_df = pd.DataFrame(combined_data)
    

    if line in data["dexmedetomidine: 0.00045"]:
        dex_45_data.append(row_df)
    elif line in data["dexmedetomidine: 0.0003"]:
        dex_30_data.append(row_df)
    elif line in data['propofol']:
        prop_data.append(row_df)
    elif line in data['sleep']:
        sleep_data.append(row_df)
    elif line in data["ketamine"]:
        ketamine_data.append(row_df)
    else:
        raise("Key error. Is this number correct?")
        

# Combine all rows into final DataFrames
dex_45_df = pd.concat(dex_45_data, ignore_index=True) if dex_45_data else None
dex_30_df = pd.concat(dex_30_data, ignore_index=True) if dex_30_data else None
sleep_df = pd.concat(sleep_data, ignore_index=True) if sleep_data else None
ketamine_df = pd.concat(ketamine_data, ignore_index=True) if ketamine_data else None
prop_df = pd.concat(prop_data, ignore_index=True) if prop_data else None


# Compute averaged DataFrames
dex_45_avg_df = compute_mean_std(dex_45_df)
dex_30_avg_df = compute_mean_std(dex_30_df)
sleep_avg_df = compute_mean_std(sleep_df)
ketamine_avg_df = compute_mean_std(ketamine_df)
prop_avg_df = compute_mean_std(prop_df)


# add p-values to each averaged dataframe
dex_45_avg_df = add_p_values(dex_45_df, dex_45_avg_df)
dex_30_avg_df = add_p_values(dex_30_df, dex_30_avg_df)
sleep_avg_df = add_p_values(sleep_df, sleep_avg_df)
ketamine_avg_df = add_p_values(ketamine_df, ketamine_avg_df)
prop_avg_df = add_p_values(prop_df, prop_avg_df)

print(f"Control: {control_list}, Treatment: {treatment_list}, Ratios: {ratios}")

# Create directory for results if it doesn't exist
results_dir = os.path.join(os.pardir, "Coherence_Analysis")
os.makedirs(results_dir, exist_ok=True)

# Save DataFrames as CSV files
def save_group(df, avg, name, channel_label):
    """Save data without overwriting — includes channel in filename."""
    if df is not None and avg is not None and not df.empty:
        
        safe_channel = channel_label.replace("/", "_").replace("\\", "_")  # sanitize
        
        df.to_csv(os.path.join(results_dir, f"{name}_{safe_channel}_data.csv"), index=False)
        avg.to_csv(os.path.join(results_dir, f"{name}_{safe_channel}_avg_data.csv"), index=False)
        
        print(f"{name.capitalize()} data saved for comparison: {channel_label}")
    else:
        print(f"{name.capitalize()} data not found or empty.")

# Create a descriptive channel label for filenames
channel_label = f"{channel_1}_vs_{channel_2}"

save_group(dex_45_df, dex_45_avg_df, "dex_45", channel_label)
save_group(dex_30_df, dex_30_avg_df, "dex_30", channel_label)
save_group(prop_df,   prop_avg_df,   "propofol", channel_label)
save_group(sleep_df,  sleep_avg_df,  "sleep", channel_label)
save_group(ketamine_df, ketamine_avg_df, "ketamine", channel_label)