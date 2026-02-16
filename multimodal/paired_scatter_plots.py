"""
Paired Scatter Plot Generation for Control vs. Treatment Comparisons
=====================================================================
Generates publication-quality scatter plots following eNeuro journal guidelines.

This script loads data directly from source (ephys and calcium imaging files)
using the same methodology as coherence_analysis_ephys_calcium.py to ensure
consistency and reproducibility.

Journal Requirements Implemented:
- Single column width: 8.5 cm (3.35 inches)
- 300 dpi minimum for grayscale
- RGB format for color, saved as TIFF and EPS
- No top/right spines (no boxed panels)
- Grayscale-friendly with texture/marker differentiation
- Colorblind accessible (no red-green combinations)

Aesthetic standards matched to isc_coherogram_analysis.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence, correlate, correlation_lags
from scipy.stats import wilcoxon
import os
from typing import Dict, List, Tuple, Optional

# Import the same modules used in coherence_analysis_ephys_calcium.py
from src import misc_functions
from src.multitaper_spectrogram_python import multitaper_spectrogram
from src2.miniscope.miniscope_data_manager import MiniscopeDataManager
from src2.ephys.ephys_api import EphysAPI
from src2.ephys.ephys_data_manager import EphysDataManager
from src2.multimodal.miniscope_ephys_alignment_utils import (
    sync_neuralynx_miniscope_timestamps, 
    find_ephys_idx_of_TTL_events
)


# ============================================================================
# EXPERIMENTAL METADATA (copied from coherence_analysis_ephys_calcium.py)
# ============================================================================

# Drug groups with their subject line numbers
data = {
    "dexmedetomidine: 0.00045": [46, 47, 64, 88, 97, 101],
    "dexmedetomidine: 0.0003": [40, 41, 48, 87, 93, 94],
    "propofol": [36, 43, 44, 86, 99, 103],
    "sleep": [35, 37, 38, 83, 90, 92],
    "ketamine": [39, 42, 45, 85, 96, 112]
}

# Time range mapping for each line number (in minutes)
# Format: [[control_start, control_end], [treatment_start, treatment_end]]
selections = { 
    46:  [[1,20], [28.24, 75]], 
    47:  [[1,20], [28.24, 75]],  
    64:  [[4,17], [28.24, 75]], 
    88:  [[1,8], [28.24, 83.24]],
    97:  [[1,13], [28.24, 83.24]], 
    101: [[1,25], [37, 90]],
    
    40:  [[8,20], [55, 75]], 
    41:  [[10,19], [60, 68]],
    48:  [[1,20], [25, 35]],
    87:  [[5,13], [73, 85]],
    93:  [[18,28], [75, 95]],
    94:  [[1,20], [75, 90]], 
    
    36:  [[1,11], [55, 67]],
    43:  [[15, 20], [40, 60]], 
    44:  [[0,21], [40, 65]],
    86:  [[5,16], [33, 65]],
    99:  [[0,19], [33, 45]],
    103: [[1,17], [38, 65]], 
    
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
    112: [[1,10], [40, 60]]
}

# Create reverse mapping from line numbers to drug types
number_to_drug = {num: drug for drug, numbers in data.items() for num in numbers}

# Drug infusion start times (populated during load_experiment)
drug_infusion_start = {}


# ============================================================================
# CONFIGURATION
# ============================================================================

# Analysis parameters
CUT = [0.5, 4]  # Frequency range for bandpass filtering (Hz)

# Data paths
CALCIUM_DATA_PATH = r"C:\Users\ericm\Desktop\meanFluorescence"

# Journal-compliant figure sizes (in inches)
SINGLE_COLUMN_WIDTH = 3.35      # 8.5 cm
ONE_POINT_FIVE_COLUMN = 4.57    # 11.6 cm  
TWO_COLUMN_WIDTH = 6.93         # 17.6 cm

# DPI settings
COLOR_DPI = 300
GRAYSCALE_DPI = 300

# Grayscale-friendly color palette (colorblind accessible)
COLORS = {
    'control_fill': '#FFFFFF',       # White fill for control points
    'treatment_fill': '#FFFFFF',     # White fill for treatment points
    'edge_color': '#000000',         # Black edges
    'connecting_line': '#666666',    # Gray connecting lines
    'unity_line': '#000000',         # Black unity line
    'violin_fill': '#CCCCCC',        # Light gray violin fill
}

# Marker styles (different shapes for accessibility)
MARKERS = {
    'control': 'o',      # Circle
    'treatment': 's',    # Square
}

# Font sizes (journal-appropriate)
FONTS = {
    'axis_label': 9,
    'tick_label': 8,
    'annotation': 7,
    'legend': 7,
}

# Output directory
RESULTS_PATH = r"C:\Users\ericm\Desktop\Correlation_poster\Scatter_Plots"


# ============================================================================
# DATA LOADING FUNCTIONS (from coherence_analysis_ephys_calcium.py)
# ============================================================================

def load_experiment(line_num: int, channel: str, calcium_signal_filepath: str = None):
    """
    Load and synchronize ephys and calcium imaging data for a given subject.
    
    This function is copied from coherence_analysis_ephys_calcium.py to ensure
    identical data loading methodology.
    """
    print(f'  Loading data for line {line_num}...')
    
    # Load miniscope data
    miniscope_data_manager = MiniscopeDataManager(
        line_num=line_num, 
        filenames=[], 
        auto_import_data=False
    )
    metadata = miniscope_data_manager._get_miniscope_metadata()
    
    if metadata:
        miniscope_data_manager.metadata.update(metadata)
        fr = miniscope_data_manager.metadata['frameRate']
    else:
        fr = 30  # Default frame rate
    
    # Load ephys data
    ephys_api = EphysAPI()
    ephys_api.run(
        line_num, 
        channel_name=channel, 
        remove_artifacts=False, 
        filter_type=None,
        filter_range=[0.5, 4], 
        plot_channel=False, 
        plot_spectrogram=False, 
        plot_phases=False, 
        logging_level="CRITICAL"
    )
    channel_object = ephys_api.ephys_data_manager.get_channel(channel_name=channel)
    
    # Extract drug infusion start time (for non-sleep experiments)
    if number_to_drug[line_num] != 'sleep':
        for idx, label in enumerate(channel_object.events['labels']):
            if 'heat' in label:
                start_time = channel_object.events['timestamps'][idx]
                drug_infusion_start[line_num] = start_time
                break
    
    # Sync timestamps between Neuralynx and miniscope
    tCaIm, low_confidence_periods, channel_object, miniscope_data_manager = \
        sync_neuralynx_miniscope_timestamps(
            channel_object, 
            miniscope_data_manager, 
            delete_TTLs=True,
            fix_TTL_gaps=True, 
            only_experiment_events=True
        )
    
    ephys_idx_all_TTL_events, ephys_idx_ca_events = find_ephys_idx_of_TTL_events(
        tCaIm, 
        channel=channel_object, 
        frame_rate=fr, 
        ca_events_idx=None, 
        all_TTL_events=True
    )
    
    # Downsample ephys to match miniscope frame rate
    channel_object.signal = channel_object.signal[ephys_idx_all_TTL_events]
    channel_object.sampling_rate = np.array(fr)
    
    # Load calcium signal from file
    if calcium_signal_filepath:
        miniscope_data_manager.mean_fluorescence_dict = np.load(calcium_signal_filepath)
    
    # Handle NaN values (common in ketamine experiments)
    if np.any(np.isnan(channel_object.signal)):
        print(f"    Replacing NaNs in EEG with zeros...")
        channel_object.signal = np.nan_to_num(channel_object.signal, nan=0.0)
    
    return channel_object, miniscope_data_manager, fr


def slice_signal(signal: np.ndarray, line_num: int, fr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Slice signal into control and treatment segments based on time selections."""
    control_start_idx = int(selections[line_num][0][0] * fr * 60)
    control_end_idx = int(selections[line_num][0][1] * fr * 60)
    treatment_start_idx = int(selections[line_num][1][0] * fr * 60)
    treatment_end_idx = int(selections[line_num][1][1] * fr * 60)
    
    sliced_control = signal[control_start_idx:control_end_idx]
    sliced_treatment = signal[treatment_start_idx:treatment_end_idx]
    
    return sliced_control, sliced_treatment


def filter_frequency(eeg_signal: np.ndarray, calcium_signal: np.ndarray, 
                     fr: float, cut: List[float] = CUT) -> Tuple[np.ndarray, np.ndarray]:
    """Apply bandpass filtering to EEG and calcium signals."""
    filtered_calcium = misc_functions.filterData(
        calcium_signal, n=2, cut=cut, 
        ftype='butter', btype='bandpass', fs=fr
    )
    filtered_eeg = EphysDataManager._filter_data(
        data=eeg_signal, n=2, cut=cut, 
        ftype='butter', fs=fr, btype='bandpass', bodePlot=False
    )
    
    return filtered_eeg, filtered_calcium


# ============================================================================
# ANALYSIS FUNCTIONS (from coherence_analysis_ephys_calcium.py)
# ============================================================================

def compute_coherence(signal1: np.ndarray, signal2: np.ndarray, fr: float) -> float:
    """Compute mean spectral coherence between two signals in the slow-wave band."""
    f, Cxy = coherence(signal1, signal2, fs=fr)
    freq_indices = np.where((f >= CUT[0]) & (f <= CUT[1]))[0]
    Cxy = Cxy[freq_indices]
    mean_coherence = np.mean(Cxy)
    
    return mean_coherence


def compute_xc(eeg_signal: np.ndarray, calcium_signal: np.ndarray, fr: float) -> Tuple[float, float]:
    """Compute normalized cross-correlation between EEG and calcium signals."""
    # Normalize signals
    norm_eeg = (eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) * len(eeg_signal))
    norm_calcium = (calcium_signal - np.mean(calcium_signal)) / np.std(calcium_signal)
    
    # Compute cross-correlation
    nxcorr = correlate(norm_eeg, norm_calcium, mode='full')
    
    # Calculate lags in seconds
    lags = correlation_lags(len(norm_eeg), len(norm_calcium), mode='full') / fr
    
    # Limit to +/-10 seconds
    max_display_lag = 10
    lag_mask = (lags >= -max_display_lag) & (lags <= max_display_lag)
    lags = lags[lag_mask]
    nxcorr = nxcorr[lag_mask]
    
    # Find maximum cross-correlation
    max_xc = nxcorr[np.argmax(nxcorr)]
    lag_at_max_xc = lags[np.argmax(nxcorr)]
    
    return max_xc, lag_at_max_xc


def compute_power(signal: np.ndarray, fr: float, windowLength: int = 60) -> float:
    """Compute mean power in the slow-wave band using multitaper spectrogram."""
    # Spectrogram parameters
    fs = fr
    timeBandwidth = 2
    numTapers = timeBandwidth * 2 - 1
    windowStep = 3
    windowParams = [windowLength, windowStep]
    freqLims = [0, 20]
    minNfft = 0
    detrendOpt = 'constant'
    multiprocess = True
    nJobs = 3
    weighting = 'unity'
    plotOn = False
    returnFig = False
    climScale = False
    verbose = False
    xyflip = False
    
    # Compute multitaper spectrogram
    power_matrix, times, frequencies = multitaper_spectrogram(
        signal, fs, freqLims, timeBandwidth, numTapers, windowParams,
        minNfft, detrendOpt, multiprocess, nJobs, weighting, plotOn,
        returnFig, climScale, verbose, xyflip
    )
    
    # Convert to decibels
    power_array = 10 * np.log10(power_matrix)
    
    # Extract slow-wave band (0.5-4 Hz)
    freq_indices = np.where((frequencies >= 0.5) & (frequencies <= 4.0))[0]
    sliced_power = power_array[freq_indices]
    mean_power = np.mean(sliced_power)
    
    return float(mean_power)


def compute_stats(eeg_signal: np.ndarray, calcium_signal: np.ndarray, fr: float) -> Dict[str, float]:
    """Compute all statistics for a pair of signals."""
    eeg_power = compute_power(eeg_signal, fr, windowLength=60)
    calcium_power = compute_power(calcium_signal, fr, windowLength=60)
    coh = compute_coherence(eeg_signal, calcium_signal, fr)
    xc, lag = compute_xc(eeg_signal, calcium_signal, fr)
    
    return {
        'EEG Power': eeg_power,
        'Calcium Power': calcium_power,
        'Coherence': coh,
        'XC': xc,
        'Lag': lag
    }


# ============================================================================
# MAIN DATA COLLECTION FUNCTION
# ============================================================================

def collect_all_data(channel: str, line_nums: List[int] = None) -> Dict[str, pd.DataFrame]:
    """Load and process all experimental data, computing statistics for each subject."""
    print("\n" + "="*80)
    print("COLLECTING DATA FROM SOURCE")
    print(f"Channel: {channel}")
    print("="*80 + "\n")
    
    # If no specific line numbers provided, use all
    if line_nums is None:
        line_nums = list(selections.keys())
    
    # Initialize storage for each drug condition
    condition_data = {drug: [] for drug in data.keys()}
    
    # Process each subject
    for line_num in line_nums:
        print(f"\n--- Processing Line {line_num} ---")
        
        drug = number_to_drug.get(line_num)
        if drug is None:
            print(f"  WARNING: Line {line_num} not found in drug mapping, skipping...")
            continue
        
        try:
            # Load experiment data
            calcium_filepath = os.path.join(
                CALCIUM_DATA_PATH, 
                f"meanFluorescence_{line_num}.npz"
            )
            
            channel_object, miniscope_data_manager, fr = load_experiment(
                line_num, 
                channel, 
                calcium_signal_filepath=calcium_filepath
            )
            
            # Extract signals
            eeg_signal = channel_object.signal
            calcium_signal = miniscope_data_manager.mean_fluorescence_dict['meanFluorescence']
            
            # Slice into control and treatment periods
            control_eeg, treatment_eeg = slice_signal(eeg_signal, line_num, fr)
            control_calcium, treatment_calcium = slice_signal(calcium_signal, line_num, fr)
            
            # Apply bandpass filtering
            filtered_control_eeg, filtered_control_calcium = filter_frequency(
                control_eeg, control_calcium, fr, cut=CUT
            )
            filtered_treatment_eeg, filtered_treatment_calcium = filter_frequency(
                treatment_eeg, treatment_calcium, fr, cut=CUT
            )
            
            # Compute statistics for control and treatment
            print("  Computing control statistics...")
            control_stats = compute_stats(filtered_control_eeg, filtered_control_calcium, fr)
            
            print("  Computing treatment statistics...")
            treatment_stats = compute_stats(filtered_treatment_eeg, filtered_treatment_calcium, fr)
            
            # Store results
            for metric in ['EEG Power', 'Calcium Power', 'Coherence', 'XC', 'Lag']:
                ctrl_val = control_stats[metric]
                treat_val = treatment_stats[metric]
                
                # Compute ratio (avoid division by zero)
                if abs(ctrl_val) < 1e-10:
                    ratio = np.nan
                else:
                    ratio = treat_val / ctrl_val
                
                condition_data[drug].append({
                    'line_num': line_num,
                    'Measurement': metric,
                    'Control': ctrl_val,
                    'Treatment': treat_val,
                    'Ratio': ratio
                })
            
            print(f"  Successfully processed line {line_num}")
            
        except Exception as e:
            print(f"  ERROR processing line {line_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert to DataFrames
    data_dict = {}
    for drug, records in condition_data.items():
        if records:
            data_dict[drug] = pd.DataFrame(records)
            print(f"\n{drug}: {len(data_dict[drug]) // 5} subjects")
    
    return data_dict


# ============================================================================
# SCATTER PLOT FUNCTIONS
# ============================================================================

def create_paired_scatter_plot(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    subject_ids: List[str],
    metric_name: str,
    condition_name: str,
    units: str = "",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create a paired scatter plot comparing control vs. treatment values."""
    
    # Create figure with single-column width
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    
    n_subjects = len(control_values)
    
    # Calculate axis limits (symmetric, with padding)
    all_values = np.concatenate([control_values, treatment_values])
    valid_values = all_values[~np.isnan(all_values)]
    
    if len(valid_values) == 0:
        print(f"Warning: No valid values for {metric_name} in {condition_name}")
        plt.close(fig)
        return fig
    
    data_min = np.min(valid_values)
    data_max = np.max(valid_values)
    data_range = data_max - data_min
    padding = data_range * 0.1
    
    axis_min = data_min - padding
    axis_max = data_max + padding
    
    # Plot unity line (y = x)
    unity_x = np.linspace(axis_min, axis_max, 100)
    ax.plot(unity_x, unity_x, 
            linestyle='--', 
            color=COLORS['unity_line'], 
            linewidth=1, 
            zorder=1,
            label='Unity (y=x)')
    
    # Plot connecting lines between paired points
    for i in range(n_subjects):
        if not (np.isnan(control_values[i]) or np.isnan(treatment_values[i])):
            ax.plot([control_values[i], control_values[i]], 
                   [control_values[i], treatment_values[i]],
                   '-', 
                   color=COLORS['connecting_line'], 
                   alpha=0.5, 
                   linewidth=0.75, 
                   zorder=2)
    
    # Plot data points
    for i in range(n_subjects):
        if not (np.isnan(control_values[i]) or np.isnan(treatment_values[i])):
            ax.plot(control_values[i], treatment_values[i],
                   marker=MARKERS['treatment'],
                   markersize=6,
                   markerfacecolor=COLORS['treatment_fill'],
                   markeredgecolor=COLORS['edge_color'],
                   markeredgewidth=0.75,
                   zorder=3)
    
    # Set axis limits
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    
    # Format axis labels
    unit_str = f" ({units})" if units else ""
    ax.set_xlabel(f'Control{unit_str}', fontsize=FONTS['axis_label'])
    ax.set_ylabel(f'Treatment{unit_str}', fontsize=FONTS['axis_label'])
    
    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=FONTS['tick_label'])
    
    # Remove top and right spines (journal requirement)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linewidth=0.5, linestyle=':')
    ax.set_axisbelow(True)
    
    # Compute statistics
    valid_mask = ~(np.isnan(control_values) | np.isnan(treatment_values))
    ctrl_valid = control_values[valid_mask]
    treat_valid = treatment_values[valid_mask]
    
    if len(ctrl_valid) > 0:
        try:
            stat, p_value = wilcoxon(ctrl_valid, treat_valid, alternative='two-sided')
            
            # Effect size (Cohen's d for paired samples)
            differences = treat_valid - ctrl_valid
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0
            
            # Format p-value
            if p_value < 0.001:
                p_str = "p < 0.001"
            elif p_value < 0.01:
                p_str = f"p = {p_value:.3f}"
            else:
                p_str = f"p = {p_value:.2f}"
            
            # Add statistical annotation
            stats_text = f"n = {len(ctrl_valid)}\n{p_str}\nd = {cohens_d:.2f}"
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=FONTS['annotation'],
                   verticalalignment='top',
                   horizontalalignment='left',
                   bbox=dict(boxstyle='round', 
                            facecolor='white',
                            edgecolor='black', 
                            linewidth=0.5, 
                            alpha=0.9))
            
        except ValueError as e:
            print(f"Could not compute statistics: {e}")
    
    # Make axes equal for proper unity line representation
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save figures
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as TIFF (journal requirement)
        tiff_path = save_path if save_path.endswith('.tiff') else save_path + '.tiff'
        fig.savefig(tiff_path, format='tiff', bbox_inches='tight', dpi=COLOR_DPI,
                   pil_kwargs={'compression': 'tiff_lzw'})
        print(f"  Saved TIFF: {tiff_path}")
        
        # Save as EPS (vector format for journal)
        eps_path = tiff_path.replace('.tiff', '.eps')
        fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi=COLOR_DPI)
        print(f"  Saved EPS: {eps_path}")
        
        # Save as SVG for editing
        svg_path = tiff_path.replace('.tiff', '.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=COLOR_DPI)
        print(f"  Saved SVG: {svg_path}")
    
    plt.show()
    
    return fig


def create_paired_dot_plot(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    subject_ids: List[str],
    metric_name: str,
    condition_name: str,
    units: str = "",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create a paired dot plot (violin + individual points) showing control vs treatment."""
    
    # Create figure with single-column width
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.9))
    
    # Filter valid pairs
    valid_mask = ~(np.isnan(control_values) | np.isnan(treatment_values))
    ctrl_valid = control_values[valid_mask]
    treat_valid = treatment_values[valid_mask]
    n_subjects = len(ctrl_valid)
    
    if n_subjects == 0:
        print(f"Warning: No valid paired values for {metric_name}")
        plt.close(fig)
        return fig
    
    # Plot violin distributions (grayscale)
    parts = ax.violinplot([ctrl_valid, treat_valid],
                          positions=[1, 2],
                          showmeans=False,
                          showmedians=True,
                          widths=0.6)
    
    # Style violins in grayscale
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['violin_fill'])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
        pc.set_linewidth(0.75)
    
    # Style median lines
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    
    # Style min/max bars
    for partname in ['cbars', 'cmins', 'cmaxes']:
        if partname in parts:
            parts[partname].set_color('black')
            parts[partname].set_linewidth(0.75)
    
    # Plot individual points with connecting lines
    for i in range(n_subjects):
        # Connecting line
        ax.plot([1, 2], [ctrl_valid[i], treat_valid[i]],
               '-',
               color=COLORS['connecting_line'],
               alpha=0.5,
               linewidth=0.75,
               zorder=1)
        
        # Control point (circle)
        ax.plot(1, ctrl_valid[i],
               marker=MARKERS['control'],
               markersize=5,
               markerfacecolor=COLORS['control_fill'],
               markeredgecolor=COLORS['edge_color'],
               markeredgewidth=0.75,
               zorder=2)
        
        # Treatment point (square)
        ax.plot(2, treat_valid[i],
               marker=MARKERS['treatment'],
               markersize=5,
               markerfacecolor=COLORS['treatment_fill'],
               markeredgecolor=COLORS['edge_color'],
               markeredgewidth=0.75,
               zorder=2)
    
    # Formatting
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Control', 'Treatment'], fontsize=FONTS['tick_label'])
    
    unit_str = f" ({units})" if units else ""
    ax.set_ylabel(f'{metric_name}{unit_str}', fontsize=FONTS['axis_label'])
    
    ax.tick_params(axis='both', which='major', labelsize=FONTS['tick_label'])
    
    # Remove top and right spines (journal requirement)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid on y-axis only
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, linestyle=':')
    ax.set_axisbelow(True)
    
    # Compute and display statistics
    try:
        stat, p_value = wilcoxon(ctrl_valid, treat_valid, alternative='two-sided')
        
        # Effect size
        differences = treat_valid - ctrl_valid
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Format p-value
        if p_value < 0.001:
            p_str = "p < 0.001"
        elif p_value < 0.01:
            p_str = f"p = {p_value:.3f}"
        else:
            p_str = f"p = {p_value:.2f}"
        
        stats_text = f"n = {n_subjects}\n{p_str}\nd = {cohens_d:.2f}"
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=FONTS['annotation'],
               verticalalignment='top',
               horizontalalignment='left',
               bbox=dict(boxstyle='round',
                        facecolor='white',
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.9))
        
    except ValueError as e:
        print(f"Could not compute statistics: {e}")
    
    # Set x-axis limits with padding
    ax.set_xlim([0.4, 2.6])
    
    plt.tight_layout()
    
    # Save figures
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        tiff_path = save_path if save_path.endswith('.tiff') else save_path + '.tiff'
        fig.savefig(tiff_path, format='tiff', bbox_inches='tight', dpi=COLOR_DPI,
                   pil_kwargs={'compression': 'tiff_lzw'})
        print(f"  Saved TIFF: {tiff_path}")
        
        eps_path = tiff_path.replace('.tiff', '.eps')
        fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi=COLOR_DPI)
        print(f"  Saved EPS: {eps_path}")
        
        svg_path = tiff_path.replace('.tiff', '.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=COLOR_DPI)
        print(f"  Saved SVG: {svg_path}")
    
    plt.show()
    
    return fig


def create_multi_condition_scatter(
    data_dict: Dict[str, pd.DataFrame],
    metric_name: str,
    units: str = "",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create a multi-panel scatter plot comparing all drug conditions."""
    
    conditions = list(data_dict.keys())
    n_conditions = len(conditions)
    
    # Determine grid layout
    if n_conditions <= 3:
        n_cols = n_conditions
        n_rows = 1
    else:
        n_cols = 3
        n_rows = int(np.ceil(n_conditions / 3))
    
    # Create figure with 2-column width
    fig_width = TWO_COLUMN_WIDTH
    fig_height = (TWO_COLUMN_WIDTH / n_cols) * n_rows * 0.9
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    if n_conditions == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Hide unused subplots
    for idx in range(n_conditions, len(axes)):
        axes[idx].set_visible(False)
    
    # Global min/max for consistent scaling
    all_control = []
    all_treatment = []
    for condition, df in data_dict.items():
        subset = df[df['Measurement'] == metric_name]
        all_control.extend(subset['Control'].dropna().values)
        all_treatment.extend(subset['Treatment'].dropna().values)
    
    if len(all_control) == 0 or len(all_treatment) == 0:
        print(f"No data found for metric: {metric_name}")
        plt.close(fig)
        return fig
    
    global_min = min(min(all_control), min(all_treatment))
    global_max = max(max(all_control), max(all_treatment))
    data_range = global_max - global_min
    padding = data_range * 0.15
    axis_min = global_min - padding
    axis_max = global_max + padding
    
    # Plot each condition
    for idx, (condition, df) in enumerate(data_dict.items()):
        ax = axes[idx]
        
        # Extract data for this metric
        subset = df[df['Measurement'] == metric_name]
        ctrl = subset['Control'].values
        treat = subset['Treatment'].values
        
        # Filter valid pairs
        valid_mask = ~(np.isnan(ctrl) | np.isnan(treat))
        ctrl_valid = ctrl[valid_mask]
        treat_valid = treat[valid_mask]
        
        # Unity line
        unity_x = np.linspace(axis_min, axis_max, 100)
        ax.plot(unity_x, unity_x,
               linestyle='--',
               color=COLORS['unity_line'],
               linewidth=0.75,
               zorder=1)
        
        # Plot data points
        for i in range(len(ctrl_valid)):
            # Connecting line
            ax.plot([ctrl_valid[i], ctrl_valid[i]],
                   [ctrl_valid[i], treat_valid[i]],
                   '-',
                   color=COLORS['connecting_line'],
                   alpha=0.4,
                   linewidth=0.5,
                   zorder=2)
            
            # Data point
            ax.plot(ctrl_valid[i], treat_valid[i],
                   marker=MARKERS['treatment'],
                   markersize=5,
                   markerfacecolor=COLORS['treatment_fill'],
                   markeredgecolor=COLORS['edge_color'],
                   markeredgewidth=0.6,
                   zorder=3)
        
        # Axis limits and labels
        ax.set_xlim([axis_min, axis_max])
        ax.set_ylim([axis_min, axis_max])
        
        # Clean up condition name for display
        display_name = condition.replace("dexmedetomidine: ", "Dex ")
        display_name = display_name.replace("0.00045", "High").replace("0.0003", "Low")
        display_name = display_name.capitalize()
        
        ax.set_title(display_name, fontsize=FONTS['axis_label'], fontweight='bold', pad=5)
        
        unit_str = f" ({units})" if units else ""
        
        # Only add axis labels to edge plots
        if idx >= (n_rows - 1) * n_cols:  # Bottom row
            ax.set_xlabel(f'Control{unit_str}', fontsize=FONTS['axis_label'])
        if idx % n_cols == 0:  # Left column
            ax.set_ylabel(f'Treatment{unit_str}', fontsize=FONTS['axis_label'])
        
        ax.tick_params(axis='both', which='major', labelsize=FONTS['tick_label'])
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid
        ax.grid(True, alpha=0.2, linewidth=0.5, linestyle=':')
        ax.set_axisbelow(True)
        
        # Statistics
        if len(ctrl_valid) > 1:
            try:
                _, p_value = wilcoxon(ctrl_valid, treat_valid, alternative='two-sided')
                
                if p_value < 0.001:
                    p_str = "p<.001"
                elif p_value < 0.01:
                    p_str = f"p={p_value:.3f}"
                else:
                    p_str = f"p={p_value:.2f}"
                
                ax.text(0.03, 0.97, f"n={len(ctrl_valid)}\n{p_str}",
                       transform=ax.transAxes,
                       fontsize=FONTS['annotation'] - 1,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white',
                                edgecolor='black', linewidth=0.4, alpha=0.9))
            except:
                pass
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save figures
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        tiff_path = save_path if save_path.endswith('.tiff') else save_path + '.tiff'
        fig.savefig(tiff_path, format='tiff', bbox_inches='tight', dpi=COLOR_DPI,
                   pil_kwargs={'compression': 'tiff_lzw'})
        print(f"  Saved TIFF: {tiff_path}")
        
        eps_path = tiff_path.replace('.tiff', '.eps')
        fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi=COLOR_DPI)
        print(f"  Saved EPS: {eps_path}")
        
        svg_path = tiff_path.replace('.tiff', '.svg')
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=COLOR_DPI)
        print(f"  Saved SVG: {svg_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_scatter_plots(
    channel: str = 'CBvsPCEEG',
    line_nums: List[int] = None,
    output_dir: str = RESULTS_PATH
):
    """Generate all scatter plots by loading data directly from source files."""
    
    print("\n" + "="*80)
    print("GENERATING PAIRED SCATTER PLOTS FROM SOURCE DATA")
    print(f"Channel: {channel}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all data from source files
    data_dict = collect_all_data(channel, line_nums)
    
    if not data_dict:
        print("ERROR: No data collected!")
        return
    
    # Define metrics to plot
    metrics = {
        'EEG Power': {'units': 'dB', 'description': 'Slow-wave EEG power'},
        'Calcium Power': {'units': 'dB', 'description': 'Slow-wave calcium power'},
        'Coherence': {'units': '', 'description': 'EEG-calcium coherence'},
        'XC': {'units': '', 'description': 'Cross-correlation'},
    }
    
    # Generate individual scatter plots for each condition and metric
    print("\n" + "="*80)
    print("GENERATING INDIVIDUAL SCATTER PLOTS")
    print("="*80 + "\n")
    
    for condition_name, df in data_dict.items():
        print(f"\nCondition: {condition_name}")
        
        # Clean condition name for filenames
        safe_condition = condition_name.replace(":", "_").replace(" ", "_").replace(".", "")
        
        for metric_name, metric_info in metrics.items():
            print(f"  Metric: {metric_name}")
            
            # Extract data for this metric
            subset = df[df['Measurement'] == metric_name]
            
            if len(subset) == 0:
                print(f"    No data found, skipping...")
                continue
            
            ctrl = subset['Control'].values
            treat = subset['Treatment'].values
            subjects = subset['line_num'].astype(str).tolist()
            
            # Generate paired scatter plot
            save_path = os.path.join(
                output_dir, 
                f"scatter_{safe_condition}_{metric_name.replace(' ', '_')}_{channel}"
            )
            
            create_paired_scatter_plot(
                control_values=ctrl,
                treatment_values=treat,
                subject_ids=subjects,
                metric_name=metric_name,
                condition_name=condition_name,
                units=metric_info['units'],
                save_path=save_path
            )
            
            # Generate paired dot plot
            save_path_dot = os.path.join(
                output_dir,
                f"dotplot_{safe_condition}_{metric_name.replace(' ', '_')}_{channel}"
            )
            
            create_paired_dot_plot(
                control_values=ctrl,
                treatment_values=treat,
                subject_ids=subjects,
                metric_name=metric_name,
                condition_name=condition_name,
                units=metric_info['units'],
                save_path=save_path_dot
            )
    
    # Generate multi-condition comparison plots
    print("\n" + "="*80)
    print("GENERATING MULTI-CONDITION COMPARISON PLOTS")
    print("="*80 + "\n")
    
    for metric_name, metric_info in metrics.items():
        print(f"Metric: {metric_name}")
        
        save_path = os.path.join(
            output_dir,
            f"multi_scatter_{metric_name.replace(' ', '_')}_{channel}"
        )
        
        create_multi_condition_scatter(
            data_dict=data_dict,
            metric_name=metric_name,
            units=metric_info['units'],
            save_path=save_path
        )
    
    print("\n" + "="*80)
    print("SCATTER PLOT GENERATION COMPLETE")
    print(f"Figures saved to: {output_dir}")
    print("="*80 + "\n")
    
    return data_dict


# ============================================================================
# FIGURE LEGEND TEMPLATES
# ============================================================================

FIGURE_LEGENDS = """
================================================================================
FIGURE LEGENDS (Copy to manuscript after reference list)
================================================================================

Figure X. Paired comparison of [METRIC] between control and treatment periods.

(A) Scatter plot showing individual subject values with control period values 
on the x-axis and treatment period values on the y-axis. The dashed diagonal 
line represents unity (y=x); points above this line indicate higher values 
during treatment compared to control. (B) Violin plots with individual paired 
data points showing the distribution of [METRIC] values during control (left) 
and treatment (right) periods. Lines connect paired observations from the same 
subject. Circles represent control values; squares represent treatment values. 
Statistical comparison performed using Wilcoxon signed-rank test 
(two-tailed, alpha=0.05). Effect size reported as Cohen's d.

Figure Contributions: [Author name] acquired the data. [Author name] performed 
the analysis. [Author name] prepared the figure.

--------------------------------------------------------------------------------

Figure Y. Multi-condition comparison of [METRIC] across drug treatments.

Scatter plots comparing control versus treatment [METRIC] values for each 
experimental condition: (A) Dexmedetomidine high dose (0.00045 mg/kg), 
(B) Dexmedetomidine low dose (0.0003 mg/kg), (C) Propofol, (D) Ketamine, 
(E) Natural sleep. Dashed lines indicate unity (y=x). Points above the unity 
line indicate treatment values exceeding control values. Statistical comparisons 
performed using Wilcoxon signed-rank test (two-tailed). n, number of subjects; 
p, p-value.

Figure Contributions: [Author name] acquired the data. [Author name] performed 
the analysis. [Author name] prepared the figure.

================================================================================
"""


if __name__ == "__main__":
    # Print figure legend templates
    print(FIGURE_LEGENDS)
    
    # Run the main analysis - loads data directly from source
    # Change channel here to analyze different electrode pairs:
    # Options: 'CBvsPCEEG', 'PFCLFPvsCBEEG', 'PFCEEGvsCBEEG'
    
    data_dict = generate_all_scatter_plots(
        channel='PFCLFPvsCBEEG',
        line_nums=None,  # None = process all subjects; or specify e.g., [46, 47, 64]
        output_dir=RESULTS_PATH
    )