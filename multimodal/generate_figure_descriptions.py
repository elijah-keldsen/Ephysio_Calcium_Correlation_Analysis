"""
Figure Description and Population Statistics Generator
=======================================================
Generates eNeuro-compliant figure descriptions and population statistics
for multimodal ephys-calcium imaging analysis.

This script follows the methodology from:
- coherence_analysis_ephys_calcium.py (coherence, power, cross-correlation)
- eeg_calcium_scatter_plots.py (hexbin correlation analysis)

Output:
- Separate .txt files for each drug condition with:
  1. Population summary statistics (table format)
  2. Prose-format text for manuscript methods/results
  3. Individual figure descriptions

Author: Generated for eNeuro manuscript preparation
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, wilcoxon, t as t_dist
from scipy.signal import butter, filtfilt, correlate, correlation_lags, coherence
from typing import Tuple, Optional, Dict, List

# Import project modules
from src import misc_functions
from src.multitaper_spectrogram_python import multitaper_spectrogram
from src2.miniscope.miniscope_data_manager import MiniscopeDataManager
from src2.ephys.ephys_api import EphysAPI
from src2.ephys.ephys_data_manager import EphysDataManager
from src2.multimodal.miniscope_ephys_alignment_utils import (
    sync_neuralynx_miniscope_timestamps, 
    find_ephys_idx_of_TTL_events
)

# Try to import statsmodels for autocorrelation estimation
try:
    from statsmodels.tsa.stattools import acf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available - using simplified effective N estimation")


# ============================================================================
# EXPERIMENTAL METADATA
# ============================================================================

# Drug groups with their subject line numbers (excluding sleep as requested)
data = {
    "dexmedetomidine_0.00045": [46, 47, 64, 88, 97, 101],
    "dexmedetomidine_0.0003": [40, 41, 48, 87, 93, 94],
    "propofol": [36, 43, 44, 86, 99, 103],
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

FILTER_CONFIG = {
    'lowcut': 0.5,              # Hz - lower bound of bandpass
    'highcut': 4.0,             # Hz - upper bound of bandpass  
    'order': 2,                 # Butterworth filter order
    'edge_trim_seconds': 5.0,   # Seconds to trim from each edge after filtering
}

STATISTICS_CONFIG = {
    'correct_autocorrelation': True,
    'confidence_level': 0.95,
    'max_acf_lags': 100,
    'bootstrap_iterations': 10000,
}

# Channel to analyze
CHANNEL = 'PFCLFPvsCBEEG'  # Options: 'PFCLFPvsCBEEG', 'PFCEEGvsCBEEG', 'CBvsPCEEG'

CHANNEL_DESCRIPTIONS = {
    'CBvsPCEEG': 'Cerebellum vs Parietal Cortex EEG',
    'PFCLFPvsCBEEG': 'Prefrontal Cortex LFP vs Cerebellum EEG',
    'PFCEEGvsCBEEG': 'Prefrontal Cortex EEG vs Cerebellum EEG'
}

# Data paths
CALCIUM_DATA_PATH = r"C:\Users\ericm\Desktop\meanFluorescence"
RESULTS_PATH = r"C:\Users\ericm\Desktop\Correlation_poster\Figure_Descriptions"


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_experiment(line_num: int, channel: str, calcium_signal_filepath: str = None):
    """
    Load and synchronize ephys and calcium imaging data for a given subject.
    Following methodology from coherence_analysis_ephys_calcium.py
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
        filter_range=[FILTER_CONFIG['lowcut'], FILTER_CONFIG['highcut']], 
        plot_channel=False, 
        plot_spectrogram=False, 
        plot_phases=False, 
        logging_level="CRITICAL"
    )
    channel_object = ephys_api.ephys_data_manager.get_channel(channel_name=channel)
    
    # Extract drug infusion start time
    if 'sleep' not in number_to_drug.get(line_num, ''):
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
    
    # Handle NaN values (as in coherence_analysis_ephys_calcium.py)
    if np.any(np.isnan(channel_object.signal)):
        print(f"    Line {line_num}: Replacing NaNs in EEG with zeros...")
        channel_object.signal = np.nan_to_num(channel_object.signal, nan=0.0)
    
    return channel_object, miniscope_data_manager, fr


# ============================================================================
# SIGNAL PROCESSING FUNCTIONS
# ============================================================================

def slice_signal(signal, line_num, fr):
    """
    Slices the signal into control and treatment segments based on time ranges.
    From coherence_analysis_ephys_calcium.py
    """
    control_start_idx = int(selections[line_num][0][0] * fr * 60)
    control_end_idx = int(selections[line_num][0][1] * fr * 60)
    treatment_start_idx = int(selections[line_num][1][0] * fr * 60)
    treatment_end_idx = int(selections[line_num][1][1] * fr * 60)
    
    sliced_control = signal[control_start_idx:control_end_idx]
    sliced_treatment = signal[treatment_start_idx:treatment_end_idx]
    
    return sliced_control, sliced_treatment


def filter_signal(signal: np.ndarray, fr: float) -> np.ndarray:
    """Apply Butterworth bandpass filter to a signal."""
    nyq = 0.5 * fr
    low = FILTER_CONFIG['lowcut'] / nyq
    high = FILTER_CONFIG['highcut'] / nyq
    
    b, a = butter(FILTER_CONFIG['order'], [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    
    return filtered


def filter_frequency(eeg_signal, calcium_signal, fr, cut):
    """
    Apply bandpass filtering to signals.
    From coherence_analysis_ephys_calcium.py
    """
    filtered_calcium_signal = misc_functions.filterData(
        calcium_signal, n=2, cut=cut, ftype='butter', btype='bandpass', fs=fr
    )
    filtered_eeg_signal = EphysDataManager._filter_data(
        data=eeg_signal, n=2, cut=cut, ftype='butter', fs=fr, btype='bandpass', bodePlot=False
    )
    
    return filtered_eeg_signal, filtered_calcium_signal


def trim_filter_edges(signal: np.ndarray, fr: float) -> np.ndarray:
    """Remove filter edge artifacts by trimming signal edges."""
    trim_samples = int(FILTER_CONFIG['edge_trim_seconds'] * fr)
    
    if 2 * trim_samples >= len(signal):
        return signal
    
    return signal[trim_samples:-trim_samples]


def zscore_signal(signal: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Z-score normalize a signal."""
    valid_mask = ~np.isnan(signal)
    
    if not np.any(valid_mask):
        return signal, np.nan, np.nan
    
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    
    if std == 0 or np.isnan(std):
        return np.zeros_like(signal), mean, std
    
    normalized = (signal - mean) / std
    
    return normalized, mean, std


# ============================================================================
# STATISTICS FUNCTIONS (From coherence_analysis_ephys_calcium.py)
# ============================================================================

def compute_power(line_num, fr, data, windowLength=60, windowStep=3, freqLims=[0,20], 
                  timeBandwidth=2, plotSpectrogram=False):
    """
    Compute power in the delta band (0.5-4 Hz) using multitaper spectrogram.
    From coherence_analysis_ephys_calcium.py
    """
    fs = fr
    numTapers = timeBandwidth * 2 - 1
    windowParams = [windowLength, windowStep]
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
    
    power_matrix, times, frequencies = multitaper_spectrogram(
        data, fs, freqLims, timeBandwidth, numTapers, windowParams, 
        minNfft, detrendOpt, multiprocess, nJobs, weighting, plotOn, 
        returnFig, climScale, verbose, xyflip
    )
    power_array = 10 * np.log10(power_matrix)  # Convert to decibels
    
    low_freq = 0.5
    high_freq = 4.0
    freq_indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0]
    
    sliced_power_array = power_array[freq_indices]      
    mean_power = np.mean(sliced_power_array)
    
    return float(mean_power)


def compute_coherence_value(signal1, signal2, fr):
    """
    Compute mean coherence between two signals in the delta band.
    From coherence_analysis_ephys_calcium.py
    """
    f, Cxy = coherence(signal1, signal2, fs=fr)
    
    freq_indices = np.where((f >= FILTER_CONFIG['lowcut']) & (f <= FILTER_CONFIG['highcut']))[0]
    
    Cxy = Cxy[freq_indices]
    mean_coherence = np.mean(Cxy)
    
    return mean_coherence


def compute_xc(eeg_signal, calcium_signal, fr):
    """
    Compute cross-correlation between EEG and calcium signals.
    From coherence_analysis_ephys_calcium.py
    """
    # Normalize signals
    norm_eeg = (eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) * len(eeg_signal))
    norm_calcium = (calcium_signal - np.mean(calcium_signal)) / np.std(calcium_signal)
    
    # Compute cross-correlation
    nxcorr = correlate(norm_eeg, norm_calcium, mode='full')
    
    # Calculate lags in seconds
    lags = correlation_lags(len(norm_eeg), len(norm_calcium), mode='full') / fr
    
    # Limit to ±10 seconds
    max_display_lag = 10
    lag_mask = (lags >= -max_display_lag) & (lags <= max_display_lag)
    lags = lags[lag_mask]
    nxcorr = nxcorr[lag_mask]
    
    # Find maximum cross-correlation and corresponding lag
    max_xc = nxcorr[np.argmax(nxcorr)]
    lag_at_max_xc = lags[np.argmax(nxcorr)]
    
    return max_xc, lag_at_max_xc


def compute_stats_coherence(eeg_signal, calcium_signal, line_num, fr):
    """
    Compute all statistics for a signal pair.
    From coherence_analysis_ephys_calcium.py methodology
    """
    eeg_power = compute_power(line_num, fr, eeg_signal, windowLength=60, plotSpectrogram=False)
    calcium_power = compute_power(line_num, fr, calcium_signal, windowLength=60, plotSpectrogram=False)
    xc, lag = compute_xc(eeg_signal, calcium_signal, fr)
    coh = compute_coherence_value(eeg_signal, calcium_signal, fr)
    
    return {
        'eeg_power': eeg_power,
        'calcium_power': calcium_power,
        'coherence': coh,
        'xc': xc,
        'lag': lag
    }


# ============================================================================
# HEXBIN STATISTICS FUNCTIONS (From eeg_calcium_scatter_plots.py)
# ============================================================================

def estimate_effective_sample_size(x: np.ndarray, y: np.ndarray, 
                                   max_lags: int = None) -> float:
    """
    Estimate effective sample size accounting for autocorrelation.
    From eeg_calcium_scatter_plots.py
    """
    if max_lags is None:
        max_lags = STATISTICS_CONFIG['max_acf_lags']
    
    n = len(x)
    max_lags = min(max_lags, n // 4)
    
    if HAS_STATSMODELS:
        try:
            acf_x = acf(x, nlags=max_lags, fft=True)
            acf_y = acf(y, nlags=max_lags, fft=True)
            
            rho_sum = np.sum(acf_x[1:] * acf_y[1:])
            n_effective = n / (1 + 2 * max(0, rho_sum))
            n_effective = max(3, min(n_effective, n))
            
        except Exception:
            n_effective = n
    else:
        # Simplified estimation without statsmodels
        n_effective = n / 10  # Conservative estimate
        n_effective = max(3, n_effective)
    
    return n_effective


def compute_confidence_interval(r: float, n_effective: float, 
                                confidence: float = None) -> Tuple[float, float]:
    """
    Compute confidence interval for correlation using Fisher z-transform.
    From eeg_calcium_scatter_plots.py
    """
    if confidence is None:
        confidence = STATISTICS_CONFIG['confidence_level']
    
    if n_effective <= 3 or np.isnan(r) or abs(r) >= 1:
        return np.nan, np.nan
    
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n_effective - 3)
    
    alpha = 1 - confidence
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    return ci_lower, ci_upper


def compute_hexbin_statistics(calcium_signal: np.ndarray, eeg_signal: np.ndarray, 
                              fr: float = 30.0) -> Dict:
    """
    Compute correlation statistics with autocorrelation correction.
    From eeg_calcium_scatter_plots.py
    """
    valid_mask = ~(np.isnan(calcium_signal) | np.isnan(eeg_signal))
    x_valid = calcium_signal[valid_mask]
    y_valid = eeg_signal[valid_mask]
    
    stats_dict = {
        'n': len(x_valid),
        'n_effective': np.nan,
        'r': np.nan,
        'p_value': np.nan,
        'p_value_corrected': np.nan,
        'ci_lower': np.nan,
        'ci_upper': np.nan,
        'slope': np.nan,
        'intercept': np.nan,
    }
    
    if len(x_valid) < 10:
        return stats_dict
    
    r, p_value = pearsonr(x_valid, y_valid)
    z = np.polyfit(x_valid, y_valid, 1)
    
    stats_dict['r'] = r
    stats_dict['p_value'] = p_value
    stats_dict['slope'] = z[0]
    stats_dict['intercept'] = z[1]
    
    if STATISTICS_CONFIG['correct_autocorrelation']:
        n_effective = estimate_effective_sample_size(x_valid, y_valid)
        stats_dict['n_effective'] = int(n_effective)
        
        if n_effective > 2:
            t_stat = r * np.sqrt((n_effective - 2) / (1 - r**2 + 1e-10))
            p_value_corrected = 2 * t_dist.sf(np.abs(t_stat), n_effective - 2)
            stats_dict['p_value_corrected'] = p_value_corrected
        
        ci_lower, ci_upper = compute_confidence_interval(r, n_effective)
        stats_dict['ci_lower'] = ci_lower
        stats_dict['ci_upper'] = ci_upper
    else:
        stats_dict['n_effective'] = stats_dict['n']
        stats_dict['p_value_corrected'] = p_value
    
    return stats_dict


# ============================================================================
# POPULATION STATISTICS FUNCTIONS
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray, paired: bool = True) -> float:
    """Compute Cohen's d effect size for paired or unpaired samples."""
    if paired:
        diff = group2 - group1
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    else:
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (np.mean(group2) - np.mean(group1)) / (pooled_std + 1e-10)
    
    return d


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude using standard thresholds."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(data: np.ndarray, statistic: str = 'mean', 
                 n_bootstrap: int = None, confidence: float = None) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic."""
    if n_bootstrap is None:
        n_bootstrap = STATISTICS_CONFIG['bootstrap_iterations']
    if confidence is None:
        confidence = STATISTICS_CONFIG['confidence_level']
    
    stat_func = np.median if statistic == 'median' else np.mean
    point_estimate = stat_func(data)
    
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)
    
    np.random.seed(42)  # For reproducibility
    for i in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_func(resample)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def format_p_value(p: float) -> str:
    """Format p-value for display."""
    if np.isnan(p):
        return "N/A"
    elif p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.4f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


def compute_population_statistics(control_values: np.ndarray, treatment_values: np.ndarray,
                                  measure_name: str) -> Dict:
    """
    Compute comprehensive population-level statistics for a measure.
    Returns statistics suitable for manuscript reporting.
    """
    valid_mask = ~(np.isnan(control_values) | np.isnan(treatment_values))
    control_valid = control_values[valid_mask]
    treatment_valid = treatment_values[valid_mask]
    n_valid = np.sum(valid_mask)
    
    stats_dict = {
        'measure': measure_name,
        'n': n_valid,
    }
    
    if n_valid < 2:
        stats_dict['error'] = "Insufficient valid subjects"
        return stats_dict
    
    # Control statistics
    ctrl_mean, ctrl_ci_low, ctrl_ci_high = bootstrap_ci(control_valid, 'mean')
    stats_dict['control'] = {
        'mean': np.mean(control_valid),
        'std': np.std(control_valid, ddof=1),
        'sem': np.std(control_valid, ddof=1) / np.sqrt(n_valid),
        'median': np.median(control_valid),
        'ci_lower': ctrl_ci_low,
        'ci_upper': ctrl_ci_high,
        'min': np.min(control_valid),
        'max': np.max(control_valid),
    }
    
    # Treatment statistics
    treat_mean, treat_ci_low, treat_ci_high = bootstrap_ci(treatment_valid, 'mean')
    stats_dict['treatment'] = {
        'mean': np.mean(treatment_valid),
        'std': np.std(treatment_valid, ddof=1),
        'sem': np.std(treatment_valid, ddof=1) / np.sqrt(n_valid),
        'median': np.median(treatment_valid),
        'ci_lower': treat_ci_low,
        'ci_upper': treat_ci_high,
        'min': np.min(treatment_valid),
        'max': np.max(treatment_valid),
    }
    
    # Ratio statistics
    ratios = treatment_valid / (control_valid + 1e-10)  # Avoid division by zero
    ratio_mean, ratio_ci_low, ratio_ci_high = bootstrap_ci(ratios, 'mean')
    stats_dict['ratio'] = {
        'mean': np.mean(ratios),
        'std': np.std(ratios, ddof=1),
        'ci_lower': ratio_ci_low,
        'ci_upper': ratio_ci_high,
    }
    
    # Paired statistical test (Wilcoxon signed-rank)
    # Skip for lag as noted in original code
    if measure_name.lower() != 'lag':
        try:
            stat, p_value = wilcoxon(control_valid, treatment_valid, 
                                     zero_method='wilcox', correction=False)
            test_name = "Wilcoxon signed-rank"
        except ValueError:
            # Fallback to t-test if Wilcoxon fails
            stat, p_value = stats.ttest_rel(control_valid, treatment_valid)
            test_name = "Paired t-test (fallback)"
    else:
        stat, p_value = np.nan, np.nan
        test_name = "Not computed for lag"
    
    # Effect size
    cohens_d = compute_cohens_d(control_valid, treatment_valid, paired=True)
    
    stats_dict['comparison'] = {
        'test_name': test_name,
        'test_statistic': stat,
        'p_value': p_value,
        'p_value_str': format_p_value(p_value),
        'cohens_d': cohens_d,
        'effect_size_interpretation': interpret_cohens_d(cohens_d),
        'mean_difference': np.mean(treatment_valid - control_valid),
    }
    
    return stats_dict


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_subject(line_num: int, channel: str) -> Dict:
    """
    Process a single subject and compute all statistics.
    """
    print(f"\n  Processing subject {line_num}...")
    
    calcium_signal_filepath = os.path.join(
        CALCIUM_DATA_PATH, f"meanFluorescence_{line_num}.npz"
    )
    
    try:
        # Load data
        channel_object, miniscope_data_manager, fr = load_experiment(
            line_num, channel, calcium_signal_filepath
        )
        
        eeg_signal = channel_object.signal
        calcium_signal = miniscope_data_manager.mean_fluorescence_dict['meanFluorescence']
        
        # Slice into control and treatment
        control_eeg, treatment_eeg = slice_signal(eeg_signal, line_num, fr)
        control_calcium, treatment_calcium = slice_signal(calcium_signal, line_num, fr)
        
        # Filter signals (for coherence/power/XC analysis)
        CUT = [FILTER_CONFIG['lowcut'], FILTER_CONFIG['highcut']]
        filtered_control_eeg, filtered_control_calcium = filter_frequency(
            control_eeg, control_calcium, fr, cut=CUT
        )
        filtered_treatment_eeg, filtered_treatment_calcium = filter_frequency(
            treatment_eeg, treatment_calcium, fr, cut=CUT
        )
        
        # Compute coherence-based statistics
        control_stats = compute_stats_coherence(
            filtered_control_eeg, filtered_control_calcium, line_num, fr
        )
        treatment_stats = compute_stats_coherence(
            filtered_treatment_eeg, filtered_treatment_calcium, line_num, fr
        )
        
        # For hexbin plots: filter and normalize signals
        # Using same methodology as eeg_calcium_scatter_plots.py
        filt_ctrl_eeg = filter_signal(control_eeg, fr)
        filt_ctrl_ca = filter_signal(control_calcium, fr)
        filt_treat_eeg = filter_signal(treatment_eeg, fr)
        filt_treat_ca = filter_signal(treatment_calcium, fr)
        
        # Trim edges
        filt_ctrl_eeg = trim_filter_edges(filt_ctrl_eeg, fr)
        filt_ctrl_ca = trim_filter_edges(filt_ctrl_ca, fr)
        filt_treat_eeg = trim_filter_edges(filt_treat_eeg, fr)
        filt_treat_ca = trim_filter_edges(filt_treat_ca, fr)
        
        # Global normalization (z-score using combined parameters)
        all_eeg = np.concatenate([filt_ctrl_eeg, filt_treat_eeg])
        all_ca = np.concatenate([filt_ctrl_ca, filt_treat_ca])
        
        eeg_mean, eeg_std = np.mean(all_eeg), np.std(all_eeg)
        ca_mean, ca_std = np.mean(all_ca), np.std(all_ca)
        
        norm_ctrl_eeg = (filt_ctrl_eeg - eeg_mean) / eeg_std
        norm_ctrl_ca = (filt_ctrl_ca - ca_mean) / ca_std
        norm_treat_eeg = (filt_treat_eeg - eeg_mean) / eeg_std
        norm_treat_ca = (filt_treat_ca - ca_mean) / ca_std
        
        # Compute hexbin statistics
        control_hexbin = compute_hexbin_statistics(norm_ctrl_ca, norm_ctrl_eeg, fr)
        treatment_hexbin = compute_hexbin_statistics(norm_treat_ca, norm_treat_eeg, fr)
        
        # Combine all results
        result = {
            'line_num': line_num,
            'fr': fr,
            'control': {
                'eeg_power': control_stats['eeg_power'],
                'calcium_power': control_stats['calcium_power'],
                'coherence': control_stats['coherence'],
                'xc': control_stats['xc'],
                'lag': control_stats['lag'],
                'hexbin_r': control_hexbin['r'],
                'hexbin_n': control_hexbin['n'],
                'hexbin_n_eff': control_hexbin['n_effective'],
                'hexbin_ci_lower': control_hexbin['ci_lower'],
                'hexbin_ci_upper': control_hexbin['ci_upper'],
                'hexbin_p': control_hexbin['p_value_corrected'],
            },
            'treatment': {
                'eeg_power': treatment_stats['eeg_power'],
                'calcium_power': treatment_stats['calcium_power'],
                'coherence': treatment_stats['coherence'],
                'xc': treatment_stats['xc'],
                'lag': treatment_stats['lag'],
                'hexbin_r': treatment_hexbin['r'],
                'hexbin_n': treatment_hexbin['n'],
                'hexbin_n_eff': treatment_hexbin['n_effective'],
                'hexbin_ci_lower': treatment_hexbin['ci_lower'],
                'hexbin_ci_upper': treatment_hexbin['ci_upper'],
                'hexbin_p': treatment_hexbin['p_value_corrected'],
            },
            'time_windows': {
                'control': selections[line_num][0],
                'treatment': selections[line_num][1],
            }
        }
        
        print(f"    Subject {line_num} processed successfully")
        return result
        
    except Exception as e:
        print(f"    ERROR processing subject {line_num}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_drug_group(drug_name: str, line_nums: List[int], channel: str) -> Dict:
    """
    Process all subjects for a drug group and compute population statistics.
    """
    print(f"\n{'='*60}")
    print(f"Processing drug group: {drug_name}")
    print(f"Subjects: {line_nums}")
    print(f"Channel: {channel}")
    print(f"{'='*60}")
    
    # Collect individual subject data
    subjects_data = []
    for line_num in line_nums:
        result = process_subject(line_num, channel)
        if result is not None:
            subjects_data.append(result)
    
    if len(subjects_data) < 2:
        print(f"Insufficient data for {drug_name}")
        return None
    
    # Extract arrays for population statistics
    measures = {
        'EEG Power (dB)': ('eeg_power', 'eeg_power'),
        'Calcium Power (dB)': ('calcium_power', 'calcium_power'),
        'Coherence': ('coherence', 'coherence'),
        'Cross-Correlation': ('xc', 'xc'),
        'Lag (s)': ('lag', 'lag'),
        'Pearson r (Hexbin)': ('hexbin_r', 'hexbin_r'),
    }
    
    population_stats = {}
    
    for measure_name, (ctrl_key, treat_key) in measures.items():
        control_values = np.array([s['control'][ctrl_key] for s in subjects_data])
        treatment_values = np.array([s['treatment'][treat_key] for s in subjects_data])
        
        pop_stats = compute_population_statistics(control_values, treatment_values, measure_name)
        population_stats[measure_name] = pop_stats
    
    # Store individual subject data for figure descriptions
    result = {
        'drug_name': drug_name,
        'channel': channel,
        'n_subjects': len(subjects_data),
        'subjects': subjects_data,
        'population_stats': population_stats,
    }
    
    return result


# ============================================================================
# OUTPUT GENERATION FUNCTIONS
# ============================================================================

def format_drug_display_name(drug_name: str) -> str:
    """Format drug name for display in reports."""
    if 'dexmedetomidine_0.00045' in drug_name:
        return "Dexmedetomidine (0.00045 mg/kg)"
    elif 'dexmedetomidine_0.0003' in drug_name:
        return "Dexmedetomidine (0.0003 mg/kg)"
    elif drug_name == 'propofol':
        return "Propofol"
    elif drug_name == 'ketamine':
        return "Ketamine"
    else:
        return drug_name.replace('_', ' ').title()


def generate_table_format(drug_data: Dict) -> str:
    """Generate population statistics in paragraph format."""
    drug_display = format_drug_display_name(drug_data['drug_name'])
    channel_desc = CHANNEL_DESCRIPTIONS.get(drug_data['channel'], drug_data['channel'])
    subjects_list = [s['line_num'] for s in drug_data['subjects']]
    
    output = []
    output.append("=" * 100)
    output.append("POPULATION STATISTICS")
    output.append("=" * 100)
    output.append("")
    
    # Header paragraph
    header_para = (
        f"Population statistics for {drug_display} were computed from {drug_data['n_subjects']} subjects "
        f"(Subject IDs: {', '.join(map(str, subjects_list))}). Electrophysiological recordings were obtained "
        f"from {channel_desc}. Analysis was performed on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}."
    )
    output.append(header_para)
    output.append("")
    
    # Generate paragraph for each measure
    for measure_name, stats in drug_data['population_stats'].items():
        if 'error' in stats:
            output.append(f"For {measure_name.lower()}, insufficient data were available for statistical analysis.")
            output.append("")
            continue
        
        ctrl = stats['control']
        treat = stats['treatment']
        ratio = stats['ratio']
        comp = stats['comparison']
        
        # Determine direction of change
        if treat['mean'] > ctrl['mean']:
            direction = "increased"
        elif treat['mean'] < ctrl['mean']:
            direction = "decreased"
        else:
            direction = "remained unchanged"
        
        # Determine significance
        if np.isnan(comp['p_value']):
            sig_statement = "Statistical comparison was not performed for this measure."
        elif comp['p_value'] < 0.05:
            sig_statement = f"This difference was statistically significant ({comp['test_name']}, {comp['p_value_str']})."
        else:
            sig_statement = f"This difference was not statistically significant ({comp['test_name']}, {comp['p_value_str']})."
        
        # Format p-value for inline use
        if np.isnan(comp['p_value']):
            p_inline = "N/A"
        elif comp['p_value'] < 0.001:
            p_inline = "p < 0.001"
        else:
            p_inline = f"p = {comp['p_value']:.4f}"
        
        # Build the paragraph
        measure_para = (
            f"{measure_name} {direction} from control to treatment. During the control period, "
            f"{measure_name.lower()} was {ctrl['mean']:.4f} ± {ctrl['std']:.4f} (mean ± SD; 95% bootstrap CI: "
            f"{ctrl['ci_lower']:.4f} to {ctrl['ci_upper']:.4f}; range: {ctrl['min']:.4f} to {ctrl['max']:.4f}). "
            f"During the treatment period, {measure_name.lower()} was {treat['mean']:.4f} ± {treat['std']:.4f} "
            f"(mean ± SD; 95% bootstrap CI: {treat['ci_lower']:.4f} to {treat['ci_upper']:.4f}; range: "
            f"{treat['min']:.4f} to {treat['max']:.4f}). The treatment-to-control ratio was {ratio['mean']:.3f} ± "
            f"{ratio['std']:.3f} (95% bootstrap CI: {ratio['ci_lower']:.3f} to {ratio['ci_upper']:.3f}). "
            f"{sig_statement} The effect size was Cohen's d = {comp['cohens_d']:.3f}, indicating a "
            f"{comp['effect_size_interpretation']} effect. The mean difference between treatment and control was "
            f"{comp['mean_difference']:.4f}."
        )
        
        output.append(measure_para)
        output.append("")
    
    return "\n".join(output)


def generate_prose_format(drug_data: Dict) -> str:
    """Generate population statistics in prose format for manuscript."""
    drug_display = format_drug_display_name(drug_data['drug_name'])
    channel_desc = CHANNEL_DESCRIPTIONS.get(drug_data['channel'], drug_data['channel'])
    n = drug_data['n_subjects']
    
    output = []
    output.append("=" * 100)
    output.append("MANUSCRIPT TEXT - RESULTS SECTION")
    output.append("=" * 100)
    output.append("")
    
    # Methods paragraph
    output.append("METHODS PARAGRAPH:")
    output.append("-" * 50)
    methods_text = f"""Electrophysiological signals were recorded from {channel_desc} and synchronized with calcium imaging data acquired at ~30 Hz. Signals were bandpass filtered ({FILTER_CONFIG['lowcut']}-{FILTER_CONFIG['highcut']} Hz, 2nd-order Butterworth) to isolate delta-band activity. For correlation analysis, both signals were z-scored using global normalization parameters (computed across control and treatment periods combined) to enable direct comparison. Effective sample sizes were estimated using autocorrelation correction (Bartlett's formula) to account for temporal dependencies in the data. Pearson correlation coefficients were computed with 95% confidence intervals calculated using Fisher's z-transformation with autocorrelation-corrected degrees of freedom. Population-level comparisons between control and treatment periods were performed using Wilcoxon signed-rank tests (two-tailed, α = 0.05) with effect sizes reported as Cohen's d for paired samples. Bootstrap confidence intervals (10,000 iterations) were computed for population means."""
    output.append(methods_text)
    output.append("")
    
    # Results paragraph
    output.append("RESULTS PARAGRAPH:")
    output.append("-" * 50)
    
    results_parts = []
    results_parts.append(f"We analyzed {n} subjects in the {drug_display} condition.")
    
    # Coherence results
    coh_stats = drug_data['population_stats'].get('Coherence', {})
    if 'error' not in coh_stats:
        ctrl = coh_stats['control']
        treat = coh_stats['treatment']
        comp = coh_stats['comparison']
        
        direction = "increased" if treat['mean'] > ctrl['mean'] else "decreased"
        sig = "significantly" if comp['p_value'] < 0.05 else "did not significantly"
        
        results_parts.append(
            f"EEG-calcium coherence {direction} from control ({ctrl['mean']:.3f} ± {ctrl['sem']:.3f}) "
            f"to treatment ({treat['mean']:.3f} ± {treat['sem']:.3f}), which {sig} differ "
            f"({comp['p_value_str']}, Cohen's d = {comp['cohens_d']:.2f}, {comp['effect_size_interpretation']} effect)."
        )
    
    # Correlation results
    r_stats = drug_data['population_stats'].get('Pearson r (Hexbin)', {})
    if 'error' not in r_stats:
        ctrl = r_stats['control']
        treat = r_stats['treatment']
        comp = r_stats['comparison']
        
        ctrl_dir = "positive" if ctrl['mean'] > 0 else "negative"
        treat_dir = "positive" if treat['mean'] > 0 else "negative"
        
        results_parts.append(
            f"The correlation between EEG and calcium signals was {ctrl_dir} during control "
            f"(r = {ctrl['mean']:.3f}, 95% CI [{ctrl['ci_lower']:.3f}, {ctrl['ci_upper']:.3f}]) "
            f"and {treat_dir} during treatment "
            f"(r = {treat['mean']:.3f}, 95% CI [{treat['ci_lower']:.3f}, {treat['ci_upper']:.3f}]). "
            f"This change was {'statistically significant' if comp['p_value'] < 0.05 else 'not statistically significant'} "
            f"({comp['p_value_str']}, Cohen's d = {comp['cohens_d']:.2f})."
        )
    
    # Cross-correlation results
    xc_stats = drug_data['population_stats'].get('Cross-Correlation', {})
    lag_stats = drug_data['population_stats'].get('Lag (s)', {})
    if 'error' not in xc_stats and 'error' not in lag_stats:
        xc_ctrl = xc_stats['control']
        xc_treat = xc_stats['treatment']
        lag_ctrl = lag_stats['control']
        lag_treat = lag_stats['treatment']
        
        results_parts.append(
            f"Peak cross-correlation occurred at {lag_ctrl['mean']:.2f} ± {lag_ctrl['sem']:.2f} s "
            f"during control (r = {xc_ctrl['mean']:.3f}) and at {lag_treat['mean']:.2f} ± {lag_treat['sem']:.2f} s "
            f"during treatment (r = {xc_treat['mean']:.3f})."
        )
    
    # Power results
    eeg_stats = drug_data['population_stats'].get('EEG Power (dB)', {})
    ca_stats = drug_data['population_stats'].get('Calcium Power (dB)', {})
    if 'error' not in eeg_stats and 'error' not in ca_stats:
        eeg_comp = eeg_stats['comparison']
        ca_comp = ca_stats['comparison']
        
        eeg_change = "increased" if eeg_stats['treatment']['mean'] > eeg_stats['control']['mean'] else "decreased"
        ca_change = "increased" if ca_stats['treatment']['mean'] > ca_stats['control']['mean'] else "decreased"
        
        results_parts.append(
            f"Delta-band power {eeg_change} in EEG ({eeg_comp['p_value_str']}) "
            f"and {ca_change} in calcium signals ({ca_comp['p_value_str']}) following drug administration."
        )
    
    output.append(" ".join(results_parts))
    output.append("")
    
    return "\n".join(output)


def generate_figure_descriptions(drug_data: Dict) -> str:
    """Generate individual figure descriptions in paragraph format."""
    drug_display = format_drug_display_name(drug_data['drug_name'])
    channel_desc = CHANNEL_DESCRIPTIONS.get(drug_data['channel'], drug_data['channel'])
    
    output = []
    output.append("=" * 100)
    output.append("FIGURE DESCRIPTIONS")
    output.append("=" * 100)
    output.append("")
    
    # Overview figure description
    output.append("OVERVIEW FIGURE DESCRIPTION")
    output.append("-" * 50)
    output.append("")
    
    overview_desc = (
        f"Figure X presents a multimodal analysis of electrophysiology-calcium coupling under {drug_display}. "
        f"Panel A shows the electrophysiology spectrogram displaying delta-band ({FILTER_CONFIG['lowcut']}-{FILTER_CONFIG['highcut']} Hz) "
        f"power over time from {channel_desc}, with vertical dashed lines indicating control (blue) and treatment (red) "
        f"analysis windows, and a solid red line marking the start of drug infusion. Panel B displays the electrophysiology "
        f"signal trace with control (blue) and treatment (red) segments overlaid on the full recording (gray). "
        f"Panel C presents a hexbin density plot showing the relationship between z-scored EEG amplitude and z-scored "
        f"calcium fluorescence during the control period, with statistics including sample size (n), effective sample "
        f"size accounting for autocorrelation (n_eff), Pearson correlation coefficient (r), 95% confidence interval, "
        f"and p-value, along with a red dashed line indicating the linear regression fit. Panel D shows the calcium "
        f"spectrogram displaying delta-band power over time. Panel E presents the calcium signal trace with control "
        f"and treatment segments highlighted. Panel F displays the hexbin density plot for the treatment period, "
        f"formatted as in Panel C. Panel G shows the coherogram displaying time-frequency coherence between "
        f"electrophysiology and calcium signals, with the color scale indicating coherence magnitude (0-0.7). "
        f"Panel H presents the cross-correlation function between EEG and calcium signals during the control period, "
        f"with the vertical dashed line indicating the lag at maximum correlation. Panel I shows the cross-correlation "
        f"function during the treatment period. Statistical comparisons between control and treatment periods were "
        f"performed using Wilcoxon signed-rank tests with autocorrelation-corrected effective sample sizes. "
        f"See Table X for population statistics."
    )
    
    output.append(overview_desc)
    output.append("")
    
    # Individual subject descriptions
    output.append("INDIVIDUAL SUBJECT DESCRIPTIONS")
    output.append("-" * 50)
    output.append("")
    
    for subject in drug_data['subjects']:
        line_num = subject['line_num']
        ctrl = subject['control']
        treat = subject['treatment']
        windows = subject['time_windows']
        
        # Format p-values for inline text
        ctrl_p_str = "p < 0.001" if ctrl['hexbin_p'] < 0.001 else f"p = {ctrl['hexbin_p']:.4f}"
        treat_p_str = "p < 0.001" if treat['hexbin_p'] < 0.001 else f"p = {treat['hexbin_p']:.4f}"
        
        subj_para = (
            f"Subject {line_num} was recorded under {drug_display}. During the control period "
            f"({windows['control'][0]:.1f} to {windows['control'][1]:.1f} minutes), EEG power was {ctrl['eeg_power']:.2f} dB "
            f"and calcium power was {ctrl['calcium_power']:.2f} dB. The coherence between signals was {ctrl['coherence']:.4f}, "
            f"and the cross-correlation peaked at r = {ctrl['xc']:.4f} with a lag of {ctrl['lag']:.2f} seconds. "
            f"The hexbin correlation analysis revealed r = {ctrl['hexbin_r']:.4f} (95% CI: {ctrl['hexbin_ci_lower']:.3f} to "
            f"{ctrl['hexbin_ci_upper']:.3f}; n = {ctrl['hexbin_n']:,}; n_effective = {ctrl['hexbin_n_eff']:,}; {ctrl_p_str}). "
            f"During the treatment period ({windows['treatment'][0]:.1f} to {windows['treatment'][1]:.1f} minutes), "
            f"EEG power was {treat['eeg_power']:.2f} dB and calcium power was {treat['calcium_power']:.2f} dB. "
            f"The coherence between signals was {treat['coherence']:.4f}, and the cross-correlation peaked at "
            f"r = {treat['xc']:.4f} with a lag of {treat['lag']:.2f} seconds. The hexbin correlation analysis revealed "
            f"r = {treat['hexbin_r']:.4f} (95% CI: {treat['hexbin_ci_lower']:.3f} to {treat['hexbin_ci_upper']:.3f}; "
            f"n = {treat['hexbin_n']:,}; n_effective = {treat['hexbin_n_eff']:,}; {treat_p_str})."
        )
        
        output.append(subj_para)
        output.append("")
    
    return "\n".join(output)


def save_drug_output(drug_data: Dict, output_dir: str):
    """Save all output files for a drug group."""
    drug_name = drug_data['drug_name']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all content
    table_content = generate_table_format(drug_data)
    prose_content = generate_prose_format(drug_data)
    figure_content = generate_figure_descriptions(drug_data)
    
    # Combine into single file
    full_content = []
    full_content.append(table_content)
    full_content.append("\n\n")
    full_content.append(prose_content)
    full_content.append("\n\n")
    full_content.append(figure_content)
    
    # Save combined file
    output_path = os.path.join(output_dir, f"{drug_name}_statistics_and_descriptions.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(full_content))
    
    print(f"Saved: {output_path}")
    
    # Also save JSON for programmatic access
    json_path = os.path.join(output_dir, f"{drug_name}_data.json")
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        else:
            return obj
    
    json_data = convert_for_json(drug_data)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved: {json_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("FIGURE DESCRIPTION AND POPULATION STATISTICS GENERATOR")
    print("=" * 80)
    print(f"Channel: {CHANNEL} ({CHANNEL_DESCRIPTIONS.get(CHANNEL, CHANNEL)})")
    print(f"Filter: {FILTER_CONFIG['lowcut']}-{FILTER_CONFIG['highcut']} Hz")
    print(f"Output Directory: {RESULTS_PATH}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Process each drug group
    all_results = {}
    
    for drug_name, line_nums in data.items():
        drug_data = process_drug_group(drug_name, line_nums, CHANNEL)
        
        if drug_data is not None:
            all_results[drug_name] = drug_data
            save_drug_output(drug_data, RESULTS_PATH)
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Processed {len(all_results)} drug groups")
    print(f"Output saved to: {RESULTS_PATH}")
    
    return all_results


if __name__ == "__main__":
    results = main()