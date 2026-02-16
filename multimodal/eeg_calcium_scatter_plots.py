"""
EEG vs Calcium Fluorescence Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import scipy
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, t as t_dist
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional, Dict, List

# Import the same modules used in coherence_analysis_ephys_calcium.py
from src import misc_functions
from src2.miniscope.miniscope_data_manager import MiniscopeDataManager
from src2.ephys.ephys_api import EphysAPI
from src2.ephys.ephys_data_manager import EphysDataManager
from src2.multimodal.miniscope_ephys_alignment_utils import (
    sync_neuralynx_miniscope_timestamps, 
    find_ephys_idx_of_TTL_events
)

# Try to import statsmodels for additional statistical tests
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available - multiple comparison correction will be skipped")


# EXPERIMENTAL METADATA

# Drug groups with their subject line numbers
data = {
    "dexmedetomidine: 0.00045": [46, 47, 64, 88, 97, 101],
    "dexmedetomidine: 0.0003": [40, 41, 48, 87, 93, 94],
    "propofol": [36, 43, 44, 86, 99, 103],
    "sleep": [35, 37, 38, 83, 90, 92],
    "ketamine": [39, 42, 45, 85, 96, 112]
}

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


# SIGNAL PROCESSING CONFIGURATION

FILTER_CONFIG = {
    'lowcut': 0.5,              # Hz - lower bound of bandpass
    'highcut': 4.0,             # Hz - upper bound of bandpass  
    'order': 2,                 # Butterworth filter order
    'filter_type': 'butter',    # Filter type
    'edge_trim_seconds': 5.0,   # Seconds to trim from each edge after filtering
}

NORMALIZATION_CONFIG = {
    'method': 'zscore',         # 'zscore', 'minmax', or 'none'
    'global_normalization': True,  # Use same parameters for control and treatment
}

NAN_HANDLING_CONFIG = {
    'max_interpolation_gap_seconds': 0.5,  # Interpolate gaps shorter than this
    'interpolation_method': 'linear',       # 'linear' or 'zero'
}

STATISTICS_CONFIG = {
    'correct_autocorrelation': True,  # Compute effective sample size
    'confidence_level': 0.95,          # For confidence intervals
    'max_acf_lags': 100,               # Maximum lags for autocorrelation estimation
}


# POPULATION ANALYSIS CONFIGURATION

POPULATION_CONFIG = {
    'bootstrap_iterations': 10000,      # For CI estimation
    'confidence_level': 0.95,           # 95% CI
    'paired_test': 'wilcoxon',          # 'wilcoxon' or 'ttest'
    'multiple_comparison_method': 'fdr_bh',  # 'bonferroni' or 'fdr_bh'
    'min_subjects_for_stats': 4,        # Minimum N for statistical tests
}



# DATA PATHS AND DISPLAY CONFIGURATION


CALCIUM_DATA_PATH = r"C:\Users\Path\To\meanFluorescence"

# Channel descriptions for figure legends
CHANNEL_DESCRIPTIONS = {
    'CBvsPCEEG': 'Cerebellum vs Parietal Cortex EEG',
    'PFCLFPvsCBEEG': 'Prefrontal Cortex LFP vs Cerebellum EEG',
    'PFCEEGvsCBEEG': 'Prefrontal Cortex EEG vs Cerebellum EEG'
}

# Units
UNITS = {
    'calcium': 'z-score',
    'eeg': 'z-score'
}

# Journal-compliant figure sizes
SINGLE_COLUMN_WIDTH = 3.35      # 8.5 cm
DOUBLE_COLUMN_WIDTH = 7.0       # 17.8 cm

# DPI settings
COLOR_DPI = 300

# Grayscale-friendly color palette (colorblind accessible)
COLORS = {
    'control': '#969696',        
    'treatment': '#252525',      
    'edge_color': '#000000',     
    'regression': '#000000',     
    'regression_ctrl': '#666666',  
    'regression_treat': '#000000', 
}

# Population plot colors
POPULATION_COLORS = {
    'violin_control': '#A8D5BA',    
    'violin_treatment': '#F4A6A6',  
    'violin_alpha': 0.7,
    'point_color': '#2C3E50',       
    'point_size': 30,
    'line_color': '#7F8C8D',        
    'line_alpha': 0.5,
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
    'title': 10,
    'stats_text': 7,
}

# Output directory
RESULTS_PATH = r"C:\Users\Path\To\EEG_Calcium_Complete_Analysis"


# DATA LOADING FUNCTIONS

def load_experiment(line_num: int, channel: str, calcium_signal_filepath: str = None):
    """
    Load and synchronize ephys and calcium imaging data for a given subject.
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
    
    channel_object.signal = channel_object.signal[ephys_idx_all_TTL_events]
    channel_object.sampling_rate = np.array(fr)
    
    # Load calcium signal from file
    if calcium_signal_filepath:
        miniscope_data_manager.mean_fluorescence_dict = np.load(calcium_signal_filepath)
    
    return channel_object, miniscope_data_manager, fr


# SIGNAL PROCESSING FUNCTIONS

def handle_nans(signal: np.ndarray, fr: float, 
                max_gap_seconds: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle NaN values: interpolate short gaps, mark long gaps for exclusion.
    """
    if max_gap_seconds is None:
        max_gap_seconds = NAN_HANDLING_CONFIG['max_interpolation_gap_seconds']
    
    nan_mask = np.isnan(signal)
    if not np.any(nan_mask):
        return signal.copy(), np.ones(len(signal), dtype=bool)
    
    max_gap_samples = int(max_gap_seconds * fr)
    valid_mask = np.ones(len(signal), dtype=bool)
    signal_clean = signal.copy()
    
    # Find contiguous NaN regions
    nan_int = nan_mask.astype(int)
    nan_diff = np.diff(np.concatenate([[0], nan_int, [0]]))
    starts = np.where(nan_diff == 1)[0]
    ends = np.where(nan_diff == -1)[0]
    
    for start, end in zip(starts, ends):
        gap_length = end - start
        
        if gap_length <= max_gap_samples:
            # Linear interpolation for short gaps
            if start > 0 and end < len(signal):
                interp_values = np.linspace(
                    signal_clean[start - 1], 
                    signal_clean[end] if end < len(signal) and not np.isnan(signal_clean[end]) else signal_clean[start - 1],
                    gap_length + 2
                )[1:-1]
                signal_clean[start:end] = interp_values
            elif start == 0 and end < len(signal):
                first_valid = signal_clean[end] if not np.isnan(signal_clean[end]) else 0
                signal_clean[start:end] = first_valid
            elif end >= len(signal) and start > 0:
                signal_clean[start:end] = signal_clean[start - 1]
        else:
            valid_mask[start:end] = False
            signal_clean[start:end] = 0
    
    n_interpolated = np.sum(nan_mask) - np.sum(~valid_mask)
    n_excluded = np.sum(~valid_mask)
    
    if n_interpolated > 0 or n_excluded > 0:
        print(f"    NaN handling: {n_interpolated} samples interpolated, {n_excluded} samples excluded")
    
    return signal_clean, valid_mask


def slice_signal(signal: np.ndarray, line_num: int, fr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Slice signal into control and treatment segments based on time selections."""
    control_start_idx = int(selections[line_num][0][0] * fr * 60)
    control_end_idx = int(selections[line_num][0][1] * fr * 60)
    treatment_start_idx = int(selections[line_num][1][0] * fr * 60)
    treatment_end_idx = int(selections[line_num][1][1] * fr * 60)
    
    sliced_control = signal[control_start_idx:control_end_idx]
    sliced_treatment = signal[treatment_start_idx:treatment_end_idx]
    
    return sliced_control, sliced_treatment


def filter_signal(signal: np.ndarray, fr: float, 
                  lowcut: float = None, highcut: float = None,
                  order: int = None) -> np.ndarray:
    """Apply Butterworth bandpass filter to a signal."""
    if lowcut is None:
        lowcut = FILTER_CONFIG['lowcut']
    if highcut is None:
        highcut = FILTER_CONFIG['highcut']
    if order is None:
        order = FILTER_CONFIG['order']
    
    nyq = 0.5 * fr
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    
    return filtered


def trim_filter_edges(signal: np.ndarray, fr: float, 
                      trim_seconds: float = None) -> np.ndarray:
    """Remove filter edge artifacts by trimming signal edges."""
    if trim_seconds is None:
        trim_seconds = FILTER_CONFIG['edge_trim_seconds']
    
    trim_samples = int(trim_seconds * fr)
    
    if 2 * trim_samples >= len(signal):
        print(f"    WARNING: Signal too short for edge trimming, skipping")
        return signal
    
    return signal[trim_samples:-trim_samples]


def filter_and_trim(eeg_signal: np.ndarray, calcium_signal: np.ndarray, 
                    fr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply bandpass filtering and edge trimming to both signals."""
    filtered_eeg = filter_signal(eeg_signal, fr)
    filtered_calcium = filter_signal(calcium_signal, fr)
    
    filtered_eeg = trim_filter_edges(filtered_eeg, fr)
    filtered_calcium = trim_filter_edges(filtered_calcium, fr)
    
    return filtered_eeg, filtered_calcium


def zscore_signal(signal: np.ndarray, mean: float = None, 
                  std: float = None) -> Tuple[np.ndarray, float, float]:
    """Z-score normalize a signal."""
    valid_mask = ~np.isnan(signal)
    
    if not np.any(valid_mask):
        return signal, np.nan, np.nan
    
    if mean is None:
        mean = np.nanmean(signal)
    if std is None:
        std = np.nanstd(signal)
    
    if std == 0 or np.isnan(std):
        return np.zeros_like(signal), mean, std
    
    normalized = (signal - mean) / std
    
    return normalized, mean, std


def normalize_signals_global(control_eeg: np.ndarray, treatment_eeg: np.ndarray,
                             control_calcium: np.ndarray, treatment_calcium: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Normalize all signals using global (combined) parameters."""
    if NORMALIZATION_CONFIG['method'] == 'none':
        return (control_eeg, treatment_eeg, control_calcium, treatment_calcium,
                {'method': 'none'})
    
    if NORMALIZATION_CONFIG['global_normalization']:
        all_eeg = np.concatenate([control_eeg, treatment_eeg])
        all_calcium = np.concatenate([control_calcium, treatment_calcium])
        
        eeg_mean = np.nanmean(all_eeg)
        eeg_std = np.nanstd(all_eeg)
        calcium_mean = np.nanmean(all_calcium)
        calcium_std = np.nanstd(all_calcium)
        
        norm_control_eeg, _, _ = zscore_signal(control_eeg, eeg_mean, eeg_std)
        norm_treatment_eeg, _, _ = zscore_signal(treatment_eeg, eeg_mean, eeg_std)
        norm_control_calcium, _, _ = zscore_signal(control_calcium, calcium_mean, calcium_std)
        norm_treatment_calcium, _, _ = zscore_signal(treatment_calcium, calcium_mean, calcium_std)
        
        norm_params = {
            'method': 'zscore',
            'global': True,
            'eeg_mean': eeg_mean,
            'eeg_std': eeg_std,
            'calcium_mean': calcium_mean,
            'calcium_std': calcium_std
        }
    else:
        norm_control_eeg, eeg_ctrl_mean, eeg_ctrl_std = zscore_signal(control_eeg)
        norm_treatment_eeg, eeg_treat_mean, eeg_treat_std = zscore_signal(treatment_eeg)
        norm_control_calcium, ca_ctrl_mean, ca_ctrl_std = zscore_signal(control_calcium)
        norm_treatment_calcium, ca_treat_mean, ca_treat_std = zscore_signal(treatment_calcium)
        
        norm_params = {
            'method': 'zscore',
            'global': False,
        }
    
    return (norm_control_eeg, norm_treatment_eeg, 
            norm_control_calcium, norm_treatment_calcium, norm_params)


# INDIVIDUAL SUBJECT STATISTICS FUNCTIONS


def estimate_effective_sample_size(x: np.ndarray, y: np.ndarray, 
                                   max_lags: int = None) -> float:
    """Estimate effective sample size accounting for autocorrelation."""
    if max_lags is None:
        max_lags = STATISTICS_CONFIG['max_acf_lags']
    
    n = len(x)
    max_lags = min(max_lags, n // 4)
    
    try:
        from statsmodels.tsa.stattools import acf
        
        acf_x = acf(x, nlags=max_lags, fft=True)
        acf_y = acf(y, nlags=max_lags, fft=True)
        
        rho_sum = np.sum(acf_x[1:] * acf_y[1:])
        n_effective = n / (1 + 2 * max(0, rho_sum))
        n_effective = max(3, min(n_effective, n))
        
    except ImportError:
        n_effective = n
    
    return n_effective


def compute_confidence_interval(r: float, n_effective: float, 
                                confidence: float = None) -> Tuple[float, float]:
    """Compute confidence interval for correlation using Fisher z-transform."""
    if confidence is None:
        confidence = STATISTICS_CONFIG['confidence_level']
    
    if n_effective <= 3 or np.isnan(r) or abs(r) >= 1:
        return np.nan, np.nan
    
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n_effective - 3)
    
    alpha = 1 - confidence
    z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
    
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    return ci_lower, ci_upper


def compute_statistics_robust(x: np.ndarray, y: np.ndarray, 
                              fr: float = 30.0) -> Dict:
    """Compute correlation statistics with autocorrelation correction."""
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    stats = {
        'n': len(x_valid),
        'n_effective': np.nan,
        'r': np.nan,
        'p_value': np.nan,
        'p_value_corrected': np.nan,
        'ci_lower': np.nan,
        'ci_upper': np.nan,
        'slope': np.nan,
        'intercept': np.nan,
        'p_value_str': 'N/A',
        'p_value_corrected_str': 'N/A'
    }
    
    if len(x_valid) < 10:
        return stats
    
    r, p_value = pearsonr(x_valid, y_valid)
    z = np.polyfit(x_valid, y_valid, 1)
    
    stats['r'] = r
    stats['p_value'] = p_value
    stats['slope'] = z[0]
    stats['intercept'] = z[1]
    
    if STATISTICS_CONFIG['correct_autocorrelation']:
        n_effective = estimate_effective_sample_size(x_valid, y_valid)
        stats['n_effective'] = int(n_effective)
        
        if n_effective > 2:
            t_stat = r * np.sqrt((n_effective - 2) / (1 - r**2 + 1e-10))
            p_value_corrected = 2 * t_dist.sf(np.abs(t_stat), n_effective - 2)
            stats['p_value_corrected'] = p_value_corrected
        
        ci_lower, ci_upper = compute_confidence_interval(r, n_effective)
        stats['ci_lower'] = ci_lower
        stats['ci_upper'] = ci_upper
    else:
        stats['n_effective'] = stats['n']
        stats['p_value_corrected'] = p_value
    
    stats['p_value_str'] = format_p_value(p_value)
    stats['p_value_corrected_str'] = format_p_value(stats['p_value_corrected'])
    
    return stats


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


# POPULATION STATISTICS FUNCTIONS

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray, paired: bool = True) -> float:
    """Compute Cohen's d effect size."""
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
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(data: np.ndarray, statistic: str = 'median', 
                 n_bootstrap: int = None, confidence: float = None) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic."""
    if n_bootstrap is None:
        n_bootstrap = POPULATION_CONFIG['bootstrap_iterations']
    if confidence is None:
        confidence = POPULATION_CONFIG['confidence_level']
    
    stat_func = np.median if statistic == 'median' else np.mean
    point_estimate = stat_func(data)
    
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_func(resample)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def paired_statistical_test(control: np.ndarray, treatment: np.ndarray, 
                            test_type: str = None) -> Tuple[float, float, str]:
    """Perform paired statistical test comparing control vs treatment."""
    if test_type is None:
        test_type = POPULATION_CONFIG['paired_test']
    
    n = len(control)
    
    if n < POPULATION_CONFIG['min_subjects_for_stats']:
        return np.nan, np.nan, "insufficient_n"
    
    if test_type == 'wilcoxon':
        try:
            statistic, p_value = stats.wilcoxon(control, treatment, alternative='two-sided')
            test_name = "Wilcoxon signed-rank"
        except ValueError:
            statistic, p_value = stats.ttest_rel(control, treatment)
            test_name = "Paired t-test (fallback)"
    else:
        statistic, p_value = stats.ttest_rel(control, treatment)
        test_name = "Paired t-test"
    
    return statistic, p_value, test_name


def format_p_value_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if np.isnan(p):
        return ""
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


class PopulationCorrelationCollector:
    """Collects correlation statistics across subjects for population-level analysis."""
    
    def __init__(self, drug_name: str):
        self.drug_name = drug_name
        self.subjects = []
        self.control_correlations = []
        self.treatment_correlations = []
        self.control_ci_lower = []
        self.control_ci_upper = []
        self.treatment_ci_lower = []
        self.treatment_ci_upper = []
        self.control_n_effective = []
        self.treatment_n_effective = []
        
    def add_subject(self, line_num: int, 
                    control_stats: Dict, treatment_stats: Dict):
        """Add a subject's correlation statistics."""
        self.subjects.append(line_num)
        
        self.control_correlations.append(control_stats.get('r', np.nan))
        self.treatment_correlations.append(treatment_stats.get('r', np.nan))
        
        self.control_ci_lower.append(control_stats.get('ci_lower', np.nan))
        self.control_ci_upper.append(control_stats.get('ci_upper', np.nan))
        self.treatment_ci_lower.append(treatment_stats.get('ci_lower', np.nan))
        self.treatment_ci_upper.append(treatment_stats.get('ci_upper', np.nan))
        
        self.control_n_effective.append(control_stats.get('n_effective', np.nan))
        self.treatment_n_effective.append(treatment_stats.get('n_effective', np.nan))
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return control and treatment correlations as numpy arrays."""
        return (np.array(self.control_correlations), 
                np.array(self.treatment_correlations))
    
    def compute_population_statistics(self) -> Dict:
        """Compute comprehensive population-level statistics."""
        control = np.array(self.control_correlations)
        treatment = np.array(self.treatment_correlations)
        
        valid_mask = ~(np.isnan(control) | np.isnan(treatment))
        control_valid = control[valid_mask]
        treatment_valid = treatment[valid_mask]
        n_valid = np.sum(valid_mask)
        
        stats_dict = {
            'drug_name': self.drug_name,
            'n_subjects': len(self.subjects),
            'n_valid': n_valid,
            'subjects': self.subjects,
        }
        
        if n_valid < 2:
            stats_dict['error'] = "Insufficient valid subjects"
            return stats_dict
        
        # Control statistics with bootstrap CI
        ctrl_median, ctrl_ci_low, ctrl_ci_high = bootstrap_ci(control_valid, 'median')
        stats_dict['control'] = {
            'median': ctrl_median,
            'mean': np.mean(control_valid),
            'std': np.std(control_valid, ddof=1),
            'sem': np.std(control_valid, ddof=1) / np.sqrt(n_valid),
            'ci_lower': ctrl_ci_low,
            'ci_upper': ctrl_ci_high,
            'min': np.min(control_valid),
            'max': np.max(control_valid),
            'values': control_valid.tolist(),
        }
        
        # Treatment statistics with bootstrap CI
        treat_median, treat_ci_low, treat_ci_high = bootstrap_ci(treatment_valid, 'median')
        stats_dict['treatment'] = {
            'median': treat_median,
            'mean': np.mean(treatment_valid),
            'std': np.std(treatment_valid, ddof=1),
            'sem': np.std(treatment_valid, ddof=1) / np.sqrt(n_valid),
            'ci_lower': treat_ci_low,
            'ci_upper': treat_ci_high,
            'min': np.min(treatment_valid),
            'max': np.max(treatment_valid),
            'values': treatment_valid.tolist(),
        }
        
        # Paired comparison
        test_stat, p_value, test_name = paired_statistical_test(control_valid, treatment_valid)
        cohens_d = compute_cohens_d(control_valid, treatment_valid, paired=True)
        
        stats_dict['comparison'] = {
            'test_name': test_name,
            'test_statistic': test_stat,
            'p_value': p_value,
            'p_value_str': format_p_value_stars(p_value),
            'cohens_d': cohens_d,
            'effect_size_interpretation': interpret_cohens_d(cohens_d),
            'mean_difference': np.mean(treatment_valid - control_valid),
            'median_difference': np.median(treatment_valid - control_valid),
        }
        
        return stats_dict


# FIGURE DESCRIPTION GENERATORS

def format_drug_name(drug: str) -> str:
    """Format drug name for display in figure descriptions."""
    if 'dexmedetomidine' in drug.lower():
        dose = drug.split(': ')[1] if ': ' in drug else ''
        return f"dexmedetomidine ({dose} mg/kg)" if dose else "dexmedetomidine"
    elif drug.lower() == 'sleep':
        return "natural sleep"
    else:
        return drug.lower()


def format_drug_name_short(drug: str) -> str:
    """Format drug name for short display in population plots."""
    if 'dexmedetomidine' in drug.lower():
        dose = drug.split(': ')[1] if ': ' in drug else ''
        return f"Dex {dose}" if dose else "Dex"
    elif drug.lower() == 'sleep':
        return "Sleep"
    else:
        return drug.capitalize()


def generate_figure_description(
    line_num: int, channel: str, drug: str, period: str,
    fr: float, stats: Dict, time_window: list, plot_type: str = 'scatter'
) -> str:
    """Generate an eNeuro-compliant figure description."""
    channel_desc = CHANNEL_DESCRIPTIONS.get(channel, channel)
    drug_display = format_drug_name(drug)
    
    n = stats.get('n', 0)
    n_eff = stats.get('n_effective', n)
    r = stats.get('r', float('nan'))
    p_str = stats.get('p_value_corrected_str', stats.get('p_value_str', 'N/A'))
    ci_lower = stats.get('ci_lower', float('nan'))
    ci_upper = stats.get('ci_upper', float('nan'))
    slope = stats.get('slope', float('nan'))
    p_value = stats.get('p_value_corrected', stats.get('p_value', 1.0))
    
    if not np.isnan(r):
        corr_direction = "positive" if r > 0 else "negative"
        corr_strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
        significance = "significant" if p_value < 0.05 else "not statistically significant"
    else:
        corr_direction = "undetermined"
        corr_strength = "undetermined"
        significance = "undetermined"
    
    ci_str = f", 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else ""
    filter_desc = f"{FILTER_CONFIG['lowcut']}-{FILTER_CONFIG['highcut']} Hz"
    
    description = f"""
{'-'*80}
FIGURE DESCRIPTION - Subject {line_num} - {period} - {plot_type.capitalize()}
{'-'*80}

Relationship between electrophysiology and calcium fluorescence signals during the {period.lower()} period for Subject {line_num} ({drug_display}). {plot_type.capitalize()} plot showing bandpass-filtered ({filter_desc}) and z-scored EEG amplitude versus calcium fluorescence from {channel_desc}. A {corr_strength} {corr_direction} correlation was observed (r = {r:.4f}{ci_str}, {p_str}, n = {n:,}, n_effective = {n_eff:,}), which was {significance} at α = 0.05.

{'-'*80}
"""
    return description


def generate_combined_figure_description(
    line_num: int, channel: str, drug: str, fr: float,
    stats_control: Dict, stats_treatment: Dict,
    time_window_control: list, time_window_treatment: list, plot_type: str = 'scatter'
) -> str:
    """Generate figure description for combined plots."""
    channel_desc = CHANNEL_DESCRIPTIONS.get(channel, channel)
    drug_display = format_drug_name(drug)
    
    r_ctrl = stats_control.get('r', float('nan'))
    r_treat = stats_treatment.get('r', float('nan'))
    n_eff_ctrl = stats_control.get('n_effective', 0)
    n_eff_treat = stats_treatment.get('n_effective', 0)
    
    if not (np.isnan(r_ctrl) or np.isnan(r_treat)):
        r_change = r_treat - r_ctrl
        change_direction = "increased" if r_change > 0 else "decreased"
        change_descriptor = f"{change_direction} by {abs(r_change):.4f}"
    else:
        change_descriptor = "could not be determined"
    
    description = f"""
{'-'*80}
FIGURE DESCRIPTION - Subject {line_num} - Combined - {plot_type.capitalize()}
{'-'*80}

Comparison of electrophysiology-calcium fluorescence relationships for Subject {line_num} ({drug_display}). Control: r = {r_ctrl:.4f} (n_eff = {n_eff_ctrl:,}). Treatment: r = {r_treat:.4f} (n_eff = {n_eff_treat:,}). The correlation {change_descriptor} from control to treatment.

{'-'*80}
"""
    return description


# INDIVIDUAL SUBJECT PLOTTING FUNCTIONS

def create_scatter_plot(
    calcium_signal: np.ndarray, eeg_signal: np.ndarray,
    line_num: int, condition: str, period: str, channel: str,
    fr: float, time_window: list, save_path: Optional[str] = None
) -> Tuple[plt.Figure, Dict]:
    """Create a scatter plot of EEG vs Calcium fluorescence."""
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    
    if period == 'Control':
        color = COLORS['control']
        marker = MARKERS['control']
    else:
        color = COLORS['treatment']
        marker = MARKERS['treatment']
    
    ax.scatter(calcium_signal, eeg_signal, c=color, marker=marker,
               s=1, alpha=0.3, edgecolors='none', rasterized=True)
    
    stats = compute_statistics_robust(calcium_signal, eeg_signal, fr)
    
    if stats['n'] > 2 and not np.isnan(stats['r']):
        valid_mask = ~(np.isnan(calcium_signal) | np.isnan(eeg_signal))
        calcium_valid = calcium_signal[valid_mask]
        
        p = np.poly1d([stats['slope'], stats['intercept']])
        x_line = np.linspace(np.min(calcium_valid), np.max(calcium_valid), 100)
        ax.plot(x_line, p(x_line), color=COLORS['regression'], 
                linestyle='--', linewidth=1)
        
        ci_str = ""
        if not np.isnan(stats['ci_lower']):
            ci_str = f"\n95% CI [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]"
        
        stats_text = (f"n = {stats['n']:,}\nn_eff = {stats['n_effective']:,}\n"
                      f"r = {stats['r']:.3f}{ci_str}\n{stats['p_value_corrected_str']}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=FONTS['annotation'], verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='black', linewidth=0.5, alpha=0.9))
    
    ax.set_xlabel(f'Calcium ({UNITS["calcium"]})', fontsize=FONTS['axis_label'])
    ax.set_ylabel(f'EEG ({UNITS["eeg"]})', fontsize=FONTS['axis_label'])
    ax.tick_params(axis='both', which='major', labelsize=FONTS['tick_label'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5, linestyle=':')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + '.tiff', format='tiff', dpi=COLOR_DPI,
                   bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
        fig.savefig(save_path + '.svg', format='svg', dpi=COLOR_DPI, bbox_inches='tight')
    
    description = generate_figure_description(
        line_num, channel, condition, period, fr, stats, time_window, 'scatter')
    print(description)
    
    plt.show()
    plt.close(fig)
    
    return fig, stats


def create_hexbin_plot(
    calcium_signal: np.ndarray, eeg_signal: np.ndarray,
    line_num: int, condition: str, period: str, channel: str,
    fr: float, time_window: list, save_path: Optional[str] = None
) -> Tuple[plt.Figure, Dict]:
    """Create a hexbin plot of EEG vs Calcium fluorescence."""
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    
    stats = compute_statistics_robust(calcium_signal, eeg_signal, fr)
    
    valid_mask = ~(np.isnan(calcium_signal) | np.isnan(eeg_signal))
    calcium_valid = calcium_signal[valid_mask]
    eeg_valid = eeg_signal[valid_mask]
    
    if len(calcium_valid) < 10:
        print(f"    WARNING: Not enough valid data points for hexbin plot")
        plt.close(fig)
        return fig, stats
    
    hb = ax.hexbin(calcium_valid, eeg_valid, gridsize=50, cmap='plasma',
                   mincnt=1, edgecolors='none')
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count', fontsize=FONTS['axis_label'])
    cb.ax.tick_params(labelsize=FONTS['tick_label'])
    
    if stats['n'] > 2 and not np.isnan(stats['r']):
        p = np.poly1d([stats['slope'], stats['intercept']])
        x_line = np.linspace(np.min(calcium_valid), np.max(calcium_valid), 100)
        ax.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=1.5)
        
        ci_str = ""
        if not np.isnan(stats['ci_lower']):
            ci_str = f"\n95% CI [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]"
        
        stats_text = (f"n = {stats['n']:,}\nn_eff = {stats['n_effective']:,}\n"
                      f"r = {stats['r']:.3f}{ci_str}\n{stats['p_value_corrected_str']}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=FONTS['annotation'], verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor='black', linewidth=0.5, alpha=0.9))
    
    ax.set_xlabel(f'Calcium ({UNITS["calcium"]})', fontsize=FONTS['axis_label'])
    ax.set_ylabel(f'EEG ({UNITS["eeg"]})', fontsize=FONTS['axis_label'])
    ax.tick_params(axis='both', which='major', labelsize=FONTS['tick_label'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + '.tiff', format='tiff', dpi=COLOR_DPI,
                   bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
        fig.savefig(save_path + '.svg', format='svg', dpi=COLOR_DPI, bbox_inches='tight')
    
    description = generate_figure_description(
        line_num, channel, condition, period, fr, stats, time_window, 'hexbin')
    print(description)
    
    plt.show()
    plt.close(fig)
    
    return fig, stats


def create_combined_scatter_plot(
    calcium_control: np.ndarray, eeg_control: np.ndarray,
    calcium_treatment: np.ndarray, eeg_treatment: np.ndarray,
    line_num: int, condition: str, channel: str, fr: float,
    time_window_control: list, time_window_treatment: list,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Dict, Dict]:
    """Create combined scatter plot for control and treatment."""
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    
    ax.scatter(calcium_control, eeg_control, c=COLORS['control'],
               marker=MARKERS['control'], s=1, alpha=0.2, edgecolors='none',
               rasterized=True, label='Control')
    ax.scatter(calcium_treatment, eeg_treatment, c=COLORS['treatment'],
               marker=MARKERS['treatment'], s=1, alpha=0.2, edgecolors='none',
               rasterized=True, label='Treatment')
    
    stats_control = compute_statistics_robust(calcium_control, eeg_control, fr)
    stats_treatment = compute_statistics_robust(calcium_treatment, eeg_treatment, fr)
    
    stats_text = ""
    
    if stats_control['n'] > 2 and not np.isnan(stats_control['r']):
        valid_ctrl = ~(np.isnan(calcium_control) | np.isnan(eeg_control))
        calcium_ctrl_valid = calcium_control[valid_ctrl]
        p_ctrl = np.poly1d([stats_control['slope'], stats_control['intercept']])
        x_line_ctrl = np.linspace(np.min(calcium_ctrl_valid), np.max(calcium_ctrl_valid), 100)
        ax.plot(x_line_ctrl, p_ctrl(x_line_ctrl), color=COLORS['regression_ctrl'],
                linestyle='--', linewidth=1.5)
        stats_text += f"Ctrl: r={stats_control['r']:.3f} (n_eff={stats_control['n_effective']:,})\n"
    
    if stats_treatment['n'] > 2 and not np.isnan(stats_treatment['r']):
        valid_treat = ~(np.isnan(calcium_treatment) | np.isnan(eeg_treatment))
        calcium_treat_valid = calcium_treatment[valid_treat]
        p_treat = np.poly1d([stats_treatment['slope'], stats_treatment['intercept']])
        x_line_treat = np.linspace(np.min(calcium_treat_valid), np.max(calcium_treat_valid), 100)
        ax.plot(x_line_treat, p_treat(x_line_treat), color=COLORS['regression_treat'],
                linestyle='-', linewidth=1.5)
        stats_text += f"Treat: r={stats_treatment['r']:.3f} (n_eff={stats_treatment['n_effective']:,})"
    
    if stats_text:
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=FONTS['annotation'], verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor='black', linewidth=0.5, alpha=0.9))
    
    ax.legend(loc='lower right', fontsize=FONTS['annotation'], frameon=True, markerscale=5)
    ax.set_xlabel(f'Calcium ({UNITS["calcium"]})', fontsize=FONTS['axis_label'])
    ax.set_ylabel(f'EEG ({UNITS["eeg"]})', fontsize=FONTS['axis_label'])
    ax.tick_params(axis='both', which='major', labelsize=FONTS['tick_label'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5, linestyle=':')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + '.tiff', format='tiff', dpi=COLOR_DPI,
                   bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
        fig.savefig(save_path + '.svg', format='svg', dpi=COLOR_DPI, bbox_inches='tight')
    
    description = generate_combined_figure_description(
        line_num, channel, condition, fr, stats_control, stats_treatment,
        time_window_control, time_window_treatment, 'scatter')
    print(description)
    
    plt.show()
    plt.close(fig)
    
    return fig, stats_control, stats_treatment


def create_combined_hexbin_plot(
    calcium_control: np.ndarray, eeg_control: np.ndarray,
    calcium_treatment: np.ndarray, eeg_treatment: np.ndarray,
    line_num: int, condition: str, channel: str, fr: float,
    time_window_control: list, time_window_treatment: list,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Dict, Dict]:
    """Create combined hexbin plot with side-by-side panels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    
    stats_control = compute_statistics_robust(calcium_control, eeg_control, fr)
    stats_treatment = compute_statistics_robust(calcium_treatment, eeg_treatment, fr)
    
    valid_ctrl = ~(np.isnan(calcium_control) | np.isnan(eeg_control))
    calcium_ctrl_valid = calcium_control[valid_ctrl]
    eeg_ctrl_valid = eeg_control[valid_ctrl]
    
    valid_treat = ~(np.isnan(calcium_treatment) | np.isnan(eeg_treatment))
    calcium_treat_valid = calcium_treatment[valid_treat]
    eeg_treat_valid = eeg_treatment[valid_treat]
    
    all_calcium = np.concatenate([calcium_ctrl_valid, calcium_treat_valid])
    all_eeg = np.concatenate([eeg_ctrl_valid, eeg_treat_valid])
    
    x_min, x_max = np.percentile(all_calcium, [1, 99])
    y_min, y_max = np.percentile(all_eeg, [1, 99])
    
    if len(calcium_ctrl_valid) >= 10:
        hb1 = ax1.hexbin(calcium_ctrl_valid, eeg_ctrl_valid, gridsize=40,
                        cmap='plasma', mincnt=1, edgecolors='none',
                        extent=[x_min, x_max, y_min, y_max])
        
        if stats_control['n'] > 2 and not np.isnan(stats_control['r']):
            p_ctrl = np.poly1d([stats_control['slope'], stats_control['intercept']])
            x_line = np.linspace(x_min, x_max, 100)
            ax1.plot(x_line, p_ctrl(x_line), 'r--', linewidth=1.5)
            ax1.text(0.02, 0.98, f"r = {stats_control['r']:.3f}\nn_eff = {stats_control['n_effective']:,}",
                    transform=ax1.transAxes, fontsize=FONTS['annotation'], verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))
    
    ax1.set_title('Control', fontsize=FONTS['title'], fontweight='bold')
    ax1.set_xlabel(f'Calcium ({UNITS["calcium"]})', fontsize=FONTS['axis_label'])
    ax1.set_ylabel(f'EEG ({UNITS["eeg"]})', fontsize=FONTS['axis_label'])
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.tick_params(axis='both', which='major', labelsize=FONTS['tick_label'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    if len(calcium_treat_valid) >= 10:
        hb2 = ax2.hexbin(calcium_treat_valid, eeg_treat_valid, gridsize=40,
                        cmap='plasma', mincnt=1, edgecolors='none',
                        extent=[x_min, x_max, y_min, y_max])
        
        cb = fig.colorbar(hb2, ax=ax2)
        cb.set_label('Count', fontsize=FONTS['axis_label'])
        cb.ax.tick_params(labelsize=FONTS['tick_label'])
        
        if stats_treatment['n'] > 2 and not np.isnan(stats_treatment['r']):
            p_treat = np.poly1d([stats_treatment['slope'], stats_treatment['intercept']])
            x_line = np.linspace(x_min, x_max, 100)
            ax2.plot(x_line, p_treat(x_line), 'r--', linewidth=1.5)
            ax2.text(0.02, 0.98, f"r = {stats_treatment['r']:.3f}\nn_eff = {stats_treatment['n_effective']:,}",
                    transform=ax2.transAxes, fontsize=FONTS['annotation'], verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))
    
    ax2.set_title('Treatment', fontsize=FONTS['title'], fontweight='bold')
    ax2.set_xlabel(f'Calcium ({UNITS["calcium"]})', fontsize=FONTS['axis_label'])
    ax2.set_ylabel('')
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.tick_params(axis='both', which='major', labelsize=FONTS['tick_label'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + '.tiff', format='tiff', dpi=COLOR_DPI,
                   bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
        fig.savefig(save_path + '.svg', format='svg', dpi=COLOR_DPI, bbox_inches='tight')
    
    description = generate_combined_figure_description(
        line_num, channel, condition, fr, stats_control, stats_treatment,
        time_window_control, time_window_treatment, 'hexbin')
    print(description)
    
    plt.show()
    plt.close(fig)
    
    return fig, stats_control, stats_treatment



# POPULATION PLOTTING FUNCTIONS

def create_population_violin_plot(
    collector: PopulationCorrelationCollector,
    output_dir: str, channel: str, save_plot: bool = True
) -> Tuple[plt.Figure, Dict]:
    """Create violin plot comparing control vs treatment for a single drug."""
    pop_stats = collector.compute_population_statistics()
    
    if 'error' in pop_stats:
        print(f"    WARNING: {pop_stats['error']} for {collector.drug_name}")
        return None, pop_stats
    
    control, treatment = collector.get_arrays()
    
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH))
    
    # Create violin plots
    parts_ctrl = ax.violinplot([control[~np.isnan(control)]], positions=[1], 
                                showmeans=False, showmedians=False, showextrema=False)
    parts_treat = ax.violinplot([treatment[~np.isnan(treatment)]], positions=[2], 
                                 showmeans=False, showmedians=False, showextrema=False)
    
    for pc in parts_ctrl['bodies']:
        pc.set_facecolor(POPULATION_COLORS['violin_control'])
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(POPULATION_COLORS['violin_alpha'])
    
    for pc in parts_treat['bodies']:
        pc.set_facecolor(POPULATION_COLORS['violin_treatment'])
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(POPULATION_COLORS['violin_alpha'])
    
    # Add individual data points with jitter
    np.random.seed(42)
    jitter_ctrl = np.random.normal(0, 0.04, size=len(control))
    jitter_treat = np.random.normal(0, 0.04, size=len(treatment))
    
    ax.scatter(1 + jitter_ctrl, control, c=POPULATION_COLORS['point_color'],
               s=POPULATION_COLORS['point_size'], alpha=0.8, zorder=3,
               edgecolors='white', linewidths=0.5)
    ax.scatter(2 + jitter_treat, treatment, c=POPULATION_COLORS['point_color'],
               s=POPULATION_COLORS['point_size'], alpha=0.8, zorder=3,
               edgecolors='white', linewidths=0.5)
    
    # Connect paired observations
    for i in range(len(control)):
        if not (np.isnan(control[i]) or np.isnan(treatment[i])):
            ax.plot([1 + jitter_ctrl[i], 2 + jitter_treat[i]], 
                    [control[i], treatment[i]],
                    color=POPULATION_COLORS['line_color'],
                    alpha=POPULATION_COLORS['line_alpha'],
                    linewidth=0.8, zorder=2)
    
    # Add median lines
    ctrl_median = pop_stats['control']['median']
    treat_median = pop_stats['treatment']['median']
    
    ax.hlines(ctrl_median, 0.75, 1.25, colors='black', linewidth=2, zorder=4)
    ax.hlines(treat_median, 1.75, 2.25, colors='black', linewidth=2, zorder=4)
    
    # Statistical annotation
    p_value = pop_stats['comparison']['p_value']
    stars = pop_stats['comparison']['p_value_str']
    cohens_d = pop_stats['comparison']['cohens_d']
    
    y_max = max(np.nanmax(control), np.nanmax(treatment))
    y_min = min(np.nanmin(control), np.nanmin(treatment))
    y_range = y_max - y_min
    bracket_y = y_max + 0.1 * y_range
    
    if p_value < 0.05:
        ax.plot([1, 1, 2, 2], [bracket_y, bracket_y + 0.03 * y_range, 
                               bracket_y + 0.03 * y_range, bracket_y], 
                'k-', linewidth=1)
        ax.text(1.5, bracket_y + 0.05 * y_range, stars, 
                ha='center', va='bottom', fontsize=FONTS['annotation'])
    
    title_str = format_drug_name_short(collector.drug_name)
    median_str = f"median: {ctrl_median:.2f}±{pop_stats['control']['std']:.2f} → {treat_median:.2f}±{pop_stats['treatment']['std']:.2f}"
    ax.set_title(f"{title_str}\n{median_str}", fontsize=FONTS['title'])
    
    stats_text = (f"n = {pop_stats['n_valid']}\n"
                  f"p = {p_value:.4f} ({stars})\n"
                  f"Cohen's d = {cohens_d:.2f}")
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=FONTS['stats_text'], verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Control', 'Treatment'], fontsize=FONTS['axis_label'])
    ax.set_ylabel('Pearson Correlation (r)', fontsize=FONTS['axis_label'])
    ax.tick_params(axis='y', labelsize=FONTS['tick_label'])
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim([y_min - 0.15 * y_range, y_max + 0.25 * y_range])
    ax.set_xlim([0.5, 2.5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot:
        safe_drug = collector.drug_name.replace(":", "_").replace(" ", "_").replace(".", "")
        save_path = os.path.join(output_dir, f"population_violin_{safe_drug}_{channel}")
        
        fig.savefig(save_path + '.tiff', format='tiff', dpi=COLOR_DPI,
                   bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
        fig.savefig(save_path + '.svg', format='svg', dpi=COLOR_DPI, bbox_inches='tight')
        fig.savefig(save_path + '.eps', format='eps', dpi=COLOR_DPI, bbox_inches='tight')
        
        print(f"    Saved population violin plot: {save_path}")
    
    plt.show()
    plt.close(fig)
    
    return fig, pop_stats


def create_all_drugs_summary_plot(
    all_collectors: Dict[str, PopulationCorrelationCollector],
    output_dir: str, channel: str, save_plot: bool = True
) -> Tuple[plt.Figure, Dict]:
    """Create comprehensive summary plot comparing all drug conditions."""
    n_drugs = len(all_collectors)
    
    if n_drugs == 0:
        print("    WARNING: No drug data to plot")
        return None, {}
    
    all_stats = {}
    for drug_name, collector in all_collectors.items():
        all_stats[drug_name] = collector.compute_population_statistics()
    
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH))
    ax1, ax2 = axes
    
    x = np.arange(n_drugs)
    width = 0.35
    
    drug_names = list(all_collectors.keys())
    control_medians = []
    treatment_medians = []
    control_errors = []
    treatment_errors = []
    
    for drug in drug_names:
        stats = all_stats[drug]
        if 'error' not in stats:
            control_medians.append(stats['control']['median'])
            treatment_medians.append(stats['treatment']['median'])
            control_errors.append(stats['control']['std'])
            treatment_errors.append(stats['treatment']['std'])
        else:
            control_medians.append(np.nan)
            treatment_medians.append(np.nan)
            control_errors.append(0)
            treatment_errors.append(0)
    
    bars1 = ax1.bar(x - width/2, control_medians, width, yerr=control_errors, capsize=3,
                    label='Control', color=POPULATION_COLORS['violin_control'],
                    edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, treatment_medians, width, yerr=treatment_errors, capsize=3,
                    label='Treatment', color=POPULATION_COLORS['violin_treatment'],
                    edgecolor='black', linewidth=1)
    
    for i, drug in enumerate(drug_names):
        stats = all_stats[drug]
        if 'error' not in stats and stats['comparison']['p_value'] < 0.05:
            y_pos = max(control_medians[i] + control_errors[i],
                       treatment_medians[i] + treatment_errors[i]) + 0.05
            ax1.text(i, y_pos, stats['comparison']['p_value_str'],
                    ha='center', fontsize=FONTS['annotation'])
    
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Median Correlation (r)', fontsize=FONTS['axis_label'])
    ax1.set_xticks(x)
    ax1.set_xticklabels([format_drug_name_short(d) for d in drug_names], 
                        fontsize=FONTS['tick_label'], rotation=45, ha='right')
    ax1.legend(fontsize=FONTS['annotation'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('Population Correlations by Drug Condition', fontsize=FONTS['title'])
    
    effect_sizes = []
    effect_colors = []
    
    for drug in drug_names:
        stats = all_stats[drug]
        if 'error' not in stats:
            d = stats['comparison']['cohens_d']
            effect_sizes.append(d)
            effect_colors.append('#E74C3C' if d > 0 else '#3498DB')
        else:
            effect_sizes.append(0)
            effect_colors.append('gray')
    
    ax2.bar(x, effect_sizes, color=effect_colors, edgecolor='black', linewidth=1, alpha=0.7)
    
    for threshold in [0.2, -0.2, 0.5, -0.5, 0.8, -0.8]:
        ax2.axhline(threshold, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    
    ax2.text(n_drugs - 0.5, 0.2, 'small', fontsize=6, va='bottom', ha='left', color='gray')
    ax2.text(n_drugs - 0.5, 0.5, 'medium', fontsize=6, va='bottom', ha='left', color='gray')
    ax2.text(n_drugs - 0.5, 0.8, 'large', fontsize=6, va='bottom', ha='left', color='gray')
    
    ax2.set_ylabel("Cohen's d (Effect Size)", fontsize=FONTS['axis_label'])
    ax2.set_xticks(x)
    ax2.set_xticklabels([format_drug_name_short(d) for d in drug_names],
                        fontsize=FONTS['tick_label'], rotation=45, ha='right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Effect Size (Treatment vs Control)', fontsize=FONTS['title'])
    
    plt.tight_layout()
    
    if save_plot:
        save_path = os.path.join(output_dir, f"population_all_drugs_summary_{channel}")
        fig.savefig(save_path + '.tiff', format='tiff', dpi=COLOR_DPI,
                   bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
        fig.savefig(save_path + '.svg', format='svg', dpi=COLOR_DPI, bbox_inches='tight')
        fig.savefig(save_path + '.pdf', format='pdf', dpi=COLOR_DPI, bbox_inches='tight')
        print(f"    Saved all-drugs summary plot: {save_path}")
    
    plt.show()
    plt.close(fig)
    
    return fig, all_stats


# LOGGING AND EXPORT FUNCTIONS


def log_analysis_parameters(output_dir: str, line_num: int, 
                            channel: str, norm_params: Dict) -> str:
    """Save analysis parameters for reproducibility."""
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    params = {
        'analysis_date': datetime.now().isoformat(),
        'line_num': line_num,
        'drug_condition': number_to_drug.get(line_num, 'unknown'),
        'channel': channel,
        'time_selections': selections.get(line_num, []),
        'filter_config': FILTER_CONFIG,
        'normalization_config': NORMALIZATION_CONFIG,
        'normalization_params_used': norm_params,
    }
    
    params = convert_numpy(params)
    
    log_path = os.path.join(output_dir, f'analysis_params_line{line_num}.json')
    with open(log_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    return log_path


def save_population_statistics(all_stats: Dict, output_dir: str, channel: str):
    """Save population statistics to JSON."""
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    output = {
        'analysis_date': datetime.now().isoformat(),
        'channel': channel,
        'population_config': POPULATION_CONFIG,
        'statistics': convert_numpy(all_stats)
    }
    
    save_path = os.path.join(output_dir, f'population_statistics_{channel}.json')
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"  Saved population statistics: {save_path}")


def print_population_summary(all_stats: Dict):
    """Print formatted summary of population statistics."""
    print("\n" + "-"*80)
    print("POPULATION STATISTICS SUMMARY")
    print("-"*80)
    
    for drug, stats in all_stats.items():
        print(f"\n{format_drug_name_short(drug)}")
        print("-" * 40)
        
        if 'error' in stats:
            print(f"  Error: {stats['error']}")
            continue
        
        print(f"  N subjects: {stats['n_valid']}")
        print(f"  Control:    median = {stats['control']['median']:.3f} ± {stats['control']['std']:.3f}")
        print(f"              95% CI [{stats['control']['ci_lower']:.3f}, {stats['control']['ci_upper']:.3f}]")
        print(f"  Treatment:  median = {stats['treatment']['median']:.3f} ± {stats['treatment']['std']:.3f}")
        print(f"              95% CI [{stats['treatment']['ci_lower']:.3f}, {stats['treatment']['ci_upper']:.3f}]")
        print(f"  Comparison: {stats['comparison']['test_name']}")
        print(f"              p = {stats['comparison']['p_value']:.4f} {stats['comparison']['p_value_str']}")
        print(f"              Cohen's d = {stats['comparison']['cohens_d']:.3f} ({stats['comparison']['effect_size_interpretation']})")
    
    print("\n" + "-"*80)


# MAIN EXECUTION FUNCTIONS


def process_subject(
    line_num: int, channel: str, output_dir: str,
    collector: PopulationCorrelationCollector = None
) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
    """Process a single subject and optionally collect stats for population analysis."""
    print(f"\n{'-'*80}")
    print(f"Processing Line {line_num}")
    print(f"{'-'*80}")
    
    drug = number_to_drug.get(line_num)
    if drug is None:
        print(f"  ERROR: Line {line_num} not found in drug mapping")
        return False, None, None
    
    print(f"  Drug condition: {drug}")
    print(f"  Channel: {CHANNEL_DESCRIPTIONS.get(channel, channel)}")
    
    try:
        calcium_filepath = os.path.join(CALCIUM_DATA_PATH, f"meanFluorescence_{line_num}.npz")
        
        channel_object, miniscope_data_manager, fr = load_experiment(
            line_num, channel, calcium_signal_filepath=calcium_filepath)
        
        eeg_signal = channel_object.signal
        calcium_signal = miniscope_data_manager.mean_fluorescence_dict['meanFluorescence']
        
        print(f"  Handling NaN values...")
        eeg_signal, _ = handle_nans(eeg_signal, fr)
        calcium_signal, _ = handle_nans(calcium_signal, fr)
        
        control_eeg, treatment_eeg = slice_signal(eeg_signal, line_num, fr)
        control_calcium, treatment_calcium = slice_signal(calcium_signal, line_num, fr)
        
        time_window_control = selections[line_num][0]
        time_window_treatment = selections[line_num][1]
        
        print(f"  Applying bandpass filter ({FILTER_CONFIG['lowcut']}-{FILTER_CONFIG['highcut']} Hz)...")
        filtered_control_eeg, filtered_control_calcium = filter_and_trim(control_eeg, control_calcium, fr)
        filtered_treatment_eeg, filtered_treatment_calcium = filter_and_trim(treatment_eeg, treatment_calcium, fr)
        
        print(f"  Normalizing signals...")
        (norm_control_eeg, norm_treatment_eeg, 
         norm_control_calcium, norm_treatment_calcium, 
         norm_params) = normalize_signals_global(
            filtered_control_eeg, filtered_treatment_eeg,
            filtered_control_calcium, filtered_treatment_calcium)
        
        safe_drug = drug.replace(":", "_").replace(" ", "_").replace(".", "")
        subject_dir = os.path.join(output_dir, f"line_{line_num}_{safe_drug}")
        os.makedirs(subject_dir, exist_ok=True)
        
        log_analysis_parameters(subject_dir, line_num, channel, norm_params)
        
        print(f"\n  Generating plots for Line {line_num}...")
        
        print("\n  1/6: Control Scatter Plot")
        create_scatter_plot(norm_control_calcium, norm_control_eeg, line_num, drug, 
                           'Control', channel, fr, time_window_control,
                           save_path=os.path.join(subject_dir, f"scatter_control_{channel}"))
        
        print("\n  2/6: Control Hexbin Plot")
        _, control_stats = create_hexbin_plot(norm_control_calcium, norm_control_eeg, 
                                              line_num, drug, 'Control', channel, fr, 
                                              time_window_control,
                                              save_path=os.path.join(subject_dir, f"hexbin_control_{channel}"))
        
        print("\n  3/6: Treatment Scatter Plot")
        create_scatter_plot(norm_treatment_calcium, norm_treatment_eeg, line_num, drug,
                           'Treatment', channel, fr, time_window_treatment,
                           save_path=os.path.join(subject_dir, f"scatter_treatment_{channel}"))
        
        print("\n  4/6: Treatment Hexbin Plot")
        _, treatment_stats = create_hexbin_plot(norm_treatment_calcium, norm_treatment_eeg,
                                                line_num, drug, 'Treatment', channel, fr,
                                                time_window_treatment,
                                                save_path=os.path.join(subject_dir, f"hexbin_treatment_{channel}"))
        
        print("\n  5/6: Combined Scatter Plot")
        create_combined_scatter_plot(norm_control_calcium, norm_control_eeg,
                                    norm_treatment_calcium, norm_treatment_eeg,
                                    line_num, drug, channel, fr,
                                    time_window_control, time_window_treatment,
                                    save_path=os.path.join(subject_dir, f"scatter_combined_{channel}"))
        
        print("\n  6/6: Combined Hexbin Plot")
        create_combined_hexbin_plot(norm_control_calcium, norm_control_eeg,
                                   norm_treatment_calcium, norm_treatment_eeg,
                                   line_num, drug, channel, fr,
                                   time_window_control, time_window_treatment,
                                   save_path=os.path.join(subject_dir, f"hexbin_combined_{channel}"))
        
        if collector is not None:
            collector.add_subject(line_num, control_stats, treatment_stats)
            print(f"  Added Line {line_num} to population collector")
        
        print(f"\n  Successfully completed Line {line_num}")
        
        return True, control_stats, treatment_stats
        
    except Exception as e:
        print(f"  ERROR processing line {line_num}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def generate_all_eeg_calcium_plots(
    channel: str = 'CBvsPCEEG',
    line_nums: list = None,
    output_dir: str = None
):
    """
    Generate EEG vs Calcium plots for all subjects with population-level analysis.
    """
    if output_dir is None:
        output_dir = RESULTS_PATH
    
    print("\n" + "-"*80)
    print("EEG vs CALCIUM COMPLETE ANALYSIS PIPELINE")
    print(f"Channel: {CHANNEL_DESCRIPTIONS.get(channel, channel)}")
    print(f"Output directory: {output_dir}")
    print("-"*80)
    print("\nConfiguration:")
    print(f"  Filter: {FILTER_CONFIG['lowcut']}-{FILTER_CONFIG['highcut']} Hz")
    print(f"  Normalization: {NORMALIZATION_CONFIG['method']}, global={NORMALIZATION_CONFIG['global_normalization']}")
    print(f"  Bootstrap iterations: {POPULATION_CONFIG['bootstrap_iterations']}")
    print("-"*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    global_config = {
        'analysis_date': datetime.now().isoformat(),
        'channel': channel,
        'filter_config': FILTER_CONFIG,
        'normalization_config': NORMALIZATION_CONFIG,
        'population_config': POPULATION_CONFIG,
    }
    with open(os.path.join(output_dir, 'analysis_configuration.json'), 'w') as f:
        json.dump(global_config, f, indent=2)
    
    successful = []
    failed = []
    all_collectors = {}
    
    for drug_name, drug_line_nums in data.items():
        print("\n" + "#"*80)
        print(f"# PROCESSING DRUG: {drug_name}")
        print(f"# Subjects: {drug_line_nums}")
        print("#"*80)
        
        if line_nums is not None:
            drug_line_nums = [ln for ln in drug_line_nums if ln in line_nums]
        
        if not drug_line_nums:
            print(f"  No subjects to process for {drug_name}")
            continue
        
        collector = PopulationCorrelationCollector(drug_name)
        
        for line_num in drug_line_nums:
            success, _, _ = process_subject(line_num, channel, output_dir, collector)
            
            if success:
                successful.append(line_num)
            else:
                failed.append(line_num)
        
        # Generate population plots for this drug
        if len(collector.subjects) >= 2:
            print(f"\n{'-'*60}")
            print(f"Generating population plots for: {drug_name}")
            print(f"{'-'*60}")
            
            safe_drug = drug_name.replace(":", "_").replace(" ", "_").replace(".", "")
            drug_output_dir = os.path.join(output_dir, f"population_{safe_drug}")
            os.makedirs(drug_output_dir, exist_ok=True)
            
            create_population_violin_plot(collector, drug_output_dir, channel)
            all_collectors[drug_name] = collector
        else:
            print(f"  Skipping population plots for {drug_name} (only {len(collector.subjects)} subjects)")
    
    # Cross-drug comparison
    if len(all_collectors) >= 2:
        print("\n" + "-"*80)
        print("GENERATING CROSS-DRUG POPULATION ANALYSES")
        print("-"*80)
        
        all_stats = {}
        for drug_name, collector in all_collectors.items():
            all_stats[drug_name] = collector.compute_population_statistics()
        
        print_population_summary(all_stats)
        
        create_all_drugs_summary_plot(all_collectors, output_dir, channel)
        save_population_statistics(all_stats, output_dir, channel)
    else:
        all_stats = {name: coll.compute_population_statistics() 
                     for name, coll in all_collectors.items()}
    
    print("-"*80)
    print(f"\nSuccessfully processed: {len(successful)} subjects")
    print(f"Failed: {len(failed)} subjects")
    if failed:
        print(f"\nFailed subjects: {failed}")
    print(f"\nAll figures saved to: {output_dir}")
    
    return successful, failed, all_stats



# MAIN ENTRY POINT

if __name__ == "__main__":
    """
    Run the complete analysis pipeline including:
    """
    
    successful, failed, population_stats = generate_all_eeg_calcium_plots(
        channel='PFCLFPvsCBEEG',  # Options: 'PFCLFPvsCBEEG', 'PFCEEGvsCBEEG', 'CBvsPCEEG'
        line_nums=None,           # None = all subjects; or specify e.g., [46, 47, 64]
        output_dir=RESULTS_PATH
    )