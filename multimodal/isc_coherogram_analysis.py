"""
Population-Level Coherence Analysis - Frequency-Domain Bootstrap Methodology
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import csd, welch
from scipy.signal.windows import dpss
from scipy.stats import wilcoxon
from scipy.stats import false_discovery_control
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
    101:  [[1,25], [37, 90]],
    
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

# Create reverse mapping
number_to_drug = {num: drug for drug, numbers in data.items() for num in numbers}

# Configuration
CHANNEL_NAME = 'PFCLFPvsCBEEG'  # Change to 'PFCLFPvsCBEEG' or 'PFCEEGvsCBEEG' or 'CBvsPCEEG'
BASE_DATA_PATH = r"C:\Users\Path\To\Fluorescence_npz_files "
RESULTS_PATH = r"C:\Users\Path\To\Results\PFCLFPvsCBEEG"
FREQUENCY_ROI = (0.5, 4.0)  # Hz - Region of interest for statistical testing
FILTER_RANGE = [0.5, 4.0]  # Hz - Bandpass filter range
FREQUENCY_DISPLAY_RANGE = (0.0, 15.0)  # Hz - Display range for plots

# Multitaper parameters
MULTITAPER_NW = 3.0  # Time-bandwidth product
MULTITAPER_K = 5     # Number of tapers
SEGMENT_LENGTH = 5.0  # seconds - gives frequency resolution of 0.2 Hz


# DATA LOADING FUNCTIONS

def load_experiment(line_num: int, calcium_signal_filepath: str = None, channel: str = CHANNEL_NAME):
    """
    Load experiment data using existing data management system.
    """
    from src2.miniscope.miniscope_data_manager import MiniscopeDataManager
    from src2.ephys.ephys_api import EphysAPI
    from src2.multimodal.miniscope_ephys_alignment_utils import (
        sync_neuralynx_miniscope_timestamps, 
        find_ephys_idx_of_TTL_events
    )
    
    print(f'  Loading data for subject {line_num}...')
    
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
        fr = 30
    
    ephys_api = EphysAPI()
    ephys_api.run(
        line_num, 
        channel_name=channel, 
        remove_artifacts=False, 
        filter_type=None,
        filter_range=FILTER_RANGE, 
        plot_channel=False, 
        plot_spectrogram=False, 
        plot_phases=False, 
        logging_level="CRITICAL"
    )
    channel_object = ephys_api.ephys_data_manager.get_channel(channel_name=channel)
    
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
    
    if calcium_signal_filepath:
        miniscope_data_manager.mean_fluorescence_dict = np.load(calcium_signal_filepath)
    
    if np.any(np.isnan(channel_object.signal)):
        print(f"    Replacing NaNs in EEG with zeros...")
        channel_object.signal = np.nan_to_num(channel_object.signal, nan=0.0)
    
    return channel_object, miniscope_data_manager, fr


def slice_signal(signal, line_num, fr):
    """Slice signal into control and treatment based on time selections."""
    control_start_idx = int(selections[line_num][0][0] * fr * 60)
    control_end_idx = int(selections[line_num][0][1] * fr * 60)
    treatment_start_idx = int(selections[line_num][1][0] * fr * 60)
    treatment_end_idx = int(selections[line_num][1][1] * fr * 60)
    
    sliced_control = signal[control_start_idx:control_end_idx]
    sliced_treatment = signal[treatment_start_idx:treatment_end_idx]
    
    return sliced_control, sliced_treatment


def filter_signals(eeg_signal, calcium_signal, fr, cut=FILTER_RANGE):
    """Apply bandpass filtering to signals."""
    try:
        from src import misc_functions
        from src2.ephys.ephys_data_manager import EphysDataManager
        
        filtered_eeg = EphysDataManager._filter_data(
            data=eeg_signal, n=2, cut=cut, 
            ftype='butter', fs=fr, btype='bandpass', bodePlot=False
        )
        filtered_calcium = misc_functions.filterData(
            calcium_signal, n=2, cut=cut, 
            ftype='butter', btype='bandpass', fs=fr
        )
        return filtered_eeg, filtered_calcium
    except ImportError:
        print("    Warning: Filtering modules not available, returning unfiltered signals")
        return eeg_signal, calcium_signal


# MULTITAPER COHERENCE COMPUTATION

def compute_multitaper_coherence(signal1: np.ndarray, 
                                  signal2: np.ndarray, 
                                  fs: float,
                                  nperseg: int = None,
                                  NW: float = MULTITAPER_NW,
                                  K: int = MULTITAPER_K,
                                  return_segments: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute multitaper coherence spectrum between two signals.
    """
    if nperseg is None:
        nperseg = int(SEGMENT_LENGTH * fs)
    
    min_length = min(len(signal1), len(signal2))
    if nperseg > min_length:
        nperseg = min_length
        print(f"    Warning: Reduced segment length to {nperseg} samples due to signal length")
    
    tapers = dpss(nperseg, NW, K)
    noverlap = nperseg // 2
    
    Pxx_sum = None
    Pyy_sum = None
    Pxy_sum = None
    
    for taper in tapers:
        freq, Pxy = csd(signal1, signal2, fs=fs, window=taper, 
                        nperseg=nperseg, noverlap=noverlap, scaling='density')
        _, Pxx = welch(signal1, fs=fs, window=taper, 
                       nperseg=nperseg, noverlap=noverlap, scaling='density')
        _, Pyy = welch(signal2, fs=fs, window=taper, 
                       nperseg=nperseg, noverlap=noverlap, scaling='density')
        
        if Pxx_sum is None:
            Pxx_sum = Pxx
            Pyy_sum = Pyy
            Pxy_sum = Pxy
        else:
            Pxx_sum += Pxx
            Pyy_sum += Pyy
            Pxy_sum += Pxy
    
    Pxx_avg = Pxx_sum / K
    Pyy_avg = Pyy_sum / K
    Pxy_avg = Pxy_sum / K
    
    coherence = np.abs(Pxy_avg)**2 / (Pxx_avg * Pyy_avg + 1e-10)
    coherence = np.clip(coherence, 0, 1)
    
    segment_coherences = None
    if return_segments:
        n_segments = min_length // nperseg
        if n_segments < 3:
            print(f"    Warning: Only {n_segments} segments available")
        
        segment_coherences = []
        
        for seg_idx in range(n_segments):
            start = seg_idx * nperseg
            end = start + nperseg
            
            seg1 = signal1[start:end]
            seg2 = signal2[start:end]
            
            Pxx_seg = None
            Pyy_seg = None
            Pxy_seg = None
            
            for taper in tapers:
                _, Pxy_t = csd(seg1, seg2, fs=fs, window=taper,
                               nperseg=len(seg1), noverlap=0, scaling='density')
                _, Pxx_t = welch(seg1, fs=fs, window=taper,
                                 nperseg=len(seg1), noverlap=0, scaling='density')
                _, Pyy_t = welch(seg2, fs=fs, window=taper,
                                 nperseg=len(seg1), noverlap=0, scaling='density')
                
                if Pxx_seg is None:
                    Pxx_seg = Pxx_t
                    Pyy_seg = Pyy_t
                    Pxy_seg = Pxy_t
                else:
                    Pxx_seg += Pxx_t
                    Pyy_seg += Pyy_t
                    Pxy_seg += Pxy_t
            
            Pxx_seg /= K
            Pyy_seg /= K
            Pxy_seg /= K
            
            coh_seg = np.abs(Pxy_seg)**2 / (Pxx_seg * Pyy_seg + 1e-10)
            coh_seg = np.clip(coh_seg, 0, 1)
            
            if len(coh_seg) != len(freq):
                from scipy.interpolate import interp1d
                freq_seg = np.fft.rfftfreq(len(seg1), 1/fs)
                interp_func = interp1d(freq_seg, coh_seg, kind='linear', 
                                       fill_value='extrapolate', bounds_error=False)
                coh_seg = interp_func(freq)
            
            segment_coherences.append(coh_seg)
        
        segment_coherences = np.array(segment_coherences)
    
    return freq, coherence, segment_coherences


class PopulationCoherenceAnalyzer:
    """
    Analyzes coherence spectra across a population of subjects.
    """
    
    def __init__(self, frequency_roi: Tuple[float, float] = FREQUENCY_ROI,
                 frequency_display_range: Tuple[float, float] = FREQUENCY_DISPLAY_RANGE):
        self.frequency_roi = frequency_roi
        self.frequency_display_range = frequency_display_range
        
        self.coherence_control = []
        self.coherence_treatment = []
        self.coherence_diff = []
        self.frequencies = None
        
        self.control_segments = []
        self.treatment_segments = []
        
        self.roi_values_control = []
        self.roi_values_treatment = []
        self.subject_ids = []
        
        self.subject_stats = []
        
    def add_subject(self, 
                   control_signal1: np.ndarray,
                   control_signal2: np.ndarray,
                   treatment_signal1: np.ndarray,
                   treatment_signal2: np.ndarray,
                   fs: float,
                   subject_id: str):
        """Add a subject's data. NO individual-level CIs are computed."""
        print(f"    Processing subject {subject_id}...")
        
        print("      Computing control coherence (multitaper)...")
        freq_control, coh_control, segments_control = compute_multitaper_coherence(
            control_signal1, control_signal2, fs, return_segments=True
        )
        
        print("      Computing treatment coherence (multitaper)...")
        freq_treatment, coh_treatment, segments_treatment = compute_multitaper_coherence(
            treatment_signal1, treatment_signal2, fs, return_segments=True
        )
        
        if self.frequencies is None:
            self.frequencies = freq_control.copy()
            print(f"      Frequency axis: {self.frequencies[0]:.3f} - {self.frequencies[-1]:.3f} Hz "
                  f"({len(self.frequencies)} bins, Δf = {self.frequencies[1]-self.frequencies[0]:.3f} Hz)")
        
        self.control_segments.append(segments_control)
        self.treatment_segments.append(segments_treatment)
        
        n_ctrl_seg = len(segments_control) if segments_control is not None else 1
        n_trt_seg = len(segments_treatment) if segments_treatment is not None else 1
        print(f"      Control segments: {n_ctrl_seg}, Treatment segments: {n_trt_seg}")
        
        if len(coh_control) != len(self.frequencies):
            from scipy.interpolate import interp1d
            interp_func = interp1d(freq_control, coh_control, kind='linear', fill_value='extrapolate')
            coh_control = interp_func(self.frequencies)
        
        if len(coh_treatment) != len(self.frequencies):
            from scipy.interpolate import interp1d
            interp_func = interp1d(freq_treatment, coh_treatment, kind='linear', fill_value='extrapolate')
            coh_treatment = interp_func(self.frequencies)
        
        self.coherence_control.append(coh_control)
        self.coherence_treatment.append(coh_treatment)
        
        diff_coh = coh_treatment - coh_control
        self.coherence_diff.append(diff_coh)
        
        freq_mask = (self.frequencies >= self.frequency_roi[0]) & \
                    (self.frequencies <= self.frequency_roi[1])
        
        roi_control = np.nanmean(coh_control[freq_mask])
        roi_treatment = np.nanmean(coh_treatment[freq_mask])
        
        self.roi_values_control.append(roi_control)
        self.roi_values_treatment.append(roi_treatment)
        self.subject_ids.append(subject_id)
        
        display_mask = (self.frequencies >= self.frequency_display_range[0]) & \
                       (self.frequencies <= self.frequency_display_range[1])
        
        stats = self._compute_subject_stats(
            coh_control, coh_treatment, diff_coh,
            freq_mask, display_mask, n_ctrl_seg, n_trt_seg,
            len(control_signal1) / fs, len(treatment_signal1) / fs
        )
        stats['subject_id'] = subject_id
        self.subject_stats.append(stats)
        
        print(f"      ROI Coherence: Control = {roi_control:.4f}, Treatment = {roi_treatment:.4f}, "
              f"Δ = {roi_treatment - roi_control:+.4f}")
    
    def _compute_subject_stats(self, coh_ctrl, coh_trt, coh_diff,
                               roi_mask, display_mask, n_ctrl_seg, n_trt_seg,
                               ctrl_dur, trt_dur) -> Dict:
        """Compute descriptive statistics for a single subject."""
        freq_display = self.frequencies[display_mask]
        
        roi_ctrl = coh_ctrl[roi_mask]
        roi_trt = coh_trt[roi_mask]
        roi_diff = coh_diff[roi_mask]
        
        ctrl_display = coh_ctrl[display_mask]
        trt_display = coh_trt[display_mask]
        ctrl_peak_idx = np.argmax(ctrl_display)
        trt_peak_idx = np.argmax(trt_display)
        
        return {
            'n_control_segments': n_ctrl_seg,
            'n_treatment_segments': n_trt_seg,
            'control_duration_sec': ctrl_dur,
            'treatment_duration_sec': trt_dur,
            
            'roi_control_mean': np.nanmean(roi_ctrl),
            'roi_control_std': np.nanstd(roi_ctrl, ddof=1),
            'roi_control_median': np.nanmedian(roi_ctrl),
            'roi_control_min': np.nanmin(roi_ctrl),
            'roi_control_max': np.nanmax(roi_ctrl),
            'roi_control_iqr_25': np.nanpercentile(roi_ctrl, 25),
            'roi_control_iqr_75': np.nanpercentile(roi_ctrl, 75),
            
            'roi_treatment_mean': np.nanmean(roi_trt),
            'roi_treatment_std': np.nanstd(roi_trt, ddof=1),
            'roi_treatment_median': np.nanmedian(roi_trt),
            'roi_treatment_min': np.nanmin(roi_trt),
            'roi_treatment_max': np.nanmax(roi_trt),
            'roi_treatment_iqr_25': np.nanpercentile(roi_trt, 25),
            'roi_treatment_iqr_75': np.nanpercentile(roi_trt, 75),
            
            'roi_diff_mean': np.nanmean(roi_diff),
            'roi_diff_std': np.nanstd(roi_diff, ddof=1),
            'roi_diff_median': np.nanmedian(roi_diff),
            'roi_diff_min': np.nanmin(roi_diff),
            'roi_diff_max': np.nanmax(roi_diff),
            
            'control_peak_freq_hz': freq_display[ctrl_peak_idx],
            'control_peak_coherence': ctrl_display[ctrl_peak_idx],
            'treatment_peak_freq_hz': freq_display[trt_peak_idx],
            'treatment_peak_coherence': trt_display[trt_peak_idx],
            
            'control_mean_all_freq': np.nanmean(ctrl_display),
            'treatment_mean_all_freq': np.nanmean(trt_display),
        }
    
    def compute_population_coherence_bootstrap(self, 
                                                n_bootstrap: int = 5000,
                                                confidence_level: float = 0.95) -> Dict:
        """
        Compute population-level coherence with bootstrap CIs.
        """
        print("\n  Computing population coherence with bootstrap methodology...")
        print("  (CIs computed at population level only - individual subjects have no CIs)")
        
        if len(self.coherence_control) == 0:
            raise ValueError("No subjects added. Use add_subject() first.")
        
        control_stack = np.stack(self.coherence_control, axis=0)
        treatment_stack = np.stack(self.coherence_treatment, axis=0)
        
        n_subjects, n_freq = control_stack.shape
        alpha = 1 - confidence_level
        
        print(f"    n_subjects = {n_subjects}, n_frequencies = {n_freq}")
        print(f"    Running {n_bootstrap} bootstrap iterations...")
        
        control_median = np.zeros(n_freq)
        control_ci_lower = np.zeros(n_freq)
        control_ci_upper = np.zeros(n_freq)
        
        treatment_median = np.zeros(n_freq)
        treatment_ci_lower = np.zeros(n_freq)
        treatment_ci_upper = np.zeros(n_freq)
        
        diff_median = np.zeros(n_freq)
        diff_ci_lower = np.zeros(n_freq)
        diff_ci_upper = np.zeros(n_freq)
        
        for freq_idx in range(n_freq):
            if freq_idx % 20 == 0:
                print(f"      Processing frequency bin {freq_idx}/{n_freq}...")
            
            control_vals = control_stack[:, freq_idx]
            treatment_vals = treatment_stack[:, freq_idx]
            differences = treatment_vals - control_vals
            
            boot_control = np.zeros(n_bootstrap)
            boot_treatment = np.zeros(n_bootstrap)
            boot_diff = np.zeros(n_bootstrap)
            
            for b in range(n_bootstrap):
                boot_idx = np.random.choice(n_subjects, size=n_subjects, replace=True)
                boot_control[b] = np.median(control_vals[boot_idx])
                boot_treatment[b] = np.median(treatment_vals[boot_idx])
                boot_diff[b] = np.mean(differences[boot_idx])
            
            control_median[freq_idx] = np.median(boot_control)
            control_ci_lower[freq_idx] = np.percentile(boot_control, 100 * alpha/2)
            control_ci_upper[freq_idx] = np.percentile(boot_control, 100 * (1 - alpha/2))
            
            treatment_median[freq_idx] = np.median(boot_treatment)
            treatment_ci_lower[freq_idx] = np.percentile(boot_treatment, 100 * alpha/2)
            treatment_ci_upper[freq_idx] = np.percentile(boot_treatment, 100 * (1 - alpha/2))
            
            diff_median[freq_idx] = np.median(boot_diff)
            diff_ci_lower[freq_idx] = np.percentile(boot_diff, 100 * alpha/2)
            diff_ci_upper[freq_idx] = np.percentile(boot_diff, 100 * (1 - alpha/2))
        
        sig_mask_uncorrected = ~((diff_ci_lower <= 0) & (diff_ci_upper >= 0))
        
        p_values = np.zeros(n_freq)
        for freq_idx in range(n_freq):
            control_vals = control_stack[:, freq_idx]
            treatment_vals = treatment_stack[:, freq_idx]
            differences = treatment_vals - control_vals
            
            boot_means = np.zeros(n_bootstrap)
            for b in range(n_bootstrap):
                boot_idx = np.random.choice(n_subjects, size=n_subjects, replace=True)
                boot_means[b] = np.mean(differences[boot_idx])
            
            if np.mean(boot_means) > 0:
                p_values[freq_idx] = 2 * np.mean(boot_means <= 0)
            else:
                p_values[freq_idx] = 2 * np.mean(boot_means >= 0)
            p_values[freq_idx] = np.clip(p_values[freq_idx], 1/n_bootstrap, 1.0)
        
        display_mask = (self.frequencies >= self.frequency_display_range[0]) & \
                       (self.frequencies <= self.frequency_display_range[1])
        p_values_display = p_values[display_mask]
        
        try:
            reject_fdr = false_discovery_control(p_values_display, method='bh')
            sig_mask_fdr_display = reject_fdr < 0.05
        except TypeError:
            n_tests = len(p_values_display)
            sorted_idx = np.argsort(p_values_display)
            sorted_p = p_values_display[sorted_idx]
            bh_critical = 0.05 * np.arange(1, n_tests + 1) / n_tests
            reject = sorted_p <= bh_critical
            if np.any(reject):
                max_k = np.max(np.where(reject)[0])
                sig_mask_fdr_display = np.zeros(n_tests, dtype=bool)
                sig_mask_fdr_display[sorted_idx[:max_k+1]] = True
            else:
                sig_mask_fdr_display = np.zeros(n_tests, dtype=bool)
        
        sig_mask_fdr = np.zeros(n_freq, dtype=bool)
        sig_mask_fdr[display_mask] = sig_mask_fdr_display
        
        print(f"    Significant frequencies (uncorrected): {np.sum(sig_mask_uncorrected)} / {n_freq}")
        print(f"    Significant frequencies (FDR-corrected): {np.sum(sig_mask_fdr)} / {n_freq}")
        
        return {
            'frequencies': self.frequencies,
            'control_median': control_median,
            'control_ci_lower': control_ci_lower,
            'control_ci_upper': control_ci_upper,
            'treatment_median': treatment_median,
            'treatment_ci_lower': treatment_ci_lower,
            'treatment_ci_upper': treatment_ci_upper,
            'diff_median': diff_median,
            'diff_ci_lower': diff_ci_lower,
            'diff_ci_upper': diff_ci_upper,
            'p_values': p_values,
            'sig_mask_uncorrected': sig_mask_uncorrected,
            'sig_mask_fdr': sig_mask_fdr,
            'n_subjects': n_subjects,
            'n_bootstrap': n_bootstrap,
            'confidence_level': confidence_level,
            'control_spectra': control_stack,
            'treatment_spectra': treatment_stack
        }
    
    def perform_roi_statistical_test(self) -> Dict:
        """Perform Wilcoxon signed-rank test on ROI values."""
        print("\n  Performing ROI statistical test (Wilcoxon signed-rank)...")
        
        control = np.array(self.roi_values_control)
        treatment = np.array(self.roi_values_treatment)
        
        statistic, p_value = wilcoxon(control, treatment, alternative='two-sided')
        
        differences = treatment - control
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        results = {
            'n_subjects': len(self.subject_ids),
            'control_median': np.median(control),
            'control_iqr': (np.percentile(control, 25), np.percentile(control, 75)),
            'control_mean': np.mean(control),
            'control_std': np.std(control, ddof=1),
            'treatment_median': np.median(treatment),
            'treatment_iqr': (np.percentile(treatment, 25), np.percentile(treatment, 75)),
            'treatment_mean': np.mean(treatment),
            'treatment_std': np.std(treatment, ddof=1),
            'wilcoxon_statistic': statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_difference': mean_diff,
            'subject_ids': self.subject_ids,
            'control_values': control,
            'treatment_values': treatment
        }
        
        print(f"    Wilcoxon test: W = {statistic:.2f}, p = {p_value:.4f}")
        print(f"    Cohen's d = {cohens_d:.3f}")
        
        return results
    
    def print_descriptive_statistics_report(self, 
                                           condition_name: str,
                                           bootstrap_results: Dict,
                                           roi_results: Dict,
                                           save_path: Optional[str] = None):
        """Print comprehensive descriptive statistics report to console and optionally save to file."""
        lines = []
        lines.append("-"*80)
        lines.append(f"COHERENCE ANALYSIS REPORT - {condition_name.upper()}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("-"*80)
        
        lines.append("\n" + "-"*80)
        lines.append("ANALYSIS PARAMETERS")
        lines.append("-"*80)
        lines.append(f"  Channel: {CHANNEL_NAME}")
        lines.append(f"  Frequency ROI: {self.frequency_roi[0]}-{self.frequency_roi[1]} Hz")
        lines.append(f"  Display Range: {self.frequency_display_range[0]}-{self.frequency_display_range[1]} Hz")
        lines.append(f"  Multitaper: NW={MULTITAPER_NW}, K={MULTITAPER_K} tapers")
        lines.append(f"  Segment Length: {SEGMENT_LENGTH}s (Δf = {1/SEGMENT_LENGTH:.2f} Hz)")
        lines.append(f"  Bootstrap iterations: {bootstrap_results['n_bootstrap']}")
        lines.append(f"  Confidence level: {bootstrap_results['confidence_level']*100:.0f}%")
        
        lines.append("\n" + "-"*80)
        lines.append("POPULATION SUMMARY")
        lines.append("-"*80)
        lines.append(f"  Number of subjects: {roi_results['n_subjects']}")
        lines.append(f"  Subject IDs: {', '.join(self.subject_ids)}")
        
        lines.append("\n" + "-"*80)
        lines.append(f"ROI COHERENCE STATISTICS ({self.frequency_roi[0]}-{self.frequency_roi[1]} Hz)")
        lines.append("-"*80)
        lines.append("\n  CONTROL CONDITION:")
        lines.append(f"    Mean +/- SD:    {roi_results['control_mean']:.4f} +/- {roi_results['control_std']:.4f}")
        lines.append(f"    Median [IQR]:   {roi_results['control_median']:.4f} [{roi_results['control_iqr'][0]:.4f}, {roi_results['control_iqr'][1]:.4f}]")
        lines.append(f"    Range:          [{np.min(roi_results['control_values']):.4f}, {np.max(roi_results['control_values']):.4f}]")
        
        lines.append("\n  TREATMENT CONDITION:")
        lines.append(f"    Mean +/- SD:    {roi_results['treatment_mean']:.4f} +/- {roi_results['treatment_std']:.4f}")
        lines.append(f"    Median [IQR]:   {roi_results['treatment_median']:.4f} [{roi_results['treatment_iqr'][0]:.4f}, {roi_results['treatment_iqr'][1]:.4f}]")
        lines.append(f"    Range:          [{np.min(roi_results['treatment_values']):.4f}, {np.max(roi_results['treatment_values']):.4f}]")
        
        lines.append("\n  DIFFERENCE (Treatment - Control):")
        diff_vals = roi_results['treatment_values'] - roi_results['control_values']
        lines.append(f"    Mean +/- SD:    {roi_results['mean_difference']:+.4f} +/- {np.std(diff_vals, ddof=1):.4f}")
        lines.append(f"    Median [IQR]:   {np.median(diff_vals):+.4f} [{np.percentile(diff_vals, 25):+.4f}, {np.percentile(diff_vals, 75):+.4f}]")
        lines.append(f"    Range:          [{np.min(diff_vals):+.4f}, {np.max(diff_vals):+.4f}]")
        
        lines.append("\n" + "-"*80)
        lines.append("STATISTICAL INFERENCE")
        lines.append("-"*80)
        lines.append("\n  ROI COMPARISON (Wilcoxon Signed-Rank Test):")
        lines.append(f"    Test statistic (W): {roi_results['wilcoxon_statistic']:.2f}")
        lines.append(f"    p-value:            {roi_results['p_value']:.6f}")
        lines.append(f"    Cohen's d:          {roi_results['cohens_d']:.3f}")
        
        d = abs(roi_results['cohens_d'])
        if d < 0.2:
            effect_interp = "negligible"
        elif d < 0.5:
            effect_interp = "small"
        elif d < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        lines.append(f"    Effect size:        {effect_interp}")
        
        if roi_results['p_value'] < 0.001:
            sig_interp = "highly significant (p < 0.001)"
        elif roi_results['p_value'] < 0.01:
            sig_interp = "significant (p < 0.01)"
        elif roi_results['p_value'] < 0.05:
            sig_interp = "significant (p < 0.05)"
        else:
            sig_interp = "not significant (p >= 0.05)"
        lines.append(f"    Interpretation:     {sig_interp}")
        
        n_sig_uncorr = np.sum(bootstrap_results['sig_mask_uncorrected'])
        n_sig_fdr = np.sum(bootstrap_results['sig_mask_fdr'])
        n_freq = len(bootstrap_results['frequencies'])
        
        lines.append("\n  FREQUENCY-RESOLVED SIGNIFICANCE (Bootstrap CI):")
        lines.append(f"    Uncorrected:   {n_sig_uncorr}/{n_freq} frequencies ({100*n_sig_uncorr/n_freq:.1f}%)")
        lines.append(f"    FDR-corrected: {n_sig_fdr}/{n_freq} frequencies ({100*n_sig_fdr/n_freq:.1f}%)")
        
        lines.append("\n" + "-"*80)
        lines.append("INDIVIDUAL SUBJECT STATISTICS")
        lines.append("-"*80)
        
        lines.append("\n  Subject         | Control  | Treatmt  | Delta    | Ctrl Peak        | Trt Peak         | Ctrl Dur | Trt Dur")
        lines.append("  " + "-"*110)
        
        for stats in self.subject_stats:
            sid = stats['subject_id']
            ctrl_roi = stats['roi_control_mean']
            trt_roi = stats['roi_treatment_mean']
            diff_roi = stats['roi_diff_mean']
            ctrl_peak = f"{stats['control_peak_coherence']:.3f}@{stats['control_peak_freq_hz']:.1f}Hz"
            trt_peak = f"{stats['treatment_peak_coherence']:.3f}@{stats['treatment_peak_freq_hz']:.1f}Hz"
            ctrl_dur = f"{stats['control_duration_sec']/60:.1f}min"
            trt_dur = f"{stats['treatment_duration_sec']/60:.1f}min"
            
            lines.append(f"  {sid:15} | {ctrl_roi:8.4f} | {trt_roi:8.4f} | {diff_roi:+8.4f} | {ctrl_peak:16} | {trt_peak:16} | {ctrl_dur:8} | {trt_dur:7}")
        
        lines.append("  " + "-"*110)
        mean_ctrl = np.mean([s['roi_control_mean'] for s in self.subject_stats])
        mean_trt = np.mean([s['roi_treatment_mean'] for s in self.subject_stats])
        mean_diff = np.mean([s['roi_diff_mean'] for s in self.subject_stats])
        lines.append(f"  {'MEAN':15} | {mean_ctrl:8.4f} | {mean_trt:8.4f} | {mean_diff:+8.4f} |")
        
        std_ctrl = np.std([s['roi_control_mean'] for s in self.subject_stats], ddof=1)
        std_trt = np.std([s['roi_treatment_mean'] for s in self.subject_stats], ddof=1)
        std_diff = np.std([s['roi_diff_mean'] for s in self.subject_stats], ddof=1)
        lines.append(f"  {'SD':15} | {std_ctrl:8.4f} | {std_trt:8.4f} | {std_diff:8.4f} |")
        
        lines.append("\n" + "-"*80)
        lines.append("METHODOLOGY NOTES")
        lines.append("-"*80)
        lines.append("  - Coherence computed using multitaper method (Thomson, 1982)")
        lines.append("  - Population-level 95% CIs computed via bootstrap resampling across subjects")
        lines.append("  - Individual subject CIs NOT computed (single recording per condition)")
        lines.append("  - FDR correction uses Benjamini-Hochberg procedure")
        lines.append("  - ROI statistical test: paired Wilcoxon signed-rank (nonparametric)")
        
        lines.append("\n" + "-"*80)
        lines.append("END OF REPORT")
        lines.append("-"*80 + "\n")
        
        report = "\n".join(lines)
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"  Report saved to: {save_path}")
        
        return report
    
    def plot_coherence_spectra(self,
                               bootstrap_results: Dict,
                               roi_results: Dict,
                               save_path: Optional[str] = None,
                               condition_name: str = "Condition"):
        """Plot population coherence spectra with bootstrap CIs."""
        freq = bootstrap_results['frequencies']
        
        display_mask = (freq >= self.frequency_display_range[0]) & \
                       (freq <= self.frequency_display_range[1])
        freq_display = freq[display_mask]
        
        control_med = bootstrap_results['control_median'][display_mask]
        control_ci_lo = bootstrap_results['control_ci_lower'][display_mask]
        control_ci_hi = bootstrap_results['control_ci_upper'][display_mask]
        
        treatment_med = bootstrap_results['treatment_median'][display_mask]
        treatment_ci_lo = bootstrap_results['treatment_ci_lower'][display_mask]
        treatment_ci_hi = bootstrap_results['treatment_ci_upper'][display_mask]
        
        diff_med = bootstrap_results['diff_median'][display_mask]
        diff_ci_lo = bootstrap_results['diff_ci_lower'][display_mask]
        diff_ci_hi = bootstrap_results['diff_ci_upper'][display_mask]
        
        sig_fdr = bootstrap_results['sig_mask_fdr'][display_mask]
        
        fig, axes = plt.subplots(2, 1, figsize=(6.93, 5.5), 
                                  gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35})
        ax1, ax2 = axes
        
        ax1.plot(freq_display, control_med, 'b-', linewidth=2, label='Control')
        ax1.fill_between(freq_display, control_ci_lo, control_ci_hi, color='blue', alpha=0.2)
        ax1.plot(freq_display, treatment_med, 'r-', linewidth=2, label='Treatment')
        ax1.fill_between(freq_display, treatment_ci_lo, treatment_ci_hi, color='red', alpha=0.2)
        
        roi_lo, roi_hi = self.frequency_roi
        ax1.axvspan(roi_lo, roi_hi, alpha=0.1, color='gray', label='ROI')
        
        ax1.set_xlabel('Frequency (Hz)', fontsize=10)
        ax1.set_ylabel('Coherence', fontsize=10)
        ax1.set_title(f'{condition_name} — Population Coherence (n={roi_results["n_subjects"]})', fontsize=11, fontweight='bold')
        ax1.set_xlim([freq_display[0], freq_display[-1]])
        ax1.set_ylim([0, min(1.0, max(control_ci_hi.max(), treatment_ci_hi.max()) * 1.1)])
        ax1.legend(loc='upper right', fontsize=8)
        ax1.tick_params(axis='both', which='major', labelsize=9)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold')
        
        stats_text = f"n = {roi_results['n_subjects']}\np = {roi_results['p_value']:.4f}\nd = {roi_results['cohens_d']:.2f}"
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        
        ax2.plot(freq_display, diff_med, 'k-', linewidth=2, label='Δ Coherence')
        ax2.fill_between(freq_display, diff_ci_lo, diff_ci_hi, color='gray', alpha=0.3, label='95% CI (Bootstrap)')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        
        # Overlay red line where FDR-significant (continuous line through significant regions)
        if np.any(sig_fdr):
            sig_line = np.where(sig_fdr, diff_med, np.nan)
            ax2.plot(freq_display, sig_line, 'r-', linewidth=2.5, label='Significant (FDR)', zorder=5)
        
        ax2.axvspan(roi_lo, roi_hi, alpha=0.1, color='gray')
        ax2.set_xlabel('Frequency (Hz)', fontsize=10)
        ax2.set_ylabel('Δ Coherence\n(Treatment − Control)', fontsize=10)
        ax2.set_title('Population Coherence Difference', fontsize=11, fontweight='bold')
        ax2.set_xlim([freq_display[0], freq_display[-1]])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            for fmt, ext in [('svg', '.svg'), ('tiff', '.tiff'), ('eps', '.eps')]:
                out_path = save_path.replace('.svg', ext)
                if fmt == 'tiff':
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
                else:
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300)
                print(f"  Saved coherence spectra ({fmt.upper()}) to: {out_path}")
        
        plt.show()
    
    def plot_individual_subject_overlay(self, bootstrap_results: Dict, save_path: Optional[str] = None, condition_name: str = "Condition"):
        """Plot individual subject differences overlaid (NO individual CIs)."""
        freq = bootstrap_results['frequencies']
        diff_stack = np.array(self.coherence_diff)
        
        display_mask = (freq >= self.frequency_display_range[0]) & (freq <= self.frequency_display_range[1])
        freq_display = freq[display_mask]
        diff_display = diff_stack[:, display_mask]
        
        diff_med = bootstrap_results['diff_median'][display_mask]
        diff_ci_lo = bootstrap_results['diff_ci_lower'][display_mask]
        diff_ci_hi = bootstrap_results['diff_ci_upper'][display_mask]
        sig_fdr = bootstrap_results['sig_mask_fdr'][display_mask]
        
        n_subjects = diff_display.shape[0]
        
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_subjects, 10)))
        
        for i in range(n_subjects):
            ax.plot(freq_display, diff_display[i, :], color=colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                   label=self.subject_ids[i] if i < 10 else None)
        
        ax.fill_between(freq_display, diff_ci_lo, diff_ci_hi, color='black', alpha=0.25, label='Population 95% CI', zorder=9)
        ax.plot(freq_display, diff_med, 'k-', linewidth=3, label='Population Median', zorder=10)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        
        # Overlay red line where FDR-significant
        if np.any(sig_fdr):
            sig_line = np.where(sig_fdr, diff_med, np.nan)
            ax.plot(freq_display, sig_line, 'r-', linewidth=3.5, label='Significant (FDR)', zorder=15)
        
        ax.axvspan(self.frequency_roi[0], self.frequency_roi[1], alpha=0.08, color='gray', label='ROI')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Δ Coherence (Treatment − Control)', fontsize=11)
        ax.set_title(f'{condition_name} — Individual Subjects (no individual CIs)', fontsize=12, fontweight='bold')
        ax.set_xlim([freq_display[0], freq_display[-1]])
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7, ncol=2)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.text(0.02, 0.98, f'n = {n_subjects}', transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            for fmt, ext in [('svg', '.svg'), ('tiff', '.tiff')]:
                out_path = save_path.replace('.svg', ext)
                if fmt == 'tiff':
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
                else:
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300)
                print(f"  Saved individual overlay ({fmt.upper()}) to: {out_path}")
        
        plt.show()
    
    def plot_individual_spectra_overlay(self, save_path: Optional[str] = None, condition_name: str = "Condition"):
        """Plot individual subject coherence spectra (NO CIs)."""
        n_subjects = len(self.subject_ids)
        display_mask = (self.frequencies >= self.frequency_display_range[0]) & (self.frequencies <= self.frequency_display_range[1])
        freq_display = self.frequencies[display_mask]
        
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
        ax_ctrl, ax_trt = axes
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_subjects, 10)))
        
        for i in range(n_subjects):
            ax_ctrl.plot(freq_display, self.coherence_control[i][display_mask], color=colors[i % len(colors)], 
                        alpha=0.8, linewidth=1.5, label=self.subject_ids[i] if i < 10 else None)
        
        ctrl_mean = np.mean([self.coherence_control[i][display_mask] for i in range(n_subjects)], axis=0)
        ax_ctrl.plot(freq_display, ctrl_mean, 'k--', linewidth=2.5, label='Mean', zorder=10)
        ax_ctrl.axvspan(self.frequency_roi[0], self.frequency_roi[1], alpha=0.08, color='gray')
        ax_ctrl.set_xlabel('Frequency (Hz)', fontsize=10)
        ax_ctrl.set_ylabel('Coherence', fontsize=10)
        ax_ctrl.set_title(f'{condition_name} — Control', fontsize=11, fontweight='bold')
        ax_ctrl.set_xlim([freq_display[0], freq_display[-1]])
        ax_ctrl.set_ylim([0, 1])
        ax_ctrl.legend(loc='upper right', fontsize=7, ncol=2)
        ax_ctrl.tick_params(axis='both', which='major', labelsize=9)
        ax_ctrl.spines['top'].set_visible(False)
        ax_ctrl.spines['right'].set_visible(False)
        ax_ctrl.grid(True, alpha=0.3, linewidth=0.5)
        ax_ctrl.text(-0.08, 1.05, 'A', transform=ax_ctrl.transAxes, fontsize=12, fontweight='bold')
        
        for i in range(n_subjects):
            ax_trt.plot(freq_display, self.coherence_treatment[i][display_mask], color=colors[i % len(colors)],
                       alpha=0.8, linewidth=1.5, label=self.subject_ids[i] if i < 10 else None)
        
        trt_mean = np.mean([self.coherence_treatment[i][display_mask] for i in range(n_subjects)], axis=0)
        ax_trt.plot(freq_display, trt_mean, 'k--', linewidth=2.5, label='Mean', zorder=10)
        ax_trt.axvspan(self.frequency_roi[0], self.frequency_roi[1], alpha=0.08, color='gray')
        ax_trt.set_xlabel('Frequency (Hz)', fontsize=10)
        ax_trt.set_ylabel('Coherence', fontsize=10)
        ax_trt.set_title(f'{condition_name} — Treatment', fontsize=11, fontweight='bold')
        ax_trt.set_xlim([freq_display[0], freq_display[-1]])
        ax_trt.set_ylim([0, 1])
        ax_trt.legend(loc='upper right', fontsize=7, ncol=2)
        ax_trt.tick_params(axis='both', which='major', labelsize=9)
        ax_trt.spines['top'].set_visible(False)
        ax_trt.spines['right'].set_visible(False)
        ax_trt.grid(True, alpha=0.3, linewidth=0.5)
        ax_trt.text(-0.08, 1.05, 'B', transform=ax_trt.transAxes, fontsize=12, fontweight='bold')
        
        ax_ctrl.text(0.02, 0.98, f'n = {n_subjects}', transform=ax_ctrl.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            for fmt, ext in [('svg', '.svg'), ('tiff', '.tiff')]:
                out_path = save_path.replace('.svg', ext)
                if fmt == 'tiff':
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
                else:
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300)
                print(f"  Saved spectra overlay ({fmt.upper()}) to: {out_path}")
        
        plt.show()
    
    def plot_roi_comparison(self, results: Dict, save_path: Optional[str] = None, condition_name: str = "Condition"):
        """Plot ROI comparison violin plot."""
        fig, ax = plt.subplots(figsize=(3.35, 3.0))
        
        control_vals = results['control_values']
        treatment_vals = results['treatment_values']
        n_subjects = len(control_vals)
        
        parts = ax.violinplot([control_vals, treatment_vals], positions=[1, 2], showmeans=False, showmedians=True, widths=0.6)
        
        for pc in parts['bodies']:
            pc.set_facecolor('#CCCCCC')
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        
        for i in range(n_subjects):
            ax.plot([1, 2], [control_vals[i], treatment_vals[i]], '-', color='#666666', alpha=0.5, linewidth=0.75)
            ax.plot(1, control_vals[i], 'o', color='white', markeredgecolor='black', markeredgewidth=0.75, markersize=5)
            ax.plot(2, treatment_vals[i], 's', color='white', markeredgecolor='black', markeredgewidth=0.75, markersize=5)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Control', 'Treatment'], fontsize=8)
        ax.set_ylabel(f'Mean Coherence\n({self.frequency_roi[0]}-{self.frequency_roi[1]} Hz)', fontsize=8)
        ax.set_title(condition_name.capitalize(), fontsize=9, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, linestyle=':')
        ax.set_axisbelow(True)
        
        stats_text = f"p = {results['p_value']:.4f}\nd = {results['cohens_d']:.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=6, va='top',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            for fmt, ext in [('svg', '.svg'), ('tiff', '.tiff')]:
                out_path = save_path.replace('.svg', ext)
                if fmt == 'tiff':
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
                else:
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300)
                print(f"  Saved ROI comparison ({fmt.upper()}) to: {out_path}")
        
        plt.show()
    
    def plot_combined_figure(self, bootstrap_results: Dict, roi_results: Dict, save_path: Optional[str] = None, condition_name: str = "Condition"):
        """Plot combined figure with all panels."""
        freq = bootstrap_results['frequencies']
        display_mask = (freq >= self.frequency_display_range[0]) & (freq <= self.frequency_display_range[1])
        freq_display = freq[display_mask]
        
        control_med = bootstrap_results['control_median'][display_mask]
        control_ci_lo = bootstrap_results['control_ci_lower'][display_mask]
        control_ci_hi = bootstrap_results['control_ci_upper'][display_mask]
        treatment_med = bootstrap_results['treatment_median'][display_mask]
        treatment_ci_lo = bootstrap_results['treatment_ci_lower'][display_mask]
        treatment_ci_hi = bootstrap_results['treatment_ci_upper'][display_mask]
        diff_med = bootstrap_results['diff_median'][display_mask]
        diff_ci_lo = bootstrap_results['diff_ci_lower'][display_mask]
        diff_ci_hi = bootstrap_results['diff_ci_upper'][display_mask]
        sig_fdr = bootstrap_results['sig_mask_fdr'][display_mask]
        
        fig = plt.figure(figsize=(7.5, 7.0))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2], width_ratios=[2.5, 1], hspace=0.35, wspace=0.3)
        
        ax_spectra = fig.add_subplot(gs[0, :])
        ax_diff = fig.add_subplot(gs[1, :])
        ax_roi = fig.add_subplot(gs[2, 0])
        ax_legend = fig.add_subplot(gs[2, 1])
        ax_legend.axis('off')
        
        ax_spectra.plot(freq_display, control_med, 'b-', linewidth=2, label='Control')
        ax_spectra.fill_between(freq_display, control_ci_lo, control_ci_hi, color='blue', alpha=0.2)
        ax_spectra.plot(freq_display, treatment_med, 'r-', linewidth=2, label='Treatment')
        ax_spectra.fill_between(freq_display, treatment_ci_lo, treatment_ci_hi, color='red', alpha=0.2)
        ax_spectra.axvspan(self.frequency_roi[0], self.frequency_roi[1], alpha=0.1, color='gray', label='ROI')
        ax_spectra.set_xlabel('Frequency (Hz)', fontsize=9)
        ax_spectra.set_ylabel('Coherence', fontsize=9)
        ax_spectra.set_title(f'{condition_name} — Population Coherence', fontsize=10, fontweight='bold')
        ax_spectra.set_xlim([freq_display[0], freq_display[-1]])
        ax_spectra.set_ylim([0, min(1.0, max(control_ci_hi.max(), treatment_ci_hi.max()) * 1.1)])
        ax_spectra.legend(loc='upper right', fontsize=7)
        ax_spectra.tick_params(axis='both', which='major', labelsize=8)
        ax_spectra.spines['top'].set_visible(False)
        ax_spectra.spines['right'].set_visible(False)
        ax_spectra.grid(True, alpha=0.3, linewidth=0.5)
        ax_spectra.text(-0.08, 1.05, 'A', transform=ax_spectra.transAxes, fontsize=12, fontweight='bold')
        
        ax_diff.plot(freq_display, diff_med, 'k-', linewidth=2, label='Δ Coherence')
        ax_diff.fill_between(freq_display, diff_ci_lo, diff_ci_hi, color='gray', alpha=0.3, label='95% CI')
        ax_diff.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        if np.any(sig_fdr):
            sig_line = np.where(sig_fdr, diff_med, np.nan)
            ax_diff.plot(freq_display, sig_line, 'r-', linewidth=2.5, label='Significant (FDR)', zorder=5)
        ax_diff.axvspan(self.frequency_roi[0], self.frequency_roi[1], alpha=0.1, color='gray')
        ax_diff.set_xlabel('Frequency (Hz)', fontsize=9)
        ax_diff.set_ylabel('Δ Coherence', fontsize=9)
        ax_diff.set_title('Coherence Difference (Treatment − Control)', fontsize=10, fontweight='bold')
        ax_diff.set_xlim([freq_display[0], freq_display[-1]])
        ax_diff.legend(loc='upper right', fontsize=7)
        ax_diff.tick_params(axis='both', which='major', labelsize=8)
        ax_diff.spines['top'].set_visible(False)
        ax_diff.spines['right'].set_visible(False)
        ax_diff.grid(True, alpha=0.3, linewidth=0.5)
        ax_diff.text(-0.08, 1.05, 'B', transform=ax_diff.transAxes, fontsize=12, fontweight='bold')
        
        control_vals = roi_results['control_values']
        treatment_vals = roi_results['treatment_values']
        n_subjects = len(control_vals)
        
        parts = ax_roi.violinplot([control_vals, treatment_vals], positions=[1, 2], showmeans=False, showmedians=True, widths=0.6)
        for pc in parts['bodies']:
            pc.set_facecolor('#CCCCCC')
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        
        for i in range(n_subjects):
            ax_roi.plot([1, 2], [control_vals[i], treatment_vals[i]], '-', color='#666666', alpha=0.5, linewidth=0.75)
            ax_roi.plot(1, control_vals[i], 'o', color='white', markeredgecolor='black', markeredgewidth=0.75, markersize=4)
            ax_roi.plot(2, treatment_vals[i], 's', color='white', markeredgecolor='black', markeredgewidth=0.75, markersize=4)
        
        ax_roi.set_xticks([1, 2])
        ax_roi.set_xticklabels(['Control', 'Treatment'], fontsize=8)
        ax_roi.set_ylabel(f'Mean Coherence\n({self.frequency_roi[0]}-{self.frequency_roi[1]} Hz)', fontsize=8)
        ax_roi.set_title('ROI Comparison', fontsize=10, fontweight='bold')
        ax_roi.tick_params(axis='both', which='major', labelsize=7)
        ax_roi.spines['top'].set_visible(False)
        ax_roi.spines['right'].set_visible(False)
        ax_roi.yaxis.grid(True, alpha=0.3, linewidth=0.5, linestyle=':')
        ax_roi.set_axisbelow(True)
        ax_roi.text(-0.15, 1.05, 'C', transform=ax_roi.transAxes, fontsize=12, fontweight='bold')
        
        n_sig = np.sum(sig_fdr)
        stats_text = (f"Statistical Summary\n{'─'*25}\nn = {n_subjects} subjects\n"
                     f"ROI: {self.frequency_roi[0]}-{self.frequency_roi[1]} Hz\n\n"
                     f"Wilcoxon signed-rank:\n  W = {roi_results['wilcoxon_statistic']:.1f}\n"
                     f"  p = {roi_results['p_value']:.4f}\n  Cohen's d = {roi_results['cohens_d']:.2f}\n\n"
                     f"FDR-significant freq: {n_sig}")
        ax_legend.text(0.1, 0.9, stats_text, transform=ax_legend.transAxes, fontsize=8, va='top', family='monospace',
                      bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        
        if save_path:
            for fmt, ext in [('svg', '.svg'), ('tiff', '.tiff'), ('eps', '.eps')]:
                out_path = save_path.replace('.svg', ext)
                if fmt == 'tiff':
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
                else:
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300)
                print(f"  Saved combined figure ({fmt.upper()}) to: {out_path}")
        
        plt.show()
    
    def plot_individual_subject(self, subject_idx: int, save_path: Optional[str] = None):
        """Plot coherence for a single subject (NO CIs, NO stats on plot)."""
        if subject_idx >= len(self.subject_ids):
            raise ValueError(f"Subject index {subject_idx} out of range")
        
        subject_id = self.subject_ids[subject_idx]
        control_coh = self.coherence_control[subject_idx]
        treatment_coh = self.coherence_treatment[subject_idx]
        diff_coh = self.coherence_diff[subject_idx]
        
        display_mask = (self.frequencies >= self.frequency_display_range[0]) & (self.frequencies <= self.frequency_display_range[1])
        freq_display = self.frequencies[display_mask]
        
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.5), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35})
        ax1, ax2 = axes
        
        ax1.plot(freq_display, control_coh[display_mask], 'b-', linewidth=2, label='Control')
        ax1.plot(freq_display, treatment_coh[display_mask], 'r-', linewidth=2, label='Treatment')
        ax1.axvspan(self.frequency_roi[0], self.frequency_roi[1], alpha=0.1, color='gray', label='ROI')
        ax1.set_xlabel('Frequency (Hz)', fontsize=10)
        ax1.set_ylabel('Coherence', fontsize=10)
        ax1.set_title(f'{subject_id} — Coherence Spectra', fontsize=11, fontweight='bold')
        ax1.set_xlim([freq_display[0], freq_display[-1]])
        ax1.set_ylim([0, min(1.0, max(control_coh[display_mask].max(), treatment_coh[display_mask].max()) * 1.15)])
        ax1.legend(loc='upper right', fontsize=8)
        ax1.tick_params(axis='both', which='major', labelsize=9)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.text(-0.08, 1.05, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold')
        
        ax2.plot(freq_display, diff_coh[display_mask], 'k-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        ax2.axvspan(self.frequency_roi[0], self.frequency_roi[1], alpha=0.1, color='gray')
        ax2.set_xlabel('Frequency (Hz)', fontsize=10)
        ax2.set_ylabel('Δ Coherence\n(Treatment − Control)', fontsize=10)
        ax2.set_title('Coherence Difference', fontsize=11, fontweight='bold')
        ax2.set_xlim([freq_display[0], freq_display[-1]])
        ax2.tick_params(axis='both', which='major', labelsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.text(-0.08, 1.05, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            for fmt, ext in [('svg', '.svg'), ('tiff', '.tiff')]:
                out_path = save_path.replace('.svg', ext)
                if fmt == 'tiff':
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
                else:
                    plt.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300)
            print(f"    Saved {subject_id} plot")
        
        plt.close()
    
    def plot_all_individual_subjects(self, output_dir: str, channel_name: str = ""):
        """Generate individual plots for all subjects."""
        individual_dir = os.path.join(output_dir, "individual_subjects")
        os.makedirs(individual_dir, exist_ok=True)
        print(f"\n  Generating individual subject plots (no CIs)...")
        
        for i, subject_id in enumerate(self.subject_ids):
            filename = f"{subject_id}_coherence_{channel_name}.svg" if channel_name else f"{subject_id}_coherence.svg"
            self.plot_individual_subject(i, save_path=os.path.join(individual_dir, filename))
        
        print(f"    Generated {len(self.subject_ids)} individual plots")
    
    def save_individual_subject_data(self, output_dir: str, channel_name: str = ""):
        """Save individual subject data to CSV (no CIs)."""
        individual_dir = os.path.join(output_dir, "individual_subjects")
        os.makedirs(individual_dir, exist_ok=True)
        print(f"\n  Saving individual subject data...")
        
        for i, subject_id in enumerate(self.subject_ids):
            df = pd.DataFrame({
                'Frequency_Hz': self.frequencies,
                'Control_Coherence': self.coherence_control[i],
                'Treatment_Coherence': self.coherence_treatment[i],
                'Difference': self.coherence_diff[i],
            })
            filename = f"{subject_id}_coherence_{channel_name}.csv" if channel_name else f"{subject_id}_coherence.csv"
            df.to_csv(os.path.join(individual_dir, filename), index=False)
        
        print(f"    Saved {len(self.subject_ids)} individual CSV files")
    
    def save_descriptive_statistics_csv(self, output_dir: str, channel_name: str = ""):
        """Save descriptive statistics to CSV."""
        df = pd.DataFrame(self.subject_stats)
        filename = f"descriptive_statistics_{channel_name}.csv" if channel_name else "descriptive_statistics.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"  Saved descriptive statistics to: {filepath}")
        return df
    
    def save_results_to_csv(self, roi_results: Dict, bootstrap_results: Dict, filepath: str):
        """Save statistical results to CSV."""
        roi_data = {
            'Subject_ID': roi_results['subject_ids'],
            'Control_Coherence': roi_results['control_values'],
            'Treatment_Coherence': roi_results['treatment_values'],
            'Difference': roi_results['treatment_values'] - roi_results['control_values']
        }
        pd.DataFrame(roi_data).to_csv(filepath.replace('.csv', '_roi.csv'), index=False)
        print(f"  Saved ROI results")
        
        freq_data = {
            'Frequency_Hz': bootstrap_results['frequencies'],
            'Control_Median': bootstrap_results['control_median'],
            'Control_CI_Lower': bootstrap_results['control_ci_lower'],
            'Control_CI_Upper': bootstrap_results['control_ci_upper'],
            'Treatment_Median': bootstrap_results['treatment_median'],
            'Treatment_CI_Lower': bootstrap_results['treatment_ci_lower'],
            'Treatment_CI_Upper': bootstrap_results['treatment_ci_upper'],
            'Diff_Median': bootstrap_results['diff_median'],
            'Diff_CI_Lower': bootstrap_results['diff_ci_lower'],
            'Diff_CI_Upper': bootstrap_results['diff_ci_upper'],
            'P_Value': bootstrap_results['p_values'],
            'Significant_FDR': bootstrap_results['sig_mask_fdr']
        }
        pd.DataFrame(freq_data).to_csv(filepath.replace('.csv', '_frequency.csv'), index=False)
        print(f"  Saved frequency results")


# MAIN ANALYSIS WORKFLOW

def analyze_drug_condition(drug_name: str, line_numbers: List[int], output_base_dir: str) -> Optional[Dict]:
    """Analyze all subjects for a given drug condition."""
    print(f"\n{'-'*80}")
    print(f"Analyzing: {drug_name}")
    print(f"Subjects: {line_numbers}")
    print(f"{'-'*80}\n")
    
    analyzer = PopulationCoherenceAnalyzer(frequency_roi=FREQUENCY_ROI, frequency_display_range=FREQUENCY_DISPLAY_RANGE)
    
    for line_num in line_numbers:
        print(f"\n--- Processing Line {line_num} ---")
        try:
            calcium_filepath = os.path.join(BASE_DATA_PATH, f"meanFluorescence_{line_num}.npz")
            channel_object, miniscope_data_manager, fr = load_experiment(line_num, calcium_filepath, CHANNEL_NAME)
            
            eeg_signal = channel_object.signal
            calcium_signal = miniscope_data_manager.mean_fluorescence_dict['meanFluorescence']
            fr = float(channel_object.sampling_rate)
            
            control_eeg, treatment_eeg = slice_signal(eeg_signal, line_num, fr)
            control_calcium, treatment_calcium = slice_signal(calcium_signal, line_num, fr)
            
            print("    Filtering...")
            control_eeg, control_calcium = filter_signals(control_eeg, control_calcium, fr, FILTER_RANGE)
            treatment_eeg, treatment_calcium = filter_signals(treatment_eeg, treatment_calcium, fr, FILTER_RANGE)
            
            analyzer.add_subject(control_eeg, control_calcium, treatment_eeg, treatment_calcium, fr, f"Line_{line_num}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    if len(analyzer.subject_ids) == 0:
        print(f"\nNo valid subjects for {drug_name}")
        return None
    
    output_dir = os.path.join(output_base_dir, drug_name.replace(":", "_").replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "-"*80)
    print("BOOTSTRAP ANALYSIS")
    print("-"*80)
    bootstrap_results = analyzer.compute_population_coherence_bootstrap()
    
    print("\n" + "-"*80)
    print("ROI STATISTICAL TEST")
    print("-"*80)
    roi_results = analyzer.perform_roi_statistical_test()
    
    print("\n" + "-"*80)
    print("DESCRIPTIVE STATISTICS REPORT")
    print("-"*80)
    analyzer.print_descriptive_statistics_report(drug_name, bootstrap_results, roi_results,
                                                  save_path=os.path.join(output_dir, f"statistics_report_{CHANNEL_NAME}.txt"))
    
    analyzer.plot_coherence_spectra(bootstrap_results, roi_results, os.path.join(output_dir, f"coherence_spectra_{CHANNEL_NAME}.svg"), drug_name)
    analyzer.plot_individual_subject_overlay(bootstrap_results, os.path.join(output_dir, f"individual_overlay_{CHANNEL_NAME}.svg"), drug_name)
    analyzer.plot_individual_spectra_overlay(os.path.join(output_dir, f"individual_spectra_overlay_{CHANNEL_NAME}.svg"), drug_name)
    analyzer.plot_combined_figure(bootstrap_results, roi_results, os.path.join(output_dir, f"combined_analysis_{CHANNEL_NAME}.svg"), drug_name)
    analyzer.plot_roi_comparison(roi_results, os.path.join(output_dir, f"roi_comparison_{CHANNEL_NAME}.svg"), drug_name)
    
    analyzer.save_results_to_csv(roi_results, bootstrap_results, os.path.join(output_dir, f"results_{CHANNEL_NAME}.csv"))
    analyzer.save_descriptive_statistics_csv(output_dir, CHANNEL_NAME)
    
    print("\n" + "-"*80)
    print("INDIVIDUAL SUBJECT OUTPUTS")
    print("-"*80)
    analyzer.plot_all_individual_subjects(output_dir, CHANNEL_NAME)
    analyzer.save_individual_subject_data(output_dir, CHANNEL_NAME)
    
    print(f"\n{'='*80}")
    print(f"Complete! Results: {output_dir}")
    print(f"{'='*80}\n")
    
    return {'bootstrap_results': bootstrap_results, 'roi_results': roi_results, 'analyzer': analyzer}


def main():
    """Main execution function."""
    print("\n" + "-"*80)
    print("POPULATION COHERENCE ANALYSIS")
    print("Bootstrap CIs at Population Level Only")
    print("-"*80)
    print(f"Channel: {CHANNEL_NAME}")
    print(f"ROI: {FREQUENCY_ROI} Hz")
    print(f"Multitaper: NW={MULTITAPER_NW}, K={MULTITAPER_K}")
    print("-"*80)
    
    os.makedirs(RESULTS_PATH, exist_ok=True)
    all_results = {}
    
    for drug_name, line_numbers in data.items():
        results = analyze_drug_condition(drug_name, line_numbers, RESULTS_PATH)
        if results:
            all_results[drug_name] = results
    
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    for drug_name, results in all_results.items():
        roi = results['roi_results']
        n_sig = np.sum(results['bootstrap_results']['sig_mask_fdr'])
        print(f"\n{drug_name}:")
        print(f"  n={roi['n_subjects']}, Control={roi['control_mean']:.4f}+/-{roi['control_std']:.4f}, "
              f"Treatment={roi['treatment_mean']:.4f}+/-{roi['treatment_std']:.4f}, p={roi['p_value']:.4f}, d={roi['cohens_d']:.3f}, FDR-sig={n_sig}")
    
    print("\n" + "-"*80)
    print(f"Results: {RESULTS_PATH}")
    print("-"*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = main()