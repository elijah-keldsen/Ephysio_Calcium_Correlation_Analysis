"""
Spatial Specificity of Calcium-Electrophysiology Coupling Analysis

This script analyzes the relationship between calcium imaging signals recorded from
the prefrontal cortex (PrL) and electrophysiological signals recorded at varying
distances from the calcium imaging site.

Expected gradient:
- PFCLFPvsCBEEG: Highest coherence (LFP closest to calcium source)
- PFCEEGvsCBEEG: Medium coherence (surface EEG)
- CBvsPCEEG: Lowest coherence (furthest from PFC calcium)
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xrscipy.signal.spectral import coherogram
from scipy.signal import coherence
from scipy.stats import f_oneway, wilcoxon, ttest_rel
import pandas as pd
import os
import warnings
from datetime import datetime

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


# Channels to analyze (ordered by expected proximity to calcium recording site)
CHANNELS = ['PFCLFPvsCBEEG', 'PFCEEGvsCBEEG', 'CBvsPCEEG']

CHANNEL_DESCRIPTIONS = {
    'PFCLFPvsCBEEG': 'PFC LFP vs CB EEG (Closest)',
    'PFCEEGvsCBEEG': 'PFC EEG vs CB EEG (Medium)',
    'CBvsPCEEG': 'CB vs Parietal EEG (Furthest)'
}

CHANNEL_SHORT_NAMES = {
    'PFCLFPvsCBEEG': 'PFC-LFP',
    'PFCEEGvsCBEEG': 'PFC-EEG',
    'CBvsPCEEG': 'CB-PC'
}

# Frequency range for coherence analysis (Hz)
FREQ_RANGE = [0.5, 4]

# Drug groups with their subject line numbers
DRUG_GROUPS = {
    "dexmedetomidine_0.00045": [46, 47, 64, 88, 97, 101],
    "dexmedetomidine_0.0003": [40, 41, 48, 87, 93, 94],
    "propofol": [36, 43, 44, 86, 99, 103],
    "ketamine": [39, 42, 45, 85, 96, 112]
}

# Display names for drugs
DRUG_DISPLAY_NAMES = {
    "dexmedetomidine_0.00045": "Dexmedetomidine (0.00045 mg/kg/min)",
    "dexmedetomidine_0.0003": "Dexmedetomidine (0.0003 mg/kg/min)",
    "propofol": "Propofol",
    "ketamine": "Ketamine"
}

SELECTIONS = { 
    # dexmedetomidine_0.00045
    46:  [[1,20], [28.24, 75]], 
    47:  [[1,20], [28.24, 75]],  
    64:  [[4,17], [28.24, 75]], 
    88:  [[1,8], [28.24, 83.24]],
    97:  [[1,13], [28.24, 83.24]], 
    101: [[1,25], [37, 90]],
    
    # dexmedetomidine_0.0003
    40:  [[8,20], [55, 75]], 
    41:  [[10,19], [60, 68]],
    48:  [[1,20], [25, 35]],
    87:  [[5,13], [73, 85]],
    93:  [[18,28], [75, 95]],
    94:  [[1,20], [75, 90]], 
    
    # propofol
    36:  [[1,11], [55, 67]],
    43:  [[15, 20], [40, 60]], 
    44:  [[0,21], [40, 65]],
    86:  [[5,16], [33, 65]],
    99:  [[0,19], [33, 45]],
    103: [[1,17], [38, 65]], 
    
    # ketamine
    39:  [[10,20], [38, 50]], 
    42:  [[1,20], [40, 51]], 
    45:  [[1,15], [40, 60]], 
    85:  [[14,24], [30, 50]], 
    96:  [[1,12], [38, 55]],
    112: [[1,10], [40, 60]]
}

# Create reverse mapping from line number to drug
LINE_TO_DRUG = {num: drug for drug, numbers in DRUG_GROUPS.items() for num in numbers}

# Output directory
OUTPUT_DIR = r"C:\Users\Path\To\Output\Directory"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def load_experiment(line_num, channel_name, calcium_signal_filepath=None):
    """
    Load and synchronize ephys and calcium data for a given subject and channel.
    
    Parameters:
    line_num : int
        Subject line number
    channel_name : str
        Name of the ephys channel to load
    calcium_signal_filepath : str, optional
        Path to pre-computed calcium signal file
        
    Returns:
    channel_object : Channel
        Synchronized ephys channel object
    calcium_signal : np.ndarray
        Calcium fluorescence signal
    fr : float
        Sampling rate (frame rate)
    """
    print(f'  Loading line {line_num}, channel {channel_name}...')
    
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
        fr = 30
    
    # Load ephys data
    ephys_api = EphysAPI()
    ephys_api.run(
        line_num, 
        channel_name=channel_name, 
        remove_artifacts=False, 
        filter_type=None, 
        filter_range=FREQ_RANGE, 
        plot_channel=False, 
        plot_spectrogram=False, 
        plot_phases=False, 
        logging_level="CRITICAL"
    )
    channel_object = ephys_api.ephys_data_manager.get_channel(channel_name=channel_name)
    
    # Sync timestamps
    tCaIm, low_confidence_periods, channel_object, miniscope_data_manager = sync_neuralynx_miniscope_timestamps(
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
    
    # Load calcium signal
    if calcium_signal_filepath:
        miniscope_data_manager.mean_fluorescence_dict = np.load(calcium_signal_filepath)
    
    calcium_signal = miniscope_data_manager.mean_fluorescence_dict['meanFluorescence']
    
    if np.any(np.isnan(channel_object.signal)):
        print(f"    Warning: Replacing NaNs in EEG with zeros for line {line_num}")
        channel_object.signal = np.nan_to_num(channel_object.signal, nan=0.0)
    
    return channel_object, calcium_signal, fr


def slice_signal(signal, line_num, fr):
    """
    Slice signal into control and treatment segments.
    
    Parameters:
    signal : np.ndarray
        Signal to slice
    line_num : int
        Subject line number
    fr : float
        Sampling rate
        
    Returns:
    control_segment : np.ndarray
        Control period signal
    treatment_segment : np.ndarray
        Treatment period signal
    """
    control_start_idx = int(SELECTIONS[line_num][0][0] * fr * 60)
    control_end_idx = int(SELECTIONS[line_num][0][1] * fr * 60)
    treatment_start_idx = int(SELECTIONS[line_num][1][0] * fr * 60)
    treatment_end_idx = int(SELECTIONS[line_num][1][1] * fr * 60)
    
    control_segment = signal[control_start_idx:control_end_idx]
    treatment_segment = signal[treatment_start_idx:treatment_end_idx]
    
    return control_segment, treatment_segment


def filter_signals(eeg_signal, calcium_signal, fr, freq_range=FREQ_RANGE):
    """
    Apply bandpass filtering to EEG and calcium signals.
    
    Parameters:
    eeg_signal : np.ndarray
        EEG/LFP signal
    calcium_signal : np.ndarray
        Calcium fluorescence signal
    fr : float
        Sampling rate
    freq_range : list
        [low_freq, high_freq] for bandpass filter
        
    Returns:
    filtered_eeg : np.ndarray
        Filtered EEG signal
    filtered_calcium : np.ndarray
        Filtered calcium signal
    """
    filtered_calcium = misc_functions.filterData(
        calcium_signal, 
        n=2, 
        cut=freq_range, 
        ftype='butter', 
        btype='bandpass', 
        fs=fr
    )
    filtered_eeg = EphysDataManager._filter_data(
        data=eeg_signal, 
        n=2, 
        cut=freq_range, 
        ftype='butter', 
        fs=fr, 
        btype='bandpass', 
        bodePlot=False
    )
    
    return filtered_eeg, filtered_calcium


def compute_coherence(signal1, signal2, fr, freq_range=FREQ_RANGE):
    """
    Compute mean coherence between two signals in the specified frequency range.
    
    Parameters:
    signal1 : np.ndarray
        First signal (EEG/LFP)
    signal2 : np.ndarray
        Second signal (calcium)
    fr : float
        Sampling rate
    freq_range : list
        [low_freq, high_freq] for analysis
        
    Returns:
    mean_coherence : float
        Mean coherence value in the frequency range
    """
    f, Cxy = coherence(signal1, signal2, fs=fr)
    freq_indices = np.where((f >= freq_range[0]) & (f <= freq_range[1]))[0]
    Cxy_band = Cxy[freq_indices]
    mean_coherence = np.mean(Cxy_band)
    
    return mean_coherence


# MAIN ANALYSIS FUNCTIONS

def run_full_analysis():
    """
    Run the complete spatial specificity analysis.
    
    Returns:
    results_df : pd.DataFrame
        DataFrame containing all coherence results
    """
    ensure_output_dir()
    
    results = []
    
    print("-" * 70)
    print("SPATIAL SPECIFICITY OF CALCIUM-ELECTROPHYSIOLOGY COUPLING ANALYSIS")
    print("-" * 70)
    print(f"Frequency range: {FREQ_RANGE[0]} - {FREQ_RANGE[1]} Hz")
    print(f"Channels: {', '.join(CHANNELS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 70)
    
    # Loop through each drug group
    for drug_name, line_nums in DRUG_GROUPS.items():
        print(f"\n{'='*50}")
        print(f"Processing drug: {DRUG_DISPLAY_NAMES[drug_name]}")
        print(f"{'='*50}")
        
        # Loop through each subject
        for line_num in line_nums:
            print(f"\n--- Subject {line_num} ---")
            
            calcium_signal_filepath = f"C:/Users/ericm/Desktop/meanFluorescence/meanFluorescence_{str(line_num)}.npz"
            
            # Loop through each channel
            for channel in CHANNELS:
                try:
                    # Load data
                    channel_object, calcium_signal, fr = load_experiment(
                        line_num, 
                        channel, 
                        calcium_signal_filepath
                    )
                    
                    eeg_signal = channel_object.signal
                    
                    # Slice into control and treatment
                    control_eeg, treatment_eeg = slice_signal(eeg_signal, line_num, fr)
                    control_calcium, treatment_calcium = slice_signal(calcium_signal, line_num, fr)
                    
                    # Filter signals
                    filtered_control_eeg, filtered_control_calcium = filter_signals(
                        control_eeg, control_calcium, fr
                    )
                    filtered_treatment_eeg, filtered_treatment_calcium = filter_signals(
                        treatment_eeg, treatment_calcium, fr
                    )
                    
                    # Compute coherence for control and treatment
                    control_coherence = compute_coherence(
                        filtered_control_eeg, filtered_control_calcium, fr
                    )
                    treatment_coherence = compute_coherence(
                        filtered_treatment_eeg, filtered_treatment_calcium, fr
                    )
                    
                    # Store results
                    results.append({
                        'Drug': drug_name,
                        'Drug_Display': DRUG_DISPLAY_NAMES[drug_name],
                        'Subject': line_num,
                        'Channel': channel,
                        'Channel_Short': CHANNEL_SHORT_NAMES[channel],
                        'Channel_Description': CHANNEL_DESCRIPTIONS[channel],
                        'Control_Coherence': control_coherence,
                        'Treatment_Coherence': treatment_coherence,
                        'Coherence_Change': treatment_coherence - control_coherence,
                        'Coherence_Ratio': treatment_coherence / control_coherence if control_coherence != 0 else np.nan
                    })
                    
                    print(f"    {CHANNEL_SHORT_NAMES[channel]}: Control={control_coherence:.4f}, Treatment={treatment_coherence:.4f}")
                    
                except Exception as e:
                    print(f"    ERROR processing {channel}: {str(e)}")
                    results.append({
                        'Drug': drug_name,
                        'Drug_Display': DRUG_DISPLAY_NAMES[drug_name],
                        'Subject': line_num,
                        'Channel': channel,
                        'Channel_Short': CHANNEL_SHORT_NAMES[channel],
                        'Channel_Description': CHANNEL_DESCRIPTIONS[channel],
                        'Control_Coherence': np.nan,
                        'Treatment_Coherence': np.nan,
                        'Coherence_Change': np.nan,
                        'Coherence_Ratio': np.nan
                    })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def compute_statistics(results_df):
    """
    Compute statistical tests on the results.
    
    Parameters:
    results_df : pd.DataFrame
        DataFrame with coherence results
        
    Returns:
    stats_results : dict
        Dictionary containing all statistical results
    """
    stats_results = {
        'channel_comparison': {},
        'control_vs_treatment': {},
        'descriptive_stats': {}
    }
    
    print("\n" + "-" * 70)
    print("STATISTICAL ANALYSIS")
    print("-" * 70)
    
    # For each drug, compare coherence across channels (ANOVA)
    for drug in DRUG_GROUPS.keys():
        drug_data = results_df[results_df['Drug'] == drug]
        
        print(f"\n--- {DRUG_DISPLAY_NAMES[drug]} ---")
        
        # Descriptive statistics
        desc_stats = drug_data.groupby('Channel').agg({
            'Control_Coherence': ['mean', 'std', 'count'],
            'Treatment_Coherence': ['mean', 'std', 'count']
        }).round(4)
        
        stats_results['descriptive_stats'][drug] = desc_stats
        print(f"\nDescriptive Statistics:")
        print(desc_stats.to_string())
        
        # ANOVA for channel comparison - Control epoch
        control_by_channel = [
            drug_data[drug_data['Channel'] == ch]['Control_Coherence'].dropna().values
            for ch in CHANNELS
        ]
        
        # Check if we have enough data
        if all(len(x) >= 2 for x in control_by_channel):
            f_stat_control, p_val_control = f_oneway(*control_by_channel)
            stats_results['channel_comparison'][f'{drug}_control'] = {
                'F_statistic': f_stat_control,
                'p_value': p_val_control
            }
            print(f"\nANOVA (Control, across channels): F={f_stat_control:.4f}, p={p_val_control:.4f}")
        else:
            print(f"\nANOVA (Control): Insufficient data")
            stats_results['channel_comparison'][f'{drug}_control'] = {'F_statistic': np.nan, 'p_value': np.nan}
        
        # ANOVA for channel comparison - Treatment epoch
        treatment_by_channel = [
            drug_data[drug_data['Channel'] == ch]['Treatment_Coherence'].dropna().values
            for ch in CHANNELS
        ]
        
        if all(len(x) >= 2 for x in treatment_by_channel):
            f_stat_treatment, p_val_treatment = f_oneway(*treatment_by_channel)
            stats_results['channel_comparison'][f'{drug}_treatment'] = {
                'F_statistic': f_stat_treatment,
                'p_value': p_val_treatment
            }
            print(f"ANOVA (Treatment, across channels): F={f_stat_treatment:.4f}, p={p_val_treatment:.4f}")
        else:
            print(f"ANOVA (Treatment): Insufficient data")
            stats_results['channel_comparison'][f'{drug}_treatment'] = {'F_statistic': np.nan, 'p_value': np.nan}
        
        # Within-channel: Control vs Treatment (paired t-test or Wilcoxon)
        stats_results['control_vs_treatment'][drug] = {}
        
        for channel in CHANNELS:
            channel_data = drug_data[drug_data['Channel'] == channel]
            control_vals = channel_data['Control_Coherence'].dropna().values
            treatment_vals = channel_data['Treatment_Coherence'].dropna().values
            
            # Ensure paired data
            min_len = min(len(control_vals), len(treatment_vals))
            if min_len >= 3:
                control_vals = control_vals[:min_len]
                treatment_vals = treatment_vals[:min_len]
                
                # Paired t-test
                t_stat, p_val_t = ttest_rel(control_vals, treatment_vals)
                
                # Wilcoxon signed-rank test (non-parametric)
                try:
                    w_stat, p_val_w = wilcoxon(control_vals, treatment_vals)
                except ValueError:
                    w_stat, p_val_w = np.nan, np.nan
                
                stats_results['control_vs_treatment'][drug][channel] = {
                    't_statistic': t_stat,
                    'p_value_ttest': p_val_t,
                    'w_statistic': w_stat,
                    'p_value_wilcoxon': p_val_w
                }
                
                print(f"\n{CHANNEL_SHORT_NAMES[channel]} Control vs Treatment:")
                print(f"  Paired t-test: t={t_stat:.4f}, p={p_val_t:.4f}")
                print(f"  Wilcoxon: W={w_stat:.4f}, p={p_val_w:.4f}" if not np.isnan(w_stat) else "  Wilcoxon: N/A")
            else:
                stats_results['control_vs_treatment'][drug][channel] = {
                    't_statistic': np.nan,
                    'p_value_ttest': np.nan,
                    'w_statistic': np.nan,
                    'p_value_wilcoxon': np.nan
                }
                print(f"\n{CHANNEL_SHORT_NAMES[channel]} Control vs Treatment: Insufficient data")
    
    return stats_results


def create_violin_plots(results_df):
    """
    Create violin plots showing coherence across channels for each drug.
    
    Parameters:
    results_df : pd.DataFrame
        DataFrame with coherence results
    """
    print("\n" + "-" * 70)
    print("CREATING VIOLIN PLOTS")
    print("-" * 70)
    
    # Color scheme for channels
    channel_colors = {
        'PFCLFPvsCBEEG': '#2ecc71',  # Green (closest)
        'PFCEEGvsCBEEG': '#3498db',  # Blue (medium)
        'CBvsPCEEG': '#e74c3c'        # Red (furthest)
    }
    
    for drug in DRUG_GROUPS.keys():
        drug_data = results_df[results_df['Drug'] == drug]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{DRUG_DISPLAY_NAMES[drug]}\nSpatial Specificity of Calcium-Electrophysiology Coupling', 
                     fontsize=14, fontweight='bold')
        
        # Plot for Control epoch
        ax1 = axes[0]
        positions = [1, 2, 3]
        
        for i, channel in enumerate(CHANNELS):
            channel_data = drug_data[drug_data['Channel'] == channel]['Control_Coherence'].dropna().values
            
            if len(channel_data) > 0:
                parts = ax1.violinplot([channel_data], positions=[positions[i]], 
                                       showmeans=True, showmedians=True)
                
                # Color the violin
                for pc in parts['bodies']:
                    pc.set_facecolor(channel_colors[channel])
                    pc.set_alpha(0.7)
                
                # Scatter individual points
                jitter = np.random.normal(0, 0.05, len(channel_data))
                ax1.scatter(positions[i] + jitter, channel_data, 
                           c=channel_colors[channel], alpha=0.8, s=50, edgecolors='black', linewidth=0.5)
        
        ax1.set_xticks(positions)
        ax1.set_xticklabels([CHANNEL_SHORT_NAMES[ch] for ch in CHANNELS], fontsize=12)
        ax1.set_ylabel('Coherence', fontsize=14)
        ax1.set_title('Control Epoch', fontsize=12)
        ax1.set_ylim(0, None)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot for Treatment epoch
        ax2 = axes[1]
        
        for i, channel in enumerate(CHANNELS):
            channel_data = drug_data[drug_data['Channel'] == channel]['Treatment_Coherence'].dropna().values
            
            if len(channel_data) > 0:
                parts = ax2.violinplot([channel_data], positions=[positions[i]], 
                                       showmeans=True, showmedians=True)
                
                # Color the violin
                for pc in parts['bodies']:
                    pc.set_facecolor(channel_colors[channel])
                    pc.set_alpha(0.7)
                
                # Scatter individual points
                jitter = np.random.normal(0, 0.05, len(channel_data))
                ax2.scatter(positions[i] + jitter, channel_data, 
                           c=channel_colors[channel], alpha=0.8, s=50, edgecolors='black', linewidth=0.5)
        
        ax2.set_xticks(positions)
        ax2.set_xticklabels([CHANNEL_SHORT_NAMES[ch] for ch in CHANNELS], fontsize=12)
        ax2.set_ylabel('Coherence', fontsize=14)
        ax2.set_title('Treatment Epoch', fontsize=12)
        ax2.set_ylim(0, None)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=channel_colors[ch], alpha=0.7, 
                                 label=CHANNEL_DESCRIPTIONS[ch]) for ch in CHANNELS]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save figure
        safe_drug_name = drug.replace(":", "_").replace(" ", "_")
        save_path = os.path.join(OUTPUT_DIR, f"violin_plot_{safe_drug_name}.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        
        plt.show()
        plt.close()


def create_combined_violin_plot(results_df):
    """
    Create a combined violin plot with all drugs in one figure.
    
    Parameters:
    results_df : pd.DataFrame
        DataFrame with coherence results
    """
    print("\nCreating combined violin plot...")
    
    # Color scheme for channels
    channel_colors = {
        'PFCLFPvsCBEEG': '#2ecc71',  # Green (closest)
        'PFCEEGvsCBEEG': '#3498db',  # Blue (medium)
        'CBvsPCEEG': '#e74c3c'        # Red (furthest)
    }
    
    drugs = list(DRUG_GROUPS.keys())
    n_drugs = len(drugs)
    
    fig, axes = plt.subplots(2, n_drugs, figsize=(4*n_drugs, 10))
    fig.suptitle('Spatial Specificity of Calcium-Electrophysiology Coupling\nAcross Drug Conditions', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    for d_idx, drug in enumerate(drugs):
        drug_data = results_df[results_df['Drug'] == drug]
        
        for epoch_idx, (epoch, epoch_col) in enumerate([('Control', 'Control_Coherence'), 
                                                         ('Treatment', 'Treatment_Coherence')]):
            ax = axes[epoch_idx, d_idx]
            positions = [1, 2, 3]
            
            for i, channel in enumerate(CHANNELS):
                channel_data = drug_data[drug_data['Channel'] == channel][epoch_col].dropna().values
                
                if len(channel_data) > 0:
                    parts = ax.violinplot([channel_data], positions=[positions[i]], 
                                          showmeans=True, showmedians=True, widths=0.7)
                    
                    for pc in parts['bodies']:
                        pc.set_facecolor(channel_colors[channel])
                        pc.set_alpha(0.7)
                    
                    # Scatter individual points
                    jitter = np.random.normal(0, 0.05, len(channel_data))
                    ax.scatter(positions[i] + jitter, channel_data, 
                              c=channel_colors[channel], alpha=0.8, s=40, 
                              edgecolors='black', linewidth=0.5)
            
            ax.set_xticks(positions)
            ax.set_xticklabels([CHANNEL_SHORT_NAMES[ch] for ch in CHANNELS], fontsize=10)
            ax.set_ylim(0, None)
            ax.grid(axis='y', alpha=0.3)
            
            if d_idx == 0:
                ax.set_ylabel(f'{epoch} Coherence', fontsize=12)
            
            if epoch_idx == 0:
                # Shorten drug names for title
                short_name = drug.replace('dexmedetomidine_', 'Dex ').replace('_', ' ')
                ax.set_title(short_name.title(), fontsize=11, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=channel_colors[ch], alpha=0.7, 
                             label=CHANNEL_DESCRIPTIONS[ch]) for ch in CHANNELS]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.93)
    
    # Save figure
    save_path = os.path.join(OUTPUT_DIR, "violin_plot_combined_all_drugs.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def save_results(results_df, stats_results):
    """
    Save all results to CSV and TXT files.
    
    Parameters:
    results_df : pd.DataFrame
        DataFrame with coherence results
    stats_results : dict
        Dictionary with statistical results
    """
    print("\n" + "-" * 70)
    print("SAVING RESULTS")
    print("-" * 70)
    
    # Save raw data
    csv_path = os.path.join(OUTPUT_DIR, "spatial_specificity_raw_data.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved raw data: {csv_path}")
    
    # Save summary statistics for each drug
    for drug in DRUG_GROUPS.keys():
        drug_data = results_df[results_df['Drug'] == drug]
        summary = drug_data.groupby('Channel').agg({
            'Control_Coherence': ['mean', 'std', 'count'],
            'Treatment_Coherence': ['mean', 'std', 'count'],
            'Coherence_Change': ['mean', 'std'],
            'Coherence_Ratio': ['mean', 'std']
        }).round(6)
        
        safe_drug_name = drug.replace(":", "_").replace(" ", "_")
        summary_path = os.path.join(OUTPUT_DIR, f"summary_stats_{safe_drug_name}.csv")
        summary.to_csv(summary_path)
        print(f"Saved summary: {summary_path}")
    
    # Save comprehensive text report
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("-" * 80 + "\n")
        f.write("SPATIAL SPECIFICITY OF CALCIUM-ELECTROPHYSIOLOGY COUPLING\n")
        f.write("Analysis Report\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Frequency range: {FREQ_RANGE[0]} - {FREQ_RANGE[1]} Hz\n")
        f.write(f"Channels analyzed (ordered by proximity to calcium recording):\n")
        for ch in CHANNELS:
            f.write(f"  - {ch}: {CHANNEL_DESCRIPTIONS[ch]}\n")
        f.write("\n")
        
        f.write("HYPOTHESIS\n")
        f.write("-" * 40 + "\n")
        f.write("Expected gradient of calcium-EEG coherence:\n")
        f.write("PFC-LFP > PFC-EEG > CB-PC\n")
        f.write("(Coherence should decrease with distance from calcium recording site)\n\n")
        
        f.write("RESULTS BY DRUG\n")
        f.write("-" * 80 + "\n\n")
        
        for drug in DRUG_GROUPS.keys():
            f.write(f"\n{DRUG_DISPLAY_NAMES[drug]}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Subjects: {DRUG_GROUPS[drug]}\n\n")
            
            drug_data = results_df[results_df['Drug'] == drug]
            
            # Descriptive statistics
            f.write("Descriptive Statistics:\n")
            for epoch in ['Control', 'Treatment']:
                col = f'{epoch}_Coherence'
                f.write(f"\n  {epoch} Epoch:\n")
                for ch in CHANNELS:
                    ch_data = drug_data[drug_data['Channel'] == ch][col].dropna()
                    if len(ch_data) > 0:
                        f.write(f"    {CHANNEL_SHORT_NAMES[ch]}: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, n={len(ch_data)}\n")
            
            # ANOVA results
            f.write("\nANOVA (comparing channels):\n")
            for epoch in ['control', 'treatment']:
                key = f'{drug}_{epoch}'
                if key in stats_results['channel_comparison']:
                    stats = stats_results['channel_comparison'][key]
                    f.write(f"  {epoch.title()}: F={stats['F_statistic']:.4f}, p={stats['p_value']:.4f}\n")
            
            # Control vs Treatment
            f.write("\nControl vs Treatment (within channel):\n")
            if drug in stats_results['control_vs_treatment']:
                for ch in CHANNELS:
                    if ch in stats_results['control_vs_treatment'][drug]:
                        stats = stats_results['control_vs_treatment'][drug][ch]
                        f.write(f"  {CHANNEL_SHORT_NAMES[ch]}:\n")
                        f.write(f"    Paired t-test: t={stats['t_statistic']:.4f}, p={stats['p_value_ttest']:.4f}\n")
                        if not np.isnan(stats['w_statistic']):
                            f.write(f"    Wilcoxon: W={stats['w_statistic']:.4f}, p={stats['p_value_wilcoxon']:.4f}\n")
            
            f.write("\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("-" * 80 + "\n")
    
    print(f"Saved report: {report_path}")
    
    # Save statistical results as separate file
    stats_path = os.path.join(OUTPUT_DIR, "statistical_results.txt")
    with open(stats_path, 'w') as f:
        f.write("STATISTICAL RESULTS SUMMARY\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("ANOVA: Comparing coherence across channels\n")
        f.write("-" * 40 + "\n")
        for key, stats in stats_results['channel_comparison'].items():
            f.write(f"{key}: F={stats['F_statistic']:.4f}, p={stats['p_value']:.4f}\n")
        
        f.write("\n\nControl vs Treatment (paired comparisons)\n")
        f.write("-" * 40 + "\n")
        for drug, channels in stats_results['control_vs_treatment'].items():
            f.write(f"\n{DRUG_DISPLAY_NAMES[drug]}:\n")
            for ch, stats in channels.items():
                f.write(f"  {CHANNEL_SHORT_NAMES[ch]}: t={stats['t_statistic']:.4f}, p={stats['p_value_ttest']:.4f}\n")
    
    print(f"Saved statistics: {stats_path}")


# MAIN EXECUTION

if __name__ == "__main__":
    # Run full analysis
    results_df = run_full_analysis()
    
    # Compute statistics
    stats_results = compute_statistics(results_df)
    
    # Create visualizations
    create_violin_plots(results_df)
    create_combined_violin_plot(results_df)
    
    # Save all results
    save_results(results_df, stats_results)
    
    # Generate figure description
    generate_figure_description()
    
    print("\n" + "-" * 70)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("-" * 70)