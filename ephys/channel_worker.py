from src2.ephys.channel import Channel
from src2.ephys.spectrogram import Spectrogram
from src2.ephys.visualizer import Visualizer
from src2.shared.multitaper_spectrogram_python import multitaper_spectrogram
import numpy as np
import matplotlib.pyplot as plt

class ChannelWorker:
    def __init__ (self, channel: Channel):
        self.channel = channel
        self.visualizer = Visualizer()
        self.spectrogram = None

    def plot_channel(self, use_filtered=False):
        self.visualizer.plot_channel(self.channel, use_filtered)


    def plot_spectrogram(self, window_length=30, window_step=3,
                           freq_limits=[0, 50], time_bandwidth=2, plot_events=False, use_filtered=False):
        
        spectrogram: Spectrogram = self.compute_spectrogram(self.channel, window_length, window_step, freq_limits, time_bandwidth, use_filtered)

        events = None if not plot_events else self.channel.events
        self.visualizer.plot_spectrogram(spectrogram, events = events)
        return self.spectrogram
    

    def compute_spectrogram(self, channel:Channel, window_length=30, window_step=3,
                           freq_limits=[0, 50], time_bandwidth=2, use_filtered=False):
        
        signal = channel.signal_filtered if use_filtered else channel.signal
        fs = int(channel.sampling_rate)

        # Spectrogram parameters
        num_tapers = time_bandwidth * 2 - 1  
        window_params = [window_length, window_step]  # [window length (s), step size (s)]

        psd_matrix, time_points, freq_points = multitaper_spectrogram( # psd_matrix = power_spectral_density_matrix
            signal, fs, freq_limits, time_bandwidth, num_tapers, window_params, plot_on=False
        )
        
        # Convert to decibel scale (dB re 1 µV²/Hz)
        psd_matrix_db = 10 * np.log10(psd_matrix)

        # Create Spectrogram object and save it to self.spectrogram
        self.spectrogram = Spectrogram(psd_matrix_db, time_points, freq_points)
        return self.spectrogram

    def plot_phases(self):
        # Step 1: Verify data and sampling rate
        print("Sampling rate:", self.channel.sampling_rate)
        print("Length of phases:", len(self.channel.phases))
        print("Min and Max phase:", np.min(self.channel.phases), np.max(self.channel.phases))
        print("Any NaN values?", np.any(np.isnan(self.channel.phases)))
        print("Any infinite values?", np.any(np.isinf(self.channel.phases)))
    
        # Step 2: Plot a histogram of phase values
        plt.figure(figsize=(10, 6), facecolor='white')  # Create a figure with white background
        plt.hist(self.channel.phases, bins=50, color='blue', alpha=0.7, label=f'Phase Distribution of {self.channel.name}')
        plt.title(f"Histogram of Phase Values for {self.channel.name}")
        plt.xlabel("Phase (radians)")
        plt.ylabel("Frequency")
        plt.xlim(-np.pi, np.pi)  # Phase range: -π to +π
        plt.grid(True, alpha=0.3)  # Light grid for readability
        plt.legend(loc='upper right')  # Explicit legend location
        plt.show()
