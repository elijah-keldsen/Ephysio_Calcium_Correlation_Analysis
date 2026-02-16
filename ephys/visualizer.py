#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from src.multitaper_spectrogram_python import multitaper_spectrogram
from src2.ephys.channel import  Channel
import logging


class Visualizer:
    """Class for visualizing ephys data."""
    def __init__(self, level='CRITICAL'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)


    def plot_channel(self, channel:Channel, use_filtered=False):
        """Visualize the processed channel."""
        self.logger.info(f"Plotting channel...")

        signal = channel.signal_filtered if use_filtered else channel.signal
        self.logger.debug(f"signal to be plotted: {signal}")
        plt.figure()
        plt.plot(channel.time_vector, signal)
        plt.title(channel.name)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (µV)')
        plt.show()
    

    def plot_spectrogram_helper(self, psd_matrix_db, time_points, freq_points, events):
        self.logger.info(f"Plotting spectrogram...")
        
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(time_points / 60, freq_points, psd_matrix_db,
                            shading='gouraud', cmap='inferno')
        plt.colorbar(mesh, ax=ax, label='Power (dB)')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Frequency (Hz)')
        
        if events:
            self._markEvents(ax, events)
            
        plt.show()
        return fig, ax
    
    def plot_spectrogram(self, spectrogram, events = None):
        """Plot the spectrogram."""
        self.plot_spectrogram_helper(spectrogram.psd_matrix_db, spectrogram.time_points, spectrogram.freq_points, events)
    
    def _markEvents(self, axisHandle, events):

        # TODO this should only mark user-made events.  Also the structure of events has changed, and will probably change again before we ever use this function


        """Mark events with labels on a given plot."""
        self.logger.info(f"Marking events on plot...")
        yLimits = axisHandle.get_ylim()
        xLimits = axisHandle.get_xlim()
        lineLength = np.diff(yLimits)
        lineOffset = yLimits[0] + (lineLength / 2)

        events = events[:10]
        
        for label, timestamp in events:
            # Plot the event line
            axisHandle.axvline(x=timestamp, color='k', linestyle='--', alpha=0.7)
            # Add the label
            axisHandle.text(timestamp, yLimits[1], label, rotation=90, verticalalignment='bottom', fontsize=8)
        
        axisHandle.axis([xLimits[0], xLimits[1], yLimits[0], yLimits[1]])