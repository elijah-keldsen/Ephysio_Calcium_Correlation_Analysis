#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from neo.io import NeuralynxIO
from src2.shared.path_finder import PathFinder
import os
from src2.ephys.block_processor import BlockProcessor
from src2.ephys.channel import Channel
import logging
from scipy.signal import hilbert

class EphysDataManager():
    """
    Manages the import of raw ephys data (neo's high level object is called a "Block").
    Processes raw ephys data via EphysBlockProcessor.
    EphysBlockProcessor returns a dictionary of channels.
    Stores the processed channels in self.channels, where the key is the channel name and the value is a Channel object.
    """


    def __init__(self, ephys_directory=None, auto_import_ephys_block=True, auto_process_block=True, auto_compute_phases=True, level = "CRITICAL"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        
        self.channels = {}  # Processed channels
        self.ephys_block = None  # Raw data storage

        if (auto_import_ephys_block):
            assert ephys_directory is not None
            self.import_ephys_block(ephys_directory)

        if (auto_process_block):
            self.process_ephys_block_to_channels()
            
        if auto_compute_phases:
            self.compute_phases_all_channels()
        

    def import_ephys_block(self, ephys_directory):
        """Load raw Neuralynx data without processing."""
        print('Importing raw ephys data...')
        ephys_file_path = self._find_ephys_file_path(ephys_directory) # get most recently edited Events.nev file
        ephys_dir_path = os.path.dirname(ephys_file_path) # get its parent directory
        file_reader = NeuralynxIO(dirname=ephys_dir_path)
        self.ephys_block = file_reader.read_block(signal_group_mode='split-all')


    def process_ephys_block_to_channels(self, channels, remove_artifacts=False):
        """Process raw ephys data into channels."""

        processor = BlockProcessor(self.ephys_block, self.logger)
        self.channels = processor.process_raw_ephys(channels, remove_artifacts=remove_artifacts)

    def compute_phases_all_channels(self):
        for key, value in self.channels.items():
            self.channels[key] = self.compute_phase(value)
            

    def compute_phase(self, channel):
        print(f"Computing phase for {channel.name}")
        analytic_signal = hilbert(channel.signal)
        channel.phases = np.angle(analytic_signal)
        return channel
    

    def filter_ephys(self, channel_name, n=2, cut=[0.5, 4], ftype='butter', btype='bandpass', replace_signal=True):
        """Filter the ephys data."""
        # self.logger.info('Filtering ' + channel_name + ' with a(n) ' + ftype + ' filter ...')
        try:
            channel: Channel = self.channels[channel_name]
        except KeyError:
            raise ValueError("Channel not found in data_manager. Please import the data first.")
            
        print(f"Filtering the ephys signal: {channel_name}")
            
        filtered_data = self._filter_data(
            channel.signal,
            n=n,
            cut=cut,
            ftype=ftype,
            btype=btype,
            fs=channel.sampling_rate
        )
        
        if (replace_signal):
            self.channels[channel_name].signal = filtered_data
        else:
            self.channels[channel_name].signal_filtered = filtered_data

        return filtered_data
    
    def get_channels(self):
        return self.channels
    
    def get_channel(self, channel_name):
        return self.channels[channel_name]










    def _find_ephys_file_path(self, ephys_directory):        
        path_finder = PathFinder()
        events_path = path_finder.find( 
                        directory=ephys_directory,
                        suffix=".nev",
                        prefix="Events"
            )
        return events_path[0]
    
    @staticmethod
    def _filter_data(data, n, cut, ftype, btype, fs, bodePlot=False):
        from scipy.signal import butter, freqz, filtfilt, firwin, bode
        import logging

        # Set up logging
        logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s') # turn to DEBUG for more info
        
        # Log input variables
        logging.info(f"Input variables:")
        logging.info(f"- data: {data}")
        logging.info(f"- n: {n}")
        logging.info(f"- cut: {cut}")
        logging.info(f"- ftype: {ftype}")
        logging.info(f"- btype: {btype}")
        logging.info(f"- fs: {fs}")
        logging.info(f"- bodePlot: {bodePlot}")

        """ Use ftype to indicate FIR or Butterworth filter.
        
        For the FIR filter indicate a LowPass, HighPass, or BandPass with btype = lowpass, highpass, or bandpass, respectively. 
        n is the length of the filter (number of coefficients, i.e. the filter order + 1). numtaps must be odd if a passband includes the Nyquist frequency.
        A good value for n is 10000.
        Channel should be set to desired .ncs file
        
        The Butterworth filters have a more linear phase response in the pass-band than other types and is able to provide better group delay performance, and also a lower level of overshoot.
        Indicate the filter type by setting btype = 'low', 'high', or 'band'.
        The default for n is n = 2
        For a bandpass filter indicate the lowstop and the highstop by using an array. example: wn= ([10, 30])"""

        if ftype.lower() == 'fir':
            h = firwin(n, cut, pass_zero=btype, fs=fs)  # Build the FIR filter
            filteredData = filtfilt(h, 1, data)  # Zero-phase filter the data
            if bodePlot:
                w, a = freqz(h, worN=10000,fs=2000)
                plt.figure()
                plt.semilogx(w, abs(a))
                
                w, mag, phase = bode((h,1),w=2*np.pi*w)
                plt.figure()
                plt.semilogx(w,mag)
                plt.figure()
                plt.semilogx(w,phase)

        if ftype.lower() == 'butterworth' or ftype.lower() == 'butter':
            print(f"fs: {type(fs)}")
            b, a = butter(n, cut, btype=btype, fs=fs)
            filteredData = filtfilt(b, a, data)
            
            if bodePlot:
                w, h = freqz(b, a, worN=10000,fs=2000)
                plt.figure()
                plt.semilogx(w, abs(h))
                
                w, mag, phase = bode((b,a),w=2*np.pi*w)
                plt.figure()
                plt.semilogx(w,mag)
                plt.figure()
                plt.semilogx(w,phase)

        return filteredData
        
        
        
        