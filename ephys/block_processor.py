#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For reference, this is the structure of the neo object:


Block (entire experiment)
│
└── Segment 0 (e.g., baseline period)
    │
    ├── AnalogSignal 0 (Channel 1: e.g., "EMG.ncs")
    ├── AnalogSignal 1 (Channel 2: e.g., "PFCLFPvsCBEEG.ncs")
    └── Event (e.g., "StimulusTriggers")
│
├── Segment 1 (e.g., drug application)
│   ├── AnalogSignal 0
│   └── ...
│
└── ...

An event object in neo contains info like "light turned on: 5:48" or "drug applied: 6:00"

"""
import numpy as np
from scipy.signal.windows import hann
from src2.ephys.channel import Channel
from neo.core import Block
import logging

class BlockProcessor:
    """Processes an Ephys Block into channels."""
    
    def __init__(self, ephys_block: Block, logger: logging.Logger):
        self.logger = logger
        self.ephys_block = ephys_block

        
    def process_raw_ephys(self, channels, remove_artifacts=False):
        """Convert raw ephys data into processed Channel objects."""
        if not self.ephys_block:
            raise ValueError("Load raw data first using EphysDataManager.import_ephys_data()")
        
        if type(channels) == str:
            channels = [channels]
        
        print('Processing raw ephys data into channels...')

        channels_dict = {}
        print(f"channels: {channels}")

        for channel_name in channels:
            # self.logger.info(f"channel_name = {channel_name}")
            print(f"channel_name = {channel_name}")
            assert type(channel_name) == str
            new_channel = self._process_single_channel(channel_name)

            if remove_artifacts:
                self.remove_artifacts(new_channel)

            channels_dict[channel_name] = new_channel

            return channels_dict

            

            
    def remove_artifacts(self, channel: Channel, volt_threshold=1500, time_threshold=60, hannNum=75):
        """Remove artifacts from the specified channel."""
        print('Removing artifacts from ' + channel.name + '...')
        dt = channel.time_vector[1] - channel.time_vector[0]
        mean = np.mean(channel.signal)
        channel.signal = channel.signal - mean
        
        mask = np.abs(channel.signal) > volt_threshold
        mask = self._fill_gaps(mask, dt, time_threshold)
        han_window = self._create_hann_window(hannNum)
        
        self._apply_hann_window(channel, mask, han_window, dt)
        
            







            
            
    def _process_single_channel(self, channel_name):
        print(f"Channel name: {channel_name}")
        """Process a single channel from raw data."""
        # Get the first and last segments
        first_segment = self.ephys_block.segments[0].analogsignals
        last_segment = self.ephys_block.segments[-1].analogsignals

        # Find the channel in the first segment by name
        try:
            # Get the channel and its index in the first segment
            channel_index, channel = next(
                (i, c) for i, c in enumerate(first_segment) 
                if c.name == channel_name
            )
        except StopIteration:
            raise ValueError(f"Channel '{channel_name}' not found in the first segment.")

        # Find the corresponding channel in the last segment by name
        try:
            last_channel = next(c for c in last_segment if c.name == channel_name)
        except StopIteration:
            raise ValueError(f"Channel '{channel_name}' not found in the last segment.")

        # Extract timing details
        sampling_rate = channel.sampling_rate.magnitude.item()
        dt = 1 / sampling_rate
        t_start = channel.t_start.magnitude
        t_stop = last_channel.t_stop.magnitude  # Use last_segment's channel

        # Generate time vector
        if (t_stop - t_start) % dt <= dt / 2:
            t_stop -= 0.51 * dt
        time_vector = np.arange(t_start, t_stop, dt)

        # Build signal array using the index from the first segment, and extract events
        signal, events= self._scan_segments(channel_index, channel_name, time_vector)

        # Return processed channel
        return Channel(channel_name, signal, sampling_rate, time_vector, events)



    def _scan_segments(self, channel_index, channel_name, time_vector):
        """Construct continuous signal from raw segments."""
        n_points = len(time_vector)
        signal = np.full(n_points, np.nan)
        # events = []

        unsorted_labels = []
        unsorted_timestamps = []
        
        for seg in self.ephys_block.segments:


            # signal processing

            sig = seg.analogsignals[channel_index]
            signal_data = sig.magnitude.squeeze()  # NEW: Flatten to 1D
            # Calculate start/end indices
            start_idx = np.argmin(np.abs(time_vector - sig.t_start.magnitude))
            end_idx = start_idx + signal_data.size  # Use flattened data size
            # Avoid overfilling
            end_idx = min(end_idx, n_points)  # NEW: Prevent index overflow
            if start_idx > 0 and np.isnan(signal[start_idx - 1]):
                self._interpolate_missing_data(channel_name, signal, start_idx, time_vector, sig.t_start.magnitude)
            # Assign flattened data
            signal[start_idx:end_idx] = signal_data[:end_idx-start_idx]  # MODIFIED


            # PREVIOUSLY IMPORTNEURALYNXEVENTS()
            # event processing Luke's: 
            # for event in seg.events:
            #     for t in event.times:
            #         events.append((event.name, t.magnitude.item()))



            for e in seg.events:
                for k, l in enumerate(e.labels.astype(str)):
                    unsorted_labels.append(l)
                    unsorted_timestamps.append(e.times[k].magnitude)
        # Sort all of the events
        np_unsorted_labels = np.array(unsorted_labels)
        np_unsorted_timestamps = np.array(unsorted_timestamps)
        reordered_indices = np.argsort(np_unsorted_timestamps)
        event_labels = np_unsorted_labels[reordered_indices]
        event_timestamps = np_unsorted_timestamps[reordered_indices] # - self.zeroTime[next(iter(self.zeroTime))]

        events = {
            'labels': event_labels,
            'timestamps': event_timestamps
        }
        
        return signal, events
        
    def _interpolate_missing_data(self, channel_name, signal, start_idx, time_vector, t_start):
        """Fill gaps between segments with interpolation."""
        interp_start = np.where(np.isnan(signal))[0][0]
        interp_length = start_idx - interp_start
        x = np.linspace(signal[interp_start - 1], signal[interp_start], interp_length + 2)
        signal[interp_start:start_idx] = x[1:-1]
        

            
    def _fill_gaps(self, mask, dt, threshold):
        """Fill gaps in the mask where the time between threshold crossings is below the threshold."""
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == -1)[0]
        
        for start in starts:
            end = np.where(diff[start:] == 1)[0]
            if end.size > 0 and (end[0] * dt) < threshold:
                mask[start:start + end[0] + 1] = True
        return mask
        
    def _create_hann_window(self, size):
        """Create a Hann window for smoothing."""
        window = hann(size)
        return np.abs(window - 1)
        
    def _apply_hann_window(self, channel, mask, window, dt):
        """Apply the Hann window to the masked regions."""
        half_len = len(window) // 2
        indices = np.where(mask)[0]
        
        for idx in indices:
            start = max(0, idx - half_len)
            end = min(len(channel.signal), idx + half_len + 1)
            segment = channel.signal[start:end]
            channel.signal[start:end] = segment * window[:len(segment)]