#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src2.ephys.channel_worker import ChannelWorker
from src2.ephys.ephys_data_manager import EphysDataManager
from src2.ephys.visualizer import Visualizer
from src2.shared.experiment_data_manager import ExperimentDataManager
from typing import List
import logging


class EphysAPI:
    """Main workflow class."""

    def __init__(self):
        pass
    


    def run(
            self, 
            line_num,
            channel_name = 'PFCLFPvsCBEEG',
            remove_artifacts = False,
            filter_type = None, # if desired, enter the type, eg "butter"
            filter_range = [0.5, 4],
            compute_phases = False,
            plot_channel = False,
            plot_spectrogram = False,
            plot_phases = False,
            logging_level = "CRITICAL"
            ):
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging_level)

        # set the filter boolean based on if filter_type is None
        filter_bool = True if filter_type is not None else False

        experiment_data_manager = ExperimentDataManager(line_num, logging_level = logging_level)

        # Extract the one relevant piece of information that EphysDataManager needs from metadata--the path to the ephys directory
        ephys_directory = experiment_data_manager.get_ephys_directory()

        # Create instance of EphysDataManager, process the block into channels
        self.ephys_data_manager = EphysDataManager(ephys_directory, auto_import_ephys_block=True, auto_process_block=False, auto_compute_phases=False)
        self.ephys_data_manager.process_ephys_block_to_channels(remove_artifacts=remove_artifacts, channels = channel_name)


        logger.debug(self.ephys_data_manager.channels)

        # If filter_type is not None, filter the signal and add it to ephys_data_manager.channels[channel_name].signal_filtered
        if filter_bool:
            print('Filtering eohys data with filter type "{filter_type}" and cut {filter_range}')
            self.ephys_data_manager.filter_ephys(channel_name, ftype=filter_type, cut = filter_range, replace_signal=False)
        
        if compute_phases:
            #compute phases after filtering
            self.ephys_data_manager.compute_phases_all_channels()

        # Extract correct channel and visualize
        logger.info(f"Visualizing channel: {channel_name}")
        channel = self.ephys_data_manager.get_channel(channel_name)
        channel_worker = ChannelWorker(channel)
        

        if plot_channel:
            channel_worker.plot_channel(use_filtered = filter_bool)

        if plot_spectrogram:
            channel_worker.plot_spectrogram(use_filtered = filter_bool, plot_events=False)  
            
        if plot_phases:
            channel_worker.plot_phases()
            
























    def run_all_channels(
            self, 
            line_num,
            remove_artifacts = False,
            filter_type = None, # if desired, enter the type, eg "butter"
            filter_range = [0.5, 4],
            plot_channel = False,
            plot_spectrogram = False,
            logging_level = "CRITICAL"
            ):
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging_level)
        
        # set the filter boolean based on if filter_type is None
        filter: bool = True if filter_type is not None else False

        experiment_data_manager = ExperimentDataManager(line_num, logging_level = logging_level)

        channels_str = experiment_data_manager.metadata['LFP and EEG CSCs']
        channels_list: List = [*channels_str] # unpack




if __name__ == "__main__":
    e = EphysAPI()
    e.run(
          line_num=101,
          channel_name = 'PFCLFPvsCBEEG',
          remove_artifacts = False,
          filter_type = None,
          filter_range = [0.3,0.5],
          compute_phases = False,
          plot_channel = True,
          plot_spectrogram = True,
          plot_phases = False,
          logging_level="DEBUG"
    )
    # main()
    
    
    
    
    
    
    
    
    
    
    