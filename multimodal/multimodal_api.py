from src2.miniscope.miniscope_api import MiniscopeAPI
from src2.ephys.ephys_api import EphysAPI
from src2.multimodal.miniscope_ephys_alignment_utils import sync_neuralynx_miniscope_timestamps, find_ephys_idx_of_TTL_events, find_ca_movie_frame_num_of_ephys_idx, find_ca_movie_filenums
from src2.multimodal.calcium_ephys_visualizer import create_ca_ephys_movie
from src2.multimodal.phase_utils import ephys_phase_ca_events, miniscope_phase_ca_events, phase_ca_events_histogram

class MultimodalAPI:
        
    
    def run(self,
            line_num,
            #ephys parameters
            channel_name = 'PFCLFPvsCBEEG',
            remove_artifacts = False,
            filter_type = None,
            filter_range = [0.5, 4],
            plot_channel = False,
            plot_spectrogram = False,
            plot_phases = False,
            logging_level = "CRITICAL",
            
            #miniscope parameters
            miniscope_filenames = [],
            #preprocessing parameters
            crop = True,
              crop_with_crop = False,
              crop_square = False,
            detrend_method = 'median',
            df_over_f = False,
              #if df_over_f = True
              secs_window=5,                     
              quantile_min=8,
              df_over_f_method='delta_f_over_sqrt_f',

            #processing parameters    
            parallel = False,
            n_processes = 6,
            apply_motion_correction = True,
            inspect_motion_correction = True,
            plot_params = False,
            run_CNMFE = True,
            save_estimates=True,
              save_CNMFE_estimates_filename = 'estimates.hdf5',
            save_CNMFE_params = False,
            
            #post-processing parameters
            remove_components_with_gui=True,  
            find_calcium_events=True,
              derivative_for_estimates='first', 
              event_height = 5, 
            compute_miniscope_phase=True, 
            filter_miniscope_data=True,
              n=2, 
              cut=[0.1,1.5], 
              ftype='butter', 
              btype='bandpass', 
              inline=False,
            compute_miniscope_spectrogram=True,
              window_length = 30, 
              window_step = 3, 
              freq_lims = [0,15], 
              time_bandwidth = 2,
        
            #multimodal parameters
            delete_TTLs=True, 
            fix_TTL_gaps=False, 
            only_experiment_events=True,
            all_TTL_events=True, 
            ca_events=False,
            time_range=None
            ):
        
        
        
        ephys_api = EphysAPI()
        ephys_api.run(line_num, channel_name, remove_artifacts, filter_type, filter_range, plot_channel, plot_spectrogram, plot_phases, logging_level)
        
        
        
        miniscope_api = MiniscopeAPI()
        miniscope_api.run(line_num, miniscope_filenames, crop, crop_square, crop_with_crop, detrend_method, df_over_f, secs_window, 
                          quantile_min, df_over_f_method, parallel, n_processes, apply_motion_correction, inspect_motion_correction, plot_params,
                          run_CNMFE, save_estimates, save_CNMFE_estimates_filename, save_CNMFE_params, remove_components_with_gui, find_calcium_events, derivative_for_estimates, event_height,
                          compute_miniscope_phase, filter_miniscope_data, n, cut, ftype, btype, inline, compute_miniscope_spectrogram, window_length, window_step, freq_lims, time_bandwidth)
        



        
        #pull everything we need to run multi modal analysis:
        channel_object = ephys_api.ephys_data_manager.get_channel(channel_name)
        frame_rate = miniscope_api.miniscope_data_manager.fr
        ca_events_idx = miniscope_api.miniscope_data_manager.ca_events_idx
        miniscope_phases = miniscope_api.miniscope_data_manager.miniscope_phases
        
        
        
        tCaIm, low_confidence_periods, channel_object, miniscope_dm = sync_neuralynx_miniscope_timestamps(channel_object, miniscope_api.miniscope_data_manager, delete_TTLs=delete_TTLs, 
                                                                                                   fix_TTL_gaps=fix_TTL_gaps, only_experiment_events=only_experiment_events)
        

        
        # set changed variables
        print("\nSuccess! Setting changed variables.")
        miniscope_api.miniscope_data_manager = miniscope_dm
        ephys_api.ephys_data_manager.channels[channel_name] = channel_object
        
        
        ephys_idx_all_TTL_events, ephys_idx_ca_events = find_ephys_idx_of_TTL_events(tCaIm, channel_object, frame_rate, all_TTL_events=all_TTL_events, ca_events_idx=ca_events_idx if ca_events else None)
        ca_frame_num_of_ephys_idx = find_ca_movie_frame_num_of_ephys_idx(channel_object, ephys_idx_all_TTL_events)
        
        ca_events_phases_ephys = ephys_phase_ca_events(ephys_idx_ca_events, channel_object, neurons='all')
        ca_events_phases_miniscope = miniscope_phase_ca_events(ca_events_idx, miniscope_phases, neurons='all')
        
        
        hist1, bin_edges1 = phase_ca_events_histogram(ca_events_phases_ephys)
        
        
        hist2, bin_edges2 = phase_ca_events_histogram(ca_events_phases_miniscope)
        


if __name__ == "__main__":
    # run the API
    api = MultimodalAPI()
    api.run(
        
        
        line_num = 97,  #line number of experiment you are analyzing
        
        #ephys parameters
        channel_name = 'PFCLFPvsCBEEG',                         #channel name of EEG signal you are analyzing
        remove_artifacts = False,                               #If you want artifacts removed from the signal
        filter_type = None,                                     #if desired, enter the filter type, in the form 'butter' or 'fir', or leave it as None
        filter_range = [0.5, 4],                                #array of cutoff frequencies (that is, band edges)
        plot_channel = False,                                    #choice to plot the EEG signal
        plot_spectrogram = False,                               #choice to plot the EEG spectrogram
        plot_phases = False,                                    #choice to plot the phases of the channel_name signal
        logging_level = "CRITICAL",                             #unimportant for non-coders. This parameter does not alter the experiment analysis
            
        
        miniscope_filenames = ['0.avi'],                               #list of movies you want to analyze, filled with just file basenames like: ['0.avi', '1.avi', ... , '10.avi']. Leave empty as [] to analyze all .avi movies in line_num's directory
        
        #miniscope parameters:
        #pre-processing
        crop = True,                                            #option to crop your movie
            crop_with_crop = False,                               #displays gui with coordinates under 'crop' in the file 'data/analysis_parameters.csv' and saves any new coordinates to same place
            crop_square = True,                                   #displays gui with coordinates under 'crop_square' in the file 'data/analysis_parameters.csv' and saves any new coordinates to same place
        detrend_method = 'linear',                                         #detrend the movie. Options are 'linear', 'median', or None if you don't want to detrend
        df_over_f = True,                                      #Decide if you want to compute df_over_f of movie. Three parameters below are related to df_over_f
          secs_window=5,                                          #length of the windows used to compute the quantile
          quantile_min=8,                                         #value of the quantile
          df_over_f_method='delta_f_over_sqrt_f',                 #method should equal one of these three: 'only_baseline', 'delta_f_over_f', 'delta_f_over_sqrt_f'

        #processing parameters    
        parallel = False,                                       #option to compute CNMFE using parallel processing on your machine
        n_processes = 6,                                        #number of processing cores to use on your machine
        apply_motion_correction = True,                        #option to motion-correct your movie
        inspect_motion_correction = True,                       #option to inspect the efficacy of the motion-correction
        plot_params = False,
        run_CNMFE = True,
        save_estimates=False,
          save_CNMFE_estimates_filename = 'estimates.hdf5',
        save_CNMFE_params = False,
        
        #post-processing parameters
        remove_components_with_gui=True,  
        find_calcium_events=True,
          #calculating event parameters
          derivative_for_estimates='first', 
          event_height = 5, 
        compute_miniscope_phase=True, 
        filter_miniscope_data=True,
          #filtering parameters
          n=2,
          cut=[0.1,1.5], 
          ftype='butter', 
          btype='bandpass', 
          inline=False,
        compute_miniscope_spectrogram=False,
          #spectrogram parameters
          window_length = 30, 
          window_step = 3, 
          freq_lims = [0,15], 
          time_bandwidth = 2,
        
        #multimodal parameters
        delete_TTLs=True, 
        fix_TTL_gaps=True, 
        only_experiment_events=False,
        all_TTL_events=True, 
        ca_events=True,
        time_range=None
        
        )