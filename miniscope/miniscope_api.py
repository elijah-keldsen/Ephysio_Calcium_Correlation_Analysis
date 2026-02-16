from src2.shared.misc_functions import updateCSVCell
from src2.miniscope.miniscope_data_manager import MiniscopeDataManager
from src2.miniscope.miniscope_preprocessor import MiniscopePreprocessor
from src2.miniscope.miniscope_processor import MiniscopeProcessor
from src2.miniscope.miniscope_postprocessor import MiniscopePostprocessor
from src2.shared.paths import ANALYSIS_PARAMS
import caiman as cm
from src2.shared.misc_functions import get_coords_dict_from_analysis_params
from src2.miniscope.movie_io import MovieIO
import matplotlib
import tkinter
import os

#Adjust the path below to where you would like Caiman to store temporary files that it uses during the miniscope pipeline
os.environ["CAIMAN_DATA"] = '/Users/nathan/Desktop/K99/miniscope_data/dexmedetomidine/R230706A/2023_09_04/15_06_16/saved_movies'

class MiniscopeAPI:
    """Main workflow class for non-technical users. Adjust the paramters at the bottom and press run."""

    def __init__(self):
        pass
    
    def run(
            self, 
            line_num: int,
            filenames = [],
            
            #preprocessing parameters
            crop = True,
              #These should only be true if crop=True, and only one should be True or niether
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
            n_processes = 12,
            apply_motion_correction = False,
            inspect_motion_correction = False,
            plot_params = False,
            run_CNMFE = False,
            save_estimates=True,
              save_CNMFE_estimates_filename = 'estimates.hdf5',
            save_CNMFE_params = False,
            
            #post processing parameters
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
              time_bandwidth = 2
            ):
        
        
        self.miniscope_data_manager = MiniscopeDataManager(line_num, filenames, auto_import_data=True)
        
        
        #get previous cropping coordinates from analysis_params in case we want to use our last crop
        coords_dict, crop_job_name = get_coords_dict_from_analysis_params(self.miniscope_data_manager, crop_with_crop, crop_square)
        
        self.preprocessor = MiniscopePreprocessor(self.miniscope_data_manager)
        self.miniscope_data_manager = self.preprocessor.preprocess_calcium_movie(coords_dict, crop=crop, detrend_method=detrend_method, df_over_f=df_over_f, 
                                                                                 crop_job_name_for_file=crop_job_name, secs_window=secs_window, 
                                                                                 quantile_min=quantile_min, df_over_f_method=df_over_f_method)
        
        print("updating experiment.csv with your cropping coordinates", flush=True)
        updateCSVCell(self.miniscope_data_manager.coords, 'crop' if crop_with_crop else 'crop_square', line_num, ANALYSIS_PARAMS)
        
        
        #Ensure self.miniscope.data_manager has 'movie' and 'preprocessed_movie_filepath' filled in with the movie that you want to process before you process
        
        self.processor = MiniscopeProcessor(self.miniscope_data_manager)
        self.miniscope_data_manager = self.processor.process_calcium_movie(parallel, n_processes, apply_motion_correction, 
                                   inspect_motion_correction, plot_params, run_CNMFE,
                                   save_estimates, save_CNMFE_estimates_filename, save_CNMFE_params)
        
        
        
        if self.miniscope_data_manager.CNMFE_obj is not None:
            if tkinter._default_root:  # Check if Tkinter root exists
                tkinter._default_root.destroy()  # Force close any Tkinter root
            matplotlib.use('Qt5Agg')  # Switch to Qt backend so that we can use interactive plotting during estimate evaluation
            
            self.postprocessor = MiniscopePostprocessor(self.miniscope_data_manager)
            self.miniscope_data_manager = self.postprocessor.postprocess_calcium_movie(remove_components_with_gui, find_calcium_events, derivative_for_estimates, 
                                                                                       event_height, compute_miniscope_phase, filter_miniscope_data,n, cut, ftype, 
                                                                                       btype, inline, compute_miniscope_spectrogram, window_length, window_step, 
                                                                                       freq_lims, time_bandwidth)




if __name__ == "__main__":
    # run the API
    api = MiniscopeAPI()
    api.run(
        line_num = 97, #line number of the experiment you are analyzing
        filenames = ['0.avi'],
        
        #preprocessing parameters
        crop = True,
            #Only one below should be True if crop=True
            crop_with_crop = False,
            crop_square = True,
        detrend_method = None,
        df_over_f = False,
          #if df_over_f = True
          secs_window = 5,                     
          quantile_min = 8,
          df_over_f_method = 'delta_f_over_sqrt_f',

        #processing parameters    
        parallel = True,
        n_processes = 6,
        apply_motion_correction = False,
        inspect_motion_correction = True,
        plot_params = False,
        run_CNMFE = True,
        save_estimates=True,
          save_CNMFE_estimates_filename = 'estimates.hdf5',
        save_CNMFE_params = True,
        
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
          inline=True,
        compute_miniscope_spectrogram=True,
          window_length = 30, 
          window_step = 3, 
          freq_lims = [0,15], 
          time_bandwidth = 2
        )
        
    
    
    
    
    
    
    
    
    
    
    
    