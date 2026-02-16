import caiman as cm
import numpy as np
from src2.miniscope.projections import Projections
import matplotlib.pyplot as plt
from scipy.signal import detrend
from caiman import movie as cm_movie
from tqdm import tqdm
from src2.miniscope.gui_utils import crop_gui
from src2.miniscope.movie_io import MovieIO
tqdm.monitor_interval = 0 #may not be necessary

class MiniscopePreprocessor:

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.frame_rate = data_manager.movie.fr
    

    def preprocess_calcium_movie(self, coords_dict=None, crop=False, detrend_method=None, df_over_f=False, crop_job_name_for_file="_cropped",
                                 secs_window=5, quantile_min=8, df_over_f_method='delta_f_over_sqrt_f'):
        """Run preprocessing steps based on provided flags.
           coords_dict: is passed in and represents what you want the final coordinates for the movie to be in the form {'x0': A, 'y0': B, 'x1': C, 'y1': D}"""
        
        steps_applied = ['preprocessed']
        
        if crop:
            self.data_manager.projections = self.compute_projections(self.data_manager.movie)
            self.data_manager.movie, self.data_manager.coords = self.crop_movie(self.data_manager.movie, coords_dict, self.data_manager.projections)
            steps_applied.append(f'{crop_job_name_for_file}')

        if detrend_method:
            self.data_manager.movie = self.detrend_movie(self.data_manager.movie, method=detrend_method)
            steps_applied.append('_detrended')

        if df_over_f:
            self.data_manager.movie = self.compute_df_over_f(self.data_manager.movie, secs_window=secs_window, quantile_min=quantile_min, method=df_over_f_method)
            steps_applied.append('_dFoverF')

        movie_file_name = ''.join(steps_applied)
        
        self.data_manager.preprocessed_movie_filepath = MovieIO.save_movie(self.data_manager, movie_file_name)
        
        print(f"This is the movie shape after preprocessing: {self.data_manager.movie.shape}")
        
        return self.data_manager
    
    
    
    
    
    def compute_projections(self, movie: cm.movie=None):
        """Calculates the projections of self.movie with progress bar."""
        print("\n\nComputing projections...\n")
        
        operations = {
            'max': lambda m: np.amax(m, axis=0),
            'std': lambda m: np.std(m, axis=0),
            'min': lambda m: np.amin(m, axis=0),
            'mean': lambda m: np.mean(m, axis=0),
            'median': lambda m: np.median(m, axis=0),
            'time': lambda m: m.mean(axis=(1,2)),
        }

        results = {}
        for name, op in tqdm(operations.items(), desc='Computing Projections'):
            results[name] = op(movie)

        results['range'] = results['max'] - results['min']

        return Projections(
            results['max'],
            results['std'],
            results['min'],
            results['mean'],
            results['median'],
            results['range'],
            results['time']
        )    
    

    def crop_movie(self, movie, coords_dict, projections):
        print('Cropping...')
        movie_height = movie.shape[1]
        movie_width = movie.shape[2]
        new_coords_dict = crop_gui(coords_dict, projections, movie_height, movie_width)
        
        # Flip y-coordinates (GUI origin is bottom-left; numpy origin is top-left)
        y0_flipped = movie.shape[1] - new_coords_dict['y1']
        y1_flipped = movie.shape[1] - new_coords_dict['y0']
        
        # Sort coordinates
        y0, y1 = sorted([y0_flipped, y1_flipped])
        x0, x1 = sorted([new_coords_dict['x0'], new_coords_dict['x1']])
        
        #crop movie using our numpy coordinates
        cropped_movie = movie[:, y0:y1, x0:x1]
        
        #Keep new_coords_dict in GUI notation so that it will display properly in the GUI if you want to view them again
        new_coords_dict = f'({new_coords_dict["x0"]},{new_coords_dict["y0"]}, {new_coords_dict["x1"]},{new_coords_dict["y1"]})'
        
        return cropped_movie, new_coords_dict
        

    def detrend_movie(self, movie, method='median', plot_trend=True):
        print('Detrending...')
        try:                    
            if method == 'linear':
                detrended_movie = detrend(movie, axis=0)
            elif method == 'median':
                detrended_movie = movie.debleach()
            else:
                raise ValueError(f"Unsupported detrending method '{method}'.")
                return movie
        except:
            print("Detrending failed, returning original movie")
            return movie
        
        if plot_trend and detrended_movie is not None:
            fig, ax = plt.subplots()
            ax.set_xlabel('Frames')
            ax.set_ylabel('Mean Fluorescence')
            
            # Plot original data
            original_mean = np.mean(movie, axis=(1, 2))
            ax.plot(original_mean, label='Original Data', color='blue')
            
            # Plot detrended data
            detrended_mean = np.mean(detrended_movie, axis=(1, 2))
            ax.plot(detrended_mean, label='Detrended Data', color='red', linestyle='--')
            
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()
            
        if isinstance(detrended_movie, np.ndarray):
            print("Ensuring that the movie that is returned is a caiman movie, not a numpy array...")
            detrended_movie = cm.movie(detrended_movie, fr=self.frame_rate)
        
        print('Detrending was successful')
        return detrended_movie
        

    def compute_df_over_f(self, movie, secs_window=5, quantile_min=8, method='delta_f_over_sqrt_f'):
        print("Attempting to compute_df_over of movie...")
        try:
            if np.min(movie) < 0:
                min_val = np.min(movie)
                movie = movie - min_val
            if np.min(movie) == 0:
                movie = movie + 1
            
            if isinstance(movie, np.ndarray):
                print("Ensuring that movie is turned back into a caiman object, not a numpy array...")
                movie = cm.movie(movie, fr=self.frame_rate)
            
            processed_movie, _ = cm_movie.computeDFF(movie, secs_window, quantile_min, method)
            print("Computing df over f / sqrt f was successful")
            return processed_movie
        
        except:
            raise ValueError("Computing df over f failed, returning original movie")
            return movie

