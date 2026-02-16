import os
import numpy as np
from src2.miniscope.miniscope_data_manager import MiniscopeDataManager
from src2.miniscope.miniscope_preprocessor import MiniscopePreprocessor
from src2.shared.path_finder import PathFinder
import caiman as cm
import sys
from src2.shared.misc_functions import updateCSVCell, get_coords_dict_from_analysis_params
from src2.shared.paths import ANALYSIS_PARAMS
from pathlib import Path

#35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 64, 83, 85, 86, 87, 88, 90, 92, 93, 94, 96, 97, 99, 101, 103, 104, 105, 107, 108, 112
line_nums = [97]
for line_num in line_nums:
    
    #display a gui for you to select how to crop the movie
    miniscope_data_manager = MiniscopeDataManager(line_num, filenames = ['10.avi', '11.avi', '12.avi', '13.avi', '14.avi', '15.avi', '16.avi', '17.avi', '18.avi', '19.avi', '20.avi'])
    coords_dict, _ = get_coords_dict_from_analysis_params(miniscope_data_manager, crop=True)
    
    preprocessor = MiniscopePreprocessor(miniscope_data_manager)
    projections = preprocessor.compute_projections(miniscope_data_manager.movie)
    movie, coords = preprocessor.crop_movie(miniscope_data_manager.movie, coords_dict, projections)
    
    #clean up the coords so they aren't in GUI format
    y0_flipped = miniscope_data_manager.movie.shape[1] - coords_dict['y1']
    y1_flipped = miniscope_data_manager.movie.shape[1] - coords_dict['y0']
    y0, y1 = sorted([y0_flipped, y1_flipped])
    x0, x1 = sorted([coords_dict['x0'], coords_dict['x1']])
    
    #get all avi files
    movie_directory = miniscope_data_manager.metadata['calcium imaging directory']
    movie_filepaths = PathFinder.find(movie_directory, suffix='.avi')
    print(movie_filepaths)
    
    sorted_movie_filepaths = sorted(movie_filepaths, key=lambda p: int(p.stem))
    
    print(sorted_movie_filepaths)
    
    
    mean_fluorescence = np.array([])
    
    #load each movie one at a time into memory, crop, and compress into mean fluorescence
    for movie_filepath in sorted_movie_filepaths:
        print("cropping this movie and compressing it: ", movie_filepath)
        
        movie = cm.load(str(movie_filepath))
        movie = movie[:, y0:y1, x0:x1]
        time_projection = movie.mean(axis=(1,2))
        mean_fluorescence = np.concatenate((mean_fluorescence, time_projection))
    
    #save time projection entire recording to disk on your desktop
    desktop = Path.home() / "Desktop"
    mean_fluorescence_folder = desktop / "meanFluorescence"
    mean_fluorescence_folder.mkdir(exist_ok=True)
    save_file = mean_fluorescence_folder / f"meanFluorescence_{line_num}.npz"
    np.savez_compressed(save_file, meanFluorescence=mean_fluorescence) #saves the time projection as an array to a key named 'meanFluorescence'