#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import caiman as cm
from caiman.base.movies import movie
import numpy as np
from src2.shared.path_finder import PathFinder
from src2.shared.experiment_data_manager import ExperimentDataManager

class MiniscopeDataManager(ExperimentDataManager):
    """Manages raw Miniscope data import and storage. Processes data via Processor."""
    
    def __init__(self, line_num: int, filenames: list=[], auto_import_data=True):
        super().__init__(line_num)
        self.line_num = line_num
        self.time_stamps: list = None
        self.frame_numbers: list = None
        self.all_movie_filepaths = self._find_movie_file_paths()
        self.chosen_movie_filepaths = self._get_specific_filepaths(filenames)
        
        if (auto_import_data):
            self.load_attributes(self.chosen_movie_filepaths if self.chosen_movie_filepaths else self.all_movie_filepaths)
            
        #Attributes below are filled in automatically during the miniscope_api pipeline: preprocessing->processing->postprocessing
        
        self.projections = None
        self.preprocessed_movie_filepath = None #Your preprocessed movie must be saved to disk and its filepath stored here before processing
        self.coords = None #contains the coordinates/shape of your cropped movie
        self.motion_corrected_movie_filepath = None
        self.CNMFE_obj = None
        self.estimates_filepath = None
        self.dview = None
        self.opts_caiman = None
        self.ca_events_idx = None
        self.PSD_spect = None
        self.t_spect = None
        self.freqs_spect = None
        self.p_spect = None
        self.miniscope_phases = None
        self.filter_object = None
            

    def load_attributes(self, filepaths):
        self.metadata.update(self._get_miniscope_metadata()) # add miniscope metadata to overall metadata
        self.time_stamps, self.frame_numbers = self._get_timestamps()  # import timestamps and frame numbers
        self.movie: movie = self._get_movies(filepaths)  # import calcium imaging data
        self.miniscope_events = self._get_miniscope_events()
        self.fr = self.metadata['frameRate']





    def convert_ca_movies(self, filenames=None, new_file_type='.tif', join_movies=False, metadata_convert=True):
        """
        Convert calcium movies from one type to another. File types must be supported by CaImAn.
        
        The new filename(s) is based on the first filename in 'filenames', with new_file_type appended.
        'join_movies' determines whether all movie files in 'filenames' are combined into a single movie,
        or whether each file is converted separately.
        
        If 'filenames' is None, the method will attempt to load filenames from self.movieFilePaths.
        """
        print("Converting movies...")
        original_filenames = filenames  # Preserve the original argument
        error_videos = []

        # If no filenames provided, try to load from self.movieFilePaths
        if filenames is None:
            if hasattr(self, 'movie'):
                print("self.movie exists, but no filenames were provided. Loading filenames from self.experiment['calcium imaging directory']")
            self.find_movie_file_paths()  # Assumes this method updates self.movieFilePaths
            filenames = self.movieFilePaths

        # Ensure filenames is a list
        if not isinstance(filenames, list):
            filenames = [filenames]

        # Use the first filename as a basis for the new filename.
        base_new_filename = os.path.splitext(filenames[0])[0]

        # If self.movie exists and no filenames were explicitly provided, use the existing movie.
        if hasattr(self, 'movie') and original_filenames is None:
            new_filename = f"{base_new_filename}{new_file_type}"
            self.movie.save(new_filename, compress=0)
        else:
            if join_movies:
                # Join movies: load the movie chain and save as a single new movie.
                try:
                    movies = cm.load_movie_chain(filenames)
                    new_filename = f"{base_new_filename}{new_file_type}"
                    movies.save(new_filename)
                except Exception as e:
                    print(f"Error converting joined movies: {e}")
                    error_videos.extend(filenames)
            else:
                # Convert each movie separately.
                for filename in filenames:
                    try:
                        # If the file doesn't exist, assume it might be in a default Miniscope directory.
                        if not os.path.isfile(filename):
                            default_dir = os.path.join(self.experiment['calcium imaging directory'], 'Miniscope')
                            filename = os.path.join(default_dir, filename)
                        movie = cm.load(filename)
                        new_filename = f"{os.path.splitext(filename)[0]}{new_file_type}"
                        movie.save(new_filename, compress=0)
                    except Exception as e:
                        print(f"Error converting movie '{filename}': {e}")
                        error_videos.append(filename)

        if metadata_convert:
            self._meta_data_converter()

        if error_videos:
            print(f"ERRORS with: {error_videos}")
            print("Consider investigating")


    def _meta_data_converter(self):  
        # Suggestion: it might be good to start combining the metaDatas and marking them with the animalID or some experiment identifier (not sure how this program will work if you have a lot of different videos/metaData)
        fileExts = self._find_metadata_paths()
        for fileExt in fileExts:
            with open(fileExt) as f:
                data = json.loads(f.read())
                if 'animalID' in data:
                    ext = fileExt.replace('\\metaData.json', '\\Miniscope\\metaData.json')
                    animalID = data['animalID']
                    timeStamp = data['recordingStartTime']
                    year = str(timeStamp['year'])
                    month = str('%02d' % timeStamp['month'])
                    day = str('%02d' % timeStamp['day'])
                    second = str('%02d' % timeStamp['second'])
                    minute = str('%02d' % timeStamp['minute'])
                    hour = str('%02d' % timeStamp['hr'])
                    date = year + month + day + '_' + hour + minute + second
                    with open(ext) as d:
                        data2 = json.loads(d.read())
                        if 'frameRate' in data2:
                            try:
                                frameRate = float(data2['frameRate'])
                            except ValueError:
                                frameRate = float(data2['frameRate'].replace('FPS', ''))
                        jdict = {'origin': animalID, 'fps': frameRate, 'date': date,
                                    'orig_meta': [data, data2]}
                        jsonFile = json.dumps(jdict, indent=4)
                        newFileName = ext.replace('\\metaData.json', '\\metaDataTif.json')
                        n = open(newFileName, 'w')
                        n.write(jsonFile)
                        n.close()


    def _get_miniscope_metadata(self) -> dict:
        """
        Imports miniscope metadata from a JSON file or multiple located at the paths returned by self._find_metadata_paths().
        Converts the 'frameRate' value to a float (removing any 'FPS' suffix if necessary).

        Returns:
            Dict: The metadata dictionary with a converted 'frameRate' value if present and converts any whole numbers to ints.
        """
        metadata_paths = self._find_metadata_paths()
        if not isinstance(metadata_paths, list):
            metadata_paths = [metadata_paths]
        print(f"Reading metadata from {metadata_paths}...")
        
        if metadata_paths[0] is None:
            print("No metadata paths found in your miniscope directory")
            return None
        
        
        for metadata_path in metadata_paths:
            try:
                with open(metadata_path, 'r') as file:
                    metadata = json.load(file)
            except (IOError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Error reading or parsing metadata file '{metadata_path}': {e}")
    
            # If 'frameRate' exists, try to convert it to a float
            if 'frameRate' in metadata:
                value = metadata['frameRate']
                try:
                    metadata['frameRate'] = float(value)
                except ValueError:
                    # Remove 'FPS' and any surrounding whitespace, then convert again
                    cleaned_value = value.replace('FPS', '').strip()
                    try:
                        metadata['frameRate'] = float(cleaned_value)
                    except ValueError:
                        raise ValueError(f"Unable to convert frameRate value '{value}' to float.")
        return metadata


    def _get_timestamps(self):
         #print('Reading miniscope software timestamps from ' + os.path.abspath(timeStampsFilename) + '...')
        file_path = self._find_timestamps_path()
        time_stamps = []
        frame_numbers = []
        with open(file_path, newline='') as t:
            next(t)
            reader = csv.reader(t)
            for row in reader:
                frame_numbers.append(int(row[0]))
                time_stamps.append(float(row[1]))
        time_stamps = np.divide(np.asarray(time_stamps), 1000)  # convert from ms to s
        return time_stamps, frame_numbers


    def _get_movies(self, filenames=None):
        """Import calcium imaging data. Not necessary if using processCaMovies().
        FILENAMES can be a single movie file or a list of movie files (in the order that you want them). 
        If FILENAMES doesn't point to a file (either absolute or relative path from the PWD), 
        it will append the path to the calcium imaging directory to the front of the filename."""
        
        print(f"Converting these filepaths into caiman movies: {filenames}")
        if filenames == None:
            filenames = self.all_movie_filepaths

        # Convert PosixPath objects to strings if necessary.
        if isinstance(filenames, list):
            filenames = [str(f) for f in filenames]
        else:
            filenames = str(filenames)
        
        print(f"Converting these filepaths into caiman movies: {filenames}")
        # Load a movie chain if there are multiple files, otherwise load a single movie.
        if type(filenames) is list:
            print(filenames)
            movie = cm.load_movie_chain(filenames)
        else:
            movie = cm.load(filenames)

        return movie
    
    def _get_specific_filepaths(self, filenames):
        if filenames is None or not isinstance(filenames, list) or len(filenames) == 0:
            return None
        
        matched_paths = []
        for path in self.all_movie_filepaths:
            basename = os.path.basename(path)  # Extract basename (e.g., '0.avi' from '/path/to/0.avi')
            if basename in filenames:
                matched_paths.append(path)
        return matched_paths


    def _get_miniscope_events(self):
        """Import calcium imaging experiment events."""
        miniscope_events_filepaths = PathFinder.find(self.metadata['calcium imaging directory'], '.csv', 'notes')
        
        if miniscope_events_filepaths is not None and len(miniscope_events_filepaths) == 1:
            miniscope_events_filepath = str(miniscope_events_filepaths[0])
        else:
            raise ValueError('Found zero or multiple event files')
            
        miniscope_events = {}
        miniscope_events['timestamps'] = []
        miniscope_events['labels'] = []
        try:
            with open(miniscope_events_filepath, newline='') as t:
                next(t)
                reader = csv.reader(t)
                for row in reader:
                    miniscope_events['timestamps'].append(int(row[0]))
                    miniscope_events['labels'].append(row[1])
            miniscope_events['timestamps'] = np.divide(np.asarray(miniscope_events['timestamps']), 1000)  # converts from ms to s
        except:
            print("Failed to extract events from notes.csv. Storing an empty dictionary in miniscope_dm.miniscope_events...")
        return miniscope_events
            
    



    @property
    def _calcium_imaging_directory(self):
        """Returns the directory where calcium imaging data is stored."""
        return self.metadata['calcium imaging directory']

    def _find_file_paths(self, suffix: str, prefix: str = ""):
        """Generalized helper to find files with the given suffix and prefix."""
        filepaths = PathFinder.find(directory=self._calcium_imaging_directory, suffix=suffix, prefix=prefix)
        
        #handle the case where filepaths is a list with only one item
        if isinstance(filepaths, list) and len(filepaths) == 1:
            filepaths = str(filepaths[0])
            
        return filepaths

    def _find_metadata_paths(self) -> list:
        """Finds and returns the metadata JSON file path."""
        return self._find_file_paths(suffix=".json", prefix="metaData")

    def _find_timestamps_path(self) -> str:
        """Finds and returns the timestamps CSV file path."""
        return self._find_file_paths(suffix=".csv", prefix="timeStamps")

    def _find_movie_file_paths(self) -> list:
        """Finds and returns the list of movie file paths (.avi)."""
        return self._find_file_paths(suffix=".avi")
        
    def _extract_numeric_suffix(self, filename: str) -> str:
        """
        Extracts and returns the substring of filename starting from the first digit.
        If no digit is found, returns the original filename.
        """
        for i, char in enumerate(filename):
            if char.isdigit():
                return filename[i:]
        return filename
        