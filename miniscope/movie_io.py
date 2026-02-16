import os
import caiman as cm

class MovieIO:
    @staticmethod
    def save_movie(dm, movie_file_name, movie=None): 
        """Saves the movie stored in dm.movie to disk, or you can pass in your own movie to save into the argument 'movie'"""
        
        # create the saved_movies directory if it doesn't exist
        miniscope_dir_path = dm.metadata['calcium imaging directory']
        
        saved_movies_dir = os.path.join(miniscope_dir_path, 'saved_movies')
        os.makedirs(saved_movies_dir, exist_ok=True)

        # create the filename
        file_name = os.path.join(saved_movies_dir, movie_file_name) + '.avi'

        # save movie
        print(f"saving movie: {file_name}\n\n")
        #uses caiman movie method .save() to save what is stored in data_manager.movie
        if movie is not None:
            movie.save(file_name, compress=0)
        else:
            dm.movie.save(file_name, compress=0)
        
        # return the full file path
        return file_name
    
    @staticmethod
    def load_movie(miniscope_dir_path, movie_file_name):
        # Load the movie from the specified file path
        path = os.path.join(miniscope_dir_path, 'saved_movies', movie_file_name)
        return cm.load(path)
    