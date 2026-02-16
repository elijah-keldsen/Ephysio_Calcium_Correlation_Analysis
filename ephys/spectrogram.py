import numpy as np
class Spectrogram():
    def __init__(self, psd_matrix_db: np.array, time_points: np.array, freq_points: np.array):
        self.psd_matrix_db = psd_matrix_db
        self.time_points = time_points
        self.freq_points = freq_points

