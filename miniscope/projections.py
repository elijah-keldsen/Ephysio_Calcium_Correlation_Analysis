import numpy as np

class Projections:
    def __init__(self, max: np.ndarray, std: np.ndarray, min: np.ndarray, mean: np.ndarray, median: np.ndarray, range: np.ndarray, time: np.ndarray):
        self.max = max
        self.std = std
        self.min = min
        self.mean = mean
        self.median = median
        self.range = range
        self.time = time
        
        
        
    