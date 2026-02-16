from src2.shared.misc_functions import filterData


class FilterMiniscopeData:
    """Class in which to store filtered miniscope data during post-processing"""
    def __init__(self, projections, frame_rate, n=2, cut=[0.1,1.5], ftype='butter', btype='bandpass'):
        self.data = projections.time
        self.filtered_data = []
        self.frame_rate = frame_rate
        self.n = n
        self.cut = cut
        self.ftype = ftype
        self.btype = btype
    



    def filter_miniscope_data(self):
        self.filtered_data = filterData(self.data, n=self.n, cut=self.cut, ftype=self.ftype, btype=self.btype, fs=self.frame_rate)