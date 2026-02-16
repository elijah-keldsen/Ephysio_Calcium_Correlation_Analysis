#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src2.shared.paths import ANALYSIS_PARAMS, EXPERIMENTS
import logging
from src2.shared.csv_worker import CSVWorker


class ExperimentDataManager:
    """
    This class manages the import and storage of metadata and analysis parameters.  Formerly the experiment class.

    metadata: a row, indexed by line_num of the experiments.csv document.  Each column is a key, and each cell is a value in the metadata dictionary.  Formerly self.experiment
    analysis_params: A similarly dictionary constructed dictionary, but with the corresponding row taken from analysis_parameters.csv.
    """

    def __init__(self, line_num, auto_import_metadata=True, auto_import_analysis_params=True, logging_level = "CRITICAL"):
        self.line_num = line_num
        self.metadata: dict = None
        self.analysis_params: dict = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        if auto_import_metadata:
            self.import_metadata()

        if auto_import_analysis_params:
            self.import_analysis_parameters()



    def import_metadata(self):
        metadata_unconverted = CSVWorker.csv_row_to_dict(EXPERIMENTS, self.line_num)
        metadata_converted = CSVWorker.convert_data_types(metadata_unconverted)
        self.metadata = metadata_converted

    def import_analysis_parameters(self):
        analysis_params_unconverted = CSVWorker.csv_row_to_dict(ANALYSIS_PARAMS, self.line_num)
        analysis_params_converted = CSVWorker.convert_data_types(analysis_params_unconverted)
        self.analysis_params = analysis_params_converted

    def get_ephys_directory(self):
        return self.metadata['ephys directory']
    
    def get_miniscope_directory(self):
        return self.metadata['calcium imaging directory']