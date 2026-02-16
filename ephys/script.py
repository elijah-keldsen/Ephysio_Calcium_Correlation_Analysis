from src2.ephys.channel_worker import ChannelWorker
from src2.ephys.ephys_data_manager import EphysDataManager
from src2.shared.experiment_data_manager import ExperimentDataManager
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# set the filter boolean based on if filter_type is None
line_num = 97
logging_level = "INFO"
remove_artifacts = False
filter_type = None  # if desired, enter the type, eg "butter"
filter_range = [0.5, 4]
plot_channel = True
plot_spectrogram = True
channel_name = 'PFCLFPvsCBEEG'

experiment_data_manager = ExperimentDataManager(line_num, logging_level = logging_level)
ephys_directory = experiment_data_manager.get_ephys_directory()
ephys_data_manager = EphysDataManager(ephys_directory, auto_import_ephys_block=True, auto_process_block=False)
ephys_data_manager.process_ephys_block_to_channels(remove_artifacts=remove_artifacts, channels = channel_name)
channel = ephys_data_manager.get_channel(channel_name)
channel_worker = ChannelWorker(channel)
channel_worker.plot_spectrogram(plot_events=True)   