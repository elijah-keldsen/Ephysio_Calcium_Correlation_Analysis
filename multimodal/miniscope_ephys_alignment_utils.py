import sys
import numpy as np

from src2.ephys.channel import Channel
from src2.ephys.ephys_data_manager import EphysDataManager
from src2.shared.path_finder import PathFinder


def sync_neuralynx_miniscope_timestamps(channel: Channel, miniscope_dm: EphysDataManager, delete_TTLs=True, fix_TTL_gaps=False, only_experiment_events=True):
    print('Syncing calcium movie times...')
    frame_acq_idx = (channel.events['labels'] == 'TTL Input on AcqSystem1_0 board 0 port 0 value (0x0000).') | (channel.events['labels'] == 'TTL Input on AcqSystem1_0 board 0 port 0 value (0x0001).')
    tCaIm = channel.events['timestamps'][frame_acq_idx]

    low_confidence_periods = np.empty((0,2))
    tCaIm, low_confidence_periods, miniscope_dm = _correct_tCaIm(channel.events['labels'][frame_acq_idx], tCaIm, low_confidence_periods, miniscope_dm, fix_TTL_gaps=fix_TTL_gaps)

        # make an array of the Neuralynx events with the TTL events removed
    if only_experiment_events:
        experiment_event_idx = np.invert(frame_acq_idx)
        channel.events['labels'] = channel.events['labels'][experiment_event_idx]
        channel.events['timestamps'] = channel.events['timestamps'][experiment_event_idx]
    
    # delete the TTL events that correspond to dropped frames in the saved calcium movie, specified in analysis_parameters.csv. This currently assumes that any gaps in the TTL events have been corrected already.
    #TODO Add a method that plots the 3 figures of the timestamps for help in deciding which events to drop, then writes to analysis_parameters.csv and self._analysisParamsDict['indices of TTL events to delete'].
    
    #make sure there are indices stored for deletion, then delete them from tCaIm
    if delete_TTLs and miniscope_dm.analysis_params.get('indices of TTL events to delete', None) is not None:
        tCaIm = np.delete(tCaIm, miniscope_dm.analysis_params['indices of TTL events to delete'])

    return tCaIm, low_confidence_periods, channel, miniscope_dm     


def _correct_tCaIm(event_labels, tCaIm, low_confidence_periods, miniscope_dm, threshold=0.065, fix_TTL_gaps=False):
    """This method first confirms that the TTL events alternate and then checks for missing TTL events. If there are any, the method guesses their timing and inserts them into the calcium imaging time vector.
    event_labels is the array of imported Neuralynx event labels.
    THRESHOLD is the time threshold, in seconds, for detecting gaps in the TTL events."""
    print('Checking that TTL events alternate...')
    # Print a message if the TTL event labels do not alternate between HIGH and LOW
    alternating = []
    for q in range(0,len(event_labels)-2):
        alternating.append(np.char.equal(event_labels[q+2], event_labels[q]))
    alternating.append(np.char.not_equal(event_labels[-1], event_labels[-2]))
    if sum(alternating) != (len(event_labels) - 1):
        print('TTL does not alternate!')
        sys.exit() # Exit program execution if this condition is reached.
    
    print('Finding any gaps in the TTL events...')
    dtCaIm = np.diff(tCaIm)
    idx_TTL_gap = np.where(dtCaIm > threshold)[0] # indices of gaps in the TTL events
    if len(idx_TTL_gap) == 0:
        print('No gaps were found with a threshold of ' + str(threshold*1000) + ' ms.')
    elif fix_TTL_gaps:
        print('Fixing any gaps in the TTL events...')
        flippedidx_TTL_gap = np.flip(idx_TTL_gap) # Reverse the order of idx_TTL_gap so that inserting TTLs in the loop doesn't affect the indices of the next iteration of the loop.
        gap_length = [] # number dropped frames per index
        for k, gap_idx in enumerate(flippedidx_TTL_gap):
            gap_length.append(round(dtCaIm[gap_idx]/(1/miniscope_dm.metadata['frameRate']))) # Guesses how many timesteps occur in the gap. E.g., a 30 Hz video with a gap of 67 ms will have 2 timesteps in the gap.
            print(str(gap_length[k]-1) + ' TTL event(s) is/are missing between index numbers ' + str(gap_idx) + ' and ' + str(gap_idx + 1) + '.')
            estimated_event_times = np.linspace(tCaIm[gap_idx], tCaIm[gap_idx+1], gap_length[k]+1) # Estimates the timing of the TTLs, beginning at the one before the gap and ending at the one after the gap.
            tCaIm = np.insert(tCaIm, gap_idx+1, estimated_event_times[1:-1])
            low_confidence_periods = np.append(low_confidence_periods, [[gap_idx, gap_idx+gap_length[k]]], axis=0)
    else:
        print('Gaps were found. Review tCaIm before proceeding.')
        sys.exit() # Exit program execution if this condition is reached.

    return tCaIm, low_confidence_periods, miniscope_dm


def find_ephys_idx_of_TTL_events(tCaIm, channel, frame_rate, ca_events_idx=None, all_TTL_events=True):
    """Finds the index of a calcium event in the Neuralynx timespace. If the miniscope class method to find the timing of calcium events has not been run yet, it runs that first.
    CHANNEL is the ephys channel with which to compare the timing of the ephys samples to the calcium event timing."""
    # Match up all calcium movie timestamps with their corresponding ephys timestamps.
    if all_TTL_events:
        print('Finding the indices of ephys timestamps that are closest to all calcium movie frame acquisition TTL events...')
        ephys_idx_all_TTL_events = np.empty(len(tCaIm),dtype=int)
        # Choose a number of indices after the last_index before which you are confident that the next index will be.
        # I am choosing the number of ephys indices during the time it takes for two calcium imaging frames.
        endPoint = round(int(channel.sampling_rate) * 2 / int(frame_rate))
        last_index = 0
        
        for k, CaIm_TTL_Event in enumerate(tCaIm):
            if k == 0:
                ephys_idx_all_TTL_events[k] = np.abs(channel.time_vector[last_index:] - CaIm_TTL_Event).argmin() + last_index
            elif len(channel.time_vector[last_index:]) - endPoint < 0:
                ephys_idx_all_TTL_events[k] = np.abs(channel.time_vector[last_index:] - CaIm_TTL_Event).argmin() + last_index
            else:
                ephys_idx_all_TTL_events[k] = np.abs(channel.time_vector[last_index:(last_index + endPoint)] - CaIm_TTL_Event).argmin() + last_index
            last_index = ephys_idx_all_TTL_events[k]
            
    # Look for the indices of the ephys timestamps that are closest to the calcium event (Neuralynx) timestamps.
    if ca_events_idx:
        print('Finding the indices of ephys timestamps that are closest to the calcium event (Neuralynx) timestamps...')
        ephys_idx_ca_events = {}
        for k in list(ca_events_idx.keys()):
            ephys_idx_ca_events[k] = []
            last_index = 0
            for j in range(len(ca_events_idx[k])):
                ephys_idx_ca_events[k].append(np.abs(channel.time_vector[last_index:] - tCaIm[ca_events_idx[k][j]]).argmin() + last_index)
                # Check to see if the gap between the calcium event time and the corresponding ephys timestamp is reasonable (within 1 frame's timestep).
                if np.abs(channel.time_vector[ephys_idx_ca_events[k][j]]-tCaIm[ca_events_idx[k][j]]) > (1/frame_rate):
                    print('There are no ephys timestamps closer to the calcium event timestamp than the duration of a calcium movie frame!')
                last_index = ephys_idx_ca_events[k][j]
            ephys_idx_ca_events[k] = np.array(ephys_idx_ca_events[k])
    
    return ephys_idx_all_TTL_events if all_TTL_events else None, ephys_idx_ca_events if ca_events_idx else None


def find_ca_movie_frame_num_of_ephys_idx(channel, ephys_idx_all_TTL_events): #TODO Make sure this method works correctly (e.g., it handles ephys indices before and after the start of the movie correctly), is placed in the most logical place in this file, and is fleshed out more.
    """Method to create an array the same size as obj.ephys[channel], where each element is the frame number of the corresponding calcium movie frame."""
    ca_frame_num_of_ephys_idx = np.zeros(np.shape(channel.signal),dtype=int)

    # Assign a frame number to each element of ca_frame_num_of_ephys_idx. I'm not sure if the sample of obj.ephys that's closest to the TTL event should be paired with the preceding frame or not.
    for k, i in enumerate(ephys_idx_all_TTL_events[1:]):
        ca_frame_num_of_ephys_idx[i:ephys_idx_all_TTL_events[k]:-1] = k+1
    
    return ca_frame_num_of_ephys_idx


def find_ca_movie_filenums(channel, ephys_idx_all_TTL_events, miniscope_dm, time_range=None):
    """Determine the calcium imaging movie file(s) that correspond to a specified time period (in seconds) in the electrophysiological signal.
    TIME_RANGE is a list specifing the boundaries of the time period.
    """
    if time_range == None:
        time_sec_start = miniscope_dm.analysis_params['periods of high slow wave power (s)'][0]
        time_sec_end = miniscope_dm.analysis_params['periods of high slow wave power (s)'][-1]
    else:
        time_sec_start = time_range[0]
        time_sec_end = time_range[1]

    print('Finding the miniscope frames and movie corresponding to the specified time period...')
    movie_frames = np.zeros(2, dtype=int)
    movie_frames[0] = np.where(channel.time_vector[ephys_idx_all_TTL_events]>=time_sec_start)[0][0] # Start frame
    movie_frames[1] = np.where(channel.time_vector[ephys_idx_all_TTL_events]<=time_sec_end)[0][-1] # End frame
    first_movie = int(movie_frames[0]/miniscope_dm.metadata['framesPerFile']) # Truncates result to just the integer part
    last_movie = int(movie_frames[1]/miniscope_dm.metadata['framesPerFile']) # Truncates result to just the integer part

    print('The first movie in the sequence is ' + str(first_movie) + '.avi.')
    print('The last movie in the sequence is ' + str(last_movie) + '.avi.')
    
    movie_range = tuple([str(x) for x in range(first_movie, last_movie+1)])
    movie_file_paths_in_this_range = PathFinder.find(directory=miniscope_dm.metadata['calcium imaging directory'], suffix=".avi", prefix=movie_range, file_and_directory=False)
    print(f'Found these movies in the specified range: {movie_file_paths_in_this_range}')
    return movie_file_paths_in_this_range, movie_frames
    
    