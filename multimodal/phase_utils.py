import numpy as np
from src2.shared import misc_functions
import pandas as pd


def ephys_phase_ca_events(ephys_idx_ca_events, channel_object, neurons='all'):
    """Compare calcium events to the phase extracted from a specified ephys channel.
    channel_object: channel object containing channel_name, signal, time_vector, smapling_rate, and the phases of the signal
    neurons: the neuron number (i.e., the integer row number in self.estimates.C, starting with 0) of the neuron you want to compare. If you want to compare all of the neurons in the recording, pass 'all' as the argument."""
    print('Comparing the calcium events to the corresponding phase of ' + channel_object.name + '...')
    ca_events_phases_ephys = {}
    if neurons == 'all':
        neurons = list(ephys_idx_ca_events.keys())
    elif type(neurons) != list:
        neurons = [neurons]
    for k in neurons:
        ca_events_phases_ephys[k] = []
        for j in range(len(ephys_idx_ca_events[k])):
            ca_events_phases_ephys[k].append(channel_object.phases[ephys_idx_ca_events[k][j]])
        ca_events_phases_ephys[k] = np.array(ca_events_phases_ephys[k])
        
    return ca_events_phases_ephys


def miniscope_phase_ca_events(ca_events_idx, miniscope_phases, neurons='all'):
    """Compare calcium events to the phase extracted from the mean fluorescence of the (cropped) miniscope recording.
    miniscope_phases: the phase calculatied from the movie that is given to MiniscopePostprocessor
    neurons: the neuron number (i.e., the integer row number in self.estimates.C, starting with 0) of the neuron you want to compare. If you want to compare all of the neurons in the recording, pass 'all' as the argument."""
    print('Comparing the calcium events to the corresponding phase of the mean fluorescence of the (cropped) miniscope recording...')
    ca_events_phases_miniscope = {}
    if neurons == 'all':
        neurons = list(ca_events_idx.keys())
    elif type(neurons) != list:
        neurons = [neurons]
    for k in neurons:
        ca_events_phases_miniscope[k] = []
        for j in range(len(ca_events_idx[k])):
            ca_events_phases_miniscope[k].append(miniscope_phases[ca_events_idx[k][j]])
        ca_events_phases_miniscope[k] = np.array(ca_events_phases_miniscope[k])
    
    return ca_events_phases_miniscope

    
    
def phase_ca_events_histogram(ca_events_phases, neurons='all', bins=18, hist_range=(-np.pi,np.pi), density=False, mean_density=False, combined=True, plot_histogram=True):
    """Compute the histogram of calcium events/probability density vs phase.
    Leave the parameters in their default values if you want to plot a combined histogram of phase counts of all neurons

    neuron: a list of the neuron indexes to compare. All neurons can be selected with 'all'.
    bins: the number of bins to sort the data into.
    hist_range: a tuple of the range that the bins should cover.
    density: determines whether the data will be bins will represent a probability density or a count.
    mean_density: provides the mean of the density histograms of all specified neurons.
    plot_histogram: chooses whether or not to plot the computed histogram.
    combined: a boolean that determines whether to combine the data from all of the specified neurons or whether to create histograms for each of the specified neurons.
    
    """
        
    if neurons != 'all' and not isinstance(neurons, list):
        neurons = [neurons]
        
    if not plot_histogram:
            #calculates histograms but does not plot them
            hist, bin_edges = _calculate_histograms_without_plotting(ca_events_phases, neurons, bins, hist_range, density, combined)
            return hist, bin_edges
        
    if mean_density and neurons == 'all':
        #plot the mean of the density histogram of phases of all neurons
        mean_neuron_vector_theta, mean_neuron_vector_radius = _mean_density_histogram(ca_events_phases, neurons, bins, hist_range)
        return hist, bin_edges, mean_neuron_vector_theta, mean_neuron_vector_radius

            
    if combined:
        
        if neurons == 'all':
            
            if density:
                #plot the probability density of phases of all neurons
                hist, bin_edges = _density_histogram(ca_events_phases, neurons, bins, hist_range)
            else:
                #plot the counts of each phase of all neurons
                hist, bin_edges = _counts_histogram(ca_events_phases, neurons, bins, hist_range)
                
        elif neurons != 'all':
            
            if density:
                #plot probability density of phases of your selected neurons WARNING THIS FUNCTION MAY NOT WORK AS INTENDED
                hist, bin_edges = _neuron_subset_density_histogram(ca_events_phases, neurons, bins, hist_range)
            else:
                #plot the counts of each phase of your selected neurons
                hist, bin_edges = _neuron_subset_counts_histogram(ca_events_phases, neurons, bins, hist_range)
                
    elif not combined:
        
        # Plot each of the neurons as separate histograms either all together or a subset of the neurons
        if neurons == 'all':
            hist, bin_edges = _individual_neuron_histograms(ca_events_phases, bins, hist_range, density)
        else:
            hist, bin_edges = _neuron_subset_individual_neuron_histograms(ca_events_phases, neurons, bins, hist_range, density)
    
    return hist, bin_edges
                    

#The following helper methods are all intended to compute histograms using the parameters of the above function
#This method directly below, in addition to computing and plotting a histogram, also computes mean vectors
def _mean_density_histogram(ca_events_phases, neurons, bins, hist_range):
    # Mean density histogram across neurons
    ca_events_phases_hist = np.empty((0, bins))
    mean_ca_events_vectors = np.empty((0, 2))
    mean_ca_events_vector_theta = []
    mean_ca_events_vector_radius = []
    for i, k in enumerate(list(ca_events_phases.keys())):
        hist, bin_edges = np.histogram(ca_events_phases[k], bins=bins, range=hist_range, density=True)
        ca_events_phases_hist = np.concatenate((ca_events_phases_hist, hist.reshape((1,-1))), axis=0)
        # Find the mean vector for each neuron's calcium events
        ca_events_vectors = np.concatenate((np.cos(ca_events_phases[k]).reshape((-1,1)), np.sin(ca_events_phases[k]).reshape((-1,1))), axis=1)
        mean_ca_events_vectors = np.concatenate((mean_ca_events_vectors, np.mean(ca_events_vectors ,axis=0).reshape((1,2))), axis=0)
        mean_ca_events_vector_theta.append(np.arctan2(mean_ca_events_vectors[i,1], mean_ca_events_vectors[i,0]))
        mean_ca_events_vector_radius.append(np.sqrt(mean_ca_events_vectors[i,0]**2 + mean_ca_events_vectors[i,1]**2))
    hist = np.mean(ca_events_phases_hist, axis=0) # Take the mean across the neurons at each bin.
    hist_error = np.std(ca_events_phases_hist, axis=0) / np.sqrt(np.shape(ca_events_phases_hist)[0]) # Take the standard error of the mean at each bin.
    h, ax = misc_functions._prepAxes(xLabel='Phase (rad)', yLabel='Mean Event Probability', title='Neuron(s): ' + str(neurons))
    ax.hist(bin_edges[:-1], bin_edges, weights=hist)
    bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    ax.errorbar(bin_midpoints, hist, yerr=hist_error, fmt='none', capsize=3)
    # Find the mean vector of all of the neurons
    mean_neuron_vector = np.mean(mean_ca_events_vectors, axis=0)
    mean_neuron_vector_theta = np.arctan2(mean_neuron_vector[1], mean_neuron_vector[0])
    mean_neuron_vector_radius = np.sqrt(mean_neuron_vector[0]**2 + mean_neuron_vector[1]**2)
    
    return hist, bin_edges, mean_neuron_vector_theta, mean_neuron_vector_radius
        
    
def _density_histogram(ca_events_phases, neurons, bins, hist_range):
    # Barstacked density histogram across neurons
    ca_events_phases_hist = list(ca_events_phases.values())
    h, ax = misc_functions._prepAxes(xLabel='Phase (rad)', yLabel='Event Probability', title='Neuron(s): ' + str(neurons))
    hist, bin_edges, _ = ax.hist(ca_events_phases_hist, bins=bins, range=hist_range, density=True, histtype='barstacked')
    return hist, bin_edges


def _counts_histogram(ca_events_phases, neurons, bins, hist_range):
    all_ca_events_phases = np.array([])
    for k in list(ca_events_phases.keys()):
        all_ca_events_phases = np.concatenate((all_ca_events_phases, ca_events_phases[k]))
    h, ax = misc_functions._prepAxes(xLabel='Phase (rad)', yLabel='Event Count', title='Neuron(s): ' + str(neurons))
    hist, bin_edges, _ = ax.hist(all_ca_events_phases, bins=bins, range=hist_range)
    return hist, bin_edges


def _neuron_subset_density_histogram(ca_events_phases, neurons, bins, hist_range):
    ca_events_phases_hist = {}
    for k in neurons:
        ca_events_phases_hist[k] = ca_events_phases[k]
    h, ax = misc_functions._prepAxes(xLabel='Phase (rad)', yLabel='Event Probability', title='Neuron(s): ' + str(neurons))
    hist, bin_edges, _ = ax.hist(ca_events_phases_hist, bins=bins, range=hist_range, density=True, histtype='barstacked')
    return hist, bin_edges


def _neuron_subset_counts_histogram(ca_events_phases, neurons, bins, hist_range):
    all_ca_events_phases = np.array([])
    for k in neurons:
        all_ca_events_phases = np.concatenate((all_ca_events_phases, ca_events_phases[k]))
    h, ax = misc_functions._prepAxes(xLabel='Phase (rad)', yLabel='Event Count', title='Neuron(s): ' + str(neurons))
    hist, bin_edges, _ = ax.hist(all_ca_events_phases, bins=bins, range=hist_range)
    return hist, bin_edges


def _individual_neuron_histograms(ca_events_phases, bins, hist_range, density):
    ax = []
    hist = {}
    bin_edges = {}
    for i, k in enumerate(list(ca_events_phases.keys())):
        if density:
            new_h, new_ax = misc_functions._prepAxes(xLabel='Phase (rad)', yLabel='Event Probability', title='Neuron(s): ' + str(k))
        else:
            new_h, new_ax = misc_functions._prepAxes(xLabel='Phase (rad)', yLabel='Event Count', title='Neuron(s): ' + str(k))
        ax.append(new_ax)
        hist[k], bin_edges[k], _ = ax[i].hist(ca_events_phases[k], bins=bins, range=hist_range, density=density)
    return hist, bin_edges


def _neuron_subset_individual_neuron_histograms(ca_events_phases, neurons, bins, hist_range, density):
    ax = []
    hist = {}
    bin_edges = {}
    for i, k in enumerate(neurons):
        if density:
            new_h, new_ax = misc_functions._prepAxes(xLabel='Phase (rad)', yLabel='Event Probability', title='Neuron(s): ' + str(k))
        else:
            new_h, new_ax = misc_functions._prepAxes(xLabel='Phase (rad)', yLabel='Event Count', title='Neuron(s): ' + str(k))
        ax.append(new_ax)
        hist[k], bin_edges[k], _ = ax[i].hist(ca_events_phases[k], bins=bins, range=hist_range, density=density)
    return hist, bin_edges


def _calculate_histograms_without_plotting(ca_events_phases, neurons, bins, hist_range, density, combined):
    if combined:
        all_ca_events_phases = np.array([])
        if neurons == 'all':
            for k in list(ca_events_phases.keys()):
                all_ca_events_phases = np.concatenate((all_ca_events_phases, ca_events_phases[k]))
            hist, bin_edges = np.histogram(all_ca_events_phases, bins=bins, range=hist_range, density=density)
        else:
            for k in neurons:
                all_ca_events_phases = np.concatenate((all_ca_events_phases, ca_events_phases[k]))
            hist, bin_edges = np.histogram(all_ca_events_phases, bins=bins, range=hist_range, density=density)
    else:
        # Plot each of the neurons as separate histograms
        hist = {}
        bin_edges = {}
        if neurons == 'all':
            for k in list(ca_events_phases.keys()):
                hist[k], bin_edges[k] = np.histogram(ca_events_phases[k], bins=bins, range=hist_range, density=density)
        else:
            for k in neurons:
                hist[k], bin_edges[k] = np.histogram(ca_events_phases[k], bins=bins, range=hist_range, density=density)
    
    return hist, bin_edges
        
        

    
    
    
    
    
    
    