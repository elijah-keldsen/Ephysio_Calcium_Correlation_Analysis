import caiman as cm

# Functions that don't apply to a single experiment
def find_same_neurons(session_list, FOV_dims, template_list = None, background=None, plot_results=False):
    '''
    Tracks ROIs across multiple imaging sessions
    
    Args:
        session_list: A numpy array of spacial footprints (estimates.A) from each session
        template_list: A numpy array of one frame from each session's movie
        FOV_dims: Dimensions of the field of view as a tuple
        background: If there are only two sessions being compared and plot_results is true,
                    this is the background that results will be plotted over.
        plot_results: If only two sessions are being compared, set true if you want the results to be plotted
        
    Returns:
        If only 2 sessions:
            matched_ROIs1: list
                indices of matched ROIs from session 1
            matched_ROIs2: list
                indices of matched ROIs from session 2
            non_matched1: list
                indices of non-matched ROIs from session 1
            non_matched2: list
                indices of non-matched ROIs from session 2
            performance:  list
                (precision, recall, accuracy, f_1 score) with A1 taken as ground truth
            A2: csc_matrix  # pixels x # of components
                ROIs from session 2 aligned to session 1
                
        If more than 2 sessions:
            A_union: csc_matrix # pixels x # of total distinct components
                union of all kept ROIs 
            assignments: ndarray int of size # of total distinct components x # sessions
                element [i,j] = k if component k from session j is mapped to component
                i in the A_union matrix. If there is no much the value is NaN
            matchings: list of lists
                matchings[i][j] = k means that component j from session i is represented
                by component k in A_union
    '''
    print('Finding common neurons between recordings...')
    if session_list.size > 2:
        return cm.base.rois.register_multisession(session_list, FOV_dims, templates=template_list)
    else:
        if template_list is not None:
            return cm.base.rois.register_ROIs(session_list[0], session_list[1], FOV_dims, template1=template_list[0], template2=template_list[1], Cn=background, plot_results=plot_results)
        else:
            return cm.base.rois.register_ROIs(session_list[0], session_list[1], FOV_dims, Cn=background, plot_results=plot_results)