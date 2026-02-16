import numpy as np
import FreeSimpleGUI as sg
import matplotlib.pyplot as plt
import caiman as cm
import io
import base64
import sys
from src2.miniscope.projections import Projections
from PIL import Image


def _create_contour_fig(sfootprints, background, estimates_obj, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                         cmap=None, unselectcolor='w', selectcolor='r', coordinates=None,
                         contour_args={}, number_args={}):

    if thr is None:
        try:
            thr = {'nrg': nrgthr, 'max': maxthr}[thr_method]
        except KeyError:
            thr = maxthr
    else:
        thr_method = 'nrg'

    h, w = np.shape(background)
    dpi = 100 # Keep dpi consistent, it affects the final pixel dimensions of the saved image relative to figsize

    # Create figure and axes
    # fig, ax = plt.subplots(1, 1, figsize=(w/dpi, h/dpi), dpi=dpi) # Original line

    # FIX: Explicitly set the axes position to cover the entire figure
    fig = plt.Figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]) # [left, bottom, width, height] in figure coordinates (0 to 1)

    ax.imshow(background, interpolation='nearest', cmap=cmap)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect('equal', adjustable='box') # Keep aspect ratio equal

    # Remove axis ticks, labels, and borders
    ax.set_axis_off()
    # No need for fig.tight_layout here, as we've explicitly set the axes to fill the figure

    if coordinates is None:
        coordinates = cm.utils.visualization.get_contours(sfootprints, np.shape(background), thr, thr_method, swap_dim=False)

    bad_components_0based_set = set(estimates_obj.idx_components_bad)

    for c_idx, c in enumerate(coordinates):
        v = c['coordinates']
        component_id_1based = c.get('neuron_id') # This is the 1-based ID from get_contours
        if component_id_1based is None:
            continue

        component_id_0based_for_check = component_id_1based - 1

        is_bad = component_id_0based_for_check in bad_components_0based_set

        if is_bad:
            ax.plot(*v.T, c=selectcolor, **contour_args) # Red for rejected
        else:
            ax.plot(*v.T, c=unselectcolor, **contour_args) # White for good

    if display_numbers:
        d1, d2 = np.shape(background)
        d, nr = np.shape(sfootprints)
        comp = cm.base.rois.com(sfootprints, d1, d2)
        if max_number is None:
            max_number = sfootprints.shape[1]
        for i in range(np.minimum(nr, max_number)):
            # The 'i' here is already 0-based, corresponding to estimates.A columns
            num_color = selectcolor if i in bad_components_0based_set else unselectcolor
            ax.text(comp[i, 1], comp[i, 0], str(i + 1), color=num_color, **number_args)

    return fig


def _component_image(estimates, projections, movie, graph, max=False, min=False, STD=False, mean=False, median=False, range=False, cmap='viridis'):
    graph.erase()

    pic_IObytes = io.BytesIO()
    background_to_display = None
    if max:
        background_to_display = projections.max
    elif min:
        background_to_display = projections.min
    elif STD:
        background_to_display = projections.std
    elif mean:
        background_to_display = projections.mean
    elif median:
        background_to_display = projections.median
    elif range:
        background_to_display = projections.range

    if background_to_display is not None:
        if cmap not in plt.colormaps():
            print(f"Invalid colormap {cmap}, using 'viridis'")
            cmap = 'viridis'

        fig = _create_contour_fig(estimates.A, background_to_display, estimates, cmap=cmap)
        fig.savefig(pic_IObytes, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        pic_IObytes.seek(0)
        pic_data = pic_IObytes.read()
        if not pic_data:
            print("Error: No image data generated")
            return
        
        pic_hash = base64.b64encode(pic_data)

        graph.draw_image(data=pic_hash, location=(0, 0))

    else:
        print("No background to display")


def component_gui(movie, estimates, projections):
    if estimates.idx_components_bad is None:
        estimates.idx_components_bad = [] 

    print(f'This is the movie shape after processing: {movie.shape}')
    print(f'Initial estimates.idx_components_bad: {estimates.idx_components_bad}')
    
    cmapOptions = ['viridis', 'jet', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens',
                   'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray','bone','pink', 'spring', 'summer', 'autumn', 'winter', 'cool','Wistia', 'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu','RdYlBu','RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2','Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b','tab20c']

    initial_listbox_selections_1based = [idx + 1 for idx in estimates.idx_components_bad]


    
    layout = [[sg.Text('Components', key='-TITLE-')],
              [sg.Graph((movie.shape[2], movie.shape[1]), (0, movie.shape[1]), (movie.shape[2], 0), key='-GRAPH-',
                         enable_events=True)],
              [sg.Text("Projection Type:"),
               sg.Combo(['Max', 'Min', 'Mean', 'Median', 'STD', "Range"], key='-OPTION-', default_value='Max',
                                 readonly=True,
                                 auto_size_text=True, enable_events=True)],
              [sg.Text("CMAP:"), sg.Combo(cmapOptions, key='-CMAP-', default_value='viridis', readonly=True,
                                                 auto_size_text=True, enable_events=True)],
              [sg.Text("Select to reject: ")],
              [sg.Listbox(values=[i + 1 for i in range(len(estimates.C))], 
                           default_values=initial_listbox_selections_1based, 
                           size=(3, 3), key='-LISTCOMP-', select_mode='multiple', 
                           background_color="white", highlight_background_color="red", enable_events = True)],
              [sg.Button('Cancel', key="-CANCEL-"), sg.Button('Submit', key="-SUBMIT-")]]
    
    window = sg.Window('Components', layout, finalize=True, resizable=True,
                         element_justification='center', font='Helvetica 18')

    graph = window['-GRAPH-']
    
    plt.close('all') 
    
    # Initial drawing of the image
    event, values = window.read(timeout=100) 
    _component_image(estimates, projections, movie, graph, max=True, cmap=values['-CMAP-'])
    
    while True:
        event, values = window.read() 
        

        if event == '-LISTCOMP-':
            selected_gui_values_to_reject = np.array(values['-LISTCOMP-'], dtype=int)
            estimates.idx_components_bad = sorted(list(selected_gui_values_to_reject - 1)) 
    
            window['-LISTCOMP-'].update(set_to_index=[x for x in estimates.idx_components_bad], 
                                        scroll_to_index=estimates.idx_components_bad[0] if estimates.idx_components_bad else 0)
        if event == sg.WINDOW_CLOSED or event == '-CANCEL-':
            break

        proj_type_flags = {
            'max': False, 'min': False, 'std': False, 
            'mean': False, 'median': False, 'range': False
        }
        
        selected_proj = values['-OPTION-'].lower()
        
        if selected_proj in proj_type_flags:
            proj_type_flags[selected_proj] = True

        # Redraw the component image on any relevant event
        if event in ('-OPTION-', '-CMAP-', '-LISTCOMP-'):
            _component_image(
                estimates, projections, movie, graph, 
                max=proj_type_flags['max'], min=proj_type_flags['min'], 
                STD=proj_type_flags['std'], mean=proj_type_flags['mean'], 
                median=proj_type_flags['median'], range=proj_type_flags['range'], 
                cmap=values['-CMAP-']
            )

        elif event == '-SUBMIT-':
            selected_0_based_to_reject = set(estimates.idx_components_bad)
            all_indices_0_based = np.arange(len(estimates.C))
            good_components_indices = [idx for idx in all_indices_0_based if idx not in selected_0_based_to_reject]
            estimates = estimates.select_components(idx_components=good_components_indices)
            break
            
    plt.close('all')
    window.close()
    return estimates

        


    
    
    
    
    
    
def crop_gui(coords_dict, projections: Projections, movie_height, movie_width, previous_coords=None) -> dict:
    """
    Creates and handles all events for the pysimplegui cropping application.  Returns a dictionary of coordinates!
    """

    # The whole point of this function is to get the coordinates that will crop the movie

    # define the window layout
    cmapOptions = ['viridis', 'jet', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                  'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                  'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                  'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                  'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                  'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                  'tab20c']

    boxOptions = ['red/white', 'blue/white', 'red/yellow', 'blue/yellow', 'blue/green',
                  'green/yellow', 'red/green', 'green/white']

    layout = [[sg.Text('Max Projection', key='-TITLE-')],
              [sg.Graph((movie_width, movie_height), (0, 0), (movie_height, movie_width), key='-GRAPH-', drag_submits=True, enable_events=True)],
              [sg.Text("Start: None", key="-START-"), sg.Text("Stop: None", key="-STOP-"),
               sg.Text("Box: None", key="-BOX-")],
              [sg.Text("Projection Type:"), sg.Combo(['Max', 'Min', 'Mean', 'Median', 'STD', "Range"], key='-OPTION-', default_value='Max', readonly=True,
                        auto_size_text=True, enable_events=True)],
              [sg.Text("CMAP:"), sg.Combo(cmapOptions, key='-CMAP-', default_value='viridis', readonly=True,
                        auto_size_text=True, enable_events=True)],
              [sg.Text("Box Colors:"), sg.Combo(boxOptions, key='-COLORBOX-', default_value='red/white', readonly=True,
                                                auto_size_text=True, enable_events=True)],
              [sg.Button('Cancel', key="-CANCEL-"), sg.Button('Submit', key="-SUBMIT-")]]

    # create the form and show it without the plot
    window: sg.Window = sg.Window('CropGUI', layout, finalize=True, resizable=True,
                       element_justification='center', font='Helvetica 18')

    # add the plot to the window
    graph = window['-GRAPH-']
    x0, y0 = None, None
    colors = ['red', 'white']
    index = False
    box = None

    #adds image to window
    _update_image(graph, movie_height, projections.max,)
    if coords_dict is not None:
        try:
            #This code seems like we are changing the coords, but only temporarily so the rectangle is drawn correctly. This function correctly saves crop coords
            box = graph.draw_rectangle((coords_dict['x0'], coords_dict['y0']),
                                   (coords_dict['x1'], coords_dict['y1']),
                                   line_color=colors[index])
        except:
            print("Failed to draw intial box on GUI with the given coords")
    else:
        if coords_dict is None or not coords_dict:
            coords_dict = {
                'x0': 0,
                'y0': 0,
                'x1': movie_width,
                'y1': movie_height
            }

    while True:
        #controls events to update window
        event, values = window.read(timeout=100)

        if event == sg.WINDOW_CLOSED or event in '-CANCEL-':
            # Make sure that nothing gets cropped
            coords_dict['x0'] = 0
            coords_dict['y0'] = 0
            coords_dict['x1'] = 0
            coords_dict['y1'] = 0
            break

        #color of box options
        elif event in '-COLORBOX-':
            if values['-COLORBOX-'] == 'red/white':
                colors = ['red', 'white']
            elif values['-COLORBOX-'] == 'blue/white':
                colors = ['blue', 'white']
            elif values['-COLORBOX-'] == 'red/yellow':
                colors = ['red', 'yellow']
            elif values['-COLORBOX-'] == 'blue/yellow':
                colors = ['blue', 'yellow']
            elif values['-COLORBOX-'] == 'blue/green':
                colors = ['blue', 'green']
            elif values['-COLORBOX-'] == 'green/yellow':
                colors = ['green', 'yellow']
            elif values['-COLORBOX-'] == 'red/green':
                colors = ['red', 'green']
            elif values['-COLORBOX-'] == 'green/white':
                colors = ['green', 'white']
            # Redraw crop rectangle
            if box:
                graph.delete_figure(box)
            index = not index
            box = graph.draw_rectangle((coords_dict['x0'], coords_dict['y0']),
                                       (coords_dict['x1'], coords_dict['y1']),
                                       line_color=colors[index])

        #Type of image options
        elif event in '-OPTION-' or event in '-CMAP-':
            if event in '-OPTION-':
                window['-TITLE-'].update(values['-OPTION-'] + " Projection")

            if values['-OPTION-'] == 'Max':
                _update_image(graph, movie_height, projections.max, cmap=values['-CMAP-'])
            elif values['-OPTION-'] == 'Min':
                _update_image(graph, movie_height, projections.min, cmap=values['-CMAP-'])
            elif values['-OPTION-'] == 'STD':
                _update_image(graph, movie_height, projections.std, cmap=values['-CMAP-'])
            elif values['-OPTION-'] == 'Mean':
                _update_image(graph, movie_height, projections.mean, cmap=values['-CMAP-'])
            elif values['-OPTION-'] == 'Median':
                _update_image(graph, movie_height, projections.median, cmap=values['-CMAP-'])
            elif values['-OPTION-'] == 'Range':
                _update_image(graph, movie_height, projections.range, cmap=values['-CMAP-'])

            # Redraw crop rectangle
            if box:
                graph.delete_figure(box)
            index = not index
            box = graph.draw_rectangle((coords_dict['x0'], coords_dict['y0']),
                                       (coords_dict['x1'], coords_dict['y1']),
                                       line_color=colors[index])

        elif event in '-SUBMIT-':
            break

        #drawing the box and getting x/y values
        elif event in '-GRAPH-':
            if (x0, y0) == (None, None):
                x0, y0 = values['-GRAPH-']
                if values['-GRAPH-'][0] < 0:
                    x0 = 0
                if values['-GRAPH-'][0] > movie_width:
                    x0 = movie_width
                if values['-GRAPH-'][1] < 0:
                    y0 = 0
                if values['-GRAPH-'][1] > movie_height:
                    y0 = movie_height
            x1, y1 = values['-GRAPH-']
            if values['-GRAPH-'][0] < 0:
                x1 = 0
            if values['-GRAPH-'][0] > movie_width:
                x1 = movie_width
            if values['-GRAPH-'][1] < 0:
                y1 = 0
            if values['-GRAPH-'][1] > movie_height:
                y1 = movie_height
            coords_dict = _update_coords(window, x0, y0, x1, y1, coords_dict)
            if box:
                graph.delete_figure(box)
            if None not in (x0, y0, x1, y1):
                box = graph.draw_rectangle((x0, y0), (x1, y1), line_color=colors[index])
                index = not index
        elif event.endswith('+UP'):
             x0, y0 = None, None
    
    plt.close()
    window.close()

    return coords_dict


def _update_image(graph, movie_height, projection, cmap='viridis'):
    """
    Redraws the desired projection(image) to the pysimplegui graph object
    """
    # adds projection to GUI
    pic_IObytes = io.BytesIO()
    plt.imsave(pic_IObytes, projection, format='png', cmap=cmap)
    plt.close()
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    # Draw image in graph
    graph.draw_image(data=pic_hash, location=(0, movie_height))


def _update_coords(window, x0, y0, x1, y1, coords_dict) -> dict:
    """
    Update cropping rectangle information
    """

    if x0 is not None:
        coords_dict['x0'] = x0
    if y0 is not None:
        coords_dict['y0'] = y0
    if x1 is not None:
        coords_dict['x1'] = x1
    if y1 is not None:
        coords_dict['y1'] = y1
    window['-START-'].update(f'Start: ({x0}, {y0})')
    window['-STOP-'].update(f'Stop: ({x1}, {y1})')
    window['-BOX-'].update(f'Box: ({abs(x1 - x0 + 1)}, {abs(y1 - y0 + 1)})')

    return coords_dict
    
    
    

        
        