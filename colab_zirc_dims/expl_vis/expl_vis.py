# -*- coding: utf-8 -*-
"""
Code for running an exploratory dataset plotting ui in a Google Colab or
Jupyter notebook. Works with colab_zirc_dims grain_dimension project folders.
"""

import copy
import os
import random
import ipywidgets as widgets

from IPython.display import display
from matplotlib import pyplot as plt

from .. import czd_utils
from . import exp_plot_fxns

__all__ = ['czd_expl_plot_ui',
           'pick_run_dir_ui',
           'filter_data_ui',
           'exploratory_plot_ui']

def filter_data(input_dict, input_sel_samples, filter_failed, str_chckbox,
                inc_strs, exc_strs, tag_chckbox, tag_drop):
    """Filter a dataset dict (as returned by czd_utils.czd_csvs_to_dict())
       based on input arguments.

    Parameters
    ----------
    input_dict : dict
        Dict of format {sample1: {header1: [row1, row2,...],...}, ...}.
    input_sel_samples : list[str]
        Names of samples selected for inclusion in output dataset dict.
    filter_failed : bool
        If True, filter out dimensionless (e.g., "failed") analyses.
    str_chckbox : bool
        If True, filter dataset by analyses names based on args
        inc_strs and exc_strs.
    inc_strs : list[str]
        Strings to be used for inclusive filtering of dataset Analysis names.
    exc_strs : list[str]
        Strings to be used for exclusive filtering of dataset Analysis names.
    tag_chckbox : bool
        If True, filter dataset based on values in a 'tagged?' dataset column.
    tag_drop : str
        Indicates method for tag-based filtering if tag_chckbox is True.
        Self explanatory options: 'Filter out tagged scans' or
        'Filter out untagged scans'

    Returns
    -------
    output_dict : dict
        Dict of format {sample1: {header1: [row1, row2,...],...}, ...}. Only
        contains scans/samples that have survived filtering.
    samples_included : int
        Number of samples that survived filtering.
    samples_excluded : int
        Number of samples completely removed by filtering.
    spots_included : int
        Number of analytical spots that survived filtering.
    spots_excluded : int
        Number of analytical spots removed by filtering.

    """
    output_dict = {}
    input_dict_copy = copy.deepcopy(input_dict)
    samples_included = 0
    samples_excluded = 0
    spots_included = 0
    spots_excluded = 0
    #loop through selected samples
    for each_sample in [sample for sample in input_sel_samples
                        if sample in list(input_dict_copy.keys())]:
        each_sample_vals = input_dict_copy[each_sample]
        sample_added = False
        samples_excluded += 1
        for each_spot_idx, each_spot in enumerate(each_sample_vals['Analysis']):

            #filter out if dimensionless (i.e., 'failed' colab_zirc_dims measurement)
            filter_dimensions_pass = True
            if filter_failed:
                if float(each_sample_vals['Area (µm^2)'][each_spot_idx]) <= 0:
                    filter_dimensions_pass = False

              #inclusive, exclusive spot name filtering
            filter_str_inc_pass = True
            filter_str_exc_pass = True
            if str_chckbox:
                #inclusive filtering
                if all(inc_strs):
                    if not any([include_str in str(each_spot) for include_str in inc_strs]):
                        filter_str_inc_pass = False
                #exclusive filtering
                if all(exc_strs):
                    if any([exclude_str in str(each_spot) for exclude_str in exc_strs]):
                        filter_str_exc_pass = False

              #filtering based on user-added-scans
            filter_tag_pass = True
            if tag_chckbox:
                each_tagged_val = str(each_sample_vals['tagged?'][each_spot_idx])
                #inclusive filtering
                if tag_drop == 'Filter out tagged scans':
                    if each_tagged_val == 'True':
                        filter_tag_pass = False
                #exclusive filtering
                if tag_drop == 'Filter out untagged scans':
                    if each_tagged_val != 'True':
                        filter_tag_pass = False

            include_each_spot = all([filter_dimensions_pass,
                                     filter_str_inc_pass,
                                     filter_str_exc_pass,
                                     filter_tag_pass])
            if include_each_spot:
                if not sample_added:
                    output_dict[each_sample] = {eachkey: [] for eachkey in
                                                each_sample_vals.keys()}
                    samples_included += 1
                    samples_excluded -= 1
                    sample_added = True
                for eachkey in each_sample_vals.keys():
                    add_val = each_sample_vals[eachkey][each_spot_idx]
                    output_dict[each_sample][eachkey].append(add_val)
                spots_included += 1
            else:
                spots_excluded += 1

    #count spots not included in selected samples
    for each_sample in [sample for sample in input_dict_copy.keys()
                        if not sample in input_sel_samples]:
        samples_excluded += 1
        spots_excluded += len(input_dict_copy[each_sample]['Analysis'])

    return output_dict, samples_included, samples_excluded, spots_included, spots_excluded

def check_dataset_tagged(inpt_dataset):
    """Check if a dataset has a 'tagged?' column (i.e., if it is from manual
       segmentation using the colab_zirc_dims GUI).

    Parameters
    ----------
    inpt_dataset : dict
        Dict of format {sample1: {header1: [row1, row2,...],...}, ...}.

    Returns
    -------
    tag_available : bool
        True if 'tagged' column present, False if not or if dataset dict empty.

    """
    tag_available = False
    dataset_keys = list(inpt_dataset.keys())
    for key in dataset_keys[:1]:
        if 'tagged?' in list(inpt_dataset[key].keys()):
            tag_available = True
    return tag_available

def create_update_plot_color_dict(input_dataset, input_mutable_color_dict):
    """Create or update (in place) a dictionary with colors for each sample
       in the input_dataset dict.

    Parameters
    ----------
    input_dataset : dict
        Dict of format {sample1: {header1: [row1, row2,...],...}, ...}.
    input_mutable_color_dict : dict
        Either empty (for creation) or with format
        {sample1: 'hex_color_string', ...} for in-place addition
        of new samples. Edited in-place.

    Returns
    -------
    None.

    """
    copy_mutable = copy.deepcopy(input_mutable_color_dict)
    input_mutable_color_dict.clear()
    #use default matplotlib color cycle for initial sample colors
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for sample_idx, sample in enumerate(list(input_dataset.keys())):
        if sample not in copy_mutable.keys():
            #use default colors for 1st 10 samples if possible
            use_default = False
            if sample_idx < 10:
                if default_colors[sample_idx] not in list(copy_mutable.values()):
                    use_default = True

            if use_default:
                copy_mutable[sample] = default_colors[sample_idx]
            #otherwise, generate and use a random (hex) color
            else:
                pos_characters = '0123456789ABCDEF'
                rand_color = '#'+''.join(random.sample(pos_characters,6))
                copy_mutable[sample] = rand_color
    input_mutable_color_dict.update(copy_mutable)


def pick_run_dir_ui(mutable_imp_dict, cur_proj_dir = '', run_dir_str=None,
                    semi_run_dir_str = None):
    """Launch a UI for interactively picking and loading a dataset from a
       colab_zirc_dims directory using Jupyter widgets.

    Parameters
    ----------
    mutable_imp_dict : dict
        Mutable (edited in-place) dictionary where dataset dict will be loaded.
    cur_proj_dir : str, optional
        Directory where dataset(s) can be found in an 'outputs' subdirectory.
        The default is ''.
    run_dir_str : str, optional
        Directory name for a recent (i.e., from same notebook) fully-automated
        processing run. The default is None.
    semi_run_dir_str : str, optional
        Directory name for a recent (i.e., from same notebook) semi-automated
        processing run. The default is None.

    Returns
    -------
    None.

    """
    selection_dict = {['Most recent auto-processing run',
                      'Most recent semi-auto (GUI) processing run'][idx]: dir
                      for idx, dir in enumerate([run_dir_str, semi_run_dir_str])
                      if dir}
    available_dirs = [f.name for f in os.scandir(os.path.join(cur_proj_dir, 'outputs'))
                      if f.is_dir()]
    # a mutable list for communicating between widgets
    mutable_run_dir_lst = []

    dropdown_vals = [*list(selection_dict.keys()), 'Manual selection']
    cur_dirs_available = dropdown_vals[0] != 'Manual selection'
    manual_str = available_dirs[0]
    if cur_dirs_available:
        manual_str = selection_dict[dropdown_vals[0]]
    select_dirs_dropdown = widgets.Dropdown(options=dropdown_vals,
                                            value=dropdown_vals[0],
                                            layout = {'width': 'max-content'},
                                            disabled = not cur_dirs_available)
    Hbox1 = widgets.HBox([widgets.Label('Directory selection:'),
                          select_dirs_dropdown])
    manual_dir_entry = widgets.Dropdown(value=manual_str,
                                        options=available_dirs,
                                        layout = {'width': 'max-content'},
                                        disabled = cur_dirs_available)
    Hbox2 = widgets.HBox([widgets.Label('Manual directory selection:'),
                          manual_dir_entry])
    imp_button = widgets.Button(description='Import dataset',
                                disabled = all([manual_str]))
    out_button = widgets.Output()

    update_args_dict= {'dir_sel': select_dirs_dropdown,
                       'man_sel':  manual_dir_entry}
    def update_proc_list(**kwargs):
        mutable_run_dir_lst.clear()
        for key, val in kwargs.items():
            if key == 'dir_sel':
                if val == 'Manual selection':
                    manual_dir_entry.disabled = False
                else:
                    mutable_run_dir_lst.append(selection_dict[val])
                    manual_dir_entry.value = selection_dict[val]
                    manual_dir_entry.disabled = True
                    imp_button.disabled = False
            elif key == 'man_sel':
                if not manual_dir_entry.disabled:
                    #clear input if switching from dropdown selection to manual
                    if manual_dir_entry.value in list(selection_dict.values()):
                        imp_button.disabled = True

                    #only enable import button for manual selection if some\
                    # val entered
                    if len(manual_dir_entry.value) > 0:
                        mutable_run_dir_lst.append(manual_dir_entry.value)
                        imp_button.disabled = False
                    else:
                        imp_button.disabled = True
    def import_on_click(_):
        out_button.clear_output()
        entered_dir = mutable_run_dir_lst[0]
        mutable_imp_dict.clear()

        with out_button:
            print('Directory selected for import:', str(entered_dir))
            print('Importing...')
        imp_dict = czd_utils.czd_csvs_to_dict(os.path.join(cur_proj_dir,
                                                           'outputs',
                                                           entered_dir))
        mutable_imp_dict.update(imp_dict)
        with out_button:
            print('Directory imported; dimensions for',
                  str(len(list(imp_dict.keys()))), 'samples in directory.')
    imp_button.on_click(import_on_click)
    out = widgets.interactive_output(update_proc_list, update_args_dict)
    dir_select_desc = "".join(["Select an automated or semi-automated processing",
                               " directory (from current project directory) for",
                               " exploratory measurement visualization."])
    ui = widgets.VBox([widgets.HTML(value = '''<font size="+2"> <b>Dataset selection</b> </font>'''),
                       widgets.HTML(value = dir_select_desc),
                       Hbox1, Hbox2, imp_button])
    display(ui, out, out_button)


def filter_data_ui(mutable_full_dict, mutable_filtered_dict,
                   selected_samples, default_inc_strings=''):
    """Launch a UI for interactively filtering/slicing a dataset loaded from a
       colab_zirc_dims directory using Jupyter widgets.

    Parameters
    ----------
    mutable_full_dict : dict
        Dict of format {sample1: {header1: [row1, row2,...],...}, ...}.
        In practice, loaded and exposed within pick_run_dir_ui.
    mutable_filtered_dict : dict
        An empty (at input) dict object to hold filtering outputs (i.e., of
        mutable_full dict). Edited in place.
    selected_samples : list[str]
        List of samples that will be included in the filtered dict
        (i.e., mutable_filtered_dict).
    default_inc_strings : str, optional
        Default string for the inclusive filtering UI field. The default is ''.

    Returns
    -------
    None.

    """
    #full_dict_copy = copy.deepcopy(mutable_full_dict)
    failed_chckbox = widgets.Checkbox(value=True,
                                      description= 'Filter out dimensionless scans',
                                      disabled=False,
                                      indent=False)
    str_based_chckbox = widgets.Checkbox(value=True,
                                         description= 'Filter data by scan name',
                                         disabled=False,
                                         indent=False)
    include_strings = widgets.Text(value=default_inc_strings,
                                   disabled = not str_based_chckbox.value)
    inc_Hbox = widgets.HBox([widgets.Label('Plot only spots with names including:'),
                              include_strings])
    exclude_strings = widgets.Text(value='',
                                   disabled = not str_based_chckbox.value)
    exc_Hbox = widgets.HBox([widgets.Label('Plot only spots with names NOT including:'),
                              exclude_strings])
    str_filter_desc = ''.join(["Enter string(s) into the boxes below for inclusive (1st box)",
                               " and/or exclusive (2nd box) scan-name-based filtering",
                               " of the dataset for",
                               " visualization (e.g., to remove standards grains).",
                               " If you need to filter based on multiple strings,",
                               " enter them separated by comma(s).",
                               " Leave box empty to ignore."])
    str_filter_vbox = widgets.VBox([
                                    widgets.HTML(value = str_filter_desc),
                                    inc_Hbox,
                                    exc_Hbox])
    str_filter_vbox_full = widgets.VBox([str_based_chckbox, str_filter_vbox])

    tag_based_chckbox = widgets.Checkbox(value=False,
                                         description= 'Filter data by manual tags',
                                         disabled= not check_dataset_tagged(mutable_full_dict),
                                         indent=False)
    tag_filter_dropdown_vals = ['Filter out tagged scans',
                                'Filter out untagged scans']
    tag_opt = widgets.Dropdown(options=tag_filter_dropdown_vals,
                                       value=tag_filter_dropdown_vals[0],
                                       layout = {'width': 'max-content'})
    tag_Hbox = widgets.HBox([widgets.Label('Tag filtering:'),
                              tag_opt])
    tag_filter_desc = ''.join(["Select from the options below to define",
                               " how tagging (done manually by user during",
                               " semi-automated GUI segmentation) is handled",
                               " during filtering for dataset visualization."])
    tag_filter_vbox = widgets.VBox([widgets.HTML(value = tag_filter_desc),
                                    tag_Hbox])
    tag_filter_vbox_full = widgets.VBox([tag_based_chckbox, tag_filter_vbox])
    filter_args_io_dict = {'filter_failed': failed_chckbox,
                           'str_chckbox': str_based_chckbox,
                           'inc_strs': include_strings,
                           'exc_strs': exclude_strings,
                           'tag_chckbox': tag_based_chckbox,
                           'tag_drop': tag_opt}

    filter_fxn_args_dict = {'filter_failed': failed_chckbox.value,
                            'str_chckbox': str_based_chckbox.value,
                            'inc_strs': list(include_strings.value.split(',')),
                            'exc_strs': list(exclude_strings.value.split(',')),
                            'tag_chckbox': tag_based_chckbox.value,
                            'tag_drop': tag_opt.value}

    def handle_filtering_vis(chckbox_key, chkbox_val):
        box_objs = [str_filter_vbox, tag_filter_vbox]
        cur_obj = box_objs[['str_chckbox', 'tag_chckbox'].index(chckbox_key)]
        if chkbox_val is False:
            cur_obj.layout.display = 'none'
        else:
            cur_obj.layout.display = None

    def handle_filtering_out(**kwargs):
        test_toggle_tag_filter_disabled()
        for key, val in kwargs.items():
            if 'chckbox' in key:
                handle_filtering_vis(key, val)

            #update actual args dict
            if 'strs' in key:
                filter_fxn_args_dict[key] = list(val.split(','))
            else:
                filter_fxn_args_dict[key] = val


    filter_button = widgets.Button(description = 'Filter data')
    filter_output = widgets.Output()

    def test_toggle_tag_filter_disabled():
        if check_dataset_tagged(mutable_full_dict):
            tag_based_chckbox.disabled = False
        else:
            tag_based_chckbox.value = False
            tag_based_chckbox.disabled = True

    def filter_button_fxn(_):
        filter_output.clear_output()
        mutable_filtered_dict.clear()
        test_toggle_tag_filter_disabled()
        with filter_output:
            print('Filtering dataset...')
        filtering_output = filter_data(mutable_full_dict, selected_samples,
                                       **filter_fxn_args_dict)
        mutable_filtered_dict.update(filtering_output[0])
        n_filt_samples = filtering_output[1]
        n_total_samples = filtering_output[1] + filtering_output[2]
        n_filt_spots = filtering_output[3]
        n_total_spots = filtering_output[3] + filtering_output[4]
        with filter_output:
            print('Filtering done.')
            print('Dataset for plotting includes',
                  '{}/{} available samples,'.format(n_filt_samples, n_total_samples),
                  '{}/{} available analyses.'.format(n_filt_spots, n_total_spots))
    filter_button.on_click(filter_button_fxn)
    filter_data_desc = "".join(["Filter your loaded dataset for visualization.",
                                " Only samples that a) have measurements available",
                                " in the loaded directory and b) are selected in",
                                " the sample selection UI near the top of the notebook",
                                " (i.e., 'Data Loading') will be included for visualization. To",
                                " change sample selection, modify the sample selection",
                                " using that UI then, re-click the 'Filter data'",
                                " button."])
    ui = widgets.VBox([widgets.HTML(value = '''<br> <font size="+2"> <b>Dataset filtering</b> </font>'''),
                       widgets.HTML(value = filter_data_desc),
                       failed_chckbox,
                       str_filter_vbox_full,
                       tag_filter_vbox_full])


    out = widgets.interactive_output(handle_filtering_out, filter_args_io_dict)
    display(ui, out, filter_button, filter_output)


def exploratory_plot_ui(dataset_for_plot):
    """Launch a UI for exploratory plotting of a loaded, filtered
       colab_zirc_dims measurement dataset using Jypyter widgets
       and matplotlib.

    Parameters
    ----------
    dataset_for_plot : dict
        A dict that holds a dataset for plotting.
        Format will be {sample1: {header1: [row1, row2,...],...}, ...}.
        Can be empty on calling function (e.g., if it will be created
        in-place by filter_data_ui().

    Returns
    -------
    None.

    """

    exp_plot_expl_str = "".join(["After loading and filtering your dataset,",
                                 " use this interface to create and view",
                                 " exploratory plots of colab_zirc_dims",
                                 " per-scan grain measurement results. Plots",
                                 " will appear below the 'Create/update plot'",
                                 " button."])
    color_dict = {}
    plot_types = ['Bar-whisker', 'Histogram', 'X-Y']
    plot_type_dropdown = widgets.Dropdown(options = plot_types,
                                          value = plot_types[0],
                                          layout = {'width': 'max-content'})
    plot_type_Hbox = widgets.HBox([widgets.Label('Plot type:'),
                                   plot_type_dropdown])
    plot_dpi_entry = widgets.BoundedIntText(value=150,
                                            min=30,
                                            max=450,
                                            step=10,
                                            disabled=False,
                                            layout = {'width': '100px'})
    plot_dpi_Hbox = widgets.HBox([widgets.Label('Plot resolution (dpi):'),
                                  plot_dpi_entry])

    plottable_params = ['Area (µm^2)',
                        'Convex area (µm^2)',
                        'Eccentricity',
                        'Equivalent diameter (µm)',
                        'Perimeter (µm)',
                        'Major axis length (µm)',
                        'Minor axis length (µm)',
                        'Circularity']

    #bar-whisker plotting UI
    bp_expl_str = ''.join(["Use the interface below to parameterize a bar",
                           " and whisker plot of your dataset measurements.",
                           " The boxes on the resulting plot will extend from",
                           " Q1 to Q3, with whiskers from Q1 - 1.5 * (Q3 - Q1)",
                           " to Q3 + 1.5 * (Q3 + Q1). Medians will be indicated",
                           " by black horizontal lines, and individual",
                           " datapoints will be indicated by darker point",
                           " symbols with some random X scatter."])
    bp_param_dropdown = widgets.Dropdown(options=plottable_params,
                                         value='Major axis length (µm)',
                                         layout = {'width': 'max-content'})
    bp_param_Hbox = widgets.HBox([widgets.Label('Plot measurement:'),
                                 bp_param_dropdown])
    bp_manual_sort_chckbox = widgets.Checkbox(value=False,
                                              description="Custom sample ordering?",
                                              indent=False)
    bp_sort_param_dropdown = widgets.Dropdown(options=plottable_params,
                                              value=bp_param_dropdown.value,
                                              disabled = True,
                                              layout = {'width': 'max-content'})
    bp_sort_param_Hbox = widgets.HBox([widgets.Label('Order samples by median:'),
                                       bp_sort_param_dropdown])
    bp_sort_style_dropdown = widgets.Dropdown(options=['Ascending', 'Descending'],
                                              value='Ascending',
                                              disabled = True,
                                              layout = {'width': 'max-content'})
    bp_sort_style_Hbox = widgets.HBox([widgets.Label('Sort:'),
                                       bp_sort_style_dropdown])


    bp_args_io_dict = {'plot_param': bp_param_dropdown,
                       'sort_chckbox': bp_manual_sort_chckbox,
                       'sort_param': bp_sort_param_dropdown,
                       'sorting_style': bp_sort_style_dropdown}
    bp_fxn_args_dict = {'plot_param': bp_param_dropdown.value,
                        'sort_param': bp_sort_param_dropdown.value,
                        'sorting_style': bp_sort_style_dropdown.value}
    def handle_bp_ui(**kwargs):
        #print(kwargs)
        for key, val in kwargs.items():
            if key == 'sort_chckbox':
                if val is True:
                    bp_sort_param_dropdown.disabled = False
                    bp_sort_style_dropdown.disabled = False
                else:
                    bp_sort_param_dropdown.disabled = False
                    bp_sort_param_dropdown.value = bp_param_dropdown.value
                    bp_sort_param_dropdown.disabled = True

                    bp_sort_style_dropdown.disabled = False
                    bp_sort_style_dropdown.value = 'Ascending'
                    bp_sort_style_dropdown.disabled = True
            else:
                bp_fxn_args_dict.update({key: val})
                for manual_key, manual_widget in bp_args_io_dict.items():
                    if manual_key != 'sort_chckbox':
                        bp_fxn_args_dict[manual_key] = manual_widget.value


    bp_ui = widgets.VBox([widgets.HTML(value = '''<br> <font size="+1"><b>Bar-whisker plot options:</b> </font> <br>'''),
                          widgets.HTML(value=bp_expl_str),
                          bp_param_Hbox, bp_manual_sort_chckbox,
                          bp_sort_param_Hbox, bp_sort_style_Hbox])

    #histogram plotting UI
    hist_expl_str = ''.join(["Use the interface below to parameterize a histogram",
                             ' plot of your dataset measurements.'])
    hist_param_dropdown = widgets.Dropdown(options=plottable_params,
                                          value='Major axis length (µm)',
                                         layout = {'width': 'max-content'})
    hist_param_Hbox = widgets.HBox([widgets.Label('Plot measurement:'),
                                    hist_param_dropdown])
    hist_bin_sel = widgets.IntText(
                                    value=20,
                                    disabled=False,
                                   layout = {'width': '100px'}
                                  )
    hist_bin_sel_Hbox = widgets.HBox([widgets.Label('Number of bins:'),
                                    hist_bin_sel])
    hist_type_options = ['bar', 'barstacked']
    hist_type_dropdown = widgets.Dropdown(options=hist_type_options,
                                          value='bar',
                                         layout = {'width': 'max-content'})
    hist_type_Hbox = widgets.HBox([widgets.Label('Histogram type:'),
                                    hist_type_dropdown])
    hist_alpha_sel = widgets.BoundedFloatText(value=0.2,
                                              min=0,
                                              max=1.0,
                                              step=0.1,
                                              layout = {'width': '100px'})
    hist_alpha_Hbox = widgets.HBox([widgets.Label('Histogram fill alpha (transparency):'),
                                    hist_alpha_sel])

    hist_args_io_dict = {'plot_param': hist_param_dropdown,
                        'num_bins': hist_bin_sel,
                        'histtype': hist_type_dropdown,
                        'hist_alpha': hist_alpha_sel}
    hist_fxn_args_dict = {'plot_param': hist_param_dropdown.value,
                          'num_bins': hist_bin_sel.value,
                          'histtype': hist_type_dropdown.value,
                          'hist_alpha': hist_alpha_sel.value}
    def handle_hist_ui(**kwargs):
        #print(kwargs)
        for key, val in kwargs.items():
            hist_fxn_args_dict.update({key: val})
            for manual_key, manual_widget in hist_args_io_dict.items():
                hist_fxn_args_dict[manual_key] = manual_widget.value


    hist_ui = widgets.VBox([widgets.HTML(value = '''<br> <font size="+1"><b>Histogram plot options:</b> </font> <br>'''),
                            widgets.HTML(value = hist_expl_str),
                            hist_param_Hbox, hist_bin_sel_Hbox,
                            hist_type_Hbox, hist_alpha_Hbox])

    #X-Y plotting UI
    XY_expl_str = ''.join(["Use the interface below to parameterize an X-Y plot",
                           "of your dataset measurements."])
    XY_x_param_dropdown = widgets.Dropdown(options=plottable_params,
                                          value='Major axis length (µm)',
                                         layout = {'width': 'max-content'})
    XY_x_param_Hbox = widgets.HBox([widgets.Label('X axis measurement:'),
                                    XY_x_param_dropdown])
    XY_y_param_dropdown = widgets.Dropdown(options=plottable_params,
                                          value='Minor axis length (µm)',
                                         layout = {'width': 'max-content'})
    XY_y_param_Hbox = widgets.HBox([widgets.Label('Y axis measurement:'),
                                    XY_y_param_dropdown])
    XY_alpha_sel = widgets.BoundedFloatText(value=0.2,
                                            min=0,
                                            max=1.0,
                                            step=0.1,
                                            layout = {'width': '100px'})
    XY_alpha_Hbox = widgets.HBox([widgets.Label('Marker alpha (transparency):'),
                                    XY_alpha_sel])
    XY_ax_equal_chckbox = widgets.Checkbox(value=False,
                                           description="Equal plot axes?",
                                           indent=False)

    XY_args_io_dict = {'x_param': XY_x_param_dropdown,
                        'y_param': XY_y_param_dropdown,
                        'xy_alpha': XY_alpha_sel,
                        'equal_axes': XY_ax_equal_chckbox}
    XY_fxn_args_dict = {'x_param': XY_x_param_dropdown.value,
                        'y_param': XY_y_param_dropdown.value,
                        'xy_alpha': XY_alpha_sel.value,
                        'equal_axes': XY_ax_equal_chckbox.value}
    def handle_XY_ui(**kwargs):
        #print(kwargs)
        for key, val in kwargs.items():
            XY_fxn_args_dict.update({key: val})
            for manual_key, manual_widget in XY_args_io_dict.items():
                XY_fxn_args_dict[manual_key] = manual_widget.value


    XY_ui = widgets.VBox([widgets.HTML(value = '''<br> <font size="+1"><b>X-Y plot options:</b> </font> <br>'''),
                          widgets.HTML(value=XY_expl_str),
                          XY_x_param_Hbox, XY_y_param_Hbox,
                          XY_alpha_Hbox, XY_ax_equal_chckbox])

    plot_ui_io_dict = {'sel_plot': plot_type_dropdown}

    plot_ui_vis_dict = {'Bar-whisker': bp_ui, 'Histogram': hist_ui, 'X-Y': XY_ui}


    def handle_plot_ui_visibility(**kwargs):
        for _, val in kwargs.items():
            for ui_key, ui_obj in plot_ui_vis_dict.items():
                if val == ui_key:
                    ui_obj.layout.display = None
                else:
                    ui_obj.layout.display = 'none'

    ui = widgets.VBox([widgets.HTML(value = '''<br> <font size="+2"> <b>Exploratory plotting</b> </font>'''),
                       widgets.HTML(value = exp_plot_expl_str),
                       plot_type_Hbox, plot_dpi_Hbox, bp_ui, hist_ui, XY_ui])
    plot_output = widgets.Output()
    plot_button = widgets.Button(description = 'Create/update plot')

    def plot_button_fxn(_):
        plot_output.clear_output()
        if dataset_for_plot:
            #update 'color_dict' with any new vals from dataset
            create_update_plot_color_dict(dataset_for_plot, color_dict)
            plot_selection = plot_type_dropdown.value
            dpi_entered = plot_dpi_entry.value
            with plot_output:
                fig = None
                if plot_selection == 'Bar-whisker':
                    fig = exp_plot_fxns.plot_exp_bar_whisker(dataset_for_plot,
                                                             **bp_fxn_args_dict,
                                                             plot_dpi=dpi_entered)
                if plot_selection == 'Histogram':
                    fig = exp_plot_fxns.plot_exp_histogram(dataset_for_plot,
                                                           color_dict,
                                                           **hist_fxn_args_dict,
                                                           plot_dpi=dpi_entered)
                if plot_selection == 'X-Y':
                    fig = exp_plot_fxns.plot_exp_XY(dataset_for_plot,
                                                    color_dict,
                                                    **XY_fxn_args_dict,
                                                    plot_dpi=dpi_entered)
                if fig:
                    plt.show(fig)
        else:
            with plot_output:
                print('No dataset loaded; plotting unavailable.')
    plot_button.on_click(plot_button_fxn)

    plt_type_out = widgets.interactive_output(handle_plot_ui_visibility,
                                              plot_ui_io_dict)
    bp_out = widgets.interactive_output(handle_bp_ui, bp_args_io_dict)
    hist_out = widgets.interactive_output(handle_hist_ui, hist_args_io_dict)
    xy_out = widgets.interactive_output(handle_XY_ui, XY_args_io_dict)

    display(ui, plt_type_out, bp_out, hist_out, xy_out, plot_button,
            widgets.HTML(value = '''<br>'''),
            plot_output)



def czd_expl_plot_ui(project_dir, recent_auto_dir, recent_semi_auto_dir,
                     selected_samples, default_include_strings = ''):
    """Run a UI for loading, filtering, and plotting colab_zirc_dims automated
       or semi-automated measurments in a project directory.

    Parameters
    ----------
    project_dir : str
        Directory where dataset(s) can be found in an 'outputs' subdirectory.
    recent_auto_dir : str or None
        Directory name for a recent (i.e., from same notebook) fully-automated
        processing run. If not available, set to None.
    recent_semi_auto_dir : str or None
        Directory name for a recent (i.e., from same notebook) semi-automated
        processing run. If not available, set to None.
    selected_samples : list[str]
        List of samples that will be included in the filtered dict
        (i.e., created by filter_data_ui using user input).
    default_include_strings : str, optional
        Default string for the inclusive filtering UI field of filter_data_ui.
        The default is ''.

    Returns
    -------
    None.

    """

    mutable_full_dataset_dict, mutable_filt_dataset_dict = {}, {}
    pick_run_dir_ui(mutable_full_dataset_dict, project_dir, recent_auto_dir,
                    recent_semi_auto_dir)
    filter_data_ui(mutable_full_dataset_dict, mutable_filt_dataset_dict,
                   selected_samples, default_include_strings)
    exploratory_plot_ui(mutable_filt_dataset_dict)
