# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:09:49 2022

@author: 6600K-PC
"""
import os
import random
import gc
import datetime
import urllib.request
from IPython.display import display
import ipywidgets as widgets
import skimage.io as skio
import pandas as pd

from . import gen_czd_utils

def gen_data_load_interface(root_proj_dir, mutable_list,
                            mutable_output_data_dict,
                            name_splitting_fxs_dict):
    """ALlow users to select options for single-shot image dataset import via
       Ipython widget display in Notebook.

    Parameters
    ----------
    root_proj_dir : str
        Path to current project directory.
    mutable_list : list
        An empty list that will be passed into the function and edited in place
        (pushes user inputs to global scope).
    mutable_output_data_dict : dict
        An empty dict that will be passed into the function and edited in place
        (pushes outputs from data loading to global scope).
    name_splitting_fxs_dict : dict(str: function)
        A dict (see gen_filename_fxns.default_name_fxns) with functions for
        extracting sample, shot info from image filenames.

    Returns
    -------
    None.

    """

    align_options = {'.Align files': [True, False],
                     'sample_info.csv': [False, True],
                     'sample_info.csv if .Align not found': [True, True],
                     'Use default scale (1 μm/pixel)': [False, False]}
    folder_type_options = {'Option A (sample info in scan image filenames)': 'A',
                           'Option B (scan images organized into sample folders)': 'B'}

    folder_option_keys = list(folder_type_options.keys())
    folder_type_dropdown = widgets.Dropdown(options=folder_option_keys,
                                            value=folder_option_keys[0],
                                            layout={'width': 'max-content'})
    hbx1 = widgets.HBox([widgets.Label('Project folder format:'),
                       folder_type_dropdown])

    name_split_keys = list(name_splitting_fxs_dict.keys())
    process_name_dropdown = widgets.Dropdown(options=name_split_keys,
                                             value=name_split_keys[0],
                                             layout={'width': 'max-content'},
                                             disabled = False)
    hbx2 = widgets.HBox([widgets.Label('Select function for extracting image sample, scan names:'),
                       process_name_dropdown])

    image_type = widgets.Text(value = '.png')
    hbx3 = widgets.HBox([widgets.Label('Image file type (include "."):'),
                         image_type])

    align_opts_keys = list(align_options.keys())
    align_opts_dropdown = widgets.Dropdown(options=align_opts_keys,
                                           value=align_opts_keys[0],
                                           layout = {'width': 'max-content'})
    hbx4 = widgets.HBox([widgets.Label('Scale(s) from:'), align_opts_dropdown])

    load_samples_button = widgets.Button(description = 'Click to import dataset',
                                         button_style = '')
    out_button = widgets.Output()
    ui = widgets.VBox([hbx1, hbx2, hbx3, hbx4, load_samples_button])
    update_args_dict= {'dirtype': folder_type_dropdown,
                       'proc_fxn':  process_name_dropdown,
                       'img_str':  image_type,
                       'align_opts': align_opts_dropdown}
    def toggle_proc_dropdown(ftype_val):
        if 'Option A' in ftype_val:
            process_name_dropdown.disabled = False
        else:
            process_name_dropdown.value = name_split_keys[-1]
            process_name_dropdown.disabled = True
    def update_proc_list(**kwargs):
        mutable_list.clear()
        for key in kwargs:
            if key == 'dirtype':
                mutable_list.append(folder_type_options[kwargs[key]])
            elif key == 'proc_fxn':
                mutable_list.append(name_splitting_fxs_dict[kwargs[key]])
            elif key == 'align_opts':
                mutable_list.append(align_options[kwargs[key]])
            else:
                mutable_list.append(kwargs[key])
    def load_on_click(button_obj):
        out_button.clear_output()
        with out_button:
            print('Starting dataset loading')
            #pass mutable list as some args to load_gen_data_dict
            out_dict = gen_czd_utils.load_gen_data_dict(root_proj_dir,
                                                        *mutable_list)
            mutable_output_data_dict.update(out_dict)
            print('Move to next cell to select samples for processing')
    load_samples_button.on_click(load_on_click)
##  previously, this widget turned off the splitting function selector \
##  when folder format B was selected. No longer neccessary, splitting \
##  function sample_name outputs are simply ignored.
#  out_toggle = widgets.interactive_output(toggle_proc_dropdown,
#                                          {'ftype_val': folder_type_dropdown})

    out = widgets.interactive_output(update_proc_list, update_args_dict)

    display(ui, out, out_button)
