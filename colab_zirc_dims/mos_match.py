#!/usr/bin/env python
# coding: utf-8

# In[7]:

import os

from . import czd_utils

def get_mos_bounds(align_file_path):
    """Get bounding coordinates for a mosaic image from .Align file

    Parameters
    ----------
    align_file_path : str
        Full path to the .Align file for which mosaic bounds will be returned

    Returns
    -------
    list
        Bounds of mosaic corresponding to input .Align file:
            [min_x, max_x, min_y, max_y]

    """
    Align_data = czd_utils.get_Align_center_size(align_file_path)

    align_x_center, align_y_center, align_x_size, align_y_size = Align_data

    min_x, max_x = align_x_center - align_x_size/2, align_x_center + align_x_size/2
    min_y, max_y = align_y_center - align_y_size/2, align_y_center + align_y_size/2
    return [min_x, max_x, min_y, max_y]


def check_shots_vs_bounds(shot_dict, mosaic_bounds, max_out_of_bounds = 3):
    """Checks whether all but *max_out_of_bounds* shots are within mosaic bounds

    Parameters
    ----------
    shot_dict : dict
        A dictionary (see czd_utils.scancsv_to_dict()) with coordinates of all
        shots in a .scancsv file:
            {shot: [x_coords, y_coords], ...}
    mosaic_bounds : list
        A list of bounds to a .Align file (see get_mos_bounds()):
            [min_x, max_x, min_y, max_y]
    max_out_of_bounds : int, optional
        Max number of out-of-bounds shots allowed for a \
        'match' between mosaic and .scancsv. The default is 3.

    Returns
    -------
    Boolean
        True or False, depending on whether all but *max_out_of_bounds* \
        shots are within mosaic bounds

    """
    total_out_of_bounds = 0

    min_x, max_x, min_y, max_y = mosaic_bounds

    for eachcoords in shot_dict.values():
        if not min_x <= eachcoords[0] <= max_x or not min_y <= eachcoords[1] <= max_y:
            total_out_of_bounds += 1

    return total_out_of_bounds <= max_out_of_bounds


def check_scan_mos_matches(scancsv_dir, mos_dir, max_out_of_bounds=1):
    """Checks which scancsv files in one dir have shots \
        within bounds of mosaic .Align files in another dir.

    Parameters
    ----------
    scancsv_dir : str
        Full path to directory containing .scancsv files to check
    mos_dir : str
        Full path to directory containing mosaic (.bmp) and \
        corresponding alignment (.Align) files to check
    max_out_of_bounds : int
        Max number of 'out of bounds' shots before a mosaic is not a match. \
        The default is 1.

    Returns
    -------
    matches_dict : dict
        A dictionary with .scancsv files and possible matching .bmp mosaic files:
            {'example.scancsv': ['possible_match_1.bmp', 'possible_match_2.bmp'...], ...}

    """
    scancsv_file_list = [file for file in os.listdir(scancsv_dir) if file.endswith('.scancsv')]
    mos_file_list = [file for file in os.listdir(mos_dir) if file.endswith('.Align')]

    #creates a dict with all shots
    all_shots_dict = {}
    for each_scan_file in scancsv_file_list:
        all_shots_dict[each_scan_file] = czd_utils.scancsv_to_dict(os.path.join(scancsv_dir,
                                                                                each_scan_file))

    #creates a dict with all mosaic bounds
    all_mos_bounds_dict = {}
    for each_mos_file in mos_file_list:
        all_mos_bounds_dict[each_mos_file] = get_mos_bounds(os.path.join(mos_dir,
                                                                         each_mos_file))

    #creates a matches dict
    matches_dict = {}

    #loops through shotlists and gets possible matches
    for each_scancsv, each_shotlist in all_shots_dict.items():
        each_pos_matches = []
        for each_mos_file, each_mos_bounds in all_mos_bounds_dict.items():
            if check_shots_vs_bounds(each_shotlist, each_mos_bounds,
                                     max_out_of_bounds=int(max_out_of_bounds)):
                each_pos_matches.append(each_mos_file.strip('Align') + 'bmp')
        matches_dict[each_scancsv] = each_pos_matches

    return matches_dict

def matches_to_mos_info(input_matches_dict, def_sample_name = '',
                        def_zirc_size = 500, def_x_offset = 0,
                        def_y_offset = 0):
    """Converts a dict returned by check_scan_mos_matches() to a dict\
        with values and format {header: [data]} of a mos_info.csv file.

    Parameters
    ----------
    input_matches_dict : dict
        A dict of .scancsv files and corresponding possible .bmp files, \
        as returned by check_scan_mos_matches().
    def_sample_name : str, optional
        Default name for samples. The default is ''.
    def_zirc_size : int, optional
        Default max size for zircon crystals (µm). The default is 500.
    def_x_offset : int, optional
        Default x offset correction (µm) for scancsv files. The default is 0.
    def_y_offset : int, optional
        Default y offset correction (µm) for scancsv files. The default is 0.

    Returns
    -------
    new_mos_info_dict : dict
        A dict with format {header: [data]} (see below for headers). \
        Headers are those needed in mosaic_info.csv files for automatic \
        processing. The default .bmp files for each scancsv are simply \
        the first match found, so these should be verified by users. \
    act_matches_dict : dict
        A version of input_matches_dict with all samples without matches \
        removed.

    """
    #removes samples without matches
    no_matches_list = []
    act_matches_dict = {}
    for each_key, each_list in input_matches_dict.items():
        if each_list:
            act_matches_dict[each_key] = each_list
        else:
            no_matches_list.append(str(each_key))
    if no_matches_list:
        print('No matches found:', ', '.join(no_matches_list))


    headers = ['Sample', 'Scanlist', 'Mosaic',
               'Max_zircon_size', 'X_offset', 'Y_offset']

    def_exp_vals = {'Sample': def_sample_name,
                    'Max_zircon_size': def_zirc_size,
                    'X_offset': def_x_offset,
                    'Y_offset': def_y_offset}

    #creates new dict using new vals
    new_mos_info_dict = {}
    for each_column in headers:
        new_mos_info_dict[each_column] = []
        if each_column not in ['Scanlist', 'Mosaic']:
            for i in range(0, len(list(act_matches_dict.keys()))):
                new_mos_info_dict[each_column].append(def_exp_vals[each_column])
        if each_column == 'Scanlist':
            for each_scanlist in act_matches_dict:
                new_mos_info_dict[each_column].append(each_scanlist)
        if each_column == 'Mosaic':
            for each_mosaics in act_matches_dict.values():
                new_mos_info_dict[each_column].append(each_mosaics[0])
    return new_mos_info_dict, act_matches_dict