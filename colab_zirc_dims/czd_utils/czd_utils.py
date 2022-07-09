#!/usr/bin/env python
# coding: utf-8

"""
Utilities and functions for working with ALC, other datasets. Includes functions
for reading .Align xml scaling files saved during LA-ICP-MS analysis and
calculating scale factors for images from .Align data.
"""

import os
import operator
import json
import copy

from urllib.request import urlopen

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


__all__ = ['check_url',
           'read_json',
           'save_json',
           'json_from_path_or_url',
           'save_csv',
           'list_if_endswith',
           'list_if_in',
           'check_any_str',
           'join_1st_match',
           'round_to_even',
           'list_of_val',
           'check_mos_csv_keys',
           'prediction_to_np',
           'mask_list_to_np',
           'rescale_2d_arr',
           'rotate_pt',
           'crop_nd_arr',
           'mask_to_3D_arr_size',
           'scancsv_to_dict',
           'get_Align_center_size',
           'calc_scale_factor',
           'load_data_dict',
           'alc_calc_scans_n']

### Various functions and classes for file processing and code simplification
### in other modules below.

def check_url(input_str):
    """Check if a string is a 'https' url.

    Parameters
    ----------
    input_str : str
        An input string or path.

    Returns
    -------
    bool
        True if input_string is url, else False.

    """
    return 'https' in str(input_str)

def read_json(json_path):
    """Read a .json file.

    Parameters
    ----------
    json_path : str
        Path to json file for reading.

    Returns
    -------
    js : any
        Contents of json file. For purposes of this package, this will be a list or dict.

    """
    with open(json_path, 'r') as f:
        js = json.load(f)
    return js

def read_json_url(json_url):
    """Read a .json file from a url.

    Parameters
    ----------
    json_url : str
        url to .json file for loading; if from Github should be 'raw' link.

    Returns
    -------
    Type: any
        Whatever data was retrieved from the .json file at the url (probably a
        list or dict for purposes of this package).

    """
    #get url data response
    url_data = urlopen(json_url)
    #return loaded json
    return json.loads(url_data.read())

def json_from_path_or_url(path_or_url_str):
    """Check if input str is a url and load .json from url if so. Else,
       assume that input str is a path and load a .json from that path.

    Parameters
    ----------
    path_or_url_str : str
        A url (with 'https') or a path to a .json file.

    Returns
    -------
    Type: any
        Whatever data was retrieved from the .json file at the url/path; likely
        a dict or list if called in this project.


    """
    if check_url(path_or_url_str):
        return read_json_url(path_or_url_str)
    #if not url, assumes path
    else:
        return read_json(path_or_url_str)


def save_json(json_path, item_for_save):
    """Save an item to a json file (will overwrite existing file with same path).

    Parameters
    ----------
    json_path : str
        Save path for json file.
    item_for_save : any
        Item for saving to json file. For purposes of this package, this will
        be a list or dict.

    Returns
    -------
    None.

    """
    if os.path.isfile(json_path):
        os.remove(json_path)
    with open(json_path, 'w') as f:
        json.dump(item_for_save, f)
    return

def save_csv(path, pandas_table):
    """Save a pandas table as a .csv file

    Parameters
    ----------
    path : str
        Full save path (including '.csv') for the pandas table.
    pandas_table : pandas DataFrame
        A pandas DataFrame with headers.

    Returns
    -------
    None.

    """
    pandas_table.to_csv(path, index=False, header=True, encoding='utf-8-sig')


def list_if_endswith(input_list, end_string):
    """Returns a list without input items that don't end with an input string.

    Parameters
    ----------
    input_list : list[str]
        A list of strings.
    end_string : str
        A string to check items in the list.

    Returns
    -------
    list
        A copy of input_list w/o strings that do not end w/ end_str.

    """
    return [val for val in input_list if val.endswith(end_string)]



def list_if_in(input_list, string_in):
    """Return a list without input items not containing an input string.

    Parameters
    ----------
    input_list : list[str]
        A list of strings.
    string_in : str
        A string to check items in the list.

    Returns
    -------
    list
        A copy of input list w/o strings not containing string_in.

    """
    return [string for string in input_list if string_in in string]


def check_any_str(list_to_check, input_string):
    """Check if any items in a list have a string in them.

    Parameters
    ----------
    list_to_check : list[str]
        A list of strings.
    input_string : str
        A string to check items in the list.

    Returns
    -------
    Boolean
        True or False, depending on whether input_string is found in >= list item.

    """
    return any(input_string in string for string in list_to_check)


def join_1st_match(input_list, input_string, input_join_path):
    """Check whether any items in a list contain a string; join first match
        to a directory if so.

    Parameters
    ----------
    input_list : list[str]
        A list of strings (ideally filenames).
    input_string : str
        A string (ideally a filename) to match.
    input_join_path : str
        A path to join the first match in input_list to.

    Returns
    -------
    output_file_pth : str
        A path *input_join_path*/first_match.

    """
    output_file_pth = ''
    if check_any_str(input_list, input_string):
        first_match = list_if_in(input_list, input_string)[0]
        output_file_pth = os.path.join(input_join_path, first_match)
    return output_file_pth


def round_to_even(number):
    """Round a number to the nearest even integer.

    Parameters
    ----------
    number : float or int
        A number for rounding.

    Returns
    -------
    Int
        Even integer rounded from *number*.

    """
    return round(float(number)/2)*2


def list_of_val(val_for_list, list_len, num_lists = 1):
    """Generate a list or list of lists containing a single value.

    Parameters
    ----------
    val_for_list : any
        Value that will be repeated in list or lists of lists.
    list_len : int
        Length of list output, or lists within list if multiple.
    num_lists : int, optional
        If > 1, number of lists within list of lists output. The default is 1.

    Returns
    -------
    list or list of lists
        A list [val_for_list, val_for_list, ...] or list of such lists.

    """
    output_list = []
    temp_list = []
    for _ in range(0, int(list_len)):
        temp_list.append(val_for_list)
    if num_lists <= 1:
        return temp_list
    else:
        for _ in range(0, int(num_lists)):
            output_list.append(temp_list)
    return output_list


def check_mos_csv_keys(input_mos_csv_dict):
    """Check whether a dict has keys matching required headers in a mosaic_info.csv.

    Parameters
    ----------
    input_mos_csv_dict : dict
        A dict with (or without) mosaic_info.csv headers as keys.

    Returns
    -------
    Bool
        True or False, depending on whether dict keys match required headers.

    """
    req_keys = ['Sample', 'Scanlist', 'Mosaic',
                'Max_grain_size', 'X_offset', 'Y_offset']
    input_keys = list(input_mos_csv_dict.keys())

    if all(key in input_keys for key in req_keys):
        return True
    #backwards compatability with original header "Max_zircon_size"
    else:
        req_keys = ['Sample', 'Scanlist', 'Mosaic',
                    'Max_zircon_size', 'X_offset', 'Y_offset']
        return all(key in input_keys for key in req_keys)


def prediction_to_np(input_results):
    """Stack Detectron prediction results to np array.

    Parameters
    ----------
    input_results : Detectron2 Prediction
        Prediction results from a Detectron2 predictor.

    Returns
    -------
    arr : np array
        All instances in the input predictions stacked to a
        np array along axis 2. Empty list if no instances.

    """
    arr = []
    if len(input_results['instances']) > 0:
        instances = input_results['instances'].get('pred_masks')
        arr = np.stack([instance.cpu().numpy() for instance in instances], 2)
    return arr


def mask_list_to_np(input_mask_list):
    """Stack a list of mask arrays (e.g., from Otsu segmentation)
        to a single, larger array.

    Parameters
    ----------
    input_mask_list : list[array]
        A list of binary mask arrays.

    Returns
    -------
    arr : np array
        A stacked array from input masks. Empty list if no masks in input.

    """
    arr = []
    if input_mask_list:
        arr = np.stack(input_mask_list, 2)
    return arr

# from https://stackoverflow.com/a/58567022
def rescale_2d_arr(im, nR, nC):
    """Rescale a 2d array to input size (nR x nC)

    Parameters
    ----------
    im : array
        Input 2D array (likely a mask image).
    nR : int
        Rows for rescaled array/image.
    nC : int
        Columns for rescaled array/image.

    Returns
    -------
    array
        Input array im resized to nR rows, nC columns.

    """
    nR0 = len(im)     # source number of rows
    nC0 = len(im[0])  # source number of columns
    return np.asarray([[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                        for c in range(nC)] for r in range(nR)])

def rotate_pt(pt, theta, center):
    """Rotate an x-y coordinate point clockwise by theta degrees around x-y
       coordinate point theta.

    Parameters
    ----------
    pt : tuple (x, y)
        Coordinates of point for rotation.
    theta : float or int
        Degrees for clockwise rotation of pt around center.
    center : tuple (x, y)
        Coordinates of center for rotation operation.

    Returns
    -------
    x_new : float
        X coordinate of rotated point.
    y_new : float
        Y coordinate of rotated point.

    """
    cos_theta, sin_theta = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    x_new = (pt[0] - center[0]) * cos_theta + (pt[1] - center[1]) * sin_theta + center[0]
    y_new = -(pt[0] - center[0]) * sin_theta + (pt[1] - center[1]) * cos_theta + center[1]
    return (x_new, y_new)


# from https://stackoverflow.com/a/50322574
def crop_nd_arr(img, bounding):
    """Crop the central portion of an array so that it matches shape 'bounding'.

    Parameters
    ----------
    img : array
        An nd array that needs to be cropped.
    bounding : tuple
        Shape tupple smaller than shape of img to crop img to.

    Returns
    -------
    array
        A version of img with its edges uniformly cropped to
        match size of 'bounding' input.

    """
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def mask_to_3D_arr_size(input_mask, input_arr):
    """Check whether a mask (2D array) is the same (x, y) shape as an input
       image; resizes or crops it to match if not.

    Parameters
    ----------
    input_mask : array
        A 2D array (presumably a mask).
    input_arr : array
        A 3D array (presumably a 3-channel image).

    Returns
    -------
    output_arr : array
        Either input_mask or input_mask resized to match the x,y of input_arr.

    """
    output_arr = input_mask

    #crop if mask is larger than original image
    if all([input_mask.shape[0] > input_arr.shape[0],
            input_mask.shape[1] > input_arr.shape[1]]):
        output_arr = crop_nd_arr(input_mask, input_arr.shape[:2])
    #rescale if mask is smaller in x or y than original image
    elif input_mask.shape != input_arr.shape[:2]:
        output_arr = rescale_2d_arr(input_mask, *input_arr.shape[:2])
    return output_arr


def scancsv_to_dict(scancsv_path):
    """Convert a .scancsv file to a dictionary.

    Parameters
    ----------
    scancsv_path : str
        File path to .scancsv for conversion.

    Returns
    -------
    temp_coords_dict : dict
        A dict with format {SCANNAME: [SCAN_X_COORD, SCAN_Y_COORD], ...}.

    """
    # dictionary for info on individual scans and their coordinates
    temp_coords_dict = {}
    # scanlist from .scancsv file, loaded as a dict
    each_scanlist = pd.read_csv(scancsv_path, header=0, index_col=False,
                                encoding='cp1252').to_dict('list')
    added_scans_unchanged = []  # list of scans added to output dictionary

    # loops through shotlist, gets coordinates for each scan, \
    # and numbers repeated instances
    for eachscan_index, eachscan in enumerate(each_scanlist['Description']):
        if each_scanlist['Scan Type'][eachscan_index] == 'Spot':

            splt_scan = each_scanlist['Vertex List'][eachscan_index].split(',')
            eachscan_coords = [float(data) for data in splt_scan][:2]

            if eachscan in added_scans_unchanged:
                count_str = str(added_scans_unchanged.count(eachscan) + 1)
                temp_scanname = str(eachscan) + '-' + count_str
                temp_coords_dict[temp_scanname] = eachscan_coords
            else:
                temp_coords_dict[str(eachscan)] = eachscan_coords
            added_scans_unchanged.append(eachscan)
    return temp_coords_dict


def get_Align_center_size(align_file_path):
    """Gets the data tagged as 'Center', 'Size' (microns), and 'Rotation'
       (degrees, clockwise around center) from a .Align alignment xml file.
       Used for scaling and mapping shots to mosaic image(s).

    Parameters
    ----------
    align_file_path : str
        Path to a .Align alignment file.

    Returns
    -------
    list
        A list of data extracted from .Align file. Format:
            [x_center, y_center, x_size, y_size, rotation]

    """

    x_center, y_center, x_size, y_size, rotation = 0, 0, 0, 0, 0
    align_tree = ET.parse(align_file_path)
    align_root = align_tree.getroot()
    #loop through xml file to get image center, size, rotation if present
    for eachchild in align_root:
        if eachchild.tag == 'Alignment':
            for each_align_data in eachchild:
                if each_align_data.tag == 'Center':
                    centers = [float(data) for data
                               in each_align_data.text.split(',')]
                    x_center, y_center = centers
                if each_align_data.tag == 'Size':
                    sizes = [float(data) for data
                             in each_align_data.text.split(',')]
                    x_size, y_size = sizes
                if each_align_data.tag == 'Rotation':
                    rotation = float(each_align_data.text)
    return [x_center, y_center, x_size, y_size, rotation]

def calc_scale_factor(Align_x_y_sizes, mosaic_x_y_sizes):
    """Calculate the scale factor for a mosaic image in microns/pixel
       by comparing the image size with data from its .Align file.
       Scale factors are calculated along the x and y axes then averaged.

    Parameters
    ----------
    Align_x_y_sizes : list
        A list [x size from .Align file, y size from .Align file].
    mosaic_x_y_sizes : list
        A list [mosaic x pixel dimension, mosaic y pixel dimension].

    Returns
    -------
    float
        A scale factor for an image in microns/pixel.

    """
    x_scale_fact = Align_x_y_sizes[0]/mosaic_x_y_sizes[0]
    y_scale_fact = Align_x_y_sizes[1]/mosaic_x_y_sizes[1]
    return (x_scale_fact+ y_scale_fact)/2

# a function for loading mosaic and shot data into a dictionary
def load_data_dict(project_dir_string):
    """Load data from a colab_zirc_dims project folder into a dictionary.

    Parameters
    ----------
    project_dir_string : str
        Path to a colab_zirc_dims project folder.

    Returns
    -------
    Dict
        A dict of dicts containing data from project folder w/ format:

        {'SAMPLE NAME': {'Scanlist': SCANLIST (.SCANCSV) PATH,
                         'Mosaic': MOSAIC .BMP PATH,
                         'Align_file': MOSAIC ALIGN FILE PATH,
                         'Max_grain_size': MAX USER-INPUT GRAIN SIZE,
                         'Offsets': [USER X OFFSET, USER Y OFFSET],
                         'Scan_dict': DICT LOADED FROM .SCANCSV FILE},
         ...}.

    """

    # initializes output dict
    temp_output_dict = {}

    # file paths
    mosaic_path = os.path.join(project_dir_string, 'mosaics')
    scanlist_path = os.path.join(project_dir_string, 'scanlists')
    mos_csv_path = os.path.join(project_dir_string, 'mosaic_info.csv')

    # loads info csv as dictionary
    mos_csv_dict = pd.read_csv(mos_csv_path, header=0, index_col=False
                               ).to_dict('list')

    if not check_mos_csv_keys(mos_csv_dict):
        print('Incorrect mosaic_info.csv headers: correct and re-save')
        return {}

    #backwards compatability with old "Max_zircon_size" mosaic_info header
    if "Max_zircon_size" in list(mos_csv_dict.keys()):
        mos_csv_dict['Max_grain_size'] = copy.deepcopy(mos_csv_dict['Max_zircon_size'])

    # lists of files in directories
    mosaic_bmp_filenames = list_if_endswith(os.listdir(mosaic_path),
                                                      '.bmp')
    mosaic_align_filenames = list_if_endswith(os.listdir(mosaic_path),
                                                        '.Align')
    scanlist_filenames = list_if_endswith(os.listdir(scanlist_path),
                                                    '.scancsv')

    # loops through mos_csv_dict in order to collect data, \
    # verify that all files given in mosaic_info.csv are present
    for eachindex, eachsample in enumerate(mos_csv_dict['Sample']):
        each_include_bool = True
        each_csv_scanlist_name = mos_csv_dict['Scanlist'][eachindex]
        # mosaic name without file extension
        each_csv_mosaic_name = mos_csv_dict['Mosaic'][eachindex].split('.')[0]

        # checks if files are in directories, gets full file paths if so
        act_scn_file = join_1st_match(scanlist_filenames,
                                                each_csv_scanlist_name,
                                                scanlist_path)
        act_mos_file = join_1st_match(mosaic_bmp_filenames,
                                                each_csv_mosaic_name,
                                                mosaic_path)
        act_align_file = join_1st_match(mosaic_align_filenames,
                                                  each_csv_mosaic_name,
                                                  mosaic_path)

        # verifies that matches found, provides user feedback if not
        error_strings = ['scanlist', 'mosaic .bmp file', 'mosaic .Align file']
        for error_idx, pth_var in enumerate([act_scn_file, act_mos_file,
                                             act_align_file]):
            if each_include_bool and not pth_var:
                print(eachsample, ': matching', error_strings[error_idx],
                      'not found')
                each_include_bool = False

        if each_include_bool:

            # dictionary for info on individual scans and their coordinates
            coords_dict = scancsv_to_dict(act_scn_file)

            #adds collected data to output dict
            temp_output_dict[eachsample] = {}

            temp_output_dict[eachsample]['Scanlist'] = act_scn_file
            temp_output_dict[eachsample]['Mosaic'] = act_mos_file
            temp_output_dict[eachsample]['Align_file'] = act_align_file
            temp_output_dict[eachsample]['Max_grain_size'] = mos_csv_dict['Max_grain_size'][eachindex]
            temp_output_dict[eachsample]['Offsets'] = [mos_csv_dict['X_offset'][eachindex],
                                                       mos_csv_dict['Y_offset'][eachindex]]
            temp_output_dict[eachsample]['Scan_dict'] = coords_dict

    return temp_output_dict

def alc_calc_scans_n(inpt_mos_data_dict, inpt_selected_samples):
    """Get the total number of scans that will be run (i.e., represented in
       both the Notebook data dict and in list[selected samples]). For ALC/
       mosaic datasets/Notebooks.

    Parameters
    ----------
    inpt_mos_data_dict : dict
        A dict of dicts containing data from project folder w/ format:

        {'SAMPLE NAME': {'Scanlist': SCANLIST (.SCANCSV) PATH,
                         'Mosaic': MOSAIC .BMP PATH,
                         'Align_file': MOSAIC ALIGN FILE PATH,
                         'Max_grain_size': MAX USER-INPUT ZIRCON SIZE,
                         'Offsets': [USER X OFFSET, USER Y OFFSET],
                         'Scan_dict': DICT LOADED FROM .SCANCSV FILE},
         ...}.
    inpt_selected_samples : list(str)
        A list of samples selected by a user for running; these should be keys
        in the input data dict.

    Returns
    -------
    n : int
        Total number of scans/sub-images that will be processed.

    """
    n = 0
    avl_samples = list(inpt_mos_data_dict.keys())
    for sample in [sample for sample in inpt_selected_samples
                   if sample in avl_samples]:
        for _ in inpt_mos_data_dict[sample]['Scan_dict'].keys():
            n += 1
    return n
