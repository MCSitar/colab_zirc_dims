#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


### Various functions and classes for file processing and code simplification
### in other modules below.

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
    for i in range(0, int(list_len)):
        temp_list.append(val_for_list)
    if num_lists <= 1:
        return temp_list
    else:
        for i in range(0, int(num_lists)):
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
                'Max_zircon_size', 'X_offset', 'Y_offset']
    input_keys = list(input_mos_csv_dict.keys())

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
                                squeeze=False, encoding='cp1252'
                                ).to_dict('list')
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
    """Gets the data tagged as 'Center' and as 'Size' (microns) from a
       .Align alignment xml file. Used for scaling and mapping shots to
       mosaic image(s).

    Parameters
    ----------
    align_file_path : str
        Path to a .Align alignment file.

    Returns
    -------
    list
        A list of data extracted from .Align file. Format:
            [x_center, y_center, x_size, y_size]

    """

    x_center, y_center, x_size, y_size = 0, 0, 0, 0
    align_tree = ET.parse(align_file_path)
    align_root = align_tree.getroot()
    #loop through xml file to get image center, size
    ##UPDATE TO INCLUDE ROTATION IF NEEDED##
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
    return [x_center, y_center, x_size, y_size]

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
                         'Max_zircon_size': MAX USER-INPUT ZIRCON SIZE,
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
    mos_csv_dict = pd.read_csv(mos_csv_path, header=0, index_col=False,
                               squeeze=False).to_dict('list')

    if not check_mos_csv_keys(mos_csv_dict):
        print('Incorrect mosaic_info.csv headers: correct and re-save')
        return {}

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
            temp_output_dict[eachsample]['Max_zircon_size'] = mos_csv_dict['Max_zircon_size'][eachindex]
            temp_output_dict[eachsample]['Offsets'] = [mos_csv_dict['X_offset'][eachindex],
                                                       mos_csv_dict['Y_offset'][eachindex]]
            temp_output_dict[eachsample]['Scan_dict'] = coords_dict

    return temp_output_dict


# class for generating points, moving ~radially outwards from center of image, \
# to find central mask if not at actual center
class PointGenerator:
    """Class for generating points moving ~radially outwards from center
        of an image. Used for checking whether zircon masks appear in
        masked images.

    Parameters
    ----------
    x_center : int
        x coordinate of the center of an image, plot, etc..
    y_center : int
        y coordinate of the center of an image, plot, etc..
    pixel_increment : int
        Number of pixels (or other points) to increase search radius by
        after each rotation.
    n_incs : int, optional
        Max number of increments before point generator stops. The default is 20.
    n_pts : int, optional
        Number of points to return around each circle. The default is 18.

    Returns
    -------
    None.

    """
    def __init__(self, x_center, y_center, pixel_increment,
                 n_incs = 20, n_pts=18):
        self.x_center, self.y_center = x_center, y_center
        self.pixel_increment = pixel_increment
        self.n_pts = n_pts
        self.max_inc = n_incs

        # current, x, y for output
        self.curr_pts = [self.x_center, self.y_center]

        # int from 0 to (n_pts - 1) defining location around circle
        self.rot_counter = 0

        #degree increment for points around circle
        self.deg_inc = 360 / int(self.n_pts)

        #pixel increment multiplier, current pixel radius
        self.inc_multiplier, self.curr_radius = 0, 0

        #bool changes to False if generator reaches max increments
        self.in_bounds = True

    def get_curr_pts(self):
        """Get current points from the point generator.

        Returns
        -------
        int
            x coordinate of current search location.
        int
            y coordinate of current search location.

        """
        return self.curr_pts[0], self.curr_pts[1]

    # updates pts (around circumference of circle w/ diameter curr_radius)
    def update_pts(self):
        """Update points of the point generator. Called internally.

        Returns
        -------
        None.

        """
        curr_rot_rad = np.radians(self.rot_counter * self.deg_inc)
        self.curr_pts = [
            int(self.x_center + self.curr_radius * np.cos(curr_rot_rad)),
            int(self.y_center + self.curr_radius * np.sin(curr_rot_rad))
            ]

    def next_inc(self):
        """Cycles generator to a larger pixel increment, updates pts. Internal.

        Returns
        -------
        None.

        """
        self.inc_multiplier += 1
        if self.inc_multiplier > self.max_inc:
            self.in_bounds = False
        self.curr_radius = self.inc_multiplier * self.pixel_increment

        self.update_pts()

    # cycles to a new point around center w/o changing pixel increment
    def next_pt(self):
        """Move to the next point and/or radius increment. Called by user.

        Returns
        -------
        None.

        """
        self.rot_counter += 1
        if self.rot_counter >= self.n_pts or self.inc_multiplier == 0:
            self.rot_counter = 0
            self.next_inc()
        self.update_pts()