#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import math
import random
import xml.etree.ElementTree as ET


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skimage
import skimage.measure as measure
import skimage.filters as filters
import skimage.io as skio

from skimage.morphology import binary_closing


# Various small functions that simplify larger functions below

#saves pandas as csv file
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

# More important and/or longer functions for mosaic/prediction processing below

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
    mosaic_bmp_filenames = list_if_endswith(os.listdir(mosaic_path), '.bmp')
    mosaic_align_filenames = list_if_endswith(os.listdir(mosaic_path), '.Align')
    scanlist_filenames = list_if_endswith(os.listdir(scanlist_path), '.scancsv')

    # loops through mos_csv_dict in order to collect data, \
    # verify that all files given in mosaic_info.csv are present
    for eachindex, eachsample in enumerate(mos_csv_dict['Sample']):
        each_include_bool = True
        each_csv_scanlist_name = mos_csv_dict['Scanlist'][eachindex]
        # mosaic name without file extension
        each_csv_mosaic_name = mos_csv_dict['Mosaic'][eachindex].split('.')[0]

        # checks if files are in directories, gets full file paths if so
        act_scn_file = join_1st_match(scanlist_filenames,
                                      each_csv_scanlist_name, scanlist_path)
        act_mos_file = join_1st_match(mosaic_bmp_filenames,
                                      each_csv_mosaic_name, mosaic_path)
        act_align_file = join_1st_match(mosaic_align_filenames,
                                        each_csv_mosaic_name, mosaic_path)

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
class point_generator:
    def __init__(self, x_center, y_center, pixel_increment,
                 n_incs = 20, n_pts=18):
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


# gets the size of the contiguous region of a mask at pt [coords]. \
# This is useful for PointRend models, which often produce instances \
# with small, noncontiguous patches at bounding box margins
def mask_size_at_pt(input_mask, coords):
    """Gets the contiguous area of a mask at a point. This is useful for
        PointRend models, which often produce instances
        with small, noncontiguous patches at bounding box margins.

    Parameters
    ----------
    input_mask : binary array
        Array of a mask.
    coords : list
        Coordinates [x, y] within the mask to re.

    Returns
    -------
    size_reg : int
        Contiguous pixel area of a mask region, if any, found at input point.

    """
    label_mask = measure.label(input_mask.astype(int))
    coords_x, coords_y = coords
    size_reg = 0
    reg_at_pt = int(label_mask[coords_x, coords_y])
    if reg_at_pt:
        size_reg = np.count_nonzero(label_mask == reg_at_pt)
    return size_reg


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


# function for retrieving mask at center of image, ~nearest to center of image
def get_central_mask(results):
    """Return the largest central mask region array, if any,
        from threshholding or Detectron2 prediction results.

    Parameters
    ----------
    results : Detectron2 Prediction or list[array]
        Detectron2 predictions or list of mask arrays for central mask.

    Returns
    -------
    mask_found_bool : Boolean
        True or False, depending on whether a central mask was found.
    array or list
        np array for the central mask found. Empty list if none found.

    """

    #bool indicating whether a central mask was found
    mask_found_bool = False
    cent_mask_index = 0

    #converts mask outputs to np arrays and stacks them into a np array, \
    # w/ different method for lists of masks (e.g., from Otsu segmentation)
    masks = []
    if type(results) == list:
        masks = mask_list_to_np(results)
    else:
        masks = prediction_to_np(results)

    #does not search for masks if no zircons segmented from image
    if not masks:
        print('NO ZIRCON MASKS FOUND')
        return mask_found_bool, []

    #gets central points for masks/image, number of masks created
    masks_shape = masks.shape
    x_cent, y_cent = int(masks_shape[0]/2), int(masks_shape[1]/2)
    num_masks = int(masks_shape[2])

    #counter for generating pts at and around center of image; \
    # will run through central ~1/5th of image
    pts = point_generator(x_cent, y_cent, masks_shape[0]//100)

    central_mask_indices = []

    #loops through masks output and finds whether mask includes center of image
    for i in range(0, num_masks):
        curr_x, curr_y = pts.curr_pts
        if masks[curr_x, curr_y, i] is True:
            central_mask_indices.append(i)
    if len(central_mask_indices) > 0:
        mask_found_bool = True

    #extends search if mask not found at exact center
    while mask_found_bool is False and pts.in_bounds:
        pts.next_pt()
        curr_x, curr_y = pts.curr_pts
        for i in range(0, num_masks):
            if masks[curr_x, curr_y, i]:
                central_mask_indices.append(i)
        if len(central_mask_indices) > 0:
            mask_found_bool = True
            break

    if not mask_found_bool:
        print('ERROR: NO CENTRAL ZIRCON MASK FOUND')
        return(mask_found_bool, [])

    #if only one mask at center, return it
    if len(central_mask_indices) == 1:
        cent_mask_index = central_mask_indices[0]
        return mask_found_bool, masks[:, :, cent_mask_index]

    #selects the largest mask from central_mask indices as output
    mask_size_list = [mask_size_at_pt(masks[:, :, i], pts.curr_pts)
                      for i in central_mask_indices]
    cent_mask_index = central_mask_indices[mask_size_list.index(max(mask_size_list))]

    return mask_found_bool, masks[:, :, cent_mask_index]


def get_main_region_props(input_central_mask):
    """Get properties (using skimage) for the only or
       largest contiguous region (if multiple) in a mask.

    Parameters
    ----------
    input_central_mask : array
        Binary mask array.

    Returns
    -------
    main_region_props : skimage RegionProperties instance
        A RegionProperties instance corresponding to the largest
        and/or only contiguous region in the input mask image.

    """

    label_image = measure.label(input_central_mask.astype(int))
    regions = measure.regionprops(label_image)

    #selects largest region in case central zircon mask \
    # has multiple disconnected regions
    area_list = [props.area for props in regions]
    main_region_index = area_list.index(max(area_list))
    main_region_props = regions[main_region_index]

    return main_region_props


#measures zircon mask (in pixels) using skimage, creates an image by \
# overlaying the mask atop the original subimage, and optionally displays \
# (if display_bool = True) and/or saves (if save_dir != '') this image.
def overlay_mask_and_get_props(input_central_mask, original_image, analys_name,
                               display_bool=False, save_dir='', tag_bool=False):
    """Measures zircon mask (in pixels) using skimage, creates an image by
       overlaying the mask atop the original subimage, and optionally displays
       (if display_bool = True) and/or saves (if save_dir != '') this image.

    Parameters
    ----------
    input_central_mask : array
        Binary mask for the input original image.
    original_image : array
        Image array that the input central mask was derived from.
    analys_name : str
        Name of analysis (e.g., 'Spot-213') corresponding to mask.
        Used as base save file name (.png will be added).
    display_bool : Boolean, optional
        True or False; determines whether plot is displayed in output.
        The default is False.
    save_dir : str, optional
        If entered, plot image will be saved to this directory. If blank,
        the image will not be saved. The default is ''.
    tag_bool : Boolean, optional
        If True and save_dir != '', '_tagged' will be appended to
        plot image file name. Called in zirc_dims_GUI. The default is False.

    Returns
    -------
    main_region : skimage RegionProperties instance
        A RegionProperties instance corresponding to the largest
        and/or only contiguous region in the input mask image.

    """

    main_region = get_main_region_props(input_central_mask)

    if display_bool or save_dir:

        #overlays mask on the original subimage
        fig, ax = plt.subplots()
        ax.imshow(original_image, cmap = 'gray')
        ax.imshow(input_central_mask, alpha=0.4)

        #gets figure size
        figsize_x, figsize_y = np.shape(input_central_mask)

        #plots measured axes atop image
        #for props in regions:
        y0, x0 = main_region.centroid
        orientation = main_region.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * main_region.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * main_region.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * main_region.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * main_region.major_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=1.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.5)
        ax.plot(x0, y0, '.g', markersize=10)

        ax.axis((0, figsize_x, figsize_y, 0))

        if save_dir:
            img_save_filename = os.path.join(save_dir, analys_name + '.png')
            if tag_bool:
                img_save_filename = os.path.join(save_dir,
                                                 analys_name + '_tagged.png')

            plt.savefig(img_save_filename)

        #plt.clf()

        if display_bool:
            plt.show()

        plt.close('all')
    return main_region

def parse_properties(props, img_scale_factor, analys_name, verbose = False):
    """Parses a skimage RegionProperties instance and scales image
       measurements to real-world measurments (microns).

    Parameters
    ----------
    props : skimage RegionProperties instance
        Ideally, the skimage RegionProperties instance corresponding
        to a zircon crystal.
    img_scale_factor : float
        Scale factor in microns/pixel for converting pixel measurements.
    analys_name : str
        A name for the analysis being parsed (e.g., 'Spot-210'). Added to output.
    verbose : Boolean, optional
        If True, some calculated measurments are printed. The default is False.

    Returns
    -------
    list
        A list of calculated properties. Added to export dict in
        main colab_zirc_dims Colab Notebook. Format:
            [analys_name, area, convex_area, eccent, eq_diam, perim,
             major_leng, minor_leng, roundness, scale_factor]

    """

    #if no region found, skip
    if props == []:
        if verbose:
            print('null properties entered')
        return [analys_name, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    area = props.area * img_scale_factor**2
    convex_area = props.convex_area * img_scale_factor**2
    eccent = props.eccentricity
    eq_diam = props.equivalent_diameter * img_scale_factor
    perim = props.perimeter * img_scale_factor
    major_leng = props.major_axis_length * img_scale_factor
    minor_leng = props.minor_axis_length * img_scale_factor
    roundness = 4 * math.pi * props.area / props.perimeter**2
    scale_factor = img_scale_factor

    props_list = [analys_name, area, convex_area, eccent, eq_diam, perim,
                  major_leng, minor_leng, roundness, scale_factor]

    if verbose:
        print('Major axis length =', round(major_leng, 1), 'µm,',
              'Minor axis length =', round(minor_leng, 1), 'µm')

    return props_list


# Get regionated masks from an image via Otsu threshholding
def otsu_masks(input_image):
    """Use Otsu threshholding to segment an image.

    Parameters
    ----------
    input_image : image array
        An input image array. Will be threshholded in grayscale.

    Returns
    -------
    output_masks_list : list[arr]
        A list of mask arrays (one for each region) from Otsu thresholding.

    """
    gray_img = skimage.color.rgb2gray(input_image)
    thresh_val = filters.threshold_otsu(gray_img)
    thresh_img = binary_closing(gray_img > thresh_val)
    label_mask = measure.label(thresh_img.astype(int))
    region_vals = list(np.unique(label_mask))

    #removes very small regions
    larger_region_vals = [val for val in region_vals
                          if np.count_nonzero(label_mask == val) > 100]

    output_masks_list = []
    for each_region in larger_region_vals:
        each_mask = label_mask == each_region
        output_masks_list.append(each_mask)

    return output_masks_list

#adjusts contrast if necessary via histogram normalization
def auto_inc_contrast(input_image, low_cont_threshhold =0.10):
    """Checks whether image has low contrast, automatically increases
       it via histogram normalization if so.

    Parameters
    ----------
    input_image : image array
        An input image array.
    low_cont_threshhold : float, optional
        Threshold for contrast. If under this, contrast will be increased.
        The default is 0.10.

    Returns
    -------
    image array
        The input image w/ or w/o contrast enhanced, depending on check.

    """
    #if image has low contrast, enhances contrast
    if skimage.exposure.is_low_contrast(input_image, low_cont_threshhold):

        ## code for enhancing contrast by channel: \
        ## avoids warning but produces grainy images
        #red = skimage.exposure.equalize_hist(input_image[:, :, 0])
        #green = skimage.exposure.equalize_hist(input_image[:, :, 1])
        #blue = skimage.exposure.equalize_hist(input_image[:, :, 2])
        #return skimage.img_as_ubyte(np.stack([red, green, blue], axis=2))

        #code for enhancing contrast without splitting channels: results are less grainy
        return skimage.img_as_ubyte(skimage.exposure.equalize_hist(input_image))
    else:
        return input_image

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


# a class for a mosaic image from which subimages are clipped, with functions for clipping
class mos_img:

    #shift list = adjustments [x, y] in microns
    def __init__(self, mosaic_image_path, align_file_path = '',
                 sub_img_size = 500, x_y_shift_list = [0, 0]):
        """A class instance wrapper for a mosaic image, with functions
           for calculating scale and extracting scaled subimages.

        Parameters
        ----------
        mosaic_image_path : str
            Full file path to a .bmp mosaic image file.
        align_file_path : str, optional
            Path to a .Align file corresponding to the mosaic image.
            If not entered, the image scale will be in pixels.
            The default is ''.
        sub_img_size : int, optional
            Size of subimages extracted from the mosaic in pixels or microns.
            The default is 500.
        x_y_shift_list : list[int], optional
            A list [x shift, y shift] used to correct misalignment between
            .scancsv shot coordinates and a mosaic image. The default is [0, 0].

        Returns
        -------
        None.

        """

        #the mosaic image from which subimages are clipped
        self.full_img = auto_inc_contrast(skio.imread(mosaic_image_path))

        self.mos_dims = self.full_img.shape[::-1] #image pixel dimensions (x, y)

        self.scale_factor = 1 #default scale factor (e.g., if no .Align file)

        #origins for x, y (for mapping coordinates \
        # (from align, shotlist files) to mosaic image)
        self.x_origin, self.y_origin = 0, 0

        #gets info from .Align xml file
        if align_file_path:
            #gets data (x,y centers, x,y sizes) from .Align file
            self.align_data = get_Align_center_size(align_file_path)

            #calculates a new scale factor (microns/pixel) from .Align file
            self.scale_factor = calc_scale_factor(self.align_data[2:],
                                                  self.mos_dims[1:])
            #converts shifts to pixels from microns
            self.pix_shift_list = [val/self.scale_factor
                                   for val in x_y_shift_list]
            #calculates origin based on new variables
            self.x_origin = float(self.align_data[0] - self.align_data[2]/2
                                  - self.pix_shift_list[0])
            self.y_origin = float(self.align_data[1] - self.align_data[3]/2
                                  - self.pix_shift_list[1])

        # a hacky solution to prevent mismatches beyond and at pixel bounds:
        # all subimage sizes will be even:
        self.sub_img_size = round_to_even(sub_img_size / self.scale_factor)

        #initiates variables that will be called in set_sub_img
        self.x_y_0, self.x_y_0_offsets = [0, 0], [0, 0]
        self.x_y_1, self.x_y_1_offsets = list_of_val(self.sub_img_size, 2, 2)

        #initiates sub_img (as a black np.zeroes array)
        self.sub_img = np.zeros([self.sub_img_size, self.sub_img_size,
                                 self.mos_dims[0]], dtype=np.uint8)

    def set_sub_img_size(self, new_sub_img_size):
        """Set a new sub image size.

        Parameters
        ----------
        new_sub_img_size : int
            New size for subimages.

        Returns
        -------
        None.

        """
        self.sub_img_size = round_to_even(new_sub_img_size / self.scale_factor)

    #maps coordinates (as in shotlist) to image pixels
    def coords_to_pix(self, x_coord, y_coord):
        """Map micron coordinates (as from a scanlist) to mosaic image pixels.

        Parameters
        ----------
        x_coord : float or int
            x coordinate for mapping.
        y_coord : float or int
            y coordinate for mapping.

        Returns
        -------
        list
            A list of pixel coordinates [x pixel coord, y pixel coord].

        """

        return [round(float((x_coord - self.x_origin) / self.scale_factor)),
                round(float((y_coord - self.y_origin) / self.scale_factor))]

    def set_subimg(self, x_coord, y_coord):
        """Set a clipped subimage centered on entered coordinates.
           This image can be retrieved as *mos_image INSTANCE*.sub_img.

        Parameters
        ----------
        x_coord : int or float
            x coordinate for subimage center.
        y_coord : int or float
            y coordinate for subimage center.

        Returns
        -------
        None.

        """

        #sets all vars back to base values
        self.x_y_0, self.x_y_0_offsets = [0, 0], [0, 0]
        self.x_y_1, self.x_y_1_offsets = list_of_val(self.sub_img_size, 2, 2)

        #gets spot location for image, sets corresponding bounds
        self.x_y_0 = [int(val - self.sub_img_size/2) for val
                      in self.coords_to_pix(x_coord, y_coord)]
        self.x_y_1 = [int(val + self.sub_img_size/2) for val
                      in self.coords_to_pix(x_coord, y_coord)]

        #modifies bounds in the case that initial bounds exceed image bounds
        for idx, val in enumerate(self.x_y_0):
            if val < 0:
                self.x_y_0_offsets[idx] = abs(int(0 - val))
                self.x_y_0[idx] = 0
        for idx, val in enumerate(self.x_y_1):
            if val > self.mos_dims[1:][idx]:
                self.x_y_1_offsets[idx] = int(self.sub_img_size
                                              - (val - self.mos_dims[1:][idx]))
                if self.x_y_1_offsets[idx] < 0:
                    self.x_y_1_offsets[idx] = 0
            #in case of points that are fully out of bounds, a \
            # fully black subimage will be displayed
            if val < 0:
                self.x_y_1[idx] = 0

        #default (black) subimage
        self.sub_img = np.zeros([self.sub_img_size, self.sub_img_size,
                                 self.mos_dims[0]], dtype=np.uint8)
        #maps crop from mosaic onto black image to create subimage with input \
        # coordinate point at center.
        self.sub_img[self.x_y_0_offsets[1]:self.x_y_1_offsets[1],
                     self.x_y_0_offsets[0]:self.x_y_1_offsets[0],
                     :] = self.full_img[self.x_y_0[1]:self.x_y_1[1],
                                        self.x_y_0[0]:self.x_y_1[0], :]


# shows a random sample of subimages with approximate (may vary by a pixel) \
# shot location marked with red circle.
def random_subimg_sample(num_samples, sample_name, scancsv_path, mosaic_path,
                         Align_path, subimg_size, offset_list):
    """Extract and plot a random sample of shot-subimages from a mosaic,
       given entered parameters. Approximate shot locations
       (may vary by a pixel) of shots are marked with red dots.

    Parameters
    ----------
    num_samples : int
        Number of random samples of shots/corresponding
        subimages to be plotted.
    sample_name : str
        Name of the sample. Added to plot; can be blank ('').
    scancsv_path : str
        Full path to a .scancsv file.
    mosaic_path : str
        Full path to a mosaic .bmp file.
    Align_path : str
        Full path to the .Align file corresponding to
        the mosaic .bmp file.
    subimg_size : int
        Size (microns) of subimages to be sampled from given mosaic.
    offset_list : list[int]
        Offsets [x offset, y offset] used to adjust mapping of scanlist
        shots to the mosaic images.

    Returns
    -------
    None.

    """
    each_mosaic = mos_img(mosaic_path, Align_path, subimg_size, offset_list)
    each_scan_dict = scancsv_to_dict(scancsv_path)

    #if too few shots in sample, reduces num_samples
    if num_samples > len(list(each_scan_dict.keys())):
        num_samples = len(list(each_scan_dict.keys()))

    scan_sample = random.sample(each_scan_dict.keys(), num_samples)
    num_rows = num_samples // 3
    if num_rows < num_samples / 3:
        num_rows += 1
    fig = plt.figure()
    fig.set_figheight(3.32 * num_rows)
    fig.set_figwidth(10)
    fig_title = str(sample_name) + ' (' + str(round(each_mosaic.scale_factor, 3)) + ' µm/pixel):'
    fig.suptitle(fig_title, fontsize=16)
    for each_i, eachscan in enumerate(scan_sample):
        each_mosaic.set_subimg(*each_scan_dict[eachscan])
        ax = fig.add_subplot(num_rows, 3, each_i + 1)
        plt.imshow(each_mosaic.sub_img)
        center_coords = [coord / 2 for coord in each_mosaic.sub_img.shape[:2]]
        ax.plot(*center_coords, 'or', markersize=5)
        ax.set_title(str(eachscan) + ':')
        ax.axis('off')
    fig.show()
