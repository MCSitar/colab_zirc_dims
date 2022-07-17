# -*- coding: utf-8 -*-
"""
Utility functions (e.g., loading datasets, calculating scale factors) for single
image-per-shot datasets (non-ALC).
"""
import os
import copy
import skimage.io as skio
import pandas as pd

from . import czd_utils

__all__ = ['find_Align',
           'indiv_img_scale_factor',
           'check_load_sample_info',
           'check_unused_samples',
           'unique_scan_name',
           'load_gen_data_dict',
           'gen_calc_scans_n',
           'get_save_fields']

def get_save_fields(proj_type = 'mosaic', save_type = 'auto', addit_fields = [],
                    get_nulls=False):
    """A function to standardize saving colab_zirc_dims measurements to .csv
        files and make adding new measurements quicker. Returns a list of
        output .csv column headers, depending on inputs.

    Parameters
    ----------
    proj_type : str, optional
        String ('default' or 'general') defining project type.
        The default is 'mosaic'.
    save_type : str, optional
        String ('auto' or 'GUI') defining saving interface type.
        The default is 'auto'.
    addit_fields : List[str], optional
        List with any additional fields that need to be added to headers list.
        The default is [].
    get_nulls : Bool
        If true, get a list of 0s with length of properties that won't have
        any data input unless a measurement is made.

    Returns
    -------
    ret_save_fields : TYPE
        DESCRIPTION.

    """
    default_save_fields = ['Analysis', 'Area (µm^2)',
                           'Convex area (µm^2)',
                           'Eccentricity',
                           'Equivalent diameter (µm)',
                           'Perimeter (µm)',
                           'Major axis length (µm)',
                           'Minor axis length (µm)',
                           'Circularity',
                           'Long axis rectangular diameter (µm)',
                           'Short axis rectangular diameter (µm)',
                           'Best long axis length (µm)',
                           'Best short axis length (µm)',
                           'Best axes calculated from',
                           'Scale factor (µm/pixel)']
    
    addit_gen_save_fields = ['Scale factor from:', 'Image filename']
    addit_GUI_save_fields = ['Human_or_auto', 'tagged?']
    poss_invariate_fields = ['Analysis', 'Scale factor (µm/pixel)']
    poss_non_plottable_fields = ['Analysis', 'Best axes calculated from',
                                 'Scale factor (µm/pixel)']

    if get_nulls:
        null_fields = [prop for prop in default_save_fields if prop not in poss_invariate_fields]
        return [0 for _ in range(len(null_fields))]
    if proj_type == 'plotting':
        return [field for field in default_save_fields if field not in poss_non_plottable_fields]

    ret_save_fields = default_save_fields
    if proj_type == 'general':
        ret_save_fields = [*ret_save_fields, *addit_gen_save_fields]
    if save_type == 'GUI':
        ret_save_fields = [*ret_save_fields, *addit_GUI_save_fields]
    ret_save_fields = [*ret_save_fields, *addit_fields]
    return ret_save_fields
    


def find_Align(name, all_files_list, img_suffix = '.png'):
    """Find a .Align file matching (same except for file type) an image file.

    Parameters
    ----------
    name : str
        Image file name (relative; without full path).
    all_files_list : list[str]
        A list of all files in a directory where .Align file may be present.
    img_suffix : str, optional
        File type for image file. The default is '.png'.

    Returns
    -------
    str
        Either the name of a matching .Align file (if found) or '' (if not).

    """
    Align_src = name.strip(img_suffix) + '.Align'
    if Align_src in all_files_list:
        return Align_src
    else:
        return ''

def indiv_img_scale_factor(img_path, Align_path):
    """Calculate the scale factor for an individual shot image.

    Parameters
    ----------
    img_path : str
        Full path to an image file (tested with .png, .bmp images, should work
        with others, including .tif).
    Align_path : str
        Full path to a .Align xml file with real-world dimensions for the image.

    Returns
    -------
    float
        A calculated scale factor for the image in μm/pixel.

    """
    img = skio.imread(img_path)
    img_xy = img.shape[:2][::-1]
    #No reason to expect rotation or that it will impact grain centeredness if \
    # present; rotation field ignored from .Align file.
    align_xy = czd_utils.get_Align_center_size(Align_path)[2:-1]
    return czd_utils.calc_scale_factor(align_xy, img_xy)

def check_load_sample_info(project_dir):
    """Checks if a sample_info.csv file is present in a project directory and
       whether it has correct formatting. Returns loaded scale dict if true.

    Parameters
    ----------
    project_dir : str
        Path to a project directory that may or may not contain sample_info.csv.

    Returns
    -------
    found_bool : bool
        True if a correctly-formatted .csv is found, otherwise False.
    output_dict : dict{str: float}
        A dict with values from .csv {SAMPLE_NAME1: SCALE_FACTOR1, ...} if
        usable sample_info.csv found. Otherwise, an empty dict {}.

    """
    found_bool = False
    output_dict = {}
    if 'sample_info.csv' in os.listdir(project_dir):
        csv_path = os.path.join(project_dir, 'sample_info.csv')
        orig_csv = pd.read_csv(csv_path, header=0, index_col=False,
                               squeeze=False).to_dict('list')
        orig_csv_cols = list(orig_csv.keys())
        #check to make sure that necessary columns are in imported csv
        if all(key in orig_csv_cols for key in ['Sample', 'Scale']):
            #make sure that columns are same length
            if len(orig_csv['Sample']) == len(orig_csv['Scale']):
                found_bool = True
                for each_idx, each_sample in enumerate(orig_csv['Sample']):
                    output_dict[str(each_sample)] = float(orig_csv['Scale'][each_idx])
            else:
                print('Error: sample_info.csv columns cannot be different lengths')
        else:
            print('Error: required column names not in sample_info.csv')
    else:
        print('Error: no sample_info.csv file found in project directory')
    return found_bool, output_dict

def check_unused_samples(loaded_csv_dict, loaded_img_dict):
    """Checks whether any samples loaded from a sample_info.csv file do not
       match any samples loaded from a project folder. Prints a warning if
       true.

    Parameters
    ----------
    loaded_csv_dict : dict{str: float}
        A dict with values from .csv {SAMPLE_NAME1: SCALE_FACTOR1, ...}.
    loaded_img_dict : dict
        A dict loaded from a project folder with format:
            {EACH_SAMPLE: {EACH_SPOT: {'img_file': FULL_IMG_PATH,
                                       'Align_file': FULL_ALIGN_PATH or '',
                                       'rel_file': IMG_FILENAME}, ...}, ...}.

    Returns
    -------
    None.

    """
    unused_samples_list = []
    csv_samples = list(loaded_csv_dict.keys())
    img_samples = list(loaded_img_dict.keys())
    for csv_sample in csv_samples:
        if csv_sample not in img_samples:
            unused_samples_list.append(csv_sample)
    if unused_samples_list:
        print('WARNING: sample(s) from sample_info.csv do not match',
              'sample(s) loaded from folder:', '\n',
              unused_samples_list, '\n',
              'Try checking names/capitalization in sample_info and try reloading')


def unique_scan_name(curr_scan_name, curr_sample_keys):
    """Get a unique scan name if neccesary to avoid replacing scans
       with the same name in a dict.

    Parameters
    ----------
    curr_scan_name : str
        Unalterned name of a scan that is being loaded into a sample dict.
    curr_sample_keys : list[str]
        Names of scans that have already been loaded into said dict.

    Returns
    -------
    curr_scan_name : str
        A unique name for the scan (appends a count integer to input name).

    """
    if curr_scan_name not in curr_sample_keys:
        return curr_scan_name
    else:
        orig_scan_name = curr_scan_name
        count_int = 1
        while curr_scan_name in curr_sample_keys:
            curr_scan_name = str(orig_scan_name) + '-' + str(count_int)
            count_int += 1
    return curr_scan_name

def load_gen_opt_A(scans_dir, split_fxn = None, file_type = '.png'):
    """Split image files in a folder by sample, shot names extracted from image
       file names and load them into a dict.

    Parameters
    ----------
    scans_dir : str
        Path to a 'scans' subdirectory in a project directory.
    split_fxn : function or None, optional
        A function (defined outside this one) that takes a image file name
        as input and returns a sample name and scan name. The default is None;
        in this case scan names will = file names - file_type.
    file_type : str, optional
        File type for loadable images in scans_dir. The default is '.png'.

    Returns
    -------
    output_dict : dict
        A dict loaded from a scans_dir with format:
            {EACH_SAMPLE: {EACH_SPOT: {'img_file': FULL_IMG_PATH,
                                       'Align_file': FULL_ALIGN_PATH or '',
                                       'rel_file': IMG_FILENAME}, ...}, ...}.

    """
    output_dict = {}
    all_dir_files = os.listdir(scans_dir)
    all_dir_imgs = [file for file in all_dir_files if file.endswith(file_type)]
    for each_file in all_dir_imgs:
        each_sample = 'UNSPECIFIED_SAMPLE'
        each_spot = each_file.strip(file_type)
        if split_fxn is not None:
            try:
                each_sample, each_spot = split_fxn(each_file)
            except ValueError:
                print('ERROR: SPLITTING FUNCTION INCOMPATIBLE WITH IMAGE NAMES')
                split_fxn = None
        each_img_path = os.path.join(scans_dir, each_file)
        #checks for available .Align file, returns (relative) path if possible
        each_align_path = find_Align(each_file, all_dir_files, file_type)
        if each_align_path:
            each_align_path = os.path.join(scans_dir, each_align_path)
        if each_sample not in output_dict.keys():
            output_dict[each_sample] = {}
        #makes sure that spots are unique to avoid losing data
        each_spot = unique_scan_name(each_spot,
                                     list(output_dict[each_sample].keys()))
        output_dict[each_sample][each_spot] = {'img_file': each_img_path,
                                               'Align_file': each_align_path,
                                               'rel_file': each_file}
    return output_dict

def load_gen_opt_B(scans_dir, split_fxn = None, file_type = '.png'):
    """Load image files from a project directory where image +/- .Align files
       are organized into sample-specific sub-folders.

    Parameters
    ----------
    scans_dir : str
        Path to a project folder 'scans' sub-directory.
    split_fxn : function or None, optional
        A function (defined outside this one) that takes a image file name
        as input and returns a sample name (this can be a generic value)
        and scan name. The default is None; in this case scan names will be
        file names - file_type.
    file_type : str, optional
        File type for loadable images in scans_dir. The default is '.png'.

    Returns
    -------
    output_dict : dict
        A dict loaded from a scans_dir with format:
            {EACH_SAMPLE: {EACH_SPOT: {'img_file': FULL_IMG_PATH,
                                       'Align_file': FULL_ALIGN_PATH or '',
                                       'rel_file': IMG_FILENAME}, ...}, ...}.

    """
    output_dict = {}
    # get sample subfolder paths, sample names from folders
    sample_dirs = [ f.path for f in os.scandir(scans_dir) if f.is_dir() ]
    #print(sample_dirs)
    samples = [ os.path.basename(os.path.normpath(f)) for f in sample_dirs ]
    #gets info for each file in subfolder
    for each_sample_idx, each_sample_dir in enumerate(sample_dirs):
        each_sample = samples[each_sample_idx]
        all_dir_files = os.listdir(each_sample_dir)
        all_dir_imgs = [file for file in all_dir_files if file.endswith(file_type)]
        for each_file in all_dir_imgs:
            each_spot = each_file.strip(file_type)
            if split_fxn is not None:
                try:
                    _, each_spot = split_fxn(each_file)
                except ValueError:
                    print('ERROR: SPLITTING FUNCTION INCOMPATIBLE WITH IMAGE NAMES')
                    split_fxn = None
            each_img_path = os.path.join(each_sample_dir, each_file)
            #checks for available .Align file, returns (relative) path if possible
            each_align_path = find_Align(each_file, all_dir_files, file_type)
            if each_align_path:
                each_align_path = os.path.join(each_sample_dir, each_align_path)
            if each_sample not in output_dict.keys():
                output_dict[each_sample] = {}
            #makes sure that spot names are unique to avoid overwriting data
            each_spot = unique_scan_name(each_spot,
                                         list(output_dict[each_sample].keys()))
            output_dict[each_sample][each_spot] = {'img_file': each_img_path,
                                                   'Align_file': each_align_path,
                                                   'rel_file': each_file}
    return output_dict

def gen_img_scale_factors(loaded_img_dict, scale_bools, sample_csv_dict ={},
                          verbose = False):
    """Add scale factors to an project dict with per-scan RL zircon images.

    Parameters
    ----------
    loaded_img_dict : dict
        A dict loaded from a project 'scans' subdirectory with format:
            {EACH_SAMPLE: {EACH_SPOT: {'img_file': FULL_IMG_PATH,
                                       'Align_file': FULL_ALIGN_PATH or '',
                                       'rel_file': IMG_FILENAME}, ...}, ...}.
    scale_bools : list[bool]
        User-input booleans indicating how scale factors should be
        found/calculated. Format is [bool, bool]. These correspond to options:
            [Try calculating scales from .Align files if possible,
             Try loading scales from a sample_info.csv file if possible].
        These options are tried sequentially. If neither work or if both bools
        are False, a default scale of 1.0 is used.
    sample_csv_dict : dict{str: float}, optional
        A dict with values from sample_info.csv if available:
            {SAMPLE_NAME1: SCALE_FACTOR1, ...}.
        Otherwise, default of {} (will not be used for scaling).
    verbose : bool, optional
        A bool indicating whether fxn will print sample names as it loads/
        calculates scale factors. This can take a while, so may be good
        to indicate that process is still running.

    Returns
    -------
    output_dict : dict
        A copy of loaded_img_dict with additional per-scan info:
            {'scale_factor': float(scale factor for each scan),
             'scale_from': str(method used to calculate/find scale factor)}.

    """

    #unlinked copy of input dict (it is edited in place here)
    output_dict = copy.deepcopy(loaded_img_dict)

    #if there is nothing in the input sample_info.csv dict, do not use it
    csv_keys = list(sample_csv_dict.keys())
    if not csv_keys:
        scale_bools[1] = False
    for each_sample_name, each_sample in output_dict.items():
        if verbose:
            print(str(each_sample_name))
        for each_scan in each_sample.values():

            found_any_scale = False
            if scale_bools[0] and os.path.isfile(each_scan['Align_file']):
                each_scan['scale_factor'] = indiv_img_scale_factor(each_scan['img_file'],
                                                                   each_scan['Align_file'])
                each_scan['scale_from'] = '.Align'
                found_any_scale = True
            if all([scale_bools[1], not found_any_scale,
                   each_sample_name in csv_keys]):
                each_scan['scale_factor'] = float(sample_csv_dict[each_sample_name])
                each_scan['scale_from'] = 'sample_info.csv'
                found_any_scale = True
            if not found_any_scale:
                each_scan['scale_factor'] = 1.0
                each_scan['scale_from'] = 'default (1.0)'
    return output_dict


def load_gen_data_dict(proj_dir, folder_struct = 'A', splitting_fxn = None,
                       file_type =  '.png', scale_bools = [False, False]):
    """Load a project directory with RL images into a dict. Method depends on
       dataset type, sample format.

    Parameters
    ----------
    proj_dir : str
        Path to a properly-formatted project directory with per-shot RL zircon
        images.
    folder_struct : str('A') or str('B'), optional
        Project directory structure; see non-ALC colab notebook linked for
        on colab_zirc_dims GitHub page for more details and links to
        downloadable template folders. The default is 'A'.
    splitting_fxn : function or None, optional
        A function (defined outside this one) that takes a image file name
        as input and returns a sample name and scan name. The default is None;
        in this case scan names will = file names - file_type.
    file_type : str, optional
        File type for loadable images in scans_dir. The default is '.png'.
    scale_bools : list[bools], optional
        User-input booleans indicating how scale factors should be
        found/calculated. Format is [bool, bool]. These correspond to options:
            [Try calculating scales from .Align files if possible,
             Try loading scales from a sample_info.csv file if possible].
        These options are tried sequentially. If neither work or if both bools
        are False, a default scale of 1.0 is used. The default is [False, False].

    Returns
    -------
    dict
        A standardized dict containing info that will be passed to automated
        and/or semi-automated segmentation functions, loaded from project dir.
        Format:
                {EACH_SAMPLE: {EACH_SPOT: {'img_file': FULL_IMG_PATH,
                               'Align_file': FULL_ALIGN_PATH or '',
                               'rel_file': IMG_FILENAME
                               'scale_factor': float(scale factor for each scan),
                               'scale_from': str(method used get scale factor)},
                               ...},
                 ...}

    """
    if os.path.isdir(os.path.join(proj_dir, 'scans')):
        scans_dir = os.path.join(proj_dir, 'scans')
    else:
        print("ERROR: missing 'scans' subdirectory in project folder!")
        return {}
    if folder_struct == 'A':
        loaded_dict = load_gen_opt_A(scans_dir, splitting_fxn, file_type)
    else:
        loaded_dict = load_gen_opt_B(scans_dir, splitting_fxn, file_type)
    print('Samples loaded:', list(loaded_dict.keys()))
    print('Getting scale factors...')
    check_sample = [False, {}]
    if scale_bools[1]:
        check_sample = check_load_sample_info(proj_dir)
        if not check_sample[0]:
            print('Warning: no sample_info.csv file found!')
            scale_bools[1] = False
        else:
            check_unused_samples(check_sample[1], loaded_dict)
    output_dict = gen_img_scale_factors(loaded_dict, scale_bools,
                                        check_sample[1], verbose=True)
    print('Done')
    return output_dict

def gen_calc_scans_n(inpt_data_dict, inpt_selected_samples):
    """Get the total number of scans that will be run (i.e., represented in
       both the Notebook data dict and in list[selected samples]). For single
       shot-per-image (non-ALC) datasets/Notebooks.

    Parameters
    ----------
    inpt_data_dict : dict
        A standardized dict containing info that will be passed to automated
        and/or semi-automated segmentation functions, loaded from project dir.
        Format:
                {EACH_SAMPLE: {EACH_SPOT: {'img_file': FULL_IMG_PATH,
                               'Align_file': FULL_ALIGN_PATH or '',
                               'rel_file': IMG_FILENAME
                               'scale_factor': float(scale factor for each scan),
                               'scale_from': str(method used get scale factor)},
                               ...},
                 ...}
    inpt_selected_samples : list(str)
        A list of samples selected by a user for running; these should be keys
        in the input data dict.

    Returns
    -------
    n : int
        Total number of scans/images that will be processed.

    """
    n=0
    avl_samples = list(inpt_data_dict.keys())
    for sample in [sample for sample in inpt_selected_samples
                   if sample in avl_samples]:
        for _ in inpt_data_dict[sample].keys():
            n += 1
    return n
