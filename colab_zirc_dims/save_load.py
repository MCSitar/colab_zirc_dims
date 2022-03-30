# -*- coding: utf-8 -*-
import os
import shutil

from . import czd_utils
from . import poly_utils

def check_loadable(run_dir_for_load, verbose=False):
    """Check whether a run directory has a saved polygons subdir with .json file(s).


    Parameters
    ----------
    run_dir_for_load : str
        Path to the run directory where saved polygon .json files may be stored.
    verbose : Bool, optional
        Bool indicating whether fxn will list available samples. The default is False.

    Returns
    -------
    loadable_dir : str or None
        Full path to the directory within run_dir where saved .json files are stored.
        None if no valid directory found.

    """
    success_bool = False
    loadable_dir = None
    full_load_path = os.path.join(run_dir_for_load,
                                  'saved_polygons')
    if os.path.isdir(full_load_path):
        test_files = os.listdir(full_load_path)
        if len(test_files) > 0:
            if test_files[0].endswith('.json'):
                success_bool = True
                loadable_dir = full_load_path
                print('\n')
                print('Valid loadable polygon directory selected:',
                      run_dir_for_load)
                if verbose:
                    print('Available samples:')
                    for each_file in [file.strip('.json') for file in test_files
                                      if '.json' in file]:
                        print(each_file)
    if success_bool:
        return loadable_dir
    else:
        print('Selected directory invalid or does not contain .json files')
        return loadable_dir

def save_sample_json(run_dir, sample_name, spot_names, spot_polys,
                     spots_auto_human = None, spot_tags = None):
    """Save a .json file with loadable polygons, spot info for a sample in a dictionary.

    Parameters
    ----------
    run_dir : str
        Path to the run directory where polygons will be saved.
    sample_name : str
        A unique (per-dataset) sample name. Used to create .json filenames.
    spot_names : list of str
        List of strings with unique names for each saved spot.
    spot_polys : list of lists of dicts
        A list of lists of dicts {x:, y:} with vertices for saved polygons.
        Sub-lists are empty if no polygons created for a spot.
    spots_auto_human : list of str or None, optional
        List of strings indicating whether a polygon was derived from automated
        ('auto') or human ('human') segmentation. The default is None; in this
        case it is assumed that all polygons are auto-generated.
    spot_tags : list of str or None, optional
        List of strings indicating whether a spot was tagged by a user 
        ('True') or not (''). The default is None; in this case it is assumed 
        that all spots have no tag.


    Returns
    -------
    None.

    """
    json_dir = os.path.join(run_dir, 'saved_polygons')
    os.makedirs(json_dir, exist_ok=True)
    sample_json_path = os.path.join(json_dir, str(sample_name) + '.json')
    if not spots_auto_human:
        spots_auto_human = ['auto' for spot in spot_names]
    if not spot_tags:
        spot_tags = ['' for spot in spot_names]
    czd_utils.save_json(sample_json_path,
                        {'spot_names': spot_names, 'spot_polys': spot_polys,
                         'spots_auto_human': spots_auto_human,
                         'spot_tags': spot_tags})

def new_json_save_dict(include_manual_fields = False):
    """Create an empty pre-keyed dict, meant to contain mask-derived polygons
       and spot info.

    Parameters
    ----------
    include_manual_fields : Bool, optional
        Bool determining whether empty lists with keys 'spots_auto_human' and
        'spot_tags' included in returned dict. The default is False.

    Returns
    -------
    dict
        A dict {'spot_names': [], 'spot_polys': [],
                +/- 'spots_auto_human': [],
                +/- 'spot_tags': []}.

    """
    if include_manual_fields:
        return {'spot_names': [], 'spot_polys': [],
                'spots_auto_human': [], 'spot_tags': []}
    else:
        return {'spot_names': [], 'spot_polys': []}

def auto_append_json_dict(json_dict, spot_name, spot_mask, curr_scale_factor = 1.0):
    """Convert a mask to a polygon then add it and a spot name to a dict for saving
       as a loadable .json file.

    Parameters
    ----------
    json_dict : dict of lists
        A dict (see new_json_save_dict()) with format
        {'spot_names': [], 'spot_polys': []}.
    spot_name : str
        Name of spot from which input mask is derived.
    spot_mask :  np array
        A numpy binary array representing the central zircon mask for an image,
        as returned by (successfully) running mos_proc.get_central_mask().
    curr_scale_factor : float, optional
        Scale factor for the current mosaic image. Passed to mask_to_poly(), where
        it is used to adjust polygon tolerance to microns. The default is 1.0.

    Returns
    -------
    None.

    """
    #convert mask to polygon
    new_poly = poly_utils.mask_to_poly(spot_mask, scale_factor = curr_scale_factor)

    #add spot name, polygon to input dictionary
    json_dict['spot_names'].append(spot_name)
    json_dict['spot_polys'].append(new_poly)


def null_append_json_dict(json_dict, spot_name):
    """Append a spot name and empty polygon values to a dict for saving as a
       loadable .json file. Used where auto segmentation returns no acceptable
       mask.

    Parameters
    ----------
    json_dict : dict of lists
        A dict (see new_json_save_dict()) with format
        {'spot_names': [], 'spot_polys': []}.
    spot_name : str
        Name of spot from which input mask is derived.

    Returns
    -------
    None.

    """

    #add spot name, empty polygon to input dictionary
    json_dict['spot_names'].append(spot_name)
    json_dict['spot_polys'].append([])


def transfer_json_files(curr_sample_list, curr_run_dir, load_dir, verbose=False):
    """Transfers loadable .json polygon/info files to a current run directory
       from a load directory if .josn files match currently-selected samples.
       Allows for iterative editing (e.g., in multiple sessions) of automated or
       manual segmentations.

    Parameters
    ----------
    curr_sample_list : list of str
        List of samples for editing. This is created by a user (checkbox selection)
        when running the 'Mosaic_zircon_process' notemook.
    curr_run_dir : str
        Directory for the current run (e.g. ../outputs/current_run_dir.
    load_dir : str
        Full path to directory from which .json files will be transferred;
        in normal course of operations this is first verified with check_loadable().
    verbose : Bool, optional
        Bool indicating whether info (i.e., samples copied) will be printed.
        The default is False.

    Returns
    -------
    curr_json_dir : str
        Path to the directory in the current run directory that .json polygon files
        were transferred to.

    """
    curr_json_dir = os.path.join(curr_run_dir, 'saved_polygons')
    os.makedirs(curr_json_dir, exist_ok = True)
    if verbose:
        print('Copying .json files from load directory to current run directory')
    for each_json in [file for file in os.listdir(load_dir) if file.endswith('.json')]:
        if each_json.strip('.json') in curr_sample_list:
            if verbose:
                print('Copying:', str(each_json))
            shutil.copy(os.path.join(load_dir, each_json),
                        os.path.join(curr_json_dir, each_json))
    if verbose:
        print('Done')
    return curr_json_dir

def find_load_json_polys(load_dir, sample_name, sample_shot_list):
    """Find a matching .json file for a sample and load polygons/data if possible.

    Parameters
    ----------
    load_dir : str
        Full path to directory where loadable polygon .json files are stored.
    sample_name : str
        Name of current sample. Used to find loadable json files.
    sample_shot_list : list of str
        List of shots for current sample.

    Returns
    -------
    Bool
        True if a matching .json polygon file was found for sample, else False.
    list
        List of lists of dicts {x:, y:} with vertices for loaded polygons. Empty
        if no matching .json file was found for sample.
    list
        List of strings ('human' or 'auto') loaded from .json and indicating
        source of segmenation polygons. Empty if no matching .json file was found
        for sample.
    list
        List of strings ('True' or '') loaded from .json and indicating whether
        a user has manually 'tagged' a spot.

    """
    ret_dict = {}
    json_src_name = str(sample_name) + '.json'
    if json_src_name in os.listdir(load_dir) and os.path.exists(load_dir):
        loaded_json_dict = czd_utils.read_json(os.path.join(load_dir, json_src_name))
        if all(len(each_item) == len(sample_shot_list) for
               each_item in loaded_json_dict.values()):
            ret_dict = loaded_json_dict
            print('Polygons, data loaded from', json_src_name)
        else:
            print('Saved file', json_src_name,
                  'found, but data does not match current sample shot list')
    else:
        print('No matching loadable .json file found')
    if ret_dict:
        return [True, ret_dict['spot_polys'], ret_dict['spots_auto_human'],
                ret_dict['spot_tags']]
    else:
        return [False, [], [], []]

def save_mosaic_info_copy(project_dir, run_dir, run_name):
    """Copy mosaic_info.csv and/or sample_info.csv from project dir to run 
       outputs dir to avoid permanent loss of info for shots, polygons if
       original mosaic/sample info csv is ever overwritten/deleted.

    Parameters
    ----------
    project_dir : str
        Path to project directory.
    run_dir : str
        Path to outputs directory created during automatic or semi-automatic
        processing.
    run_name : str
        Unique run name (includes date-time).

    Returns
    -------
    None.

    """
    for csv_type in ['mosaic_info.csv', 'sample_info.csv']:
        new_csv_name = run_name + '_' + csv_type.strip('.csv') + '_copy.csv'
        new_csv_path = os.path.join(run_dir, new_csv_name)
        if csv_type in os.listdir(project_dir):
            shutil.copy(os.path.join(project_dir, csv_type),
                        new_csv_path)
