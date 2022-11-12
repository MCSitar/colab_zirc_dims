# -*- coding: utf-8 -*-
"""
Functions to simplify the colab_zirc_dims auto-processing notebook for ALC
datasets. Code here was previously contained within cells in the notebook
itself.
"""

import os
import random
import gc
import datetime
import urllib.request
import shutil
import time
from IPython.display import display
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

try:
    from google.colab.patches import cv2_imshow
except ModuleNotFoundError:
    print('WARNING: google.colab not found; (machine != Colab VM?).',
          'Using local copy of patches for visualization functions.')
    from ..jupyter_colab_compat.patches import cv2_imshow
    pass
try:
    from detectron2.utils.visualizer import Visualizer
except ModuleNotFoundError:
    print('WARNING: Detectron2 not installed on (virtual?) machine;',
          'colab_zirc_dims ALC image segmentation functions unavailable')
    pass
import ipywidgets as widgets
import skimage.io as skio
import pandas as pd


from .. import czd_utils
from .. import mos_proc
from .. import save_load
from .. import segment
from .. import eta

__all__ = ['select_samples_fxn',
           'inspect_data',
           'select_download_model_interface',
           'demo_eval',
           'auto_proc_sample',
           'full_auto_proc']


def select_samples_fxn(inpt_mos_data_dict, mutable_list):
    """Open an array of checkboxes with sample names from a dataset info dict
       and modify a global list so that it contains ordered sample names
       selected by a user.

    Parameters
    ----------
    inpt_mos_data_dict : dict
        A dict (as returned by czd_utils.load_data_dict()) with all sample data
        for an ALC dataset.
    mutable_list : list
        A list with global scope that will contain user-selected sample names.

    Returns
    -------
    None.

    """
    #dynamically created checkboxes with automatically-updated output; \
    # partially based on https://stackoverflow.com/a/57230942
    checkboxes = [widgets.Checkbox(value=True, description=sample)
                  for sample in list(inpt_mos_data_dict.keys())]
    args_dict = {list(inpt_mos_data_dict.keys())[i]: checkbox for i,
                 checkbox in enumerate(checkboxes)}
    ui = widgets.GridBox(checkboxes,
                         layout=widgets.Layout(grid_template_columns="repeat(5, 200px)"))

    def select_data(**kwargs):
        mutable_list.clear()

        for key in kwargs:
            if kwargs[key] is True:
                mutable_list.append(key)
        print('Selected samples:', mutable_list)

    out = widgets.interactive_output(select_data, args_dict)
    display(ui, out)

def inspect_data(inpt_mos_data_dict, inpt_selected_samples, n_scans_sample = 3):
    """Plot n randomly-sampled, scaled scan images from each sample in a list
       of user-selected samples.

    Parameters
    ----------
    inpt_mos_data_dict : dict
        A dict (as returned by czd_utils.load_data_dict()) with all sample data
        for an ALC dataset.
    inpt_selected_samples : list[str]
        A list of strings matching user-selected sample names in inpt_mos_data_dict.
    n_scans_sample : int, optional
        Number of scans to randomly sample and display from each sample in
        dataset. The default is 3.

    Returns
    -------
    None.

    """
    for eachsample in inpt_selected_samples:
        print(3*'\n')
        each_mosaic = mos_proc.MosImg(inpt_mos_data_dict[eachsample]['Mosaic'],
                                      inpt_mos_data_dict[eachsample]['Align_file'],
                                      inpt_mos_data_dict[eachsample]['Max_grain_size'],
                                      inpt_mos_data_dict[eachsample]['Offsets'])
        scan_sample = random.sample(inpt_mos_data_dict[eachsample]['Scan_dict'].keys(),
                                    n_scans_sample)

        print(str(eachsample) + ':')
        print('Scale factor:', each_mosaic.scale_factor, 'µm/pixel')
        for eachscan in scan_sample:
            each_mosaic.set_subimg(*inpt_mos_data_dict[eachsample]['Scan_dict'][eachscan])
            print(str(eachscan) + ':')
            each_y_extent, each_x_extent = [size * each_mosaic.scale_factor for
                                            size in each_mosaic.sub_img.shape[:2]]
            skio.imshow(each_mosaic.sub_img, extent=[0, each_x_extent, 0, each_y_extent])
            skio.show()


def select_download_model_interface(mut_curr_model_d, model_lib_loc = 'default'):
    """Open user interface (dynamically-populated dropdown menu) for selection
       and download/copying of RCNN models available through either the
       colab_zirc_dims model library or a user-provided model library.

    Parameters
    ----------
    mut_curr_model_d : dict
        A dict with global scope that will contain info copied from model library
        json for the user-selected model. Modified in place.
    model_lib_loc : str, optional
        url or path (if downloaded or mounted to virtual machine) to model library
        json file. The default is 'default', which gets the current model lib
        from the colab_zirc_dims GitHub repo.
        If users want to use their own model library (with valid download links
        and/or paths to models on a mounted Google Drive and formatting
        matching colab_zirc_dims model_library.json), they can create one then
        upload the file to their Google Drive and input its path here.


    Returns
    -------
    None.

    """
    if model_lib_loc == 'default':
        if os.path.exists(os.path.join(os.getcwd(), 'czd_model_library.json')):
            model_lib_loc = os.path.join(os.getcwd(), 'czd_model_library.json')
        else:
            if not czd_utils.connected_to_internet():
                raise Exception(' '.join(['No model library file accessible.', 
                                          'Such a file is available at',
                                          ''.join(['https://raw.githubusercontent.com/',
                                          'MCSitar/colab_zirc_dims/main/',
                                          'czd_model_library.json'])+'.',
                                          'To enable this model download /',
                                          'selection UI, please either',
                                          'download it manually to your current',
                                          'working directory or run this',
                                          'notebook at least once while',
                                          'connected to the internet to',
                                          'do so automatically']))
            model_lib_loc = ''.join(['https://raw.githubusercontent.com/',
                                     'MCSitar/colab_zirc_dims/main/',
                                     'czd_model_library.json'])
            urllib.request.urlretrieve(model_lib_loc,
                                       os.path.join(os.getcwd(), 
                                                    'czd_model_library.json'))
    model_lib_list = czd_utils.json_from_path_or_url(model_lib_loc)
    model_labels = [each_dict['desc'] for each_dict in model_lib_list]
    model_picker = widgets.Dropdown(options=model_labels, value=model_labels[0],
                                    description='Model:',
                                    layout={'width': 'max-content'})
    def select_download_model(selection):
        cwd = os.getcwd()
        weights_yml_dirpath = os.path.join(cwd, 'downloaded_czd_model_files')
        os.makedirs(weights_yml_dirpath, exist_ok=True)
        if selection is not None:
            mut_curr_model_d.clear()
            mut_curr_model_d.update(model_lib_list[model_labels.index(selection)])
            print('Selected:', mut_curr_model_d['name'])
            target_weights_path = os.path.join(weights_yml_dirpath, 
                                               mut_curr_model_d['name'])
            if os.path.exists(target_weights_path):
                if czd_utils.check_url(mut_curr_model_d['model_url']):
                    print('Model already downloaded')
                else:
                    print('Model already copied to current working directory')
                mut_curr_model_d.update({'selected_model_weights':
                                         target_weights_path})
            else:
                #download weights if url (default); attempt to copy as path if not
                if czd_utils.check_url(mut_curr_model_d['model_url']):
                    if not czd_utils.connected_to_internet():
                        print(' '.join(['No model weights file accessible',
                                        'for selected model. Please either',
                                        'copy weights to directory manually',
                                        'or run this notebook while connected',
                                        'to the internet to do so automatically.',
                                        'The current model library .json file',
                                        'indicates that weights for the',
                                        'selected model are available at:', 
                                        str(mut_curr_model_d['model_url']),
                                        'They should be copied to:',
                                        str(target_weights_path)]))
                    else:
                        print('Downloading:', mut_curr_model_d['name'])
                        print('...')
                        urllib.request.urlretrieve(mut_curr_model_d['model_url'],
                                                   target_weights_path)
                        print('Download finished')
                        mut_curr_model_d.update({'selected_model_weights':
                                                 target_weights_path})
                else:
                    print('Copying:', mut_curr_model_d['name'])
                    shutil.copy(mut_curr_model_d['model_url'],
                                target_weights_path)
                    print('Done copying')
                    mut_curr_model_d.update({'selected_model_weights':
                                             target_weights_path})
            if 'full_config_yaml_name' in mut_curr_model_d.keys():
                target_yaml_path = os.path.join(weights_yml_dirpath,
                                                mut_curr_model_d['full_config_yaml_name'])
                src_yaml_path_or_url = mut_curr_model_d['yaml_url']
                if os.path.exists(target_yaml_path):
                    if czd_utils.check_url(src_yaml_path_or_url):
                        print('Config. .yaml already downloaded')
                    else:
                        print('Config. .yaml already copied to',
                              'current working directory')
                    mut_curr_model_d.update({'selected_config_yaml':
                                             target_yaml_path})
                else:
                    #download yaml if url (default); attempt to copy as path if not
                    if czd_utils.check_url(src_yaml_path_or_url):
                        if not czd_utils.connected_to_internet():
                            print(' '.join(['No config .yaml file accessible',
                                            'for selected model. Please either',
                                            'copy .yaml to directory manually',
                                            'or run this notebook while connected',
                                            'to the internet to do so automatically.',
                                            'The current model library .json file',
                                            'indicates that the .yaml config for the',
                                            'selected model is available at:', 
                                            str(src_yaml_path_or_url),
                                            'It should be copied to:',
                                            str(target_yaml_path)]))
                        else:
                            print('Downloading:', 
                                  mut_curr_model_d['full_config_yaml_name'])
                            print('...')
                            urllib.request.urlretrieve(src_yaml_path_or_url,
                                                       target_yaml_path)
                            print('Download finished')
                            mut_curr_model_d.update({'selected_config_yaml':
                                                     target_yaml_path})
                    else:
                        print('Copying:', mut_curr_model_d['name'])
                        shutil.copy(src_yaml_path_or_url,
                                    target_yaml_path)
                        print('Done copying')
                        mut_curr_model_d.update({'selected_config_yaml':
                                                 target_yaml_path})
                
    model_out = widgets.interactive_output(select_download_model,
                                           {'selection': model_picker})
    display(model_picker, model_out)

def demo_eval(inpt_selected_samples, inpt_mos_data_dict, inpt_predictor,
              d2_metadata, n_scans_sample =3, src_str = None, **kwargs):
    """Plot predictions and extract grain measurements for n randomly-selected
       scans from each selected sample in an ALC dataset.

    Parameters
    ----------
    inpt_selected_samples : list[str]
        A list of strings matching user-selected sample names in inpt_mos_data_dict.
    inpt_mos_data_dict : dict
        A dict (as returned by czd_utils.load_data_dict()) with all sample data
        for an ALC dataset.
    inpt_predictor : Detectron2 Predictor class instance
        A D2 instance segmentation predictor to apply to images.
    d2_metadata : Detectron2 catalog metadata instance
        Metadata matching predictor training for segmentation results visualization.
    n_scans_sample : int, optional
        Number of randomly-selected scans from each sample to segment and measure.
        The default is 3.
    src_str : str or None, optional
        String for selecting spot names - if not None, spots will only be
        displayed if their names match the string. The default is None.
    **kwargs :
        Plotting-related kwargs, passed in full to czd_utils.save_show_results_img.
            fig_dpi = int; will set plot dpi to input integer.
            show_ellipse = bool; will plot ellipse corresponding
                           to maj, min axes if True.
            show_legend = bool; will plot a legend on plot if
                          True.

    Returns
    -------
    None.

    """
    for eachsample in inpt_selected_samples:
        each_mosaic = mos_proc.MosImg(inpt_mos_data_dict[eachsample]['Mosaic'],
                                      inpt_mos_data_dict[eachsample]['Align_file'],
                                      inpt_mos_data_dict[eachsample]['Max_grain_size'],
                                      inpt_mos_data_dict[eachsample]['Offsets'])
        scan_sample = random.sample(inpt_mos_data_dict[eachsample]['Scan_dict'].keys(),
                                    n_scans_sample)
        #if src_str provided, ignore sample size and instead search for string
        if isinstance(src_str, type('a')):
            scan_sample = [key for key in
                           inpt_mos_data_dict[eachsample]['Scan_dict'].keys()
                           if src_str in str(key)]
        print(4 * "\n")
        print(str(eachsample) + ':')
        print('Scale factor:', each_mosaic.scale_factor, 'µm/pixel')
        for eachscan in scan_sample:
            each_mosaic.set_subimg(*inpt_mos_data_dict[eachsample]['Scan_dict'][eachscan])
            print(str(eachscan), 'processed subimage:')
            outputs = inpt_predictor(each_mosaic.sub_img[:, :, ::-1])
            central_mask = mos_proc.get_central_mask(outputs)
            v = Visualizer(each_mosaic.sub_img[:, :, ::-1],
                      metadata=d2_metadata,
                      scale=2
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2_imshow(out.get_image()[:, :, ::-1])
            if central_mask[0]:
                print(str(eachscan), 'analyzed (scanned) grain image:')
                each_props = mos_proc.overlay_mask_and_get_props(central_mask[1],
                                                                each_mosaic.sub_img,
                                                                eachscan,
                                                                display_bool = True,
                                                                scale_factor=each_mosaic.scale_factor,
                                                                **kwargs)
                _ = mos_proc.parse_properties(each_props,
                                              each_mosaic.scale_factor,
                                              eachscan, verbose = True)
            else:
                print(str(eachscan), 'analyzed (scanned) grain image:')
                mos_proc.save_show_results_img(each_mosaic.sub_img, eachscan,
                                               display_bool = True,
                                               scale_factor = each_mosaic.scale_factor)

def auto_proc_sample(run_dir, img_save_root_dir, csv_save_dir, eachsample,
                     inpt_save_polys_bool, inpt_mos_data_dict, inpt_predictor,
                     inpt_alt_methods, eta_trk, out_trk, save_spot_time=False,
                     n_jobs=4, **kwargs):
    """Automatically process and save results from a single sample in an ALC
       dataset.

    Parameters
    ----------
    run_dir : str
        Path to (Google Drive) run directory where results will be saved.
    img_save_root_dir : str
        Path to dir where mask images for each scan will be saved.
    csv_save_dir : str
        Path to dir where .csv files with grain dimensions for each scan
        will be saved.
    eachsample : str
        Sample name (must be in inpt_mos_data_dict) for the sample being processed.
    inpt_save_polys_bool : bool
        Bool indicating whether polygon approximations of central grain masks
        should be saved for future inspection/editing.
    inpt_mos_data_dict : dict
        A dict (as returned by czd_utils.load_data_dict()) with all sample data
        for an ALC dataset.
    inpt_predictor : Detectron2 Predictor class instance
        A D2 instance segmentation predictor to apply to images.
    inpt_alt_methods : list[bool]
        A list of bools corresponding to alternate methods (scale jittering,
        contrast enhancement, and/or Otsu thresholding) to iteratively try on
        a scan image if inital central grain segmentation is unsuccesful.
        Format: [Try_zoomed_out_subimage, Try_zoomed_in_subimage,
                 Try_contrast_enhanced_subimage,
                 Try_Otsu_thresholding]
    eta_trk : colab_zirc_dims.eta.EtaTracker instance
        A timer/rate calculator that calculates a running eta.
    out_trk : colab_zirc_dims.eta.OutputTracker instance
        Optionally (depending on initialization params) refreshes text output
        for every scan instead of streaming all print() data to output box.
    save_spot_time : bool, optional
        If True, push per-spot segmentation times to output data .csv file.
        The default is False.
    n_jobs : int, optional
        Number of parallel threads to run using joblib during processing.

    Returns
    -------
    None.

    """
    #directory for saving images for each sample
    each_img_save_dir = os.path.join(img_save_root_dir, str(eachsample))
    os.makedirs(each_img_save_dir)

    # a list of lists, later converted to a Pandas dataframe, \
    # which is in turn saved as a .csv file
    output_data_list = []

    #if mask polygons are being saved, creates a holder for polygons and other data
    if inpt_save_polys_bool:
        each_json_dict = save_load.new_json_save_dict()

    #parallel processing does not maintain input sequence order. We don't want \
    # that for output data. The dicts below accumulate unsorted processing outputs
    # for subsequent sorting.
    unsort_output_data_dict = {}
    unsort_each_json_dict = {}

    #loads mosaic file, automatically increasing contrast if needed. Links it \
    #to the current sample's scan dict to make it iterable for parallel proc.
    mos_iterator = mos_proc.IterableMosImg(inpt_mos_data_dict[eachsample]['Mosaic'],
                                           inpt_mos_data_dict[eachsample]['Scan_dict'],
                                           inpt_alt_methods,
                                           inpt_mos_data_dict[eachsample]['Align_file'],
                                           inpt_mos_data_dict[eachsample]['Max_grain_size'],
                                           inpt_mos_data_dict[eachsample]['Offsets'])

    #extracts grain subimage and runs predictor for each scan. Takes outputs
    # from __iter__ function of a mos_proc.IterableMosImg instance
    def parallel_proc_scans(iter_count, eachscan, imgs, each_scale_factor):

        #restrict printing to a reasonable rate to avoid strange
        if iter_count % n_jobs == 0 or out_trk.stream_outputs:
            #reset output text display, prints some useful info
            out_trk.reset_and_print([eta_trk.str_eta,
                                     ' '.join(['Processing:',
                                               str(eachsample),
                                               str(eachscan)])])
        time_start_seg = time.perf_counter()
        central_mask=segment.segment_given_imgs(imgs, inpt_predictor,
                                                try_bools=inpt_alt_methods,
                                                **kwargs)
        ##time for segmentation. Will only be accurate if n_jobs == 1.
        each_total_seg_time = time.perf_counter()-time_start_seg
        if central_mask[0]:
            #saves mask image and gets properties
            each_props = mos_proc.overlay_mask_and_get_props(central_mask[1],
                                                             imgs[0],
                                                             str(eachscan),
                                                             display_bool = False,
                                                             save_dir=each_img_save_dir,
                                                             scale_factor=each_scale_factor)

            #adds properties to output list
            temp_props_list = mos_proc.parse_properties(each_props,
                                                        each_scale_factor,
                                                        str(eachscan),
                                                        verbose = False)

            #add segmentation time dependent on user params
            if save_spot_time:
                temp_props_list.append(each_total_seg_time)

            unsort_output_data_dict[str(eachscan)] = temp_props_list
            #output_data_list.append(temp_props_list)

            #optionally converts mask to polygon and adds it to json_dict for saving
            if inpt_save_polys_bool:
                unsort_each_json_dict[str(eachscan)] = {'spot_names':[],
                                                        'spot_polys':[]}
                save_load.auto_append_json_dict(unsort_each_json_dict[str(eachscan)],
                                                str(eachscan),
                                                central_mask[1], 
                                                each_scale_factor)

        #gives empty outputs if no mask image
        else:
            null_properties = mos_proc.parse_properties([],
                                                        each_scale_factor,
                                                        str(eachscan))
            #add segmentation time dependent on user params
            if save_spot_time:
                null_properties.append(each_total_seg_time)

            unsort_output_data_dict[str(eachscan)] = null_properties
            mos_proc.save_show_results_img(imgs[0], str(eachscan),
                                           display_bool = False,
                                           save_dir = each_img_save_dir,
                                           scale_factor = each_scale_factor)
            #optionally adds empty polygons to json_dict for saving
            if inpt_save_polys_bool:
                unsort_each_json_dict[str(eachscan)] = {'spot_names':[],
                                                        'spot_polys':[]}
                save_load.null_append_json_dict(unsort_each_json_dict[str(eachscan)],
                                                str(eachscan))

        #get total time for spot
        eta_trk.stop_update_eta()

    plt.ioff()
    #run our big per-scan parallelized segmentation function for each scan
    # in the sample.
    Parallel(n_jobs=n_jobs, 
             require='sharedmem'
             )(delayed(parallel_proc_scans)(*iter_out)
                                  for iter_out in mos_iterator)

    #fix order for output data
    for eachscan in inpt_mos_data_dict[eachsample]['Scan_dict'].keys():
        match_key = str(eachscan)
        output_data_list.append(unsort_output_data_dict[match_key])
        if inpt_save_polys_bool:
            add_names = unsort_each_json_dict[match_key]['spot_names']
            add_polys = unsort_each_json_dict[match_key]['spot_polys']
            each_json_dict['spot_names'] = [*each_json_dict['spot_names'],
                                            *add_names]
            each_json_dict['spot_polys'] = [*each_json_dict['spot_polys'],
                                            *add_polys]


    #deletes mosaic class instance after processing. \
    # May or may not reduce RAM during automated processing; probably best practice.
    del mos_iterator
    
    #additional parameters to save to output (i.e., for performance monitoring)
    addit_save_fields = []
    
    if save_spot_time:
        addit_save_fields.append('spot_time')
    
    #converts collected data to pandas DataFrame, saves as .csv
    output_dataframe = pd.DataFrame(output_data_list,
                                    columns=czd_utils.get_save_fields(proj_type='mosaic',
                                                                      save_type='auto',
                                                                      addit_fields=addit_save_fields))
    csv_filename = str(eachsample) + '_grain_dimensions.csv'
    output_csv_filepath = os.path.join(csv_save_dir, csv_filename)
    czd_utils.save_csv(output_csv_filepath, output_dataframe)

    #optionally saves polygons to .json
    if inpt_save_polys_bool:
        save_load.save_sample_json(run_dir, str(eachsample),
                                  **each_json_dict)

def full_auto_proc(inpt_root_dir, inpt_selected_samples, inpt_mos_data_dict,
                   inpt_predictor, inpt_save_polys_bool, inpt_alt_methods,
                   id_string = '', stream_output=False, save_spot_time=False,
                   n_jobs=2, **kwargs):
    """Automatically segment, measure, and save results for every selected
    sample in an ALC dataset.

    Parameters
    ----------
    inpt_root_dir : str
        Path to (Google Drive mounted) project dir.
    inpt_selected_samples : list[str]
        A list of strings matching user-selected sample names in inpt_mos_data_dict
    inpt_mos_data_dict : dict
        A dict (as returned by czd_utils.load_data_dict()) with all sample data
        for an ALC dataset.
    inpt_predictor : Detectron2 Predictor class instance
        A D2 instance segmentation predictor to apply to images.
    inpt_save_polys_bool : bool
        Bool indicating whether polygon approximations of central grain masks
        should be saved for future inspection/editing.
    inpt_alt_methods : list[bool]
        A list of bools corresponding to alternate methods (scale jittering,
        contrast enhancement, and/or Otsu thresholding) to iteratively try on
        a scan image if inital central grain segmentation is unsuccesful.
        Format: [Try_zoomed_out_subimage, Try_zoomed_in_subimage,
                 Try_contrast_enhanced_subimage,
                 Try_Otsu_thresholding]
    id_string : str, optional
        A string to add to front of default (date-time) output folder name.
        The default is ''.
    stream_output : bool, optional
        A bool indicating whether output text will be 'streamed' (all text goes
        straight to output, no clearing or refreshing) or displayed in an
        automatically-refreshing block at the top of cell outputs.
        The default is False.
    n_jobs : int, optional
        Number of parallel threads to run using joblib during processing.

    Returns
    -------
    run_dir : str
        Path to Google Drive processing run results folder; available for return
        into notebook global scope for easy loading of any saved polygon .json
        files into GUI.

    """
    #main output directory path
    root_output_dir = os.path.join(inpt_root_dir, 'outputs')


    #creates output directory if it does not already exist
    if not os.path.exists(root_output_dir):
        os.makedirs(root_output_dir)

    #creates a main directory (with datetime stamp) for this processing run
    curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir_str = 'auto_zirc_proccessing_run_' + curr_datetime
    if str(id_string):
        run_dir_str = str(id_string) + '_' + run_dir_str
    run_dir = os.path.join(root_output_dir, run_dir_str)
    os.makedirs(run_dir)

    #copy mosaic info csv (for reloading at later point in case original changed)
    save_load.save_mosaic_info_copy(inpt_root_dir, run_dir, run_dir_str)

    #creates a root directory for saved images
    img_save_root_dir = os.path.join(run_dir, 'mask_images')
    os.makedirs(img_save_root_dir)

    #creates a directory for grain dimension .csv files
    csv_save_dir = os.path.join(run_dir, 'grain_dimensions')
    os.makedirs(csv_save_dir)

    #initialize class instances for ETA, other output display
    eta_trk = eta.EtaTracker(czd_utils.alc_calc_scans_n(inpt_mos_data_dict,
                                                        inpt_selected_samples))
    out_trk = eta.OutputTracker(n_blank_lines=10, stream_outputs=stream_output)
    
    #start timing for ETA
    eta_trk.start()

    #starts loop through dataset dictionary
    for eachsample in inpt_selected_samples:
        auto_proc_sample(run_dir, img_save_root_dir, csv_save_dir, eachsample,
                         inpt_save_polys_bool, inpt_mos_data_dict, inpt_predictor,
                         inpt_alt_methods, eta_trk, out_trk,
                         save_spot_time=save_spot_time, n_jobs=n_jobs, **kwargs)
        gc.collect()
    out_trk.print_txt('Done')

    return run_dir
