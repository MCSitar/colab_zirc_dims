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
from IPython.display import display

try:
    from google.colab.patches import cv2_imshow
except ModuleNotFoundError:
    print('WARNING: google.colab not found; (machine != Colab VM?).',
          'Some colab_zirc_dims visualization functions will fail.')
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
           'test_eval',
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
                                      inpt_mos_data_dict[eachsample]['Max_zircon_size'],
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
        model_lib_loc = 'https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/czd_model_library.json'
    model_lib_list = czd_utils.json_from_path_or_url(model_lib_loc)
    model_labels = [each_dict['desc'] for each_dict in model_lib_list]
    model_picker = widgets.Dropdown(options=model_labels, value=model_labels[0],
                                    description='Model:',
                                    layout={'width': 'max-content'})
    def select_download_model(selection):
        cwd = os.getcwd()
        if selection is not None:
            mut_curr_model_d.update(model_lib_list[model_labels.index(selection)])
            print('Selected:', mut_curr_model_d['name'])
            if os.path.exists(os.path.join(cwd, mut_curr_model_d['name'])):
                if czd_utils.check_url(mut_curr_model_d['model_url']):
                    print('Model already downloaded')
                else:
                    print('Model already copied to current working directory')
            else:
                #download weights if url; attempt to copy if not
                if czd_utils.check_url(mut_curr_model_d['model_url']):
                    print('Downloading:', mut_curr_model_d['name'])
                    print('...')
                    urllib.request.urlretrieve(mut_curr_model_d['model_url'],
                                              os.path.join(cwd,
                                                            mut_curr_model_d['name']))
                    print('Download finished')
                else:
                    print('Copying:', mut_curr_model_d['name'])
                    shutil.copy(mut_curr_model_d['model_url'],
                                os.path.join(cwd,
                                             mut_curr_model_d['name']))
                    print('Done copying')
    model_out = widgets.interactive_output(select_download_model,
                                           {'selection': model_picker})
    display(model_picker, model_out)

def test_eval(inpt_selected_samples, inpt_mos_data_dict, inpt_predictor,
              d2_metadata, n_scans_sample =3):
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

    Returns
    -------
    None.

    """
    for eachsample in inpt_selected_samples:
        each_mosaic = mos_proc.MosImg(inpt_mos_data_dict[eachsample]['Mosaic'],
                                      inpt_mos_data_dict[eachsample]['Align_file'],
                                      inpt_mos_data_dict[eachsample]['Max_zircon_size'],
                                      inpt_mos_data_dict[eachsample]['Offsets'])
        scan_sample = random.sample(inpt_mos_data_dict[eachsample]['Scan_dict'].keys(),
                                    n_scans_sample)
        print(4 * "\n")
        print(str(eachsample) + ':')
        print('Scale factor:', each_mosaic.scale_factor, 'µm/pixel')
        for eachscan in scan_sample:
            each_mosaic.set_subimg(*inpt_mos_data_dict[eachsample]['Scan_dict'][eachscan])
            print(str(eachscan), 'processed subimage:')
            outputs = inpt_predictor(each_mosaic.sub_img)
            central_mask = mos_proc.get_central_mask(outputs)
            v = Visualizer(each_mosaic.sub_img[:, :, ::-1],
                      metadata=d2_metadata,
                      scale=2
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2_imshow(out.get_image()[:, :, ::-1])
            if central_mask[0]:
                print(str(eachscan), 'analyzed (scanned) zircon image:')
                each_props = mos_proc.overlay_mask_and_get_props(central_mask[1],
                                                                each_mosaic.sub_img,
                                                                eachscan,
                                                                display_bool = True,
                                                                scale_factor=each_mosaic.scale_factor)
                _ = mos_proc.parse_properties(each_props,
                                              each_mosaic.scale_factor,
                                              eachscan, verbose = True)
            else:
                print(str(eachscan), 'analyzed (scanned) zircon image:')
                mos_proc.save_show_results_img(each_mosaic.sub_img, eachscan,
                                               display_bool = True,
                                               scale_factor = each_mosaic.scale_factor)

def auto_proc_sample(run_dir, img_save_root_dir, csv_save_dir, eachsample,
                     inpt_save_polys_bool, inpt_mos_data_dict, inpt_predictor,
                     inpt_alt_methods, eta_trk, out_trk):
    """Automatically process and save results from a single sample in an ALC
       dataset.

    Parameters
    ----------
    run_dir : str
        Path to (Google Drive) run directory where results will be saved.
    img_save_root_dir : str
        Path to dir where mask images for each scan will be saved.
    csv_save_dir : str
        Path to dir where .csv files with zircon dimensions for each scan
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

    #loads mosaic file, automatically increasing contrast if needed
    each_mosaic = mos_proc.MosImg(inpt_mos_data_dict[eachsample]['Mosaic'],
                                  inpt_mos_data_dict[eachsample]['Align_file'],
                                  inpt_mos_data_dict[eachsample]['Max_zircon_size'],
                                  inpt_mos_data_dict[eachsample]['Offsets'])

    #extracts zircon subimage and runs predictor for each scan
    for eachscan in inpt_mos_data_dict[eachsample]['Scan_dict'].keys():
        #start timing for spot
        eta_trk.start()

        #reset output text display
        out_trk.reset()
        out_trk.print_txt(eta_trk.str_eta)
        out_trk.print_txt(' '.join(['Processing:',
                                    str(eachsample),
                                    str(eachscan)]))
        each_mosaic.set_subimg(*inpt_mos_data_dict[eachsample]['Scan_dict'][eachscan])
        central_mask = segment.segment(each_mosaic, inpt_predictor,
                                       inpt_alt_methods, out_trk)
        if central_mask[0]:
            out_trk.print_txt('Success')

            #saves mask image and gets properties
            each_props = mos_proc.overlay_mask_and_get_props(central_mask[1],
                                                             each_mosaic.sub_img,
                                                             str(eachscan),
                                                             display_bool = False,
                                                             save_dir=each_img_save_dir,
                                                             scale_factor=each_mosaic.scale_factor)

            #adds properties to output list
            temp_props_list = mos_proc.parse_properties(each_props,
                                                        each_mosaic.scale_factor,
                                                        str(eachscan),
                                                        verbose = False)
            output_data_list.append(temp_props_list)

            #optionally converts mask to polygon and adds it to json_dict for saving
            if inpt_save_polys_bool:
                save_load.auto_append_json_dict(each_json_dict, str(eachscan),
                                                central_mask[1], each_mosaic.scale_factor)

        #gives empty outputs if no mask image
        else:
            null_properties = mos_proc.parse_properties([],
                                                        each_mosaic.scale_factor,
                                                        str(eachscan))
            output_data_list.append(null_properties)
            mos_proc.save_show_results_img(each_mosaic.sub_img, str(eachscan),
                                           display_bool = False,
                                           save_dir = each_img_save_dir,
                                           scale_factor = each_mosaic.scale_factor)
            #optionally adds empty polygons to json_dict for saving
            if inpt_save_polys_bool:
                save_load.null_append_json_dict(each_json_dict, str(eachscan))

        #get total time for spot
        eta_trk.stop_update_eta()
    #deletes mosaic class instance after processing. \
    # May or may not reduce RAM during automated processing; probably best practice.
    del each_mosaic

    #converts collected data to pandas DataFrame, saves as .csv
    output_dataframe = pd.DataFrame(output_data_list,
                                    columns=['Analysis', 'Area (µm^2)',
                                             'Convex area (µm^2)',
                                             'Eccentricity',
                                             'Equivalent diameter (µm)',
                                             'Perimeter (µm)',
                                             'Major axis length (µm)',
                                             'Minor axis length (µm)',
                                             'Circularity',
                                             'Scale factor (µm/pixel)'])
    csv_filename = str(eachsample) + '_zircon_dimensions.csv'
    output_csv_filepath = os.path.join(csv_save_dir, csv_filename)
    czd_utils.save_csv(output_csv_filepath, output_dataframe)

    #optionally saves polygons to .json
    if inpt_save_polys_bool:
        save_load.save_sample_json(run_dir, str(eachsample),
                                  **each_json_dict)

def full_auto_proc(inpt_root_dir, inpt_selected_samples, inpt_mos_data_dict,
                   inpt_predictor, inpt_save_polys_bool, inpt_alt_methods,
                   id_string = '', stream_output=False):
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

    #creates a directory for zircon dimension .csv files
    csv_save_dir = os.path.join(run_dir, 'zircon_dimensions')
    os.makedirs(csv_save_dir)

    #initialize class instances for ETA, other output display
    eta_trk = eta.EtaTracker(czd_utils.alc_calc_scans_n(inpt_mos_data_dict,
                                                        inpt_selected_samples))
    out_trk = eta.OutputTracker(n_blank_lines=10, stream_outputs=stream_output)

    #starts loop through dataset dictionary
    for eachsample in inpt_selected_samples:
        auto_proc_sample(run_dir, img_save_root_dir, csv_save_dir, eachsample,
                         inpt_save_polys_bool, inpt_mos_data_dict, inpt_predictor,
                         inpt_alt_methods, eta_trk, out_trk)
        gc.collect()

    return run_dir
