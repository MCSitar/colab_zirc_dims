# -*- coding: utf-8 -*-
"""
Functions to simplify data loading, automated processing of detrital zircon
RL images with a single image per shot (unlike ALC mosaics) in Colab Notebook.
"""
import os
import random
import gc
import datetime
import time
from IPython.display import display
from joblib import Parallel, delayed
try:
    from detectron2.utils.visualizer import Visualizer
except ModuleNotFoundError:
    print('WARNING: Detectron2 not installed on (virtual?) machine;',
          'colab_zirc_dims generalized image segmentation functions unavailable')
    pass
try:
    from google.colab.patches import cv2_imshow
except ModuleNotFoundError:
    print('WARNING: google.colab not found; (machine != Colab VM?).',
          'Using local copy of patches for visualization functions.')
    from ..jupyter_colab_compat.patches import cv2_imshow
    pass
import ipywidgets as widgets
import skimage.io as skio
import pandas as pd

from .. import czd_utils
from .. import mos_proc
from .. import save_load
from .. import segment
from .. import eta

__all__ = ['gen_data_load_interface',
           'gen_inspect_data',
           'gen_demo_eval',
           'gen_auto_proc_sample',
           'full_auto_proc']

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
            out_dict = czd_utils.load_gen_data_dict(root_proj_dir,
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

def gen_inspect_data(inpt_loaded_data_dict, inpt_selected_samples,
                     n_scans_sample = 3):
    """Plot n randomly-sampled, scaled scan images from each sample in a list
       of user-selected samples.

    Parameters
    ----------
    inpt_loaded_data_dict : dict
        A dict (as returned by czd_utils.load_gen_data_dict()) with all
        sample data for a non-ALC dataset.
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

        sample_dict = inpt_loaded_data_dict[eachsample]
        scan_sample = random.sample(list(sample_dict.keys()),
                                    n_scans_sample)
        print(str(eachsample) + ':')
        for eachscan in scan_sample:
            scale_factor = sample_dict[eachscan]['scale_factor']
            scale_from = sample_dict[eachscan]['scale_from']
            each_img = skio.imread(sample_dict[eachscan]['img_file'])
            print(str(eachscan) + ':')
            print('Scale factor:', round(scale_factor, 5),
                  'µm/pixel; scale from:', scale_from)
            each_y_extent, each_x_extent = [size * scale_factor for
                                            size in each_img.shape[:2]]
            skio.imshow(each_img, extent=[0, each_x_extent, 0, each_y_extent])
            skio.show()

def gen_demo_eval(inpt_selected_samples, inpt_loaded_data_dict, inpt_predictor,
                  d2_metadata, n_scans_sample =3, src_str=None, **kwargs):
    """Plot predictions and extract grain measurements for n randomly-selected
       scans from each selected sample in an ALC dataset.

    Parameters
    ----------
    inpt_selected_samples : list[str]
        A list of strings matching user-selected sample names in inpt_mos_data_dict.
    inpt_loaded_data_dict : dict
        A dict (as returned by czd_utils.load_gen_data_dict()) with all
        sample data for a non-ALC dataset.
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
            show_box = bool; will plot the minimum area rect.
                       if True.
            show_legend = bool; will plot a legend on plot if
                          True.

    Returns
    -------
    None.

    """
    for eachsample in inpt_selected_samples:
        sample_dict = inpt_loaded_data_dict[eachsample]
        scan_sample = random.sample(list(sample_dict.keys()),
                                    n_scans_sample)
        if isinstance(src_str, type('a')):
            scan_sample = [key for key in sample_dict.keys() if src_str in str(key)]

        print(4 * "\n")
        print(str(eachsample) + ':')
        for eachscan in scan_sample:
            scale_factor = sample_dict[eachscan]['scale_factor']
            scale_from = sample_dict[eachscan]['scale_from']
            each_img = skio.imread(sample_dict[eachscan]['img_file'])
            print(str(eachscan) + ':')
            print('Scale factor:', round(scale_factor, 5),
                  'µm/pixel; scale from:', scale_from)
            print(str(eachscan), 'processed subimage:')
            outputs = inpt_predictor(each_img[:, :, ::-1])
            central_mask = mos_proc.get_central_mask(outputs)
            v = Visualizer(each_img[:, :, ::-1],
                           metadata=d2_metadata,
                           scale=500/each_img.shape[1]
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2_imshow(out.get_image()[:, :, ::-1])
            if central_mask[0]:
                print(str(eachscan), 'analyzed (scanned) zircon image:')
                each_props = mos_proc.overlay_mask_and_get_props(central_mask[1],
                                                                 each_img,
                                                                 eachscan,
                                                                 display_bool = True,
                                                                 scale_factor=scale_factor,
                                                                 **kwargs)
                _ = mos_proc.parse_properties(each_props,
                                              scale_factor,
                                              eachscan, verbose = True)
            else:
                print(str(eachscan), 'analyzed (scanned) zircon image:')
                mos_proc.save_show_results_img(each_img, eachscan,
                                               display_bool = True,
                                               scale_factor = scale_factor)

def gen_auto_proc_sample(run_dir, img_save_root_dir, csv_save_dir, eachsample,
                         inpt_save_polys_bool, inpt_loaded_data_dict,
                         inpt_predictor, inpt_alt_methods, eta_trk, out_trk,
                         save_spot_time=False, n_jobs=2, **kwargs):
    """Automatically process and save results from a single sample in a single-
       image per shot (non-ALC) dataset.

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
    inpt_loaded_data_dict : dict
        A dict (as returned by czd_utils.load_gen_data_dict()) with all
        sample data for a non-ALC dataset.
    inpt_predictor : Detectron2 Predictor class instance
        A D2 instance segmentation predictor to apply to images.
    inpt_alt_methods : list[bools]
        A list of bools corresponding to alternate methods (contrast
        enhancement and/or Otsu thresholding) to iteratively try on a
        scan image if inital central grain segmentation is unsuccesful.
        Format: [Try_contrast_enhanced_subimage,
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

    #loads sample subdict
    sample_dict = inpt_loaded_data_dict[eachsample]

    #parallel processing does not maintain input sequence order. We don't want \
    # that for output data. The dicts below accumulate unsorted processing outputs
    # for subsequent sorting.
    unsort_output_data_dict = {}
    unsort_each_json_dict = {}


    def iterate_sample_dict(inpt_sample_dict):
        iter_count = 0
        for eachscan, eachvals in sample_dict.items():
            img=skio.imread(eachvals['img_file'])
            scale_factor = eachvals['scale_factor']
            scale_from = eachvals['scale_from']
            file_name = eachvals['rel_file']
            yield (iter_count, str(eachscan),
                   img, scale_factor,
                   scale_from, file_name)
            iter_count +=1


    #extracts zircon subimage and runs predictor for each scan
    def parallel_proc_scans(iter_count, eachscan,
                            each_img, scale_factor,
                            scale_from, file_name):

        #restrict printing to a reasonable rate to avoid strange
        if iter_count % n_jobs == 0 or out_trk.stream_outputs:
            #reset output text display, prints some useful info
            out_trk.reset_and_print([eta_trk.str_eta,
                                     ' '.join(['Processing:',
                                               str(eachsample),
                                               str(eachscan)])])

        time_start_seg = time.perf_counter()
        central_mask=segment.segment_given_imgs(each_img, inpt_predictor,
                                                try_bools=inpt_alt_methods,
                                                **kwargs)
        ##time for segmentation. Will only be accurate if n_jobs == 1.
        each_total_seg_time = time.perf_counter()-time_start_seg
        if central_mask[0]:
            #saves mask image and gets properties
            each_props = mos_proc.overlay_mask_and_get_props(central_mask[1],
                                                             each_img,
                                                             str(eachscan),
                                                             display_bool = False,
                                                             save_dir=each_img_save_dir,
                                                             scale_factor=scale_factor)

            #adds properties to output list, with additional info on scale, \
            # source img
            temp_props_list = mos_proc.parse_properties(each_props,
                                                        scale_factor,
                                                        str(eachscan),
                                                        False,
                                                        [str(scale_from),
                                                         str(file_name)])

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
                                                scale_factor)
        #gives empty outputs if no mask image
        else:
            null_properties = mos_proc.parse_properties([],
                                                        scale_factor,
                                                        str(eachscan),
                                                        False,
                                                        [str(scale_from),
                                                         str(file_name)])
            #add segmentation time dependent on user params
            if save_spot_time:
                null_properties.append(each_total_seg_time)

            unsort_output_data_dict[str(eachscan)] = null_properties
            mos_proc.save_show_results_img(each_img, str(eachscan),
                                           display_bool = False,
                                           save_dir = each_img_save_dir,
                                           scale_factor = scale_factor)
            #optionally adds empty polygons to json_dict for saving
            if inpt_save_polys_bool:
                unsort_each_json_dict[str(eachscan)] = {'spot_names':[],
                                                        'spot_polys':[]}
                save_load.null_append_json_dict(unsort_each_json_dict[str(eachscan)],
                                                str(eachscan))

        #get total time for spot
        eta_trk.stop_update_eta()

    
    #run our big per-scan parallelized segmentation function for each scan
    # in the sample.
    Parallel(n_jobs=n_jobs,
             require='sharedmem'
             )(delayed(parallel_proc_scans)(*iter_out)
                        for iter_out in 
                        iterate_sample_dict(sample_dict))

    #fix order for output data
    for eachscan in sample_dict.keys():
        match_key = str(eachscan)
        output_data_list.append(unsort_output_data_dict[match_key])
        if inpt_save_polys_bool:
            add_names = unsort_each_json_dict[match_key]['spot_names']
            add_polys = unsort_each_json_dict[match_key]['spot_polys']
            each_json_dict['spot_names'] = [*each_json_dict['spot_names'],
                                            *add_names]
            each_json_dict['spot_polys'] = [*each_json_dict['spot_polys'],
                                            *add_polys]
    #additional parameters to save to output (i.e., for performance monitoring)
    addit_save_fields = []
    
    if save_spot_time:
        addit_save_fields.append('spot_time')

    #converts collected data to pandas DataFrame, saves as .csv
    output_dataframe = pd.DataFrame(output_data_list,
                                    columns=czd_utils.get_save_fields(proj_type='general',
                                                                      save_type='auto',
                                                                      addit_fields=addit_save_fields))
    csv_filename = str(eachsample) + '_grain_dimensions.csv'
    output_csv_filepath = os.path.join(csv_save_dir, csv_filename)
    czd_utils.save_csv(output_csv_filepath, output_dataframe)

    #optionally saves polygons to .json
    if inpt_save_polys_bool:
        save_load.save_sample_json(run_dir, str(eachsample),
                                  **each_json_dict)

def full_auto_proc(inpt_root_dir, inpt_selected_samples, inpt_loaded_data_dict,
                   inpt_predictor, inpt_save_polys_bool, inpt_alt_methods,
                   id_string = '', stream_output=False, n_jobs=2, **kwargs):
    """Automatically segment, measure, and save results for every selected
    sample in an ALC dataset.

    Parameters
    ----------
    inpt_root_dir : str
        Path to (Google Drive mounted) project dir.
    inpt_selected_samples : list[str]
        A list of strings matching user-selected sample names in inpt_mos_data_dict
    inpt_loaded_data_dict : dict
        A dict (as returned by czd_utils.load_gen_data_dict()) with all
        sample data for a non-ALC dataset.
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
    csv_save_dir = os.path.join(run_dir, 'grain_dimensions')
    os.makedirs(csv_save_dir)

    #initialize class instances for ETA, other output display
    eta_trk = eta.EtaTracker(czd_utils.gen_calc_scans_n(inpt_loaded_data_dict,
                                                        inpt_selected_samples))
    out_trk = eta.OutputTracker(n_blank_lines=10, stream_outputs=stream_output)

    #start timing for eta
    eta_trk.start()
    #starts loop through dataset dictionary
    for eachsample in inpt_selected_samples:
        gen_auto_proc_sample(run_dir, img_save_root_dir, csv_save_dir,
                             eachsample, inpt_save_polys_bool,
                             inpt_loaded_data_dict, inpt_predictor,
                             inpt_alt_methods, eta_trk, out_trk,
                             n_jobs=n_jobs, **kwargs)
        gc.collect()
    out_trk.print_txt('Done')

    return run_dir
