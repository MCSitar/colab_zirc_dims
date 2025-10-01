#!/usr/bin/env python
# coding: utf-8

"""
Module with functions, class for extracting spot subimages from DZ mosaics
(only used for ALC datasets) and for finding a 'central' grain and getting
measurements from segmented images (dataset type irrelevant).
"""

import os
import math
import random


import numpy as np
import matplotlib.pyplot as plt

import skimage
import skimage.measure as measure
import skimage.io as skio

from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from .. import czd_utils
from .. import pointgen
from .. import measure_utils

__all__ = ['mask_size_at_pt',
           'get_central_mask',
           'get_main_region_props',
           'save_show_results_img',
           'overlay_mask_and_get_props',
           'parse_properties',
           'auto_inc_contrast',
           'random_subimg_sample',
           'MosImg',
           'IterableMosImg']

# Various small functions that simplify larger functions below


def mask_size_at_pt(input_mask, coords):
    """Gets the contiguous area of a mask at a point. This is useful for
        PointRend models, which often produce instances
        with small, noncontiguous patches at bounding box margins.

    Parameters
    ----------
    input_mask : binary array
        Array of a mask.
    coords : list
        Coordinates [x, y] within the mask to check for mask size.

    Returns
    -------
    size_reg : int
        Contiguous pixel area of a mask region, if any, found at input point.

    """
    label_mask = measure.label(input_mask.astype(int))
    coords_x, coords_y = coords
    size_reg = 0
    reg_at_pt = int(label_mask[coords_y, coords_x])
    if reg_at_pt:
        size_reg = np.count_nonzero(label_mask == reg_at_pt)
    return size_reg




# function for retrieving mask at center of image, ~nearest to center of image
def get_central_mask(results, verbose=True):
    """Return the largest central mask region array, if any,
        from threshholding or Detectron2 prediction results.

    Parameters
    ----------
    results : Detectron2 Prediction or list[array]
        Detectron2 predictions or list of mask arrays for central mask.
    verbose : bool, optional
        Determines whether function prints error messages (e.g., "NO CENTRAL
        GRAIN FOUND"). The default is True.
        

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
    if isinstance(results, list):
        masks = czd_utils.mask_list_to_np(results)
    else:
        masks = czd_utils.prediction_to_np(results)

    #does not search for masks if no zircons segmented from image
    if not results or "instances" not in results or len(results["instances"]) == 0:
        if verbose:
            print('NO GRAIN MASKS FOUND')
        return mask_found_bool, []

    #gets central points for masks/image, number of masks created
    masks_shape = masks.shape

    y_cent, x_cent = round(masks_shape[0]/2), round(masks_shape[1]/2)
    num_masks = int(masks_shape[2])

    #counter for generating pts at and around center of image; \
    # will run through central ~15% of image
    pts = pointgen.PointGenerator(x_cent, y_cent, round(masks_shape[0]/150))

    central_mask_indices = []

    #loops through masks output and finds whether mask includes center of image
    for i in range(0, num_masks):
        curr_x, curr_y = pts.curr_pts
        if masks[curr_y, curr_x, i] is True:
            central_mask_indices.append(i)
    if len(central_mask_indices) > 0:
        mask_found_bool = True

    #extends search if mask not found at exact center
    for each_x, each_y in pts:
        if mask_found_bool is False:
            for i in range(0, num_masks):
                if masks[each_y, each_x, i]:
                    central_mask_indices.append(i)
            if len(central_mask_indices) > 0:
                mask_found_bool = True
                break

    if not mask_found_bool:
        if verbose:
            print('ERROR: NO CENTRAL GRAIN MASK FOUND')
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

def save_show_results_img(original_image, analys_name, display_bool=False,
                          save_dir='', tag_bool=False, input_central_mask=None,
                          main_region=None, scale_factor=None, **kwargs):
    """Save and/or display a (optionally) scaled figure showing a shot-image,
       w/ or w/o mask and axes overlaid.

    Parameters
    ----------
    original_image : array
        Image array for shot image. If input_central_mask is provided, this must
        be the image that the input central mask was derived from.
    analys_name : str
        Name of analysis (e.g., 'Spot-213') corresponding to mask.
        Used as base save file name (.png will be added).
    display_bool : Boolean, optional
        True or False; determines whether plot is displayed in output.
    save_dir : str, optional
        If entered, plot image will be saved to this directory. If blank,
        the image will not be saved. The default is ''.
    tag_bool : Boolean, optional
        If True and save_dir != '', '_tagged' will be appended to
        plot image file name. Called in zirc_dims_GUI. The default is False.
    input_central_mask : array, optional
        Binary mask for the segmented input original image. If not provided,
        this mask will not be overlaid on the origninal image. The default is None.
    main_region : skimage RegionProperties instance
        A RegionProperties instance corresponding to the largest
        and/or only contiguous region in the input mask image. If not provided,
        no mask/props will be overlaid on original image. The default is None.
    scale_factor : float, optional
        Scale factor (microns/pixel) for original image and mask.
        The default is None.
    **kwargs :
        Args for plotting - fig_dpi = int; will set plot dpi to input integer.
                            show_ellipse = bool; will plot ellipse corresponding
                                           to maj, min axes if True.
                            show_legend = bool; will plot a legend on plot if
                                          True.
                            

    Returns
    -------
    None.

    """

    #gets figure size
    figext_y, figext_x = np.shape(original_image)[:2]
    #if scale factor is provided, scale figsize to microns
    if scale_factor:
        figext_y, figext_x = [size * scale_factor for size
                                in np.shape(original_image)[:2]]
    #sets interval between ticks based on image size
    tick_interval = 5
    for each_interval in [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]:
        if int(figext_x / each_interval) >= 5:
            tick_interval = each_interval
    #set axis tick locations, labels
    x_tick_labels = list(range(0, int(figext_x), tick_interval))
    x_tick_locs = [loc/scale_factor for loc in x_tick_labels]
    y_tick_labels = list(range(0, int(figext_y), tick_interval))
    y_tick_locs = [loc/scale_factor for loc in y_tick_labels]

    #set up image plot
    fig, ax = plt.subplots()
    ax.imshow(original_image)

    adj_analys_name = analys_name

    #overlay mask if mask and props provided
    overlay_bool = False
    if not isinstance(input_central_mask, type(None)):
        if not isinstance(main_region, type(None)):
            overlay_bool = True
    #check kwargs
    plot_ellipse = False
    if 'show_ellipse' in kwargs:
        plot_ellipse = True
    if overlay_bool:

        ax.imshow(input_central_mask, alpha=0.4)
        
        #plots measured axes atop image
        #for props in regions:
        y0, x0 = main_region.centroid
        orientation = main_region.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * main_region.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * main_region.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * main_region.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * main_region.major_axis_length
        if 'moment' in str(main_region.best_ax_from):
            ax.plot((x0, x1), (y0, y1), '-r', linewidth=1.5,
                    label = 'Major, minor axes')
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.5)
        else:
            ax.plot((x0, x1), (y0, y1), '--r', linewidth=1.5,
                    label = 'Major, minor axes', alpha =0.5,)
            ax.plot((x0, x2), (y0, y2), '--r', linewidth=1.5, alpha =0.5)
        ax.plot(x0, y0, '.g', markersize=10, label = 'Centroid')
        


        #plot minimum bounding rectangle atop image (source of Feret diameter)
        if plot_ellipse is True:
            el_rot_deg = math.degrees(orientation) * -1.0
            el = Ellipse((x0, y0), main_region.minor_axis_length,
                          main_region.major_axis_length, angle=el_rot_deg,
                          fill = False, linestyle='--', color='red', 
                          alpha=0.5, linewidth=1.5)
            ax.add_artist(el)


        #plot minimum bounding rectangle atop image (source of Feret diameter)
        if 'rect' in str(main_region.best_ax_from):
            ax.plot(*main_region.rect_points, color = 'b',
                    linestyle = '-', alpha =0.7, linewidth=1.0,
                    label = 'Minimum area rectangle')
        else:
            ax.plot(*main_region.rect_points, color = 'b',
                    linestyle = '--', alpha =0.35, linewidth=1.0,
                    label = 'Minimum area rectangle')

    #mark 'failed' shots
    else:
        adj_analys_name = str(analys_name) + '_failed'

    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels([str(label) for label in x_tick_labels])
    ax.set_yticks(y_tick_locs)
    ax.set_yticklabels([str(label) for label in y_tick_labels])

    #additional kwarg plotting options
    if 'show_legend' in kwargs:
        if kwargs['show_legend'] is True:
            handles, labels = ax.get_legend_handles_labels()
            if plot_ellipse is True:
                handles, labels = ax.get_legend_handles_labels()
                handles.append(Line2D([0], [0], linestyle='--', color='red',
                                      alpha=0.5, linewidth=1.5))
                labels.append("Ellipse with same $2_{nd}$ order\nmoments as grain mask")
            ax.legend(handles=handles, labels = labels)
    if 'fig_dpi' in kwargs:
        fig.set_dpi(int(kwargs['fig_dpi']))

    if save_dir:
        img_save_filename = os.path.join(save_dir, adj_analys_name + '.png')
        if tag_bool:
            img_save_filename = os.path.join(save_dir,
                                             adj_analys_name + '_tagged.png')

        fig.savefig(img_save_filename)

    #plt.clf()

    if display_bool:
        plt.show()

    plt.close(fig)


#measures zircon mask (in pixels) using skimage, creates an image by \
# overlaying the mask atop the original subimage, and optionally displays \
# (if display_bool = True) and/or saves (if save_dir != '') this image.
def overlay_mask_and_get_props(input_central_mask, original_image, analys_name,
                               display_bool=False, save_dir='', tag_bool=False,
                               scale_factor=None, **kwargs):
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
    scale_factor : float, optional
        Scale factor (microns/pixel) for original image and mask.
        The default is None.
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
    main_region : skimage RegionProperties instance
        A RegionProperties instance corresponding to the largest
        and/or only contiguous region in the input mask image.

    """
    #resize array if necessary
    input_central_mask = czd_utils.mask_to_3D_arr_size(input_central_mask,
                                                       original_image)
    #get main region properties
    main_region = measure_utils.box_main_region_props(input_central_mask)

    if display_bool or save_dir:
        save_show_results_img(original_image, analys_name, display_bool,
                              save_dir, tag_bool, input_central_mask,
                              main_region, scale_factor, **kwargs)

    return main_region

def parse_properties(props, img_scale_factor, analys_name, verbose = False,
                     addit_props = []):
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
    addit_props : list, optional
        Additonal values to append to output props list. The default is [].

    Returns
    -------
    list
        A list of calculated properties. Added to export dict in
        main colab_zirc_dims Colab Notebook. Format:
            [analys_name, area, convex_area, eccent, eq_diam, perim,
             major_leng, minor_leng, roundness, scale_factor, *addit_props]

    """
    #initialize empty (null values) props list
    props_list = [analys_name, *czd_utils.get_save_fields(get_nulls=True), img_scale_factor]
    #if no region found, skip and return null properties
    if props == []:
        if verbose:
            print('null properties entered')
        for addit_prop in addit_props:
            props_list.append(addit_prop)
        return props_list

    area = props.area * img_scale_factor**2
    convex_area = props.convex_area * img_scale_factor**2
    eccent = props.eccentricity
    eq_diam = props.equivalent_diameter * img_scale_factor
    perim = props.perimeter * img_scale_factor
    major_leng = props.major_axis_length * img_scale_factor
    minor_leng = props.minor_axis_length * img_scale_factor
    roundness = 4 * math.pi * props.area / props.perimeter**2
    Feret_diam = props.rect_major_axis_length * img_scale_factor
    orth_feret_diam = props.rect_minor_axis_length * img_scale_factor
    best_major_ax = props.best_major_ax_length * img_scale_factor
    best_minor_ax = props.best_minor_ax_length * img_scale_factor
    best_ax_from = props.best_ax_from
    scale_factor = img_scale_factor

    props_list = [analys_name, area, convex_area, eccent, eq_diam, perim,
                  major_leng, minor_leng, roundness, Feret_diam,
                  orth_feret_diam, best_major_ax, best_minor_ax,
                  best_ax_from, scale_factor]
    for addit_prop in addit_props:
        props_list.append(addit_prop)

    if verbose:
        print('Major axis length =', round(best_major_ax, 1), 'µm,',
              'Minor axis length =', round(best_minor_ax, 1), 'µm')

    return props_list


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
        red = skimage.exposure.equalize_hist(input_image[:, :, 0])
        green = skimage.exposure.equalize_hist(input_image[:, :, 1])
        blue = skimage.exposure.equalize_hist(input_image[:, :, 2])
        return skimage.img_as_ubyte(np.stack([red, green, blue], axis=2))

        #code for enhancing contrast without splitting channels: results are less grainy
        #return skimage.img_as_ubyte(skimage.exposure.equalize_hist(input_image))
    else:
        return input_image


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
    each_mosaic = MosImg(mosaic_path, Align_path, subimg_size, offset_list)
    each_scan_dict = czd_utils.scancsv_to_dict(scancsv_path)

    #if too few shots in sample, reduces num_samples
    if num_samples > len(list(each_scan_dict.keys())):
        num_samples = len(list(each_scan_dict.keys()))

    scan_sample = random.sample(list(each_scan_dict.keys()), num_samples)
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


# a class for a mosaic image from which subimages are clipped, with functions for clipping
class MosImg:
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
    #shift list = adjustments [x, y] in microns
    def __init__(self, mosaic_image_path, align_file_path = '',
                 sub_img_size = 500, x_y_shift_list = [0, 0]):

        #records original sub_img_size input; useful for later adjustments
        self.sub_img_size_input = sub_img_size
        #the mosaic image from which subimages are clipped
        self.full_img = auto_inc_contrast(skio.imread(mosaic_image_path))

        self.mos_dims = self.full_img.shape[::-1] #image pixel dimensions (x, y)

        self.scale_factor = 1 #default scale factor (e.g., if no .Align file)

        #origins for x, y (for mapping coordinates \
        # (from align, shotlist files) to mosaic image)
        self.x_origin, self.y_origin = 0, 0
        self.curr_coords = [0, 0] #current coordinates (default = 0, 0)

        #gets info from .Align xml file
        if align_file_path:
            #gets data (x,y centers, x,y sizes) from .Align file
            self.align_data = czd_utils.get_Align_center_size(align_file_path)

            #calculates a new scale factor (microns/pixel) from .Align file
            self.scale_factor = czd_utils.calc_scale_factor(self.align_data[2:],
                                                            self.mos_dims[1:])
            #initialize shift list (in microns)
            self.pix_shift_list = [val for val in x_y_shift_list]
            
            #calculates origin based on new variables
            self.x_origin = float(self.align_data[0] - self.align_data[2]/2
                                  - self.pix_shift_list[0])
            self.y_origin = float(self.align_data[1] - self.align_data[3]/2
                                  - self.pix_shift_list[1])

        # a hacky solution to prevent mismatches beyond and at pixel bounds:
        # all subimage sizes will be even:
        self.sub_img_size = czd_utils.round_to_even(sub_img_size / self.scale_factor)

        #initiates variables that will be called in set_sub_img
        self.x_y_0, self.x_y_0_offsets = [0, 0], [0, 0]
        self.x_y_1, self.x_y_1_offsets = czd_utils.list_of_val(self.sub_img_size,
                                                               2, 2)

        #initiates sub_img (as a black np.zeroes array)
        self.sub_img = np.zeros([self.sub_img_size, self.sub_img_size,
                                 self.mos_dims[0]], dtype=np.uint8)

    def set_sub_img_size(self, new_sub_img_size):
        """Set a new sub image size, updates subimage.

        Parameters
        ----------
        new_sub_img_size : int
            New size for subimages, in microns.

        Returns
        -------
        None.

        """
        self.sub_img_size = czd_utils.round_to_even(new_sub_img_size / self.scale_factor)
        self.set_subimg(*self.curr_coords)

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
        #unrotated mapping if no rotation in align file
        if float(self.align_data[4]) == 0.0:
            return [round(float((x_coord - self.x_origin) / self.scale_factor)),
                    round(float((y_coord - self.y_origin) / self.scale_factor))]
        else:
            #back-rotate coordinates by rotation given in .Align file
            x_coord, y_coord = czd_utils.rotate_pt((x_coord, y_coord),
                                                   self.align_data[4],
                                                   self.align_data[:2])
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
        #records coordinates
        self.curr_coords = [x_coord, y_coord]

        #sets all vars back to base values
        self.x_y_0, self.x_y_0_offsets = [0, 0], [0, 0]
        self.x_y_1, self.x_y_1_offsets = czd_utils.list_of_val(self.sub_img_size,
                                                               2, 2)

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
    def get_subimg(self, *coords_inpt):
        """Get the current sub_img, after optionally adjusting the current
        sub_img location to input coordinates if they are given.

        Parameters
        ----------
        coords_inpt : 2 ints and/or floats, optional
            If given, they will be treated as x_coord, y_coord and used to set
            a new sub_img location. Otherwise, location will be kept as-is.

        Returns
        -------
        np.ndarray
            Array representing the current subimage (i.e., self.sub_img).

        """
        if coords_inpt:
            self.set_subimg(*coords_inpt)
        return self.sub_img
    
    def get_zoomed_subimg(self, zoom_fact=1.1):
        """Get a version of the current subimage that is zoomed in or out by
        the given zoom_fact (i.e., decreasing or increasing spatial extent on
        the mosaic image, keeping the same center). Reset instance sub_img_size
        afterwards.

        Parameters
        ----------
        zoom_fact : int or float, optional
            Factor to multiply the current sub_img extent by. The default is
            1.1.

        Returns
        -------
        out_img: np.ndarray
            Array representing the current subimage (i.e., self.sub_img), with
            extent adjusted by zoom_fact.

        """
        self.set_sub_img_size(self.sub_img_size*zoom_fact)
        out_img = self.get_subimg()
        self.set_sub_img_size(self.sub_img_size_input)
        return out_img


class IterableMosImg:
    """Creates and links a MosImage instance to a scan dict loaded from a
    corresponding dict loaded from a .scancsv, with scan coordinates. Iterable
    for easy use with parallel processing.

    Parameters
    ----------
    mosaic_image_path : str
        Full file path to a .bmp mosaic image file.
    scan_dict : dict{Any: [float|int, float|int]}
        A dictionary with scan names: scan coordinates, as loaded from a
        .scancsv file using czd_utils.scancsv_to_dict().
    try_bools : list[bool], optional
        A list of bools corresponding to alternate methods (scale jittering,
        contrast enhancement, and/or Otsu thresholding) to iteratively try on
        a scan image if inital central grain segmentation is unsuccesful.
        Format: [Try_zoomed_out_subimage, Try_zoomed_in_subimage,
                 Try_contrast_enhanced_subimage,
                 Try_Otsu_thresholding].
        The default is [False, False, False, False].
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
    
    def __init__(self, mosaic_image_path, scan_dict,
                 try_bools=[False, False, False, False],
                 align_file_path = '',
                 sub_img_size = 500, x_y_shift_list = [0, 0]):

        #create the child mosaic image instance
        self._mos_img = MosImg(mosaic_image_path, 
                               align_file_path=align_file_path,
                               sub_img_size=sub_img_size,
                               x_y_shift_list=x_y_shift_list)
        self._scan_dict = scan_dict
        self.try_bools = try_bools

    def get_aug_subimgs(self, *inpt_pt_coords):
        """Get a list of subimgs (as np.ndarrays) from the child MosImg. These
        will depend on the first three try bools.

        Parameters
        ----------
        *inpt_pt_coords : float|int, float|int, optional
            Coordinates to set subimage location. If not included, subimages
            will be drawn from the child MosImg current subimg location.

        Returns
        -------
        out_imgs_lst : list[np.ndarray]
            Images for segmentation. These depend on the first two booleans in
            self.try_bools. The number of returned subimages in the list will
            thus range from 1 to 3. The first subimage will always be the
            unaltered (i.e., 'original') subimage clipped from the child
            MosImg instance, at the current target location.
            Format:
                [unaltered subimage at target location,
                 if try_bools[0]: subimage with extent increased w/ fact. of 1.1,
                 if try_bools[0]: subimage with extent decreased w/ fact. of 0.9,]

        """
        if inpt_pt_coords:
            self._mos_img.set_subimg(*inpt_pt_coords)
        out_imgs_lst = [self._mos_img.get_subimg()]
        #if first try bool is true, we need a 'zoomed-out' subimage (extent*1.1)
        if self.try_bools[0]:
            out_imgs_lst.append(self._mos_img.get_zoomed_subimg(1.1))
        #if second is True, we need a 'zoomed-in' subimage (extent*0.9)
        if self.try_bools[1]:
            out_imgs_lst.append(self._mos_img.get_zoomed_subimg(0.9))
        return out_imgs_lst

    def __iter__(self):
        """Iterate through the scan dict, yielding all scan-dependent variables
        needed for segmentation and central grain dimensional analysis of the
        target grain.

        Yields
        ------
        idx_count : int
            Current index-location within the scan dict.
        str
            Current scan name (from .scancsv file, by way of the scan dict).
        out_imgs : list[np.ndarray]
            List of images (as np.ndarrays).These depend on the first two booleans in
            self.try_bools. The number of returned subimages in the list will
            thus range from 1 to 3. The first subimage will always be the
            unaltered (i.e., 'original') subimage clipped from the child
            MosImg instance, at the current target location.
            Format:
                [unaltered subimage at target location,
                 if try_bools[0]: subimage with extent increased w/ fact. of 1.1,
                 if try_bools[0]: subimage with extent decreased w/ fact. of 0.9,]
        out_scale_factor : float
            Scale in um/pixel for spot. From loaded child MosImg instance.

        """
        idx_count = 0
        for eachscan, eachscan_coords in self._scan_dict.items():
            out_imgs = self.get_aug_subimgs(*eachscan_coords)
            out_scale_factor=self._mos_img.scale_factor
            yield (idx_count,str(eachscan), out_imgs, out_scale_factor)
            idx_count +=1
