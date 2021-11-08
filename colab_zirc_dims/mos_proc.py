#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import math
import random


import numpy as np
import matplotlib.pyplot as plt

import skimage
import skimage.measure as measure
import skimage.io as skio

from . import czd_utils
from . import pointgen

# Various small functions that simplify larger functions below

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
    if isinstance(results, list):
        masks = czd_utils.mask_list_to_np(results)
    else:
        masks = czd_utils.prediction_to_np(results)

    #does not search for masks if no zircons segmented from image
    if masks == []:
        print('NO ZIRCON MASKS FOUND')
        return mask_found_bool, []

    #gets central points for masks/image, number of masks created
    masks_shape = masks.shape

    x_cent, y_cent = round(masks_shape[0]/2), round(masks_shape[1]/2)
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
    while mask_found_bool is False and pts.in_bounds:
        pts.next_pt()
        curr_x, curr_y = pts.curr_pts
        for i in range(0, num_masks):
            if masks[curr_y, curr_x, i]:
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
    #resize array if necessary
    input_central_mask = czd_utils.mask_to_3D_arr_size(input_central_mask,
                                                       original_image)
    #get main region properties
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
            New size for subimages.

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
