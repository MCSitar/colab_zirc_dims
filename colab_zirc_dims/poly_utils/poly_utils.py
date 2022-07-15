# -*- coding: utf-8 -*-

"""
Functions for converting RCNN-derived zircon segmentation masks to polygons
viewable and editable in GUI and vice-versa.
"""

import numpy as np
from skimage import draw
import skimage.measure as measure

__all__ = ['mask_to_poly',
           'poly_to_mask',
           'vertex_dict_to_list',
           'poly_dicts_to_arrays']

# code for fxn below significantly modified from: \
# https://github.com/waspinator/pycococreator (covered by Apache-2.0 License)
def mask_to_poly(mask_for_conversion, tolerance = 1, scale_factor = 1.0):
    """Convert a numpy mask array to polygon suitable for GUI display, editing.

    Parameters
    ----------
    mask_for_conversion : np array
        A numpy binary array representing the central zircon mask for an image,
        as returned by (successfully) running mos_proc.get_central_mask().
    tolerance : Int, optional
        Tolerance in microns for polygon converted from input mask; resulting
        polygon will approximate the mask within *tolerance* microns.
        The default is 1.
    scale_factor : float, optional
        Scale factor for the current mosaic image. Used to adjust polygon
        tolerance to microns. The default is 1.0.

    Returns
    -------
    export_polygon
        An ordered list of dicts {x:, y:} representing vertices in a polygon.
        Point coordinates are x = x/image width, y = y/image height.
        Suitable for display/editing in manual adjustment/annotation GUI.

    """
    #print('Input shape:', mask_for_conversion.shape)

    #closes contour
    def close_contour(contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    export_polygon = []

    full_mask_h, full_mask_w = mask_for_conversion.shape #size of original mask

    #adjust tolerance to image size so that polygons are consistent during processing
    adj_tolerance = tolerance / scale_factor

    mask_labels, labels_nnum = measure.label(mask_for_conversion.astype(int), return_num=True)

    main_region_label = 1
    
    regions = measure.regionprops(mask_labels)

    if labels_nnum > 1:
        #selects largest region in case central zircon mask has multiple disconnected regions
        area_list = [props.area for props in regions]
        main_region_label = regions[area_list.index(max(area_list))].label

    #filled area for better contour finding; only relevant for Otsu masks
    filled_binary, f_bbox = [(props.image_filled, props.bbox) for props in
                             regions if props.label == main_region_label][0]
    mask_filled = mask_labels == main_region_label
    mask_filled[f_bbox[0]:f_bbox[2],f_bbox[1]:f_bbox[3]] = filled_binary

    # padding of mask is apparently necessary for contour closure.
    pad_fill = np.pad(mask_filled.astype(int), pad_width = 1,
                      mode='constant', constant_values = 0)

    #gets contours of mask
    mask_contours = measure.find_contours(pad_fill, 0.5)[0]

    mask_contours = np.subtract(mask_contours, 1)
    mask_contours = close_contour(mask_contours)
    poly_pts = measure.approximate_polygon(mask_contours, adj_tolerance) #converts contours to mask

    #flip ensures that polygons load properly (rather than mirrored) in GUI
    poly_pts = np.flip(poly_pts, axis=1)

    #converts to list of {x:, y:} dicts for JS annotation tool
    for each_pt in poly_pts:
        pt_dict = {'x': 0.0, 'y': 0.0}

        if each_pt[0] >= 0:
            pt_dict['x'] = round(each_pt[0]/full_mask_w, 3)

        if each_pt[1] >= 0:
            pt_dict['y'] = round(each_pt[1]/full_mask_h, 3)
        export_polygon.append(pt_dict)

    return export_polygon


def poly_to_mask(poly_for_conversion, original_image):
    """Converts polygons exported by JS annotation tool to masks for automated measurement.

    Parameters
    ----------
    poly_for_conversion : list of np 2d arrays
        An ordered list of arrays [x, y] representing vertices in a polygon.
    original_image : np array
        Numpy array representing the original image from which polygon was derived.

    Returns
    -------
    success_bool : Boolean
        Boolean indicating whether the polygon was successfully converted.
        Will be False if input polygon didn't exist, had under three points, or
        had no area.
    mask_output : np array or list
        If conversion successful, a numpy binary array representing the input
        polygon. Otherwise, an empty list.

    """

    success_bool = False
    if poly_for_conversion is None:
        return(success_bool, [])
    #polygon must have at least 3 points to have any area
    if np.shape(poly_for_conversion)[0] < 3:
        return(success_bool, [])

    poly_pts = np.clip(poly_for_conversion, 0, 1)

    original_image_shape = original_image.shape[:2]

    rescaled_poly = poly_pts * np.asarray(original_image_shape)

    mask_output = draw.polygon2mask(original_image_shape, rescaled_poly)

    #if polygon has no area, do not send it for measurements!
    if len(np.column_stack(np.where(mask_output > 0))) < 10:
        return(success_bool, [])
    success_bool = True

    return success_bool, mask_output

def vertex_dict_to_list(input_poly):
    """Convert polygon vertices from {x:, y:} to [x, y].

    Parameters
    ----------
    input_poly : dict
        Dict with position of x, y polygon vertex {x:, y:}.

    Returns
    -------
    Type: any
        X coordinate of vertex.
    Type: any
        Y coordinate of vertex.

    """

    return (input_poly['y'], input_poly['x'])

def poly_dicts_to_arrays(input_list):
    """Convert a list of lists of dicts {x:, y:} with polygon vertices to a list
       of arrays for same vertices.

    Parameters
    ----------
    input_list : list of lists of dicts
        List of lists (1 per polygon, 1 polygon per image) of dicts containing
        polygon vertex locations.

    Returns
    -------
    arr_list : list[arr]
        List of np arrays representing polygon vertices (1 per image).

    """
    arr_list = []
    for vertices_per_img in input_list:
        poly_as_array = [vertex_dict_to_list(vertex)
                         for vertex in vertices_per_img]
        if poly_as_array:
            arr_list.append(np.stack(poly_as_array))
        else:
            arr_list.append(None)
    return arr_list
