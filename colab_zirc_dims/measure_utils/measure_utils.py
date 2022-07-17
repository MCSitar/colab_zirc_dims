# -*- coding: utf-8 -*-
"""
Utilities for measurement of grain masks (i.e., not just from regionprops).
"""

import cv2
import numpy as np

from .. import poly_utils
from .. import mos_proc


__all__ = ['contours_to_cv2',
           'min_rect_to_lines',
           'mask_to_min_rect',
           'box_main_region_props']


def contours_to_cv2(skimage_contours):
    """Convert scikit-image contours to CV2 format.

    Parameters
    ----------
    skimage_contours : np array
        An array (n, 2) with n (row, column) coordinates for contour.

    Returns
    -------
    np array.
        A float32 array (n, 2) with n (column, row) coordinates for contour.

    """
    new_conts = []
    for cont in skimage_contours:
        new_conts.append(cont[::-1])
    return np.array(new_conts).astype(np.float32)

def min_rect_to_lines(minrect_out):
    """Convert cv2 minAreaRect output to plottable X and Y points.

    Parameters
    ----------
    minrect_out : tuple
        minAreaRect output;
        (center_y, center_x), (rect_width, rect_height), rect_rotation.

    Returns
    -------
    x_pts : list[float]
        X points for 4 + 1 corners of box (lines return to starting point).
    y_pts : list[float]
        Y points for 4 + 1 corners of box (lines return to starting point).

    """
    bpoints = cv2.boxPoints(minrect_out)
    x_pts, y_pts = [], []
    for pt in bpoints:
        x_pts.append(pt[0])
        y_pts.append(pt[1])
    #add first pts to end for line plotting
    x_pts.append(x_pts[0])
    y_pts.append(y_pts[0])
    return x_pts, y_pts

def mask_to_min_rect(inpt_mask):
    """Convert a grain mask to shape parameters from minumum circumscribing
       recangle using cv2.minAreaRect

    Parameters
    ----------
    inpt_mask : np array
        A np array mask for the target grain in an image (segmentation output).

    Returns
    -------
    tuple
        (Rectangle center x, rectangle center y).
    tuple
        (minor rectangle axis length, major rectangle axis length).
    angle : float
        Orientation in radians between the rectangle major axis and horizontal.
        Equivalent to skimage.measure.regionprops.orientation.
    box_points : tuple(list[float], list[float])
        X and Y points for 4 + 1 corners of box (lines return to starting point).

    """
    contours = poly_utils.contours_from_mask(inpt_mask, convex=True)
    minrect_out = cv2.minAreaRect(contours_to_cv2(contours))
    (c_y, c_x), (rect_w, rect_h), rect_rot = minrect_out
    min_ax, maj_ax = min([rect_w, rect_h]), max([rect_w, rect_h])
    angle = 0.0
    if rect_w < rect_h:
        angle = -np.radians(rect_rot + 180)
    else:
        angle = -np.radians(rect_rot + 90)
    box_points = min_rect_to_lines(minrect_out)
    return (c_x, c_y), (min_ax, maj_ax), angle, box_points

def best_axis_measures(moment_minor_ax, moment_major_ax,
                       rect_minor_ax, rect_major_ax, mask_convex_area):
    """Check whether shapes for moment-based (corresponds to elliptical shape) or
       minimum area circumscribing rectangle (i.e., rectangular shape) axial
       measurements fit the convex area of a mask better, and return the 'best'
       measurements.

    Parameters
    ----------
    moment_minor_ax : float
        Minor axis of a grain mask (skimage.regionprops . minor_axis_length).
    moment_major_ax : float
        Major axis of a grain mask (skimage.regionprops . minor_axis_length).
    rect_minor_ax : float
        Minor axis of a minimum circumscribing rectangle for a grain mask
        (from measure_utils.mask_to_min_rect()).
    rect_major_ax : float
        Major axis of a minimum circumscribing rectangle for a grain mask
        (from measure_utils.mask_to_min_rect()).
    mask_convex_area : float
        Convex area of a grain mask (skimage.regionprops . convex_area).

    Returns
    -------
    float
        'Best' minor axis for grain mask's convex area.
    float
        'Best' major axis for grain mask's convex area.
    str
        'minimum circumscribing rectangle' or '2nd central moments', depending
        on which measurements were chosen as 'best'.

    """
    area_rect = rect_minor_ax * rect_major_ax
    area_ellipse = moment_minor_ax * moment_major_ax * np.pi
    dif_area_rect = abs(mask_convex_area - area_rect)
    dif_area_ellipse = abs(mask_convex_area - area_ellipse)
    if dif_area_rect < dif_area_ellipse:
        return rect_minor_ax, rect_major_ax, 'minimum circumscribing rectangle'
    else:
        return moment_minor_ax, moment_major_ax, '2nd central moments'

class box_main_region_props:
    
    def __init__(self, inpt_mask):
        """A wrapper for an skimage region (i.e., skimage.measure.regionprops)
           properties instance that also calculates axial lengths from
           minimum circumscribing rectangle.

        Parameters
        ----------
        inpt_mask : np array
            A numpy binary array representing the central zircon mask for an image,
            as returned by (successfully) running mos_proc.get_central_mask().

        Returns
        -------
        None.
        
        Properties
        -------
        
        basic_region_props:
            An skimage region corresponding to the input mask. Call here to use
            skimage properties not defined at top level of function.
        rect_centroid: tuple(float, float)
            Center (X, Y) of the minimum circumscribing rectangle for the region.
        rect_minor_axis_length: float
            Length (in pixels) of the minimum circumscribing rectangle for the
            region. Equivalent to Feret diameter.
        rect_major_axis_length: float
            Width (in pixels) of the minimum circumscribing rectangle for the
            region. Equivalent to maximum object diameter orthogonal to
            maximum Feret diameter.
        rect_orientation: float
            Orientation in radians between the rectangle major axis and horizontal.
            Equivalent to skimage.measure.regionprops.orientation, but for rectangle.
        rect_points: tuple(list[float], list[float])
            X and Y points for 4 + 1 corners of box (lines return to starting point).
        Other properties
            [centroid, minor_axis_length, major_axis_length,
            orientation, area, convex_area, perimeter,
            eccentricity, equivalent_diameter]:
                These are all standard calculations as in skimage.regionprops.

        """
        
        self.basic_region_props = mos_proc.get_main_region_props(inpt_mask)
        (
            self.rect_centroid,
            (self.rect_minor_axis_length, self.rect_major_axis_length),
            self.rect_orientation, self.rect_points
            ) = mask_to_min_rect(inpt_mask)
        self.centroid = self.basic_region_props.centroid
        self.minor_axis_length = self.basic_region_props.minor_axis_length
        self.major_axis_length = self.basic_region_props.major_axis_length
        self.orientation = self.basic_region_props.orientation
        self.area = self.basic_region_props.area
        self.convex_area = self.basic_region_props.convex_area
        self.perimeter = self.basic_region_props.perimeter
        self.eccentricity = self.basic_region_props.eccentricity
        self.equivalent_diameter = self.basic_region_props.equivalent_diameter
        (self.best_minor_ax_length,
         self.best_major_ax_length,
         self.best_ax_from) = best_axis_measures(self.minor_axis_length,
                                                 self.major_axis_length,
                                                 self.rect_minor_axis_length,
                                                 self.rect_major_axis_length,
                                                 self.convex_area)