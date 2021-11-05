# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:21:23 2021

@author: 6600K-PC
"""
import numpy as np


# class for generating points, moving ~radially outwards from center of image, \
# to find central mask if not at actual center
class PointGenerator:
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
        Max number of increments before in_bounds bool = False. The default is 20.
    n_pts : int, optional
        Number of points to return around each circle. The default is 18.

    Returns
    -------
    None.

    """
    def __init__(self, x_center, y_center, pixel_increment,
                 n_incs = 20, n_pts=18):
        self.x_center, self.y_center = x_center, y_center
        self.pixel_increment = pixel_increment
        self.n_pts = n_pts
        self.max_inc = n_incs

        # corrects pixel increment if 0
        if self.pixel_increment < 1:
            self.pixel_increment = 1

        # most recent curr_pts; used to avoid repeating same point(s)
        self.last_pts = [False, False]

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

    def check_same_as_last(self):
        """Checks whether new pts are same as the last. Called internally.

        Returns
        -------
        Boolean
            True if curr_pts == last_pts, else False.

        """
        return self.curr_pts == self.last_pts

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
        self.last_pts = self.curr_pts
        curr_rot_rad = np.radians(self.rot_counter * self.deg_inc)
        self.curr_pts = [
            round(self.x_center + self.curr_radius * np.cos(curr_rot_rad)),
            round(self.y_center + self.curr_radius * np.sin(curr_rot_rad))
            ]
        if self.check_same_as_last():
            self.next_pt()


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
