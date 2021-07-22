#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import math
from skimage import io, exposure, img_as_ubyte
import skimage.measure as measure
import pandas as pd
import csv
import xml.etree.ElementTree as ET
import sys



# a function for loading mosaic and shot data into a dictionary
def load_data_dict(ROOT_DIR_STRING):
    
    #initializes output dict
    temp_output_dict = {}
    
    #file paths
    mosaic_path = os.path.join(ROOT_DIR_STRING, 'mosaics')
    scanlist_path = os.path.join(ROOT_DIR_STRING, 'scanlists')
    mos_csv_path = os.path.join(ROOT_DIR_STRING, 'mosaic_info.csv')
    #save_path = os.path.join(ROOT_DIR, 'training_images') #MODIFY FOR FINAL VERSION
    
    
    #loads info csv as dictionary
    mos_csv_dict = pd.read_csv(mos_csv_path, header=0, index_col=False, squeeze=False).to_dict('list')

    #checks that headers in info_dict are correct
    mos_csv_dict_keys = list(mos_csv_dict.keys())
    req_keys = ['Sample', 'Scanlist', 'Mosaic', 'Max_zircon_size', 'X_offset', 'Y_offset']
    if all(key in mos_csv_dict_keys for key in req_keys) != True:
        print('Incorrect mosaic_info.csv headers: correct and re-save')
        return({})

    
    
    #lists of files in directories
    mosaic_bmp_filenames = [file for file in os.listdir(mosaic_path) if file.endswith('.bmp')]
    mosaic_align_filenames = [file for file in os.listdir(mosaic_path) if file.endswith('.Align')]
    scanlist_filenames = os.listdir(scanlist_path)

    #print(mosaic_bmp_filenames)
    #print(mosaic_align_filenames)

    #full_data_dict = {'Sample':[], 'Scanlist': [], 'Mosaic': []}

    # loops through mos_csv_dict in order to collect data, verify that all files given in mosaic_info.csv are present
    for eachindex, eachsample in enumerate(mos_csv_dict['Sample']):
        each_include_bool = False
        act_scn_file, act_mos_file, act_align_file = '', '', ''
        each_csv_scanlist_name = mos_csv_dict['Scanlist'][eachindex]
        each_csv_mosaic_name = mos_csv_dict['Mosaic'][eachindex].split('.')[0] # mosaic name without file extension
        
        #checks if files are in directories, gets full file paths if so
        if any(each_csv_scanlist_name in string for string in scanlist_filenames):
            temp_scn_file = [string for string in scanlist_filenames if each_csv_scanlist_name in string][0]
            act_scn_file = os.path.join(scanlist_path, temp_scn_file)

            if any(each_csv_mosaic_name in string for string in mosaic_bmp_filenames):
                temp_mos_file = [string for string in mosaic_bmp_filenames if each_csv_mosaic_name in string][0]
                act_mos_file = os.path.join(mosaic_path, temp_mos_file)

                if any(each_csv_mosaic_name in string for string in mosaic_align_filenames):
                    
                    temp_align_file = [string for string in mosaic_align_filenames if each_csv_mosaic_name in string][0]
                    
                    
                    act_align_file = os.path.join(mosaic_path, temp_align_file)
                    each_include_bool = True

                    
                else:
                    print(eachsample, ":  matching mosaic .Align file not found")

            else:
                print(eachsample, ":  matching mosaic .bmp file not found")

        else:
            print(eachsample, ":  matching scanlist not found")
            
        if each_include_bool == True:
            
            
            #dictionary for info on individual scans and their coordinates
            temp_coords_dict = {}
            
            #scanlist from .scancsv file, loaded as a dict
            each_scanlist = pd.read_csv(act_scn_file, header=0, index_col=False, squeeze=False, encoding='cp1252').to_dict('list')
            added_scans_unchanged = [] # list of scans added to output dictionary
            
            #loops through shotlist and gets coordinates for each scan + numbers repeated instances
            for eachscan_index, eachscan in enumerate(each_scanlist['Description']):
                if each_scanlist['Scan Type'][eachscan_index] == 'Spot':
                    eachscan_coords = [float(data) for data in each_scanlist['Vertex List'][eachscan_index].split(',')][:2]
                    if (eachscan in added_scans_unchanged):
                        temp_scanname = str(eachscan) + '-' + str(added_scans_unchanged.count(eachscan) + 1)
                        temp_coords_dict[temp_scanname] = eachscan_coords
                    else:
                        temp_coords_dict[str(eachscan)] = eachscan_coords
                    added_scans_unchanged.append(eachscan)
                        
                        
            
            
            #adds collected data to output dict
            temp_output_dict[eachsample] = {}
            
            temp_output_dict[eachsample]['Scanlist'] = act_scn_file
            temp_output_dict[eachsample]['Mosaic'] = act_mos_file
            temp_output_dict[eachsample]['Align_file'] = act_align_file
            temp_output_dict[eachsample]['Max_zircon_size'] = mos_csv_dict['Max_zircon_size'][eachindex]
            temp_output_dict[eachsample]['Offsets'] = [mos_csv_dict['X_offset'][eachindex], mos_csv_dict['Y_offset'][eachindex]]
            temp_output_dict[eachsample]['Scan_dict'] = temp_coords_dict
            
    return(temp_output_dict)


#function for retrieving mask at center of image, ~nearest to center of image
def get_central_mask(results):
    
    #class for generating points, moving ~radially outwards from center of image, \
    # to find central mask if not at actual center
    class point_generator:
        def __init__(self, x_center, y_center, pixel_increment):
            self.x_center, self.y_center, self.pixel_increment = x_center, y_center, pixel_increment
            
            #current, x, y for output
            self.curr_pts = [self.x_center, self.y_center]
            
            #int from 0-7 whether x or y are added to or subtracted from; at 0 x is added to
            self.rot_counter = 0
            
            #pixel increment multiplier, current pixel margin
            self.inc_multiplier, self.curr_margin = 0, 0
            
        def get_curr_pts(self):
            return self.curr_pts[0], self.curr_pts[1]
        
        #updates pts
        def update_pts(self):
            if self.rot_counter == 0:
                self.curr_pts = [int(self.x_center + self.curr_margin), self.y_center]
            if self.rot_counter == 1:
                self.curr_pts = [self.x_center, int(self.y_center - self.curr_margin)]
            if self.rot_counter == 2:
                self.curr_pts = [int(self.x_center - self.curr_margin), self.y_center]
            if self.rot_counter == 3:
                self.curr_pts = [self.x_center, int(self.y_center + self.curr_margin)]  
            if self.rot_counter == 4:
                self.curr_pts = [int(self.x_center + self.curr_margin), int(self.y_center - self.curr_margin)]
            if self.rot_counter == 5:
                self.curr_pts = [int(self.x_center - self.curr_margin), int(self.y_center - self.curr_margin)]
            if self.rot_counter == 6:
                self.curr_pts = [int(self.x_center - self.curr_margin), int(self.y_center + self.curr_margin)]
            if self.rot_counter == 7:
                self.curr_pts = [int(self.x_center + self.curr_margin), int(self.y_center + self.curr_margin)]        
                
        #cycles to a larger pixel increment, updates pts
        def next_inc(self):
            self.inc_multiplier += 1
            self.curr_margin = self.inc_multiplier * self.pixel_increment
            
            self.update_pts()
            
        #cycles to a new point around center w/o changing pixel increment    
        def next_pt(self):
            self.rot_counter += 1
            if self.rot_counter >= 8:
                self.rot_counter = 0
            self.update_pts()
                

    #bool indicating whether a central mask was found
    mask_found_bool = False
    cent_mask_index = 0
    
    #does not search for masks if no zircons segmented from image
    if len(results['instances']) == 0:
        print('NO ZIRCON MASKS FOUND')
        return(mask_found_bool, [])

    #converts mask outputs to np arrays and stacks them into a larger np array
    masks = np.stack([eachresult.cpu().numpy() for eachresult in results['instances'].get('pred_masks')], 2)

    #gets central points for masks/image, number of masks created
    masks_shape = masks.shape
    x_cent, y_cent, num_masks = int(masks_shape[0]/2), int(masks_shape[1]/2), int(masks_shape[2])

    
    #counter for generating pts at and around center of image; will run through central ~1/5th of image
    pts = point_generator(x_cent, y_cent, masks_shape[0]//100)
    
    central_mask_indices = []

    #print(pts.curr_pts)
    #loops through masks output and finds whether mask includes center of image
    for i in range(0, num_masks):
        curr_x, curr_y = pts.curr_pts
        if masks[curr_x, curr_y, i] == True:
            #cent_mask_index = i
            central_mask_indices.append(i)
    if len(central_mask_indices) > 0:
        mask_found_bool = True


    #extends search if mask not found at exact center
    while (mask_found_bool == False) and (pts.inc_multiplier < 20):
        pts.next_inc()
        for pt_number in range(0, 7):
            curr_x, curr_y = pts.curr_pts
            for i in range(0, num_masks):
                if masks[curr_x, curr_y, i] == True:
                    central_mask_indices.append(i)
            if len(central_mask_indices) > 0:
                mask_found_bool = True
                break
            pts.next_pt()
    
    

    if mask_found_bool == False:
        print('ERROR: NO CENTRAL ZIRCON MASK FOUND')
        return(mask_found_bool, [])

    #if only one mask at center, return it
    if len(central_mask_indices) == 1:
        cent_mask_index = central_mask_indices[0]
        return(mask_found_bool, masks[:, :, cent_mask_index])

    #selects the largest mask from central_mask indices as output
    mask_size_list = [np.count_nonzero(masks[:, :, i]) for i in central_mask_indices]
    cent_mask_index = central_mask_indices[mask_size_list.index(max(mask_size_list))]


    return(mask_found_bool, masks[:, :, cent_mask_index])

#measures zircon mask (in pixels) using skimage, creates an image by overlaying the mask atop the original subimage, \
# and optionally displays (if display_bool = True) and/or saves (if save_dir != '') this image.
def overlay_mask_and_get_props(input_central_mask, original_image, analys_name, display_bool = False, save_dir = '', tag_bool = False):
    label_image = measure.label(input_central_mask.astype(int))
    regions = measure.regionprops(label_image)

    save_bool = False
    if save_dir != '':
        save_bool = True
        
    #selects largest region in case central zircon mask has multiple disconnected regions
    area_list = [props.area for props in regions]
    main_region_index = area_list.index(max(area_list))
    main_region = regions[main_region_index]
    
    
    if (display_bool == True) or (save_bool == True):

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

        if save_bool == True:
        
            img_save_filename = os.path.join(save_dir, analys_name + '.png')
            if tag_bool == True:
                img_save_filename = os.path.join(save_dir, analys_name + '_tagged.png')

            
            plt.savefig(img_save_filename)
        
        #plt.clf()

        if display_bool == True:
            plt.show()
        
        plt.close('all')
    return(main_region)
    
def parse_properties(props, img_scale_factor, analys_name, verbose = False):
    
    #if no region found, skip
    if props == []:
        if verbose == True:
            print('null properties entered')
        return([analys_name, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
    
    area = props.area * img_scale_factor**2
    convex_area = props.convex_area * img_scale_factor**2
    eccent = props.eccentricity
    eq_diam = props.equivalent_diameter * img_scale_factor
    perim = props.perimeter * img_scale_factor
    major_leng = props.major_axis_length * img_scale_factor
    minor_leng = props.minor_axis_length * img_scale_factor
    roundness = 4 * math.pi * props.area / props.perimeter**2
    scale_factor = img_scale_factor

    props_list = [analys_name, area, convex_area, eccent, eq_diam, perim, major_leng, minor_leng, roundness, scale_factor]

    if verbose == True:
        print('Major axis length =', round(major_leng, 1), 'µm,',               'Minor axis length =', round(minor_leng, 1), 'µm')

    return(props_list)


#saves pandas as csv file
def save_csv(path, pandas_table):
    pandas_table.to_csv(path, index=False, header=True, encoding='utf-8-sig')
    return()


# a class for a mosaic image from which subimages are clipped, with functions for clipping
class mos_img:
    
    #shift list = adjustments [x, y] in microns
    def __init__(self, mosaic_image_path, align_file_path = '', sub_img_size = 500, x_y_shift_list = [0, 0]):
        
        #the mosaic image from which subimages are clipped
        self.orig_full_img = skimage.io.imread(mosaic_image_path)
        
        #if image has low contrast, enhances contrast
        if skimage.exposure.is_low_contrast(self.orig_full_img, 0.10):
            
            ## code for enhancing contrast by channel: avoids warning but produces grainy images
            #self.red = skimage.exposure.equalize_hist(self.orig_full_img[:, :, 0])
            #self.green = skimage.exposure.equalize_hist(self.orig_full_img[:, :, 1])
            #self.blue = skimage.exposure.equalize_hist(self.orig_full_img[:, :, 2])
            #self.full_img = skimage.img_as_ubyte(np.stack([self.red, self.green, self.blue], axis=2))
            
            #code for enhancing contrast without splitting channels: results are less grainy
            self.full_img = skimage.img_as_ubyte(skimage.exposure.equalize_hist(self.orig_full_img))
            
            self.contrast_enchanced_bool = True
            
        else:
            self.full_img = self.orig_full_img
            self.contrast_enchanced_bool = False
        
        self.mos_dims = self.full_img.shape[::-1] #image pixel dimensions (x, y)
        
        self.scale_factor = 1 #default scale factor (e.g., if no .Align file)
        
        #origins for x, y (for mapping coordinates (from align, shotlist files) to mosaic image)
        self.x_origin, self.y_origin = 0, 0
        
        #gets info from .Align xml file
        if align_file_path != '':
            self.align_tree = ET.parse(align_file_path)
            self.align_root = self.align_tree.getroot()
            
            
            self.align_x_size, self.align_y_size = self.mos_dims[1:] #image dimensions in microns from align file
            
            #image center in microns from align file or (default) of image (in pixels) from image dimensions
            self.align_x_center, self.align_y_center = self.align_x_size//2, self.align_y_size//2
            
            #loop through xml file to get image center, size 
            ##UPDATE TO INCLUDE ROTATION IF NEEDED##
            for eachchild in self.align_root:
                if eachchild.tag == 'Alignment':
                    for each_align_data in eachchild:
                        if each_align_data.tag == 'Center':
                            self.align_x_center, self.align_y_center = [float(data) for data in each_align_data.text.split(',')]
                        if each_align_data.tag == 'Size':
                            self.align_x_size, self.align_y_size = [float(data) for data in each_align_data.text.split(',')]
            
            #sets new origin based on data in .Align file
            self.x_origin  = int(self.align_x_center - self.align_x_size/2 - x_y_shift_list[0])
            self.y_origin  = int(self.align_y_center - self.align_y_size/2 - x_y_shift_list[1])
            
            #sets new scale factor (microns(?)/pixel) based on dimensions of mosaic image and those in .Align file
            self.scale_factor = (self.align_x_size/self.mos_dims[1] + self.align_y_size/self.mos_dims[2])/2
            
            #print('Scale factor:', self.scale_factor)
            #print('New origin:', str(self.x_origin) + ",", self.y_origin)
            #print(self.align_root[0][1].text)
            
        
        self.sub_img_size = int(sub_img_size / self.scale_factor) # number of pixels for subimages from micron input
        
        #print(self.mos_dims)
        
    def set_sub_img_size(self, new_sub_img_size): # sets new sub image size
        
        self.sub_img_size = int(new_sub_img_size / self.scale_factor)
        
    #maps coordinates (as in shotlist) to image pixels
    def coords_to_pix(self, x_coord, y_coord):
        
        return((x_coord - self.x_origin) / self.scale_factor, (y_coord - self.y_origin) / self.scale_factor)
        
    def set_subimg(self, x_coord, y_coord): #x_coord and y_coord (microns(?)) are the center of the new subimage generated
        
        #initializes variables for extent of desired subimage beyond bounds of mosaic image; these will ensure that the
        # central point of subimage is that of input coordinates. Subimage pixels beyond mosaic image bounds will be black.
        self.x0_offset, self.y0_offset, self.x1_offset, self.y1_offset = 0, 0, self.sub_img_size, self.sub_img_size
        self.x0_chng_bool, self.y0_chng_bool = False, False
        
        #sets initial clipping bounds (in pixels) for a subimage of size sub_img_size^2 centered at input coordinates
        self.x0 = int((x_coord - self.x_origin) / self.scale_factor - self.sub_img_size/2)
        self.x1 = int((x_coord - self.x_origin) / self.scale_factor + self.sub_img_size/2)
        self.y0 = int((y_coord - self.y_origin) / self.scale_factor - self.sub_img_size/2)
        self.y1 = int((y_coord - self.y_origin) / self.scale_factor + self.sub_img_size/2)
        
        #modifies coordinates in the case that initial bounds exceed image bounds
        if self.x0 < 0:
            #print('x0 adjusted')
            self.x0_offset = abs(int(0 - self.x0))
            self.x0 = 0
            self.x0_chng_bool = True
        if self.y0 < 0:
            #print('y0 adjusted')
            self.y0_offset = abs(int(0 - self.y0))
            self.y0 = 0
            self.y0_chng_bool = True
            
        if self.x1 > self.mos_dims[1]:
            #print('x1 adjusted')
            self.x1_offset = int(self.sub_img_size - (self.x1 - self.mos_dims[1]))
            if self.x1_offset < 0:
                self.x1_offset = 0
            self.x1 = self.mos_dims[1]
        if self.x1 <0:
            #print('x1 under 0, adjusted')
            self.x1 = 0 #in case of points that are fully out of bounds, a fully black subimage will be displayed
            
            
        if self.y1 > self.mos_dims[2]:
            #print('y1 adjusted')
            self.y1_offset = int(self.sub_img_size - (self.y1 - self.mos_dims[2]))
            if self.y1_offset < 0:
                self.y1_offset = 0
            self.y1 = self.mos_dims[2]
            
        if self.y1 <0:
            #print('y1 under 0, adjusted')
            self.y1 = 0 #in case of points that are fully out of bounds, a fully black subimage will be displayed

        if self.x0_chng_bool and self.x1 != 0:
            self.x1 += 1
        if self.y0_chng_bool and self.y1 != 0:
            self.y1 += 1
            
            
        #print('x0:', self.x0, 'x1:', self.x1, 'y0:', self.y0, 'y1:', self.y1)
        #print('Offsets:', self.x0_offset, self.x1_offset, self.y0_offset, self.y1_offset)
        
        #default (black) subimage
        self.sub_img = np.zeros([self.sub_img_size, self.sub_img_size, self.mos_dims[0]], dtype=np.uint8)
        
        #maps crop from mosaic onto black image to create subimage with input coordinate point at center
        self.sub_img[self.y0_offset:self.y1_offset, self.x0_offset:self.x1_offset, :]         = self.full_img[self.y0:self.y1, self.x0:self.x1, :]

