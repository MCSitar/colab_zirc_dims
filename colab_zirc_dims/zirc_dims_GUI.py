#!/usr/bin/env python
# coding: utf-8

# In[1]:


import base64
import io
import json
import uuid
import os, copy, datetime

from typing import List
from typing import Union
from IPython.display import display
from IPython.display import Javascript
import numpy as np
from PIL import Image
import pandas as pd

from google.colab import output
from google.colab.output import eval_js

from . import czd_utils
from . import mos_proc
from . import poly_utils
from . import save_load

### This is a function to automatically segment all selected samples in a dataset and allow \
### users to inspect and/or replace all automatically-produced segmentations via a Javascript-based \
### polygon annotation GUI. \
###
### NOTE: The simplest way to apply the JS GUI to each sample would probably(?) be to loop through all samples and \
### run the GUI to each sample using threading and Event.wait() functions. Integrating threading with JS and Google \
### Colab seems, however, to be quite tricky; callbacks from JS to Python within threads do not seem to run and may be \
### automatically sent to the (paused) main thread. The solution to this issue implemented here is a mess of \
### callbacks that allows navigation through all samples via GUI.
###
###   Args:
###       sample_data_dict = a dictionary as produced by mos_data_dict() containing info, filepaths, etc. for a full dataset \
###
###       sample_list = a list of keys for samples in sample_data_dict. Only these samples will actually be processed/annotated. \
###                     If =None, defaults to all keys in 'sample_data_dict'. \
###	  root_dir_path = string path to root project directory \
###       Predictor = Detectron 2 predictor for automatic segmentation of zircon images. Should be initialized before running this fxn.
def run_GUI(sample_data_dict, sample_list, root_dir_path, Predictor, load_dir = None):
    """Run the colab-based GUI for automated / manual zircon segmentation and
       segmentation inspection.

    Parameters
    ----------
    sample_data_dict : dict
        A dict of dicts containing data from project folder w/ format:

        {'SAMPLE NAME': {'Scanlist': SCANLIST (.SCANCSV) PATH,
                         'Mosaic': MOSAIC .BMP PATH,
                         'Align_file': MOSAIC ALIGN FILE PATH,
                         'Max_zircon_size': MAX USER-INPUT ZIRCON SIZE,
                         'Offsets': [USER X OFFSET, USER Y OFFSET],
                         'Scan_dict': DICT LOADED FROM .SCANCSV FILE},
         ...}.
    sample_list : list of str
        A list of sample names (selected by user while running Colab notebook)
        indicating which samples they will actually be working with.
    root_dir_path : str
        Full path to project directory.
    Predictor : Detectron2 Predictor class instance
        A Detectron2 Predictor; should be initialized before running this fxn.
    load_dir : str, optional
        User-selected directory with .json files for loading polygons. The default is None.

    Raises
    ------
    TypeError
        Raised if image type is not supported (np array or array-like).

    Returns
    -------
    None

    """

    if len(sample_list) == 0:
        print('ERROR: NO SAMPLES SELECTED')
        return

    ### CODE FOR FUNCTIONS BELOW (SIGNIFICANTLY) MODIFIED FROM tensorflow.models FOR POLYGON ANNOTATION ###
    # Lint as: python3
    # Copyright 2020 The TensorFlow Authors. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================

    def image_from_numpy(image):
        """Open an image at the specified path and encode it in Base64.
        Args:
          image: np.ndarray
            Image represented as a numpy array
        Returns:
          An encoded Base64 representation of the image
        """

        with io.BytesIO() as img_output:
            Image.fromarray(image).save(img_output, format='JPEG')
            data = img_output.getvalue()
        data = str(base64.b64encode(data))[2:-1]
        return data


    def draw_polygons(image_urls, spot_names, track_list, original_polys,
                      auto_human_list_input, tag_list_input1,
                      sample_name, sample_scale_factor, callbackId1, callbackId2):  # pylint: disable=invalid-name
        """Open polygon annotation UI and send the results to a callback function.
        """
        js = Javascript('''
                    async function load_image(imgs, spot_nms, trck_list, orig_polys, inpt_auto_human, inpt_tag_list, sample_nm, sample_scl,  callbackId1, callbackId2) {
                        
                        //init current sample number displays (index + 1)
                        var curr_sample_idx = trck_list[0] + 1;
                        var max_sample_idx = trck_list[1] + 1;

                        //init organizational elements
                        const div = document.createElement('div');
                        const buttondiv = document.createElement('div');
                        buttondiv.style.alignItems = 'center';
                        var image_cont = document.createElement('div');
                        var errorlog = document.createElement('div');
                        var crosshair_h = document.createElement('div');
                        var sample_name = document.createElement('div');
                        sample_name.innerHTML = 'Sample: ' + sample_nm + " (" + curr_sample_idx + '/' + max_sample_idx + ')';
                        var sample_scale = document.createElement('div');
                        sample_scale.innerHTML = 'Scale: ' + sample_scl + ' µm/pixel';
                        var spot_name = document.createElement('div');
                        crosshair_h.style.position = "absolute";
                        crosshair_h.style.backgroundColor = "transparent";
                        crosshair_h.style.width = "100%";
                        crosshair_h.style.height = "0px";
                        crosshair_h.style.zIndex = 9998;
                        crosshair_h.style.borderStyle = "dotted";
                        crosshair_h.style.borderWidth = "2px";
                        crosshair_h.style.borderColor = "rgba(255, 0, 0, 0.75)";
                        crosshair_h.style.cursor = "crosshair";
                        var crosshair_v = document.createElement('div');
                        crosshair_v.style.position = "absolute";
                        crosshair_v.style.backgroundColor = "transparent";
                        crosshair_v.style.width = "0px";
                        crosshair_v.style.height = "100%";
                        crosshair_v.style.zIndex = 9999;
                        crosshair_v.style.top = "0px";
                        crosshair_v.style.borderStyle = "dotted";
                        crosshair_v.style.borderWidth = "2px";
                        crosshair_v.style.borderColor = "rgba(255, 0, 0, 0.75)";
                        crosshair_v.style.cursor = "crosshair";
                        crosshair_v.style.marginTop = "90px";
                        var brdiv = document.createElement('br');
                        //init control elements
                        var next = document.createElement('button');
                        var prev = document.createElement('button');
                        var submit = document.createElement('button');
                        var deleteButton = document.createElement('button');
                        var deleteAllbutton = document.createElement('button');
                        var resetButton = document.createElement('button');
                        var tagImagebutton = document.createElement('button');
                        var prevSamplebutton = document.createElement('button');
                        var nextSamplebutton = document.createElement('button');
                        //init image containers
                        var image = new Image();
                        var canvas_img = document.createElement('canvas');
                        var ctx = canvas_img.getContext("2d");
                        canvas_img.style.cursor = "crosshair";
                        canvas_img.setAttribute('draggable', false);
                        crosshair_v.setAttribute('draggable', false);
                        crosshair_h.setAttribute('draggable', false);
                        // polygon containers
                        const height = 500
                        //const width = 600
                        var allPolygons = [];
                        var all_human_auto = [];
                        var all_tags = [];
                        var curr_image = 0
                        var im_height = 0;
                        var im_width = 0;
                        var aspect_ratio = 0.0;
                        //initialize polygons, human_auto tags, user tags
                        for (var i = 0; i < imgs.length; i++) {
                          allPolygons[i] = [...orig_polys[i]];
                          all_human_auto[i] = inpt_auto_human[i];
                          all_tags[i] = inpt_tag_list[i];
                        }
                        //initialize image view
                        errorlog.id = 'errorlog';
                        image.style.display = 'block';
                        image.setAttribute('draggable', false);
                        //load the first image
                        img = imgs[curr_image];
                        //im_height = img[0].length;
                        //im_width = img.length;
                        //aspect_ratio = (im_width / im_height); //width curr image / height
                        spot_name.innerHTML = spot_nms[curr_image] + " (" + (curr_image + 1) + "/" + imgs.length + ")";
                        image.src = "data:image/png;base64," + img;
                        image.onload = function() {
                            // normalize display height and canvas
                            aspect_ratio = image.naturalWidth / image.naturalHeight
                            image.height = height;
                            image_cont.height = canvas_img.height = image.height;
                            image.width = (height*aspect_ratio).toFixed(0);
                            //image.width = (height*aspect_ratio).toFixed(0);
                            image_cont.width = canvas_img.width = image.width;
                            crosshair_v.style.height = image_cont.height + "px";
                            crosshair_h.style.width = image_cont.width + "px";
                            // draw the new image
                            ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight, 0, 0,  canvas_img.width,  canvas_img.height);
                            draw();
                        };
                        // move to next image in array
                        next.textContent = "next scan";
                        next.onclick = function(){
                            if (curr_image < imgs.length - 1){
                                // clear canvas and load new image
                                curr_image += 1;
                                errorlog.innerHTML = "";
                            }
                            else{
                                errorlog.innerHTML = "";
                            }
                            resetcanvas();
                        }
                        //move forward through list of images
                        prev.textContent = "prev scan"
                        prev.onclick = function(){
                            if (curr_image > 0){
                                // clear canvas and load new image
                                curr_image -= 1;
                                errorlog.innerHTML = "";
                            }
                            else{
                                errorlog.innerHTML = "";
                            }
                            resetcanvas();
                        }
                        //tag image on press
                        tagImagebutton.textContent = "tag scan"
                        tagImagebutton.onclick = function(){
                            var orig_tagged = false;
                            if (all_tags[curr_image] === 'True') {
                                orig_tagged = true;
                            }
                            if (orig_tagged){
                                // clear canvas and load new image
                                all_tags[curr_image] = 'False';
                            }
                            else{
                                all_tags[curr_image] = 'True';
                            }
                            resetcanvas();
                        }

                        // on undo (modified delete buttun), deletes the last polygon vertex
                        deleteButton.textContent = "undo last pt";
                        deleteButton.onclick = function(){
                          if (poly.length > 0) {
                            all_human_auto[curr_image] = 'human'; //tags polygon as drawn by human
                          }
                          poly.pop();
                          ctx.clearRect(0, 0, canvas_img.width, canvas_img.height);
                          image.src = "data:image/png;base64," + img;
                          image.onload = function() {
                              ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight, 0, 0,  canvas_img.width,  canvas_img.height);
                              drawPoly(poly);
                          };
                        }
                        // on all delete, deletes all of the polygons
                        deleteAllbutton.textContent = "clear polygon"
                        deleteAllbutton.onclick = function(){
                          if (poly.length > 0) {
                            all_human_auto[curr_image] = 'human'; //tags polygon as drawn by human
                          }
                          poly.splice(0, poly.length);
                          ctx.clearRect(0, 0, canvas_img.width, canvas_img.height);
                          image.src = "data:image/png;base64," + img;
                          image.onload = function() {
                              ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight, 0, 0,  canvas_img.width,  canvas_img.height);
                              drawPoly(poly);
                          };
                        }

                        // on reset, reverts to original (auto or user) polygon
                        resetButton.textContent = "restore orig. polygon"
                        resetButton.onclick = function(){
                          poly.splice(0, poly.length, ...orig_polys[curr_image]);
                          all_human_auto[curr_image] = inpt_auto_human[curr_image];
                          ctx.clearRect(0, 0, canvas_img.width, canvas_img.height);
                          image.src = "data:image/png;base64," + img;
                          image.onload = function() {
                              ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight, 0, 0,  canvas_img.width,  canvas_img.height);
                              drawPoly(poly);
                          };
                        }
                        // on submit, send the polygon to display
                        submit.textContent = "Analyze sample zircon dimensions and export to Drive";
                        submit.onclick = function(){
                          errorlog.innerHTML = "Segmentations sent for processing";
                          // send polygon data to callback function
                          google.colab.kernel.invokeFunction(callbackId1, [allPolygons, all_human_auto, all_tags], {});
                        }
                        // on next sample, moves to next sample (will not work if at last sample)
                        nextSamplebutton.textContent = "Next sample";
                        nextSamplebutton.onclick = function(){
                          errorlog.innerHTML = "";
                          if (trck_list[0] < trck_list[1]) {
                              // quit and move to next sample
                              div.remove();
                              google.colab.kernel.invokeFunction(callbackId2, ['next'], {});
                          }
                          if (trck_list[0] === trck_list[1]) {
                              errorlog.innerHTML = "Already at last sample!";
                          }
                        }
                        // on prev sample, moves to prev sample (will not work if at 1st sample)
                        prevSamplebutton.textContent = "Prev sample";
                        prevSamplebutton.onclick = function(){
                          errorlog.innerHTML = "";
                          if (trck_list[0] > 0) {
                              // quit and move to prev sample
                              div.remove();
                              google.colab.kernel.invokeFunction(callbackId2, ['prev'], {});
                          }
                          if (trck_list[0] === 0) {
                              errorlog.innerHTML = "Already at first sample";
                          }
                        }

                      // init template for annotations
                      const annotation = {
                            x: 0,
                            y: 0,

                      };
                      // the array of polygon points
                      let poly = allPolygons[curr_image];
                      // the actual polygon being drawn
                      let o = {};
                      // a variable to store the mouse position
                      let m = {},
                      // a variable to store the point where you begin to draw the
                      // polygon
                      start = {};
                      // a boolean variable to store the drawing state
                      let isDrawing = false;
                      var elem = null;
                      function handleClick(e) {
                        // if drawing, add new polygon point
                        if (isDrawing) {
                          //errorlog.innerHTML = "next single click started " + poly.toString();
                          const nextpt = Object.create(annotation);
                          nextpt.x = o.x;
                          nextpt.y = o.y;
                          poly.push(nextpt);
                          //errorlog.innerHTML = "next single_click successful: " + poly.toString();
                        }
                        
                        // if not drawing, clear current polygons and start new one
                        if (!isDrawing) {
                          poly.splice(0, poly.length); // clears polygon
                          all_human_auto[curr_image] = 'human'; //tags polygon as drawn by human
                          const firstpt = Object.create(annotation);
                          start = oMousePos(canvas_img, e);
                          o.x = (start.x)/image.width;  // start position of x
                          o.y = (start.y)/image.height;  // start position of y
                          firstpt.x = o.x;
                          firstpt.y = o.y;
                          poly.push(firstpt);
                          isDrawing = true;
                          //errorlog.innerHTML = "first single_click successful: " + poly.toString() + isDrawing;
                        }
                        // on mouse click set change the cursor and start tracking the mouse position
                        //start = oMousePos(canvas_img, e);
                        // configure is drawing to true
                        //isDrawing = true;
                      }
                      function handleMouseMove(e) {
                          // move crosshairs, but only within the bounds of the canvas
                          if (document.elementsFromPoint(e.pageX, e.pageY).includes(canvas_img)) {
                            crosshair_h.style.top = e.pageY + "px";
                            crosshair_v.style.left = e.pageX + "px";
                          }
                          // move the bounding polygon
                          if(isDrawing){
                            m = oMousePos(canvas_img, e);
                            draw();
                          }
                      }
                      function handleDblClick(e) {
                        //errorlog.innerHTML = "double click triggered: " + poly.toString();
                          if (isDrawing) {
                              // on double click, push current annotation to poly
                              isDrawing = false;
                              draw()
                          }
                      }
                      function draw() {
                          o.x = (m.x)/image.width;  // curr mouse position in canvas space
                          o.y = (m.y)/image.height;  // curr y mouse position in canvas space
                          //o.w = (m.x - start.x)/image.width;  // width
                          //o.h = (m.y - start.y)/image.height;  // height
                          ctx.clearRect(0, 0, canvas_img.width, canvas_img.height);
                          ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight, 0, 0,  canvas_img.width,  canvas_img.height);
                          // draw the polygon
                          drawPoly(poly);
                      }
                      
                      // add the handlers needed for dragging
                      crosshair_h.addEventListener("click", handleClick);
                      crosshair_v.addEventListener("click", handleClick);
                      document.addEventListener("mousemove", handleMouseMove);
                      document.addEventListener("dblclick", handleDblClick);
                      function resetcanvas(){
                          // clear canvas
                          ctx.clearRect(0, 0, canvas_img.width, canvas_img.height);
                          img = imgs[curr_image]
                          spot_name.innerHTML = spot_nms[curr_image] + " (" + (curr_image + 1) + "/" + imgs.length + ")";
                          if (all_tags[curr_image] === 'True') {
                            spot_name.innerHTML = spot_nms[curr_image] + " (tagged) " + (curr_image + 1) + "/" + imgs.length + ")";
                          }
                          image.src = "data:image/png;base64," + img;
                          // onload init new canvas and display image
                          image.onload = function() {
                              // normalize display height and canvas
                              aspect_ratio = image.naturalWidth / image.naturalHeight
                              image.height = height;
                              image_cont.height = canvas_img.height = image.height;
                              image.width = (height*aspect_ratio).toFixed(0);
                              image_cont.width = canvas_img.width = image.width;
                              crosshair_v.style.height = image_cont.height + "px";
                              crosshair_h.style.width = image_cont.width + "px";
                              // draw the new image
                              ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight, 0, 0,  canvas_img.width,  canvas_img.height);
                              // draw polygons
                              poly = allPolygons[curr_image];
                              drawPoly(poly);
                              //errorlog.innerHTML = "reset_run";
                          };
                      }
                      function drawPoly(polyarray){
                          // draw a predefined polygon
                          if (!(polyarray.length === 0)) {
                            ctx.strokeStyle = "rgba(255, 0, 0, 0.75)";
                            ctx.fillStyle = 'rgba(255, 255, 200, 0.2)';
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(polyarray[0].x * image.width, polyarray[0].y * image.height);
                            if (polyarray.length >= 2) {
                              for (var i = 1; i < polyarray.length; i++) {
                                ctx.lineTo(polyarray[i].x * image.width, polyarray[i].y * image.height);
                              }
                            }
                            if (isDrawing) {
                              ctx.lineTo(o.x*image.width, o.y*image.height);
                            }
                            ctx.closePath();
                            ctx.stroke();
                            ctx.fill();
                          }
                        }
                      // Function to detect the mouse position
                      function oMousePos(canvas_img, evt) {
                        let ClientRect = canvas_img.getBoundingClientRect();
                          return {
                            x: evt.clientX - ClientRect.left,
                            y: evt.clientY - ClientRect.top
                          };
                      }
                      //configure colab output display
                      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
                      //build the html document that will be seen in output
                      div.appendChild(document.createElement('br'))
                      div.appendChild(sample_name)
                      div.appendChild(sample_scale)
                      div.appendChild(document.createElement('br'))
                      div.appendChild(spot_name)
                      div.appendChild(image_cont)
                      image_cont.appendChild(canvas_img)
                      image_cont.appendChild(crosshair_h)
                      image_cont.appendChild(crosshair_v)
                      div.appendChild(document.createElement('br'))
                      div.appendChild(errorlog)
                      buttondiv.appendChild(prev)
                      buttondiv.appendChild(next)
                      buttondiv.appendChild(document.createElement('br'))
                      buttondiv.appendChild(deleteButton)
                      buttondiv.appendChild(deleteAllbutton)
                      buttondiv.appendChild(resetButton)
                      buttondiv.appendChild(tagImagebutton)
                      buttondiv.appendChild(document.createElement('br'))
                      buttondiv.appendChild(document.createElement('br'))
                      buttondiv.appendChild(submit)
                      buttondiv.appendChild(document.createElement('br'))
                      buttondiv.appendChild(brdiv)
                      buttondiv.appendChild(prevSamplebutton)
                      buttondiv.appendChild(nextSamplebutton)
                      div.appendChild(buttondiv)
                      document.querySelector("#output-area").appendChild(div);
                      return
                  }''')

        # load the images as a byte array
        bytearrays = []
        for image in image_urls:
            if isinstance(image, np.ndarray):
                bytearrays.append(image_from_numpy(image))
            else:
                raise TypeError('Image has unsupported type {}.'.format(type(image)))

        # format arrays for input
        image_data = json.dumps(bytearrays)
        del bytearrays

        # call java script function pass string byte array(image_data) as input
        display(js)

        eval_js('load_image({}, {}, {}, {}, {}, {}, \'{}\', \'{}\', \'{}\', \'{}\')'.format(image_data, spot_names, track_list, original_polys,
                                                                                            auto_human_list_input, tag_list_input1,
                                                                                            sample_name, str(sample_scale_factor),
                                                                                            callbackId1, callbackId2))

        return


    def annotate(imgs: List[Union[str, np.ndarray]],  # pylint: disable=invalid-name
                  poly_storage_pointer: List[np.ndarray],
                  auto_polygons: List[List[dict]],
                  spot_names: List[str],
                  track_list: List[int],
                  outputs_path: str,
                  predictor_input,
                  auto_human_list_input: List[str],
                  tag_list_input: List[str],
                  sample_name: str = None,
                  sample_scale_factor: float = 0.0):
        """Open the polygon annotation UI and prompt the user for input.
        """

        # Set random IDs for the callback functions
        callbackId1 = str(uuid.uuid1()).replace('-', '')
        callbackId2 = str(uuid.uuid1()).replace('-', '')

        def dictToList(input_poly):  # pylint: disable=invalid-name
            """Convert polygons.
            This function converts the dictionary from the frontend (if the format
            {x, y} as shown in callbackFunction) into a list
            ([x, y])
            """
            return (input_poly['y'], input_poly['x'])

        #def callbackFunction(annotations: List[List[Dict[str, float]]]):  # pylint: disable=invalid-name
        def savecallbackFunction(annotations, human_auto_list, tags_list):  # pylint: disable=invalid-name
            #print('callback started')
            """Callback function.
            This is the call back function to capture the data from the frontend,
            convert the data into a numpy array, and export it to linked Google Drive folders.
            """
            tags_for_export = []
            for tag in tags_list:
                if str(tag) == 'True':
                    tags_for_export.append('True')
                else:
                    tags_for_export.append('')
            #tags_for_export = [tag if tag == 'True' else '' for tag in tags_list ]
            #print('poly conversion started')
            # reset the poly list
            nonlocal poly_storage_pointer
            #print(box_storage_pointer)
            polys: List[np.ndarray] = poly_storage_pointer
            polys.clear()

            # load the new annotations into the polygon list
            for annotations_per_img in annotations:
                polys_as_arrays = [dictToList(annotation)
                                        for annotation in annotations_per_img]
                if polys_as_arrays:
                    polys.append(np.stack(polys_as_arrays))
                else:
                    polys.append(None)

            ###exports data


            output_data_list = []
            #paths for saving
            img_save_root_dir = os.path.join(outputs_path, 'mask_images')
            each_img_save_dir = os.path.join(img_save_root_dir, str(sample_name))
            csv_save_dir = os.path.join(outputs_path, 'zircon_dimensions')
            #poly_save_dir = os.path.join(outputs_path, 'saved_polys')

            #directory for saving images for each sample
            os.makedirs(each_img_save_dir, exist_ok=True)
            
            ##directory for saving polygons for current sample
            #os.makedirs(poly_save_dir, exist_ok=True)
            
            
            for eachindex, eachpoly in enumerate(poly_storage_pointer):

                poly_mask = poly_utils.poly_to_mask(eachpoly, imgs[eachindex])
                
                tag_Bool = False
                if tags_for_export[eachindex] == 'True':
                    tag_Bool = True

                #if polygon sucessfully converted into a mask w/ area >0:
                if poly_mask[0] == True:
                    each_props = mos_proc.overlay_mask_and_get_props(poly_mask[1], imgs[eachindex], spot_names[eachindex],
                                                                     display_bool = False, save_dir=each_img_save_dir,
                                                                     tag_bool = tag_Bool, scale_factor=sample_scale_factor)
                    each_props_list = mos_proc.parse_properties(each_props, sample_scale_factor,
                                                                spot_names[eachindex], verbose = False)
                    each_props_list.extend([human_auto_list[eachindex], tags_for_export[eachindex]])
                    output_data_list.append(each_props_list)
                else:
                    null_properties = mos_proc.parse_properties([], sample_scale_factor, spot_names[eachindex], verbose = False)
                    null_properties.extend([human_auto_list[eachindex], tags_for_export[eachindex]])
                    output_data_list.append(null_properties)
                    mos_proc.save_show_results_img(imgs[eachindex], spot_names[eachindex], display_bool=False,
                                                   save_dir=each_img_save_dir, tag_bool=tag_Bool,
                                                   scale_factor=sample_scale_factor)


            #converts collected data to pandas DataFrame, saves as .csv
            output_dataframe = pd.DataFrame(output_data_list,
                                            columns=['Analysis', 'Area (µm^2)', 'Convex area (µm^2)', 'Eccentricity',
                                                     'Equivalent diameter (µm)', 'Perimeter (µm)', 'Major axis length (µm)',
                                                     'Minor axis length (µm)', 'Circularity', 'Scale factor (µm/pixel)',
                                                     'Human_or_auto', 'tagged?'])
            csv_filename = str(sample_name) + '_zircon_dimensions.csv'
            output_csv_filepath = os.path.join(csv_save_dir, csv_filename)
            czd_utils.save_csv(output_csv_filepath, output_dataframe)
            
            save_load.save_sample_json(outputs_path, str(sample_name), spot_names,
                                       annotations, human_auto_list, tags_for_export)

            
            # output the annotations to the errorlog
            with output.redirect_to_element('#errorlog'):
                display('--Export_completed')

      
        #a callback function to change samples; including 
        def changesamplecallbackFunction(next_prev_str):

            nonlocal index_tracker
            nonlocal predictor_input

            proceed_bool = False

            if str(next_prev_str) == 'next': 
                if index_tracker.at_end == False:
                    index_tracker.next_sample()
                    proceed_bool = True

            if str(next_prev_str) == 'prev': 
                if index_tracker.at_begin == False:
                    index_tracker.prev_sample()
                    proceed_bool = True

            if proceed_bool:
                #output.clear()
                #eval_js(remove_prev_GUI_js) #should clear previous GUI
                load_and_annotate(Predictor)

        output.register_callback(callbackId1, savecallbackFunction)
        output.register_callback(callbackId2, changesamplecallbackFunction)
        draw_polygons(imgs, spot_names, track_list, auto_polygons, auto_human_list_input,
                      tag_list_input, sample_name, sample_scale_factor, callbackId1, callbackId2)

### END MODIFIED CODE
# ====================================================================================================

    ### Code for looping through datasets, loading data for each, etc. below\

    # a class for storing index and name of a samples from a list of sample names.\
    # An instance of this class is made nonlocal and used to track position within \
    # a dataset while using the GUI function.
    class sample_index:

        def __init__(self, input_sample_list):
            self.sample_list = input_sample_list
            self.curr_index = 0
            self.max_index = len(input_sample_list) - 1
            self.at_begin = True
            self.at_end = False
            self.track_list = [self.curr_index, self.max_index]
            if self.curr_index == self.max_index:
                self.at_end = True

            #self.curr_sample = current sample name
            self.curr_sample = self.sample_list[self.curr_index]
        
        #moves to next sample, unless at end of sample list
        def next_sample(self):
            if self.at_end == True:
                return
            self.curr_index += 1
            self.track_list = [self.curr_index, self.max_index]
            self.curr_sample = self.sample_list[self.curr_index]
            if self.curr_index == self.max_index:
                self.at_end = True
            if self.curr_index > 0:
                self.at_begin = False

        #moves to prev sample, unless at beginning of sample list
        def prev_sample(self):
            if self.at_begin == True:
                return
            self.curr_index -= 1
            self.track_list = [self.curr_index, self.max_index]
            self.curr_sample = self.sample_list[self.curr_index]
            if self.curr_index < self.max_index:
                self.at_end = False
            if self.curr_index == 0:
                self.at_begin = True
    


    # segments all zircons in a sample automatically and opens annotation GUI to inspect, modify, and/or save segmentations

    def load_and_annotate(Predictor):

        nonlocal index_tracker
        nonlocal sample_data_dict
        nonlocal run_dir
        nonlocal run_load_dir

        #lists, variables that will be called in function for loading new samples
        curr_auto_polys = [] #list of polygons from automatically generated masks (as np (N, 2) arrays)
        curr_poly_pointer = [] #pointer for polygons from GUI, automatically updated in save functions
        curr_subimage_list = [] #list of subimages (as np arrays) for loading into GUI
        #curr_scan_name_list = [] #list of spot names corresponding to each subimage
        curr_auto_human_list = [] #list of strings indicating whether segmentation \
                                  # (or lack therof) of spot was done automatically or by a human
        curr_spot_tags = [] #list of strings indicating whether user has 'tagged' each spot
        curr_scale_factor = 1.0 #current scale factor


        curr_dict_copy = copy.deepcopy(sample_data_dict[index_tracker.curr_sample])

        #loads sample mosaic
        each_mosaic = mos_proc.MosImg(curr_dict_copy['Mosaic'], curr_dict_copy['Align_file'],
                                      curr_dict_copy['Max_zircon_size'], curr_dict_copy['Offsets'])
        curr_scan_names = list(curr_dict_copy['Scan_dict'].keys())
        print('Scale factor:', each_mosaic.scale_factor, 'µm/pixel')
        print(2 * "\n")
        curr_scale_factor = each_mosaic.scale_factor
        load_outputs = [False]
        if run_load_dir is not None:
            load_outputs = save_load.find_load_json_polys(run_load_dir,
                                                          index_tracker.curr_sample,
                                                          curr_scan_names)
        if load_outputs[0] is False:
            #use predictor to automatically segment images if loadable polys unavailable
            print('Auto-processing:', index_tracker.curr_sample)
            for eachscan in curr_scan_names:
                #gets subimage, processes it, and appends subimage and results (or empty values if unsuccessful) to various lists
                each_mosaic.set_subimg(*curr_dict_copy['Scan_dict'][eachscan])
                curr_subimage_list.append(each_mosaic.sub_img)
                print(str(eachscan) + ':')
                outputs = Predictor(each_mosaic.sub_img)
                central_mask = mos_proc.get_central_mask(outputs)
                curr_auto_human_list.append('auto')
                curr_spot_tags.append('')
    
                #if a central zircon is found, does initial processing and adds polygon 
                if central_mask[0] == True:
                    print('Successful')
    
                    ## uncomment below to produce and print initial (auto) zircon measurements while samples are loading
                    #each_props = mos_proc.overlay_mask_and_get_props(central_mask[1], each_mosaic.sub_img, eachscan, display_bool = False)
                    #props_list = mos_proc.parse_properties(each_props, each_mosaic.scale_factor, eachscan, verbose = True)
                    
                    curr_auto_polys.append(poly_utils.mask_to_poly(central_mask[1], 1, 
                                                                   each_mosaic.scale_factor))
                else:
                    curr_auto_polys.append([])
            #saves polygons on initial processing so that processing does not have to repeat if navigating back to sample
            run_load_dir = os.path.join(run_dir, 'saved_polys')
            save_load.save_sample_json(run_dir, index_tracker.curr_sample,
                                       curr_scan_names, curr_auto_polys)
        else:
            #simply load polygons from .json if possible
            curr_auto_polys, curr_auto_human_list, curr_spot_tags = load_outputs[1:]
            print('Preparing grain subimages')
            for eachscan in curr_scan_names:
                #gets subimage, processes it, and appends subimage and results (or empty values if unsuccessful) to various lists
                each_mosaic.set_subimg(*curr_dict_copy['Scan_dict'][eachscan])
                curr_subimage_list.append(each_mosaic.sub_img)
            
        print('')

        #starts annotator GUI for current sample
        output.clear()
        annotate(curr_subimage_list, curr_poly_pointer, curr_auto_polys, curr_scan_names, index_tracker.track_list,
                 run_dir, Predictor, curr_auto_human_list, curr_spot_tags, str(index_tracker.curr_sample), curr_scale_factor)


    ##code below runs upon initial startup
    index_tracker = sample_index(sample_list) #initializes sample/index tracker class instance

    #directory initialization
    root_output_dir = os.path.join(root_dir_path, 'outputs') #main output directory path, can be modified if necessary

    #creates output directory if it does not already exist
    if not os.path.exists(root_output_dir):
        os.makedirs(root_output_dir)

    #creates a main directory for this processing run
    run_dir = os.path.join(root_output_dir, 'semi-auto_proccessing_run_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(run_dir)

    #creates a root directory for saved images
    img_save_root_dir = os.path.join(run_dir, 'mask_images')
    os.makedirs(img_save_root_dir)

    #creates a directory for zircon dimension .csv files
    csv_save_dir = os.path.join(run_dir, 'zircon_dimensions')
    os.makedirs(csv_save_dir)
    
    if load_dir is not None:
        run_load_dir = save_load.transfer_json_files(sample_list, run_dir, load_dir,
                                           verbose=True)
    else:
        run_load_dir = None

    #starts annotator for first time/sample
    load_and_annotate(Predictor)

