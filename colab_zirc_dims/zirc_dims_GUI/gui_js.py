# -*- coding: utf-8 -*-
"""
Javascript code for use in ALC, generalized colab_zirc_dims GUIs. Javascript
here is significantly extended from code in tensorflow.models (Apache 2.0 license).
"""

from IPython.display import Javascript

__all__ = ['js']

### CODE BELOW (SIGNIFICANTLY) MODIFIED FROM tensorflow.models FOR POLYGON ANNOTATION ###
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

js = Javascript('''
async function load_image(is_colab, imgs, spot_nms, trck_list, sample_scls, orig_polys, inpt_auto_human, inpt_tag_list, sample_nm, callbackId1, callbackId2, callbackId3, callbackId4) {
    
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
    if (is_colab == true){
      crosshair_v.style.marginTop = "90px";
    } else {
      crosshair_v.style.marginTop="0px";
    }
    var brdiv = document.createElement('br');
    //init control elements
    var next = document.createElement('button');
    var prev = document.createElement('button');
    var MeasureSample = document.createElement('button');
    var savepolys = document.createElement('button');
    var deleteButton = document.createElement('button');
    var deleteAllbutton = document.createElement('button');
    var resetButton = document.createElement('button');
    var tagImagebutton = document.createElement('button');
    var prevSamplebutton = document.createElement('button');
    var nextSamplebutton = document.createElement('button');
    var analyzeAllbutton = document.createElement('button');
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
    var img = imgs[curr_image];
    //im_height = img[0].length;
    //im_width = img.length;
    //aspect_ratio = (im_width / im_height); //width curr image / height
    sample_scale.innerHTML = 'Scale: ' + sample_scls[curr_image] + ' µm/pixel';
    spot_name.innerHTML = spot_nms[curr_image] + " (" + (curr_image + 1) + "/" + imgs.length + ")";
    if (all_tags[curr_image] === 'True') {
            spot_name.innerHTML = spot_nms[curr_image] + " (tagged) (" + (curr_image + 1) + "/" + imgs.length + ")";
    }
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
    next.textContent = "next scan [d]";
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
    prev.textContent = "prev scan [a]"
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
    tagImagebutton.textContent = "tag scan [t]"
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
    deleteButton.textContent = "undo last pt [z]";
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
    deleteAllbutton.textContent = "clear polygon [c]"
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
    resetButton.textContent = "restore orig. polygon [r]"
    resetButton.onclick = function(){
      if (isDrawing) {
          // on reset, stop drawing poly
          isDrawing = false;
          draw();
      }
      poly.splice(0, poly.length, ...orig_polys[curr_image]);
      all_human_auto[curr_image] = inpt_auto_human[curr_image];
      ctx.clearRect(0, 0, canvas_img.width, canvas_img.height);
      image.src = "data:image/png;base64," + img;
      image.onload = function() {
          ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight, 0, 0,  canvas_img.width,  canvas_img.height);
          drawPoly(poly);
      };
    }
    //jsonify arguments manually if executing in local ipynotebook
    function stringify_args(args_arr, kwargs){
        const out_args = JSON.stringify(args_arr);
        const out_kwargs = JSON.stringify(kwargs);
        return " r'" + out_args + "'," + " r'" + out_kwargs + "'";
      };
    
    //invoke a callback manually using notebook.kernel.execute
     function local_ipyinvoke(callbackid, args, kwargs){
         const args_str = "'" + callbackid + "'," + stringify_args(args, kwargs);
         const full_run_fxn_str = "output.invoke_function("+args_str + ")";
         Jupyter.notebook.kernel.execute(full_run_fxn_str);
     };
     
    //direct callback and args to either colab or notebook kernel
    function callback_director(callbackid, args, kwargs){
        if (is_colab===false){
            local_ipyinvoke(callbackid, args, kwargs);
        }
        if (is_colab===true){
            google.colab.kernel.invokeFunction(callbackid, args, kwargs);
        }
    };

    // on analyze and save (MeasureSample), send polygons for saving, conversion, size analysis
    MeasureSample.textContent = "Analyze sample zircon dimensions and export to Drive";
    MeasureSample.onclick = function(){
      errorlog.innerHTML = "Segmentations sent for processing";
      //orig polygons = new polygons after saving
      for (var i = 0; i < imgs.length; i++) {
      orig_polys[i] = [...allPolygons[i]];
      inpt_auto_human[i] = all_human_auto[i];
      }
      // send polygon data to callback function
      callback_director(callbackId1, [allPolygons, all_human_auto, all_tags], {});
    }
    // on savepolys, send polygons for saving
    savepolys.textContent = "Save changes to sample polygons";
    savepolys.onclick = function(){
      errorlog.innerHTML = "Saving sample polygons";
      //orig polygons = new polygons after saving
      for (var i = 0; i < imgs.length; i++) {
      orig_polys[i] = [...allPolygons[i]];
      inpt_auto_human[i] = all_human_auto[i];
      }
      // send polygon data to callback function
      callback_director(callbackId3, [allPolygons, all_human_auto, all_tags], {});
    }
    // on next sample, moves to next sample (will not work if at last sample)
    nextSamplebutton.textContent = "Next sample";
    nextSamplebutton.onclick = function(){
      errorlog.innerHTML = "";
      if (trck_list[0] < trck_list[1]) {
          // quit and move to next sample
          div.remove();
          callback_director(callbackId2, ['next', allPolygons, all_human_auto, all_tags], {});
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
          callback_director(callbackId2, ['prev', allPolygons, all_human_auto, all_tags], {});
      }
      if (trck_list[0] === 0) {
          errorlog.innerHTML = "Already at first sample";
      }
    }
    // on analyzeall, save current polys, clear GUI, and start callback
    analyzeAllbutton.textContent = "Analyze, export zircon dimensions from polygons for all selected samples";
    analyzeAllbutton.onclick = function(){
      errorlog.innerHTML = "";
      // close GUI
      div.remove();
      // start analyze all callback (saves current sample polygons)
      callback_director(callbackId4, [allPolygons, all_human_auto, all_tags], {});
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
      const nextpt = Object.create(annotation);
      nextpt.x = o.x;
      nextpt.y = o.y;
      poly.push(nextpt);
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
    }
    // on mouse click set change the cursor and start tracking the mouse position
    //start = oMousePos(canvas_img, e);
    // configure is drawing to true
    //isDrawing = true;
  }
  function handleMouseMove(e) {
      // move crosshairs, but only within the bounds of the canvas
    if (document.elementsFromPoint(e.pageX, e.pageY).includes(canvas_img)) {
        //Colab and local jupyter cells have different output areas and
        // require different handling of the crosshairs. Placement of
        // all other elements (including polygon vertices relative to
        // images) seems unaffected.
        if (is_colab===true){
            crosshair_h.style.top = e.pageY + "px";
            crosshair_v.style.left = e.pageX + "px";
        } else {
            var div_rect = div.getBoundingClientRect();
            var canv_rect= canvas_img.getBoundingClientRect();
            var v_top_offset = canv_rect.top - div_rect.top + 3;
            crosshair_v.style.top = v_top_offset + "px";
            var rel_x = e.pageX - canv_rect.left + 3;
            var rel_y = e.pageY - div_rect.top + 3;
            crosshair_h.style.top = rel_y + "px";
            crosshair_v.style.left = rel_x + "px";
        }
    }
      // move the bounding polygon
      if(isDrawing){
        m = oMousePos(canvas_img, e);
        draw();
      }
  }
  function handleDblClick(e) {
      if (isDrawing) {
          // on double click, push current annotation to poly
          isDrawing = false;
          draw();
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
  
  // add the handlers needed for clicking
  crosshair_h.addEventListener("click", handleClick);
  crosshair_v.addEventListener("click", handleClick);
  //add click event listeners. Limit scope if not in Colab.
  if (is_colab===true){
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("dblclick", handleDblClick);
    } else{
      image_cont.addEventListener("mousemove", handleMouseMove);
      image_cont.addEventListener("dblclick", handleDblClick);
  }

  //register hotkey shortcuts
  function hotkey_press(event){
        //hotkey e to push polygon (same as double click)
        if (event.key === 'e'){
                handleDblClick(event);
        }
        //hotkey d for next scan
        if (event.key === 'd'){
                next.click();
        }
        //hotkey a for previous scan
        if (event.key === 'a'){
                prev.click();
        }
        //hotkey r for revert to original polygon
        if (event.key === 'r'){
                resetButton.click();
        }
        //hotkey t for tag scan
        if (event.key === 't'){
                tagImagebutton.click();
        }
        //hotkey z for undo last pt
        if (event.key === 'z'){
                deleteButton.click();
        }
        //hotkey c for clear current polygon
        if (event.key === 'c'){
                deleteAllbutton.click();
        }
  }
  //adding hotkey event listener. Limit scope if not in colab.
  if (is_colab===true){
    document.addEventListener('keydown', hotkey_press);
   } else{
    div.addEventListener('keydown', hotkey_press);
   }
  function resetcanvas(){
      // clear canvas
      ctx.clearRect(0, 0, canvas_img.width, canvas_img.height);
      img = imgs[curr_image]
      spot_name.innerHTML = spot_nms[curr_image] + " (" + (curr_image + 1) + "/" + imgs.length + ")";
      if (all_tags[curr_image] === 'True') {
        spot_name.innerHTML = spot_nms[curr_image] + " (tagged) (" + (curr_image + 1) + "/" + imgs.length + ")";
      }
      image.src = "data:image/png;base64," + img;
      // on load init new canvas and display image
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
  if (is_colab===true){
    google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
  }
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
  buttondiv.appendChild(prevSamplebutton)
  buttondiv.appendChild(nextSamplebutton)
  buttondiv.appendChild(document.createElement('br'))
  buttondiv.appendChild(document.createElement('br'))
  buttondiv.appendChild(savepolys)
  buttondiv.appendChild(document.createElement('br'))
  buttondiv.appendChild(document.createElement('br'))
  buttondiv.appendChild(brdiv)
  buttondiv.appendChild(MeasureSample)
  buttondiv.appendChild(document.createElement('br'))
  buttondiv.appendChild(document.createElement('br'))
  buttondiv.appendChild(analyzeAllbutton)
  div.appendChild(buttondiv)
  if (is_colab===true){
    document.querySelector("#output-area").appendChild(div);
  } else {
    element.append(div);
  }
  
  return
};
''')