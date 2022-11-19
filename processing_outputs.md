# colab_zirc_dims processing outputs:

Processing a reflected light image dataset using a colab_zirc_dims processing notebook will result in the following data being saved to a timestamped subdirectory within the 'outputs' subdirectory (this will be created automatically if not extant) of your project directory.

## Contents:
  * **[Measurement data & metadata](https://github.com/MCSitar/colab_zirc_dims/tree/main/processing_outputs.md#measurement-data--metadata)**
    * **[Default fields](https://github.com/MCSitar/colab_zirc_dims/tree/main/processing_outputs.md#default-fields)**
    * **[Additional fields (image-per-shot datasets)](https://github.com/MCSitar/colab_zirc_dims/tree/main/processing_outputs.md#additional-fields-for-image-per-shot-eg-ucsb-datasets)**
    * **[Additional fields (semi-automated measurement)](https://github.com/MCSitar/colab_zirc_dims/tree/main/processing_outputs.md#additional-fields-when-saving-files-from-semi-automated-segmentation-gui)**
   * **[Verification images](https://github.com/MCSitar/colab_zirc_dims/tree/main/processing_outputs.md#verification-images)**
   * **[Polygon .json files (optional)](https://github.com/MCSitar/colab_zirc_dims/tree/main/processing_outputs.md#polygon-json-files-optional)**
   * **[References](https://github.com/MCSitar/colab_zirc_dims/tree/main/processing_outputs.md#references)**

## Measurement data & metadata:
This data will be saved into the file:
<br>
```.../YOUR_PROJECT_FOLDER/outputs/...processing_run_*TIMESTAMP*/grain_dimensions/SAMPLE_NAME.csv```
<br>

### Default fields:
<table>
<thead>
  <tr>
    <th>Output field</th>
    <th>Explanation</th>
    <th>Reference</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><i>Analysis</i></td>
    <td>Name of the analysis. In the case of mosaic-style datasets, this is derived from a .scancsv file (e.g., 'Spot 29'). In the case of image-per-shot datasets, this can either be derived from the image filename (e.g., "Spot_100") or just the full filename (e.g., "12345_Spot_100.png"), depending on whether the user has specified a function for splitting file names to sample and/or spot names.</td>
    <td> </td>
  </tr>
  <tr>
    <td><i>Area (µm^2)</i></td>
    <td>Number of pixels in the central grain mask as calculated using <a href="https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops" target="_blank" rel="noopener noreferrer">scikit-image</a>, multiplied by (<i>Scale factor</i>)<sup>2</sup>.</td>
    <td>Van der Walt et al. (2014)</td>
  </tr>
  <tr>
    <td><i>Convex area (µm^2)</i></td>
    <td>Number of pixels in the convex hull of the central grain mask (the smallest polygon that encloses the mask; calculated using  <a href="https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops" target="_blank" rel="noopener noreferrer">scikit-image</a>), multiplied by (<i>Scale factor</i>)<sup>2</sup>.</td>
    <td>Van der Walt et al. (2014)</td>
  </tr>
  <tr>
    <td><i>Eccentricity</i></td>
    <td>"Eccentricity of the ellipse that has the same second-moments as the region", calculated using  <a href="https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops" target="_blank" rel="noopener noreferrer">scikit-image</a>.</td>
    <td>Van der Walt et al. (2014)</td>
  </tr>
  <tr>
    <td><i>Equivalent diameter (µm)</i></td>
    <td>The square root of the pixel-area of the central grain mask divided by π, multiplied by 2x the <i>Scale factor</i>. Area is calculated using  <a href="https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops" target="_blank" rel="noopener noreferrer">scikit-image</a>.</td>
    <td>Van der Walt et al. (2014)</td>
  </tr>
  <tr>
    <td><i>Perimeter (µm)</i></td>
    <td>"Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity" as calculated using  <a href="https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops" target="_blank" rel="noopener noreferrer">scikit-image</a>, multiplied by <i>Scale factor</i>.</td>
    <td>Van der Walt et al. (2014)</td>
  </tr>
  <tr>
    <td><i>Major axis length (µm)</i></td>
    <td>Length of "major axis of the ellipse... [with] the same normalized central moments" as the central grain mask (calculated via  <a href="https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops" target="_blank" rel="noopener noreferrer">scikit-image</a>), multiplied by <i>Scale factor</i>.</td>
    <td>Van der Walt et al. (2014)</td>
  </tr>
  <tr>
    <td><i>Minor axis length (µm)</i></td>
    <td>Length of "minor axis of the ellipse... [with] the same normalized central moments" as the central grain mask (calculated via  <a href="https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops" target="_blank" rel="noopener noreferrer">scikit-image</a>), multiplied by <i>Scale factor</i>.</td>
    <td>Van der Walt et al. (2014)</td>
  </tr>
  <tr>
    <td><i>Circularity</i></td>
    <td>(4 x π x central mask area), divided by the central mask perimeter as calculated using  <a href="https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops" target="_blank" rel="noopener noreferrer">scikit-image</a>.</td>
    <td> </td>
  </tr>
  <tr>
    <td><i>Long axis rectangular diameter (µm)</i></td>
    <td>Length of the longer side of the minimum-area rectangle circumscribing the central grain mask (calculated via  <a href="https://docs.opencv.org/4.6.0/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9" target="_blank" rel="noopener noreferrer">OpenCV</a>), multiplied by <i>Scale factor</i>.</td>
    <td>Bradski (2000)</td>
  </tr>
  <tr>
    <td><i>Short axis rectangular diameter (µm)</i></td>
    <td>Length of the shorter side of the minimum-area rectangle circumscribing the central grain mask (calculated via  <a href="https://docs.opencv.org/4.6.0/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9" target="_blank" rel="noopener noreferrer">OpenCV</a>), multiplied by <i>Scale factor</i>.</td>
    <td>Bradski (2000)</td>
  </tr>
  <tr>
    <td><i>Best long axis length (µm)</i></td>
    <td>Equal to <i>Major axis length</i> if moment-based aspect ratio (<i>major axis length</i>/<i>minor axis length</i>) &lt;= 1.8. Otherwise, equal to <i>Long axis rectangular diameter</i>.</td>
    <td> </td>
  </tr>
  <tr>
    <td><i>Best short axis length (µm)</i></td>
    <td>Equal to <i>Minor axis length</i> if moment-based aspect ratio (<i>major axis length</i>/<i>minor axis length</i>) &lt;= 1.8. Otherwise, equal to <i>Short axis rectangular diameter</i>.</td>
    <td> </td>
  </tr>
  <tr>
    <td><i>Best axes calculated from</i></td>
    <td>"2nd central moments" if moment-based aspect ratio (<i>major axis length</i>/<i>minor axis length</i>) &lt;= 1.8. Otherwise, "minimum circumscribing rectangle".</td>
    <td> </td>
  </tr>
  <tr>
    <td><i>Scale factor (µm/pixel)</i></td>
    <td>Image scale factor, derived from Chromium metadata or user inputs.</td>
    <td> </td>
  </tr>
</tbody>
</table>

### Additional fields for image-per-shot (e.g., UCSB) datasets:
<table>
<thead>
  <tr>
    <th>Output field</th>
    <th>Explanation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><i>Scale factor from:</i></td>
    <td>".Align" if scale factor was loaded from a Chromium .Align metadata file. "sample_info.csv" if loaded from a user-created sample_info.csv file. If neither of those, "default (1.0)".</td>
  </tr>
  <tr>
    <td><i>Image filename</i></td>
    <td>Full filename (e.g., 'xxxx.png') for image used to produce the measured central grain mask.</td>
  </tr>
</tbody>
</table>


### Additional fields when saving files from semi-automated segmentation GUI:
<table>
<thead>
  <tr>
    <th>Output field</th>
    <th>Explanation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><i>Human_or_auto</i></td>
    <td>"auto" if the measured central grain mask was automatically generated. "human" if the mask has been manually edited.</td>
  </tr>
  <tr>
    <td><i>tagged?</i></td>
    <td>"TRUE" if user has manually tagged the scan using the semi-automated segmentation GUI. Otherwise, blank.</td>
  </tr>
</tbody>
</table>

## Verification images:

Images will be saved as:
<br>
```.../YOUR_PROJECT_FOLDER/outputs/...processing_run_*TIMESTAMP*/mask_images/SAMPLE_NAME/SPOT_NAME.png```
<br>
Axes will be scaled in µm except in cases where no Chromium scaling metadata exists and/or the user did not provide a scale factor manually. In these cases, the images will be scaled in pixels.

### Examples (with legend):
Image where 'best' axes are moment-based (see table above):
![spot_mmt_ax](https://user-images.githubusercontent.com/74220513/202790622-1be5092b-edb2-4d1a-bc40-0027151d1452.png)

Image where 'best' axes are minimum-area-bounding-rectangle-based (see table above):
![spot_rct_ax](https://user-images.githubusercontent.com/74220513/202790735-bf526b6c-51f7-4b11-9385-c985c3f1200c.png)

Image where the algorithm failed to find a 'central' grain mask to measure:
![5PS_58_Spot 235_failed](https://user-images.githubusercontent.com/74220513/202791489-c960f56b-3390-47c5-ac3b-be5a9f9adbef.png)

## Polygon .json files (optional):
If indicated by the user, per-sample .json files will be saved to:
<br>
```.../YOUR_PROJECT_FOLDER/outputs/...processing_run_*TIMESTAMP*/saved_polygons/SAMPLE_NAME.json```
<br>
These .json files allow viewing and editing of (polygonized) segmentation masks for the dataset in the your project directory.

## References:
Bradski, G.: The OpenCV Library, Dr Dobbs J. Softw. Tools, 2000.

van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., Yu, T., and contributors,  the scikit-image: scikit-image: Image processing in Python, PeerJ, 2, e453, https://doi.org/10.7717/peerj.453, 2014.
