# colab_zirc_dims
This repository contains code for dimensional analysis of heavy mineral grains in reflected light LA-ICP-MS alignment images using Google-Colab-compatable Jupyter notebooks, with or without the aid of deep-learning-based instance segmentation models. Because processing can be done in Google Colab, this method should be available to anyone with an internet connection and access to Google services. Users who lack one or both of these can instead run the notebooks on a local machine after [a lengthier installation process](https://github.com/MCSitar/colab_zirc_dims/blob/main/advanced_local_installation.md).  [Detectron2](https://github.com/facebookresearch/detectron2) was used for model training and is used for Colab/Jupyter implementation.

## Features
The code in this repo enables organization (e.g., matching zircon/heavy mineral mosaic images to .scancsv scanlists), data extraction (e.g., mapping shots to subimages of mosaics), and post-segmentation processing (e.g., extracting accurate grain dimensions from segmentation masks) of mosaic-scanlist and single-shot-per-image reflected light zircon/heavy mineral image datasets from LA-ICP-MS facilities using Chromium targeting software. RCNN instance segmentation of grains is handled by Detectron2 and implemented in ready-to-run Google Colab/Jupyter notebooks (see 'Links'). Said code and notebooks have been tested with and fully support processing of image datasets from the [University of Arizona LaserChron Center](https://sites.google.com/laserchron.org/arizonalaserchroncenter/home) and the [UCSB LA-ICP-MS facility](https://www.petrochronology.com/); datasets from other facilities using Chromium *should* work with colab_zirc_dims but have not been tested. Users with reflected image datasets lacking Chromium image metadata *can* segment their images (see 'single image-per-shot' notebook below) with some additional project folder organization and/or after manually adding image scaling information, but they (and researchers with non-reflected-light images) should also consider using [AnalyZr](https://github.com/TarynScharf/AnalyZr).

In datasets with good image quality and well-exposed grains (i.e., with full cross-sections visible above mounts(s)), automated colab_zirg processing achieves measurements comparable to those produced by humans (with average absolute error along long and short axes <10 μm vs. humans in 17/19 tested samples) in a fraction of the time. See below for an example analysis of a single spot.

[<img align="center" src="https://user-images.githubusercontent.com/74220513/202251949-f8bd3905-26b8-4aba-9726-0d6f5ef65d77.png" width="50%"/>](1qz19_sp95_withleg.png)
<table>
<thead>
  <tr>
    <th>Analysis</th>
    <th>Area (µm^2)</th>
    <th>Convex area (µm^2)</th>
    <th>Eccentricity</th>
    <th>Equivalent diameter (µm)</th>
    <th>Perimeter (µm)</th>
    <th>Major axis length (µm)</th>
    <th>Minor axis length (µm)</th>
    <th>Circularity</th>
    <th>Long axis rectangular diameter (µm)</th>
    <th>Short axis rectangular diameter (µm)</th>
    <th>Best long axis length (µm)</th>
    <th>Best short axis length (µm)</th>
    <th>Best axes calculated from</th>
    <th>Scale factor (µm/pixel)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Spot 95</td>
    <td>1037.1</td>
    <td>1078.3</td>
    <td>0.898</td>
    <td>36.3</td>
    <td>133.6</td>
    <td>54.9</td>
    <td>24.2</td>
    <td>0.730</td>
    <td>54.4</td>
    <td>24.8</td>
    <td>54.9</td>
    <td>24.2</td>
    <td>2nd central moments</td>
    <td>1.027</td>
  </tr>
</tbody>
</table>

See our [Processing Outputs page](https://github.com/MCSitar/colab_zirc_dims/blob/main/processing_outputs.md) for more details.


In sub-optimal datasets, automated processing is less successful and can produce inaccurate segmentations. To mitigate this, a semi-automated Colab/Jupyter-based GUI (extended from [TensorFlow object detection utils code](https://github.com/tensorflow/models/blob/master/research/object_detection/utils/colab_utils.py)) that allows users to view and edit automatically-generated segmentations before export has also been implemented. Semi-automated processing is recommended for production of publication-quality measurement datasets.

![auto_seg_gif_reduced](https://user-images.githubusercontent.com/74220513/139791884-b88c9854-c825-4a95-a678-598abb204eea.gif)

Various functions for processing mosaic image datasets are available in (among others) modules **czd_utils**, **mos_match**, and **mos_proc**.

## Models:
A variety of Detectron2-based instance segmentation models are available for application to images via the provided notebooks. You can check them out in our [model library](https://github.com/MCSitar/colab_zirc_dims/blob/main/model_library.md).

## Datasets:
Please refer to the [training datasets section](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets) of this repository to learn more about and/or download the datasets used to train models for application in colab_zirc_dims.

## Links
#### Per-sample mosaic image datasets (e.g., from ALC):
Colab/Jupyter notebooks are available for:
- [Matching ALC mosiacs to .scancsv files (dataset preparation)](https://colab.research.google.com/drive/1yakvH0j0ZOeBhNa5-bygT9saqvugvB-B?usp=sharing)
- [Automatically and/or semi-automatically segmenting and measuring grains from mosaics - v1.0.12](https://colab.research.google.com/drive/1XXynpwnioev3RnDR47nnd5brctsDB_Ow?usp=sharing)

[A template project folder is available here.](https://drive.google.com/drive/folders/1cFOoxp2ELt_W6bqY24EMpxQFmI00baDl?usp=sharing)

#### Single image-per-shot datasets (e.g., from UCSB):
- [A Colab/Jupyter notebook for automatically and/or semi-automatically segmenting and measuring grains is available here - v1.0.12](https://colab.research.google.com/drive/1x84iPYvVAbdYQ1tnPQgAzp4SYAB1ckus?usp=sharing)

Template project folders are available for:
- [Datasets where sample, shot information can be extracted from image filenames (e.g., from UCSB)](https://drive.google.com/drive/folders/1MkWh9PRArbV1m1eVbSTbb9C5PKC95Re3?usp=sharing)
- [Datasets where shot images have been manually organized by user(s)](https://drive.google.com/drive/folders/1VpYo5HwDUaAQ4lJ0waZJRDrWkzHv2QyM?usp=sharing)

#### Other links:
- [Video tutorial and demo - intro to Colab notebooks at 9:05](https://youtu.be/ZdO6B-dvHm0)

## Running notebooks without Google Colab (on local machines):
The notebooks provided can be run as basic Jupyter notebooks in a local Anaconda environment. Some setup is required, though. Please refer to the [advanced local installation instructions](https://github.com/MCSitar/colab_zirc_dims/blob/main/advanced_local_installation.md).

## Installation outside of provided notebooks:
[A distribution of this package is available through the Python Package Index](https://pypi.org/project/colab-zirc-dims/). It is recommended that this package only be used with the provided notebooks, but some functions could be useful to users working with mosaic or .Align files.

To install inside of Google Colab, add:
```
!pip install colab_zirc_dims
```
to a cell, then run the cell.

To install outside of Google Colab (without segmentation functionalities), open command line and enter:

```
pip install colab_zirc_dims
```

then press enter.
<br>
<br>
To install outside of Google Colab with full segmentation functionalities, please refer to the [advanced local installation instructions](https://github.com/MCSitar/colab_zirc_dims/blob/main/advanced_local_installation.md).


## Project Status (updated 12/11/2022)
- All features are functional. Bugs may exist, and are most likely to be encountered when using the package outside of the provided notebooks.
- All notebooks will now run as Jupyter notebooks in local Anaconda environments.
- New 'best axes' measurement that mitigates inaccuracies in moment-based axial measurements by defering to minimum-area rectangle measurements for most high-aspect ratio grains.
- Added new dataset, new models, and documentation pages.
- Automated processing notebooks updated on 05/19/2022 with V1.0.8.1 hotfix for incompatibility between latest Detectron2 binary release and new standard Colab PyTorch installation (v1.11).
### Possible future work:
- Any changes and/or additions necessary to ensure compatibility with datasets from LA-ICP-MS facilities other than ALC and UCSB. Contact me if you have a dataset from a non-ALC, non-UCSB facility that does not seem to work with colab-zirc-dims!
- Model retraining; will be done as needed given new architectures or hyperparameter tuning schemes that could significantly improve segmentation performance.
- Refactoring. The colab_zirc_dims package is fairly sprawling and probably doesn't need to be.
- Miscellaneous processing speed improvements. As of v1.0.10, there is a lot of room for improvement here, especially w/r/t maximizing GPU utilization.


## Additional Notes
- Training and large-n zircon measurement datasets for this project were provided by Dr. Ryan Leary (New Mexico Tech). Also, motivation; see his [recent work](https://doi.org/10.1029/2019JB019226) on the utility of augmenting LA-ICP-MS data with grain size data.
- Some additional training data are from the [UCSB Petrochronology Center](https://www.petrochronology.com/).
- Although models were trained on (and tests have only been performed on) detrital zircon mosaic images, I think that this method could probably be applied to LA-ICP-MS mosaics/samples of other minerals (e.g., monazite) without further model training.
- Any suggestions, comments, or questions about this project are also welcome.

## Citation
This project has been written up as a technical note in collaboration with Dr. Ryan Leary and published (open access) in Geochronology. You can find the technical note (along with associated links to additional datasets and replication code) [here](https://doi.org/10.5194/gchron-5-109-2023). If you use colab_zirc_dims or associated datasets/models in your work, please use the following citation:

```
Sitar, M. C. and Leary, R. J.: Technical note: colab_zirc_dims: a Google Colab-compatible toolset for automated and semi-automated measurement of mineral grains in laser ablation–inductively coupled plasma–mass spectrometry images using deep learning models, Geochronology, 5, 109–126, https://doi.org/10.5194/gchron-5-109-2023, 2023.
```

## Author
Michael Cole Sitar

email: clsitar5@gmail.com

## References

Yuxin Wu, Alexander Kirillov, Francisco Massa and Wan-Yen Lo, & Ross Girshick. (2019). Detectron2. https://github.com/facebookresearch/detectron2.

TensorFlow Developers: TensorFlow, Zenodo, https://doi.org/10.5281/zenodo.5949169, 2022.
