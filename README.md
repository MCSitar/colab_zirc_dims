# colab_zirc_dims
This repository contains code for dimensional analysis of heavy mineral grains in reflected light LA-ICP-MS alignment images using Google Colab, with or without the aid of RCNN (deep learning) segmentation. Models were trained and tested on images of detrital zircons. Because processing is done in Google Colab, this method should be available to anyone with an internet connection. [Detectron2](https://github.com/facebookresearch/detectron2) was used for model training and is used for Colab implementation.

## Features
The code in this repo enables organization (e.g., matching zircon/heavy mineral mosaic images to .scancsv scanlists), data extraction (e.g., mapping shots to subimages of mosaics), and post-segmentation processing (e.g., extracting accurate grain dimensions from segmentation masks) of mosaic-scanlist and single-shot-per-image reflected light zircon/heavy mineral image datasets from LA-ICP-MS facilities using Chromium targeting software. RCNN instance segmentation of grains is handled by Detectron2 and implemented in ready-to-run Google Colab notebooks (see 'Links'). Said code and notebooks have been tested with and fully support processing of image datasets from the [University of Arizona LaserChron Center](https://sites.google.com/laserchron.org/arizonalaserchroncenter/home) and the [UCSB LA-ICP-MS facility](https://www.petrochronology.com/); datasets from other facilities using Chromium *should* work with colab_zirc_dims but have not been tested. Users with reflected image datasets lacking Chromium image metadata *can* segment their images (see 'single image-per-shot' notebook below) with some additional project folder organization and/or after manually adding image scaling information, but they (and researchers with non-reflected-light images) should also consider using [AnalyZr](https://github.com/TarynScharf/AnalyZr).

In datasets with good image quality and well-exposed zircons (i.e., with full cross-sections visible above mounts(s)), automated colab_zirg processing achieves measurements comparable to those produced by humans (with average absolute error along long and short axes <10 μm vs. humans in 18/19 tested samples) in a fraction of the time. See below for an example analysis of a single spot.

![Spot 315_cropped](https://user-images.githubusercontent.com/74220513/139790689-a68c5cf8-7c6b-4158-b555-76b6718673b8.png)

| Analysis | Area (µm^2) | Eccentricity | Perimeter (µm) | Major axis length (µm) | Minor axis length (µm) |
|----------|-------------|--------------|----------------|------------------------|------------------------|
| Spot 315 | 2227.648    | 0.899        | 193.0          | 80.7                   | 35.4                   |

In sub-optimal datasets, automated processing is less successful and can produce inaccurate segmentations. To mitigate this, a semi-automated Colab-based GUI (extended from [TensorFlow object detection utils code](https://github.com/tensorflow/models/blob/master/research/object_detection/utils/colab_utils.py)) that allows users to view and edit automatically-generated segmentations before export has also been implemented. Semi-automated processing is recommended for production of publication-quality measurement datasets.

![auto_seg_gif_reduced](https://user-images.githubusercontent.com/74220513/139791884-b88c9854-c825-4a95-a678-598abb204eea.gif)

Various functions for processing mosaic image datasets are available in (among others) modules **czd_utils**, **mos_match**, and **mos_proc**.

## Links
#### Per-sample mosaic image datasets (e.g., from ALC):
Colab notebooks are available for:
- [Matching ALC mosiacs to .scancsv files (dataset preparation)](https://colab.research.google.com/drive/1oRgtZGHrl1nlN-8qMEnoAEy0PPesf6AR?usp=sharing)
- [Automatically and/or semi-automatically segmenting and measuring grains from mosaics - v1.0.9](https://colab.research.google.com/drive/1ZPxYZ2atEelmg-Sz30lZBQkIBI6OO4mc?usp=sharing)

[A template project folder is available here.](https://drive.google.com/drive/folders/1cFOoxp2ELt_W6bqY24EMpxQFmI00baDl?usp=sharing)

#### Single image-per-shot datasets (e.g., from UCSB):
- [A Colab notebook for automatically and/or semi-automatically segmenting and measuring grains is available here - v1.0.9](https://colab.research.google.com/drive/1aylXW-2BVEpqL3TGZc-TjB7K-5MxJFPk?usp=sharing)

Template project folders are available for:
- [Datasets where sample, shot information can be extracted from image filenames (e.g., from UCSB)](https://drive.google.com/drive/folders/1MkWh9PRArbV1m1eVbSTbb9C5PKC95Re3?usp=sharing)
- [Datasets where shot images have been manually organized by user(s)](https://drive.google.com/drive/folders/1VpYo5HwDUaAQ4lJ0waZJRDrWkzHv2QyM?usp=sharing)


#### Other links:
- [Video tutorial and demo - intro to Colab notebooks at 4:50](https://youtu.be/WM7qEjaJdgo)

## Installation outside of provided notebooks:
[A distribution of this package is available through the Python Package Index](https://pypi.org/project/colab-zirc-dims/). It is recommended that this package only be used within Google Colab, but some functions could be useful to users working with mosaic or .Align files on local machines.
To install outside of Google Colab, open command line and enter:

```
pip install colab_zirc_dims
```

then press enter.

To install inside of Google Colab, add:
```
!pip install colab_zirc_dims
```
to a cell, then run the cell.


## Project Status (updated 07/16/2022)
- All features are functional. Bugs may exist, and are most likely to be encountered when using the package outside of the provided notebooks.
- A bug in Otsu thresholding has been fixed; it now works slightly better as a backup to CNN-based segmentation.
- Fully interactive Jupyter-widget-based exploratory measurement plotting UI (expl_vis module) implemented.
- New measurements: minimum and maximum diameters measured from the minimum area circumscribing rectangle for a grain mask.
- Automated processing notebooks updated on 05/19/2022 with V1.0.8.1 hotfix for incompatibility between latest Detectron2 binary release and new standard Colab PyTorch installation (v1.11).

## Additional Notes
- Training and large-n zircon measurement datasets for this project were provided by Dr. Ryan Leary (New Mexico Tech). Also, motivation; see his [recent work](https://doi.org/10.1029/2019JB019226) on the utility of augmenting LA-ICP-MS data with grain size data.
- Some additional training data are from the [UCSB Petrochronology Center](https://www.petrochronology.com/).
- Although models were trained on (and tests have only been performed on) detrital zircon mosaic images, I think that this method could probably be applied to LA-ICP-MS mosaics/samples of other minerals (e.g., monazite) without further model training.
- Any suggestions, comments, or questions about this project are also welcome.

## Citation
This project has been written up in as a technical note in collaboration with Dr. Ryan Leary and is currently in review at Geochronology. You can find the preprint (along with associated links to additional datasets and replication code) [here](https://gchron.copernicus.org/preprints/gchron-2022-12). If you use colab_zirc_dims in your work prior to (hopeful(!)) full publication, please use the following preprint citation:

```
Sitar, M. C. and Leary, R. J.: Technical Note: colab_zirc_dims: a Google-Colab-based Toolset for Automated and Semi-automated Measurement of Mineral Grains in LA-ICP-MS Images Using Deep Learning Models, Geochronology Discuss. [preprint], https://doi.org/10.5194/gchron-2022-12, in review, 2022.
```

## Author
Michael Cole Sitar

M.S. student, Colorado State University Department of Geosciences

email: mcsitar@colostate.edu

## References

Yuxin Wu, Alexander Kirillov, Francisco Massa and Wan-Yen Lo, & Ross Girshick. (2019). Detectron2. https://github.com/facebookresearch/detectron2.

TensorFlow Developers: TensorFlow, Zenodo, https://doi.org/10.5281/zenodo.5949169, 2022.
