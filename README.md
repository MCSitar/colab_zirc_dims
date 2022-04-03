# colab_zirc_dims
This repository contains code for dimensional analysis of zircons in LA-ICP-MS alignment images using Google Colab, with or without the aid of RCNN (deep learning) segmentation. Because processing is done in Google Colab, this method should be available to anyone with an internet connection. [Detectron2](https://github.com/facebookresearch/detectron2) was used for model training and is used for Colab implementation.

## Features
The code in this repo enables organization (e.g., matching zircon mosaic images to .scancsv scanlists), data extraction (e.g., mapping shots to subimages of mosaics), and post-segmentation processing (e.g., extracting accurate zircon dimensions from segmentation masks) of mosaic-scanlist and single-shot-per-image reflected light zircon image datasets. RCNN instance segmentation of zircons is handled by Detectron2 and implemented in ready-to-run Google Colab Notebooks (see 'Links'). Said code and Notebooks have been tested with and fully support processing of image datasets from the [University of Arizona LaserChron Center](https://sites.google.com/laserchron.org/arizonalaserchroncenter/home) and the [UCSB LA-ICP-MS facility](https://www.petrochronology.com/), and users with image datasets from other sources can segment their images (see 'single image-per-shot' Notebook below) with some additional project folder organization and/or after manually adding image scaling information.

In datasets with good image quality and well-exposed zircons (i.e., with full cross-sections visible above puck(s)), automated processing achieves measurements comparable to those produced by humans (with net error along long and short axes < .5 μm vs. humans in some tested samples) in a fraction of the time. See below for an example analysis of a single spot.

![Spot 315_cropped](https://user-images.githubusercontent.com/74220513/139790689-a68c5cf8-7c6b-4158-b555-76b6718673b8.png)

| Analysis | Area (µm^2) | Eccentricity | Perimeter (µm) | Major axis length (µm) | Minor axis length (µm) |
|----------|-------------|--------------|----------------|------------------------|------------------------|
| Spot 315 | 2227.648    | 0.899        | 193.0          | 80.7                   | 35.4                   |

In sub-optimal datasets, automated processing is less successful and can produce inaccurate segmentations. To mitigate this, a semi-automated Colab-based GUI that allows users to view and edit automatically-generated segmentations before export has also been implemented.

![auto_seg_gif_reduced](https://user-images.githubusercontent.com/74220513/139791884-b88c9854-c825-4a95-a678-598abb204eea.gif)

Various functions for processing mosaic image datasets are available in (among others) modules **czd_utils**, **mos_match**, and **mos_proc**.

## Links
#### ALC datasets (per-sample mosaic images):
Colab Notebooks are available for:
- [Matching ALC mosiacs to .scancsv files (dataset preparation)](https://colab.research.google.com/drive/1aPMjSF2uGOP4Xy2dssjhk--TFQ7_hWuu?usp=sharing)
- [Automatically and/or semi-automatically segmenting and measuring zircons from ALC mosaics](https://colab.research.google.com/drive/1Y-5Z0Gf9SWZXDAh-3yH3GyyQkeoT8rv_?usp=sharing)

[A template project folder is available here.](https://drive.google.com/drive/folders/1cFOoxp2ELt_W6bqY24EMpxQFmI00baDl?usp=sharing)

#### Single image-per-shot datasets (e.g., from UCSB):
- [A Colab Notebook for automatically and/or semi-automatically segmenting, measuring zircon is available here](https://colab.research.google.com/drive/1REt2hA2TD1yp8f972XHcbK6ILI2Z7cby?usp=sharing)

Template project folders are available for:
- [Datasets where sample, shot information can be extracted from image filenames (e.g., from UCSB)](https://drive.google.com/drive/folders/1MkWh9PRArbV1m1eVbSTbb9C5PKC95Re3?usp=sharing)
- [Datasets where shot images have been manually organized by user(s)](https://drive.google.com/drive/folders/1VpYo5HwDUaAQ4lJ0waZJRDrWkzHv2QyM?usp=sharing)

## Installation outside of provided Notebooks:
[A distribution of this package is available through the Python Package Index](https://pypi.org/project/colab-zirc-dims/). It is recommended that this package only be used within Google Colab, but some functions could be useful to users working with mosaic or .Align files on local machines.
To install outside of Google Colab, open command line and enter:
'''
pip install colab_zirc_dims
'''
then press enter.

To install inside of Google Colab, enter:
'''
!pip install colab_zirc_dims
'''
then run the cell.


## Project Status (updated 04/02/2022)
- All features are functional. Bugs surely exist, and are most likely to be encountered when using the package outside of the provided Notebooks.
- New models are now available.  Models are also now downloaded directly (from AWS) in the automated processing notebook and do not need to be included in project folder(s).
- Saving and loading of automatically- and user-produced zircon segmentation polygons into the Colab GUI has been implemented. This is (I think) big for user convenience - you can automatically process a dataset, disconnect, and then view/edit segmentations in later session(s).
- Generalized segmentation functions for non-ALC datasets now implemented, with full support for datasets from the UCSB LA-ICP-MS facility.
- An example ALC dataset with two mosaics has been added to the repo.
- Package now available through PyPI (though probably of limited utility outside of Colab).

## Additional Notes
- Training and large-n zircon measurement datasets for this project were provided by Dr. Ryan Leary (New Mexico Tech). Also, motivation; see his [recent work](https://doi.org/10.1029/2019JB019226) on the utility of augmenting LA-ICP-MS data with grain size data.
- Some additional training data are from the [UCSB Petrochronology Center](https://www.petrochronology.com/).
- Although models were trained on (and tests have only been performed on) detrital zircon mosaic images, I think that this method could probably be applied to LA-ICP-MS mosaics/samples of other minerals (e.g., monazite) without further model training.
- I do plan to write this project up into some sort of publication (journal article, conference paper, conference poster, vanity license plate, etc). At that point, I will post citation info here. If you submit a publication utilizing this code in the meantime, please reach out to me.
- Any suggestions, comments, or questions about this project are also welcome.

## Author
Michael Cole Sitar

M.S. student, Colorado State University Department of Geosciences

email: mcsitar@colostate.edu

## References

Yuxin Wu, Alexander Kirillov, Francisco Massa and Wan-Yen Lo, & Ross Girshick. (2019). Detectron2. https://github.com/facebookresearch/detectron2.
