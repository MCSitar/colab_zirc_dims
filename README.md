# colab_zirc_dims
This repository contains code for dimensional anlaysis of zircons in LA-ICP-MS alignment images using Google Colab, with or without the aid of RCNN (deep learning) segmentation. Because processing is done in Google Colab, this method should be available to anyone with an internet connection. [Detectron2](https://github.com/facebookresearch/detectron2) was used for model training and is used for Colab implementation.

## Features
The code in this repo enables organization (e.g., matching zircon mosaic images to .scancsv scanlists), data extraction (e.g., mapping shots to subimages of mosaics), and post-segmentation processing (e.g., extracting accurate zircon dimensions from segmentation masks) of mosaic-scanlist datasets. Said code currently only supports processing of images and scanlists from the [UA LaserChron Center](https://sites.google.com/laserchron.org/arizonalaserchroncenter/home), but may be updated in the ~near future to support processing of datasets from other LA-ICP-MS geochronology facilities. RCNN instance segmentation of zircons is handled by Detectron2 and implemented in ready-to-run Google Colab Notebooks (see 'Links').

In datasets with good image quality and well-exposed zircons (i.e., with full cross-sections visible above puck(s)), automated processing achieves measurements comparable to those produced by humans (with net error along long and short axes < .5 μm vs. humans in several tested datasets) in a fraction of the time. See below for an example analysis of a single spot.

![Spot 315_cropped](https://user-images.githubusercontent.com/74220513/139790689-a68c5cf8-7c6b-4158-b555-76b6718673b8.png)

| Analysis | Area (µm^2) | Eccentricity | Perimeter (µm) | Major axis length (µm) | Minor axis length (µm) |
|----------|-------------|--------------|----------------|------------------------|------------------------|
| Spot 315 | 2227.648    | 0.899        | 193.0          | 80.7                   | 35.4                   |

In sub-optimal datasets, automated processing is less successful and can produce inaccurate segmentations. To mitigate this, a semi-automated Colab-based GUI that allows users to view and edit automatically-generated segmentations before export has also been implemented.

![auto_seg_gif_reduced](https://user-images.githubusercontent.com/74220513/139791884-b88c9854-c825-4a95-a678-598abb204eea.gif)

Various functions for processing mosaic image datasets are available in modules **mos_match** and **mos_proc**.

## Links
Colab Notebooks are available for:
- [Matching mosiacs to .scancsv files (dataset preparation)](https://colab.research.google.com/drive/14LiFLS6XeYjlxu4a_-VSsCaN9BS27Kjz?usp=sharing)
- [Automatically and/or semi-automatically segmenting and measuring zircons from images](https://colab.research.google.com/drive/1uJ1G4U7n0qIxGQLVggasRrBQzwAOxke_?usp=sharing)

[A template project folder with a trained model is available here.](https://drive.google.com/drive/folders/1cFOoxp2ELt_W6bqY24EMpxQFmI00baDl?usp=sharing)

## Project Status (updated 11/03/2021)
- All features are functional. Bugs surely exist, and are most likely to be encountered when using the package outside of the provided Notebooks.
- Model training/improvement is ongoing, and new, better-performing models will be made available in the near future.
- modules **czd_utils**, **mos_match**, and **mos_proc** have full docstring documentation.
- module **zirc_dims_GUI** is sort of a horror show of code, but a functional one. I won't touch this until other modules are in great condition and/or I have a lot of free time.

## Additional Notes
- Training and large-n zircon measurment datasets for this project were provided by Dr. Ryan Leary (New Mexico Tech). Also, motivation; see his [recent work](https://doi.org/10.1029/2019JB019226) on the utility of augmenting LA-ICP-MS data with grain size data.
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
