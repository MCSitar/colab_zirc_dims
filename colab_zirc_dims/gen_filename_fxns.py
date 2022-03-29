# -*- coding: utf-8 -*-
"""
This module contains functions for splitting names of single-shot image files
for non-ALC datasets by file type and scan name. Dataset loading using these
functions is implemented in the non-ALC colab_zirc_dims processing Notebook.
This is a separate module to streamline the processing of potentially extending
colab_zirc_dims dataset import capabilities to new LA-ICP-MS facilities.

Users with LA-ICP-MS RL image datasets with currently-unsupported formatting
can write their own filename-splitting function into the non-ALC
processing notebook (consider a pull request to this module if you have done
this!). If more extensive changes/additions to the package seem to be needed,
Python-naive users are encouraged to send a sample dataset (i.e., images and
any associated metadata/scaling files) to the colab_zirc_dims project manager
(contact details on GitHub) for potential package support in the future.

Functions in this module should take a image file name string as an input
and return strings SAMPLE_NAME, SCAN_NAME. The output sample name can be a
generic value if sample names are not found in file names. Output spot names
should but do not have to be unique per-sample. New functions should be added
to the default_name_fxns dict before 'None'.
"""

def split_name_ucsb(name):
    """Function for splitting out sample, spot names from filenames for images
       saved at the UCSB LA-ICP-MS facility.

    Parameters
    ----------
    name : str
        Filename (relative, without full path) for a image or .Align file.

    Returns
    -------
    sample_name : str
        Name of sample.
    spot_name : str
        Name of spot (probably an integer; returned as a string in case not).

    """
    sans_prefix = name[::-1][:name[::-1].index('ScanImage_'[::-1])][::-1]
    sans_suffix = sans_prefix[:sans_prefix.index('_Ablation')]
    spot_name = sans_suffix[::-1][:sans_suffix[::-1].index('-')][::-1]
    sample_name = sans_suffix[::-1][sans_suffix[::-1].index('-')+1:][::-1]
    return sample_name, spot_name

default_name_fxns = {'UCSB': split_name_ucsb,
                     'Use full image file name': None}
