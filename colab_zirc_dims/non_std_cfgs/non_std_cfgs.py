# -*- coding: utf-8 -*-
"""
Functions to install dependencies for non-standard models (e.g., Centermask2)
and get compatible Detectron2 configs for them.
"""

import sys
import subprocess
try:
    from detectron2.config import get_cfg
except ModuleNotFoundError:
    print('WARNING: Detectron2 not installed on (virtual?) machine;',
          'colab_zirc_dims model loading functions unavailable')

__all__ = ['get_czd_swint_cfg',
           'get_czd_centermask2_cfg']

def get_czd_swint_cfg():
    """Install dependencies for swint_detectron2 and/or get a Swin-T Mask RCNN
       Detectron2 cfg.

    Returns
    -------
    out_cfg : Detectron2 Config instance
        A D2 config for a MaskRCNN model with a Swin-T (see swint_detectron2)
        backbone. Lacks usable weights path; this must be added in main
        Notebook.

    """
    from detectron2.config import get_cfg
    try:
        import timm.utils as test_timm
    except ModuleNotFoundError:
        print('Installing module: timm')
        try:
            subpout = subprocess.run(["pip", "install", "timm"],
                                     capture_output=True, check=True)
            print(str(subpout.stdout.decode('UTF-8')))
        except subprocess.CalledProcessError as check:
            print(check)
    try:
        import swint
    except ModuleNotFoundError:
        print('Cloning module: Swint_detectron2')
        try:
            subpout = subprocess.run(["git", "clone",
                                      "https://github.com/xiaohu2015/SwinT_detectron2",
                                      "swinT_repo"], capture_output=True,
                                     check=True)
            print(str(subpout.stdout.decode('UTF-8')))
        except subprocess.CalledProcessError as check:
            print(check)
        sys.path.insert(0, '/content/swinT_repo')
        import swint
    out_cfg = get_cfg()
    swint.add_swint_config(out_cfg)
    out_cfg.merge_from_file('/content/swinT_repo/configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml')
    return out_cfg


def get_czd_centermask2_cfg():
    """Clone dependency for Centermask2 and/or get a Centermask2 VoVNet2-backbone
       Detectron2 cfg.

    Returns
    -------
    out_cfg : Detectron2 Config instance
        A D2 config for a Centermask2 model with VoVNet2 backbone. Lacks usable
        weights path; this must be added in main Notebook.

    """
    try:
        import centermask
    except ModuleNotFoundError:
        print('Cloning module: Centermask2')
        try:
            subpout = subprocess.run(["git", "clone",
                                      "https://github.com/youngwanLEE/centermask2.git",
                                      "centermask"], capture_output=True,
                                     check = True)
            print(str(subpout.stdout.decode('UTF-8')))
        except subprocess.CalledProcessError as check:
            print(check)
        sys.path.insert(0, '/content/centermask')
        import centermask
    from centermask.config import get_cfg
    out_cfg = get_cfg()
    out_cfg.merge_from_file('/content/centermask/configs/centermask/centermask_V_99_eSE_FPN_ms_3x.yaml')
    return out_cfg
