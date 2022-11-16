# -*- coding: utf-8 -*-
"""
Functions to install dependencies for non-standard models (e.g., Centermask2)
and get compatible Detectron2 configs for them.
"""

import sys
import subprocess
import os
from .. import czd_utils

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
except ModuleNotFoundError:
    print('WARNING: Detectron2 not installed on (virtual?) machine;',
          'colab_zirc_dims model loading functions unavailable')


__all__ = ['get_czd_swint_cfg',
           'setup_for_swint',
           'get_czd_centermask2_cfg',
           'setup_for_centermask',
           'get_model_arch_from_yaml',
           'smart_load_predictor']

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
    assert setup_for_swint()
    from detectron2.config import get_cfg
    out_cfg = get_cfg()
    import swint
    swint.add_swint_config(out_cfg)
    base_cfg_f_path = os.path.join(os.getcwd(),
                                   'swinT_repo/configs/SwinT',
                                   'mask_rcnn_swint_T_FPN_3x.yaml')
    out_cfg.merge_from_file(base_cfg_f_path)
    #out_cfg.merge_from_file('/content/swinT_repo/configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml')
    return out_cfg

def setup_for_swint():
    """Run setup (i.e., automated repository cloning, timm package installation,
    and/or PYTHONPATH addition) for using swint-detectron2 in current
    working directory.

    Returns
    -------
    bool
        True if swint should be importable and functional, False if not.

    """
    try:
        import timm.utils as test_timm
    except ModuleNotFoundError:
        if not czd_utils.connected_to_internet():
            print('To use CZD Swin-T models without an internet connection',
                  'please first manually install timm',
                  '(https://github.com/rwightman/pytorch-image-models)', 
                  'and then clone the D2 Swin-T repository',
                  '(https://github.com/xiaohu2015/SwinT_detectron2.git)',
                  'into your current working directory as "swinT_repo".')
            return False
        print('Installing module: timm')
        try:
            subpout = subprocess.run(["pip", "install", "timm"],
                                     capture_output=True, check=True)
            print(str(subpout.stdout.decode('UTF-8')))
        except subprocess.CalledProcessError as check:
            print(check)
    if not os.path.isdir(os.path.join(os.getcwd(), 'swinT_repo')):
        if not czd_utils.connected_to_internet():
            print('To use CZD Swin-T models without an internet connection',
                  'please manually clone the D2 Swin-T repository',
                  '(https://github.com/xiaohu2015/SwinT_detectron2.git)',
                  'into your current working directory as "swinT_repo".')
            return False
        print('Cloning module: Swint_detectron2')
        try:
            subpout = subprocess.run(["git", "clone",
                                      "https://github.com/xiaohu2015/SwinT_detectron2",
                                      "swinT_repo"], capture_output=True,
                                     check=True)
            print(str(subpout.stdout.decode('UTF-8')))
        except subprocess.CalledProcessError as check:
            print(check)
    if os.path.join(os.getcwd(), 'swinT_repo') not in sys.path:
        sys.path.insert(0, os.path.join(os.getcwd(), 'swinT_repo'))
    return True


def get_czd_centermask2_cfg():
    """Clone dependency for Centermask2 and/or get a Centermask2 VoVNet2-backbone
       Detectron2 cfg.

    Returns
    -------
    out_cfg : Detectron2 Config instance
        A D2 config for a Centermask2 model with VoVNet2 backbone. Lacks usable
        weights path; this must be added in main Notebook.

    """
    assert setup_for_centermask()
    from centermask.config import get_cfg as get_cmask_cfg
    out_cfg = get_cmask_cfg()
    base_cfg_f_path = os.path.join(os.getcwd(),
                                   'centermask/configs/centermask',
                                   'centermask_V_99_eSE_FPN_ms_3x.yaml')
    out_cfg.merge_from_file(base_cfg_f_path)
    return out_cfg


def setup_for_centermask():
    """Run setup (i.e., automated repository cloning and/or PYTHONPATH addition)
    for using Centermask2 in current working directory.

    Returns
    -------
    bool
        True if current working dir is properly setup to use Centermask2, else
        False.

    """
    if not os.path.isdir(os.path.join(os.getcwd(), 'centermask')):
        if not czd_utils.connected_to_internet():
            print('To use Centermask2 models without an internet connection',
                  'please manually copy the Centermask2 repository',
                  '(https://github.com/youngwanLEE/centermask2.git)',
                  'into your current working directory.')
            return False
        print('Cloning module: Centermask2')
        try:
            subpout = subprocess.run(["git", "clone",
                                      "https://github.com/youngwanLEE/centermask2.git",
                                      "centermask"], capture_output=True,
                                     check = True)
            print(str(subpout.stdout.decode('UTF-8')))
        except subprocess.CalledProcessError as check:
            print(check)
        #print(sys.path)
    if os.path.join(os.getcwd(), 'centermask') not in sys.path:
        sys.path.insert(0, os.path.join(os.getcwd(), 'centermask'))
    return True

def get_model_arch_from_yaml(inpt_yaml_cfg_path):
    """Find a (supported) model architecture type from a Detectron2 cfg .yaml
    file.

    Parameters
    ----------
    inpt_yaml_cfg_path : str
        Path to a .yaml cfg file for loading and use in Detectron2.

    Returns
    -------
    str
        One of ['MRCNN', 'SwinT', 'Centermask2'], dependent on whether input
        .yaml file path corresponds to a Mask-RCNN, Swin-T, or Centermask2-Vovnet
        model architecture.

    """
    out_model_arch='MRCNN'
    cfg_yam_dict = czd_utils.read_yaml(inpt_yaml_cfg_path)
    if 'swint' in cfg_yam_dict['MODEL']['BACKBONE']['NAME']:
        return 'SwinT'
    if 'fcos_vovnet_fpn' in cfg_yam_dict['MODEL']['BACKBONE']['NAME']:
        return 'Centermask2'
    return out_model_arch

def _assert_adj_thresh_inpt_valid(inpt_thresh_val, inpt_thresh_name):
    """Assert validity of manual threshhold inputs for smart_D2_cfg_load. These
    should always be None, float betweeen 0 and 1, or str('auto').

    Parameters
    ----------
    inpt_thresh_val : Any
        Argument value to check.
    inpt_thresh_name : str
        Name of the argument variable from parent function. Printed if assert
        fails.

    Returns
    -------
    None.

    """
    assert isinstance(inpt_thresh_val,
                      (float, int, str)), ' '.join([inpt_thresh_name,
                                                    'must be None, a float',
                                                    'or str("auto").'])
    if isinstance(inpt_thresh_val, (float, int)):
        assert all([inpt_thresh_val >= 0.0,
                    inpt_thresh_val <=  1.0]), ' '.join([inpt_thresh_name,
                                                        'manual float input',
                                                        'should be from',
                                                        '0 to 1, inclusive.'])
    else:
        assert inpt_thresh_val=='auto', ' '.join([inpt_thresh_name,
                                                 'string input should be',
                                                 '"auto". Are you sure',
                                                 'that you meant to input',
                                                 'a string?'])

def smart_D2_cfg_load(inpt_cfg_yaml_path, use_cpu=False,
                   adj_nms_thresh=None,
                   adj_thresh_test=None):
    """Load a Detectron2 model config file with a CZD-supported architecture
    (i.e., Mask-RCNN with or without Swin-T backbone or Centermask2-VoVnet).
    Architecture is determined automatically, and dependencies are installed
    if necessary.

    Parameters
    ----------
    inpt_cfg_yaml_path : str
        Path to config .yaml file for model.
    use_cpu : bool, optional
        If True, set the cfg MODEL.DEVICE parameter to 'cpu'. Otherwise, set it
        to 'cuda'. The default is False (i.e., use 'cuda').
    adj_nms_thresh : float or str('auto') or None, optional
        If present, manually or automatically adjust the model config NMS
        threshold. If set to 'auto', will use a empirically chosen NMS threshold
        that works well for the model: 0.2 for Mask-RCNN Resnet/Swin-T models,
        0.4 for Centermask models. Alternatively, input a float (should be
        between 0 and 1) to manually adjust the threshold. If None, use the
        threshold from the input config .yaml file. The default is None.
    adj_thresh_test : float or or str('auto') or None, optional
        If present, manually set the model config ROI_HEADS or FCOS inference
        score threshold to this setting. If set to 'auto', will use an
        empirically chosen threshhold value that works well for the model:
        0.7 for Mask-RCNN Resnet/Swin-T models, 0.5 for Centermask models.
        Alternatively, input a float (should be between 0 and 1) to manually
        adjust the threshold. If None, use the threshold from the input config
        .yaml file. The default is None.

    Returns
    -------
    cfg : Detectron2 config instance
        Loaded config instance, with all necessary parameters merged from
        input config .yaml file.

    """
    inpt_cfg_type = get_model_arch_from_yaml(inpt_cfg_yaml_path)
    #load basic mask-rcnn config
    if inpt_cfg_type == 'MRCNN':
        cfg = get_cfg()
    #otherwise load Swin-T config
    if inpt_cfg_type == 'SwinT':
        assert setup_for_swint()
        import swint
        cfg = get_cfg()
        swint.add_swint_config(cfg)
    #or centermask2 config
    if inpt_cfg_type=='Centermask2':
        assert setup_for_centermask()
        from centermask.config import get_cfg as get_centermask2_cfg
        cfg=get_centermask2_cfg()
    cfg.merge_from_file(inpt_cfg_yaml_path)
    if use_cpu:
        cfg.MODEL.DEVICE='cpu'
    else:
        cfg.MODEL.DEVICE='cuda'
    if not isinstance(adj_nms_thresh, type(None)):
        _assert_adj_thresh_inpt_valid(adj_nms_thresh, 'adj_nms_thresh')
        if isinstance(adj_nms_thresh, (float, int)):
            if inpt_cfg_type in ['MRCNN', 'SwinT']:
                cfg.MODEL.RPN.NMS_THRESH =adj_nms_thresh
                cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=adj_nms_thresh
            #assumes Centermask2 if cfg type is not Mask-RCNN w/ or w/o Swin backbone
            else:
                cfg.MODEL.FCOS.NMS_TH = adj_nms_thresh
        else:
            if inpt_cfg_type in ['MRCNN', 'SwinT']:
                cfg.MODEL.RPN.NMS_THRESH =0.2
                cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST =0.2
            #assumes Centermask2 if cfg type is not Mask-RCNN w/ or w/o Swin backbone
            else:
                cfg.MODEL.FCOS.NMS_TH = 0.4
    if not isinstance(adj_thresh_test, type(None)):
        _assert_adj_thresh_inpt_valid(adj_thresh_test, 'adj_thresh_test')
        if isinstance(adj_thresh_test, (float, int)):
            if inpt_cfg_type in ['MRCNN', 'SwinT']:
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =adj_thresh_test
            #assumes Centermask2 if cfg type is not Mask-RCNN w/ or w/o Swin backbone
            else:
                cfg.MODEL.FCOS.INFERENCE_TH_TEST = adj_thresh_test
        else:
            if inpt_cfg_type in ['MRCNN', 'SwinT']:
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =0.7
            #assumes Centermask2 if cfg type is not Mask-RCNN w/ or w/o Swin backbone
            else:
                cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.5
    return cfg


def smart_load_predictor(inpt_cfg_yaml_path, inpt_weights_path, use_cpu=False,
                         adj_nms_thresh=None, adj_thresh_test=None):
    """Load a Detectron2 DefaultPredictor instance from a config .yaml file and
    a .pth or .pkl weights file. Model should have a CZD-supported architecture
    (i.e., Mask-RCNN with or without Swin-T backbone or Centermask2-VoVnet).
    Architecture is determined automatically from the .yaml and dependencies
    are installed if necessary.

    Parameters
    ----------
    inpt_cfg_yaml_path : str
        Path to config .yaml file for model.
    inpt_weights_path : str
        Path to config .pth or .pkl weights file for model.
    use_cpu : bool, optional
        If True, set the cfg MODEL.DEVICE parameter to 'cpu'. Otherwise, set it
        to 'cuda'. The default is False (i.e., use 'cuda').
    adj_nms_thresh : float or None, optional
        If present, manually set the model config NMS threshold to this setting.
        Input floats should be between 0 and 1. The default is None.
    adj_thresh_test : float or None, optional
        If present, manually set the model config ROI_HEADS or FCOS inference
        score threshold to this setting. Input floats should be between 0 and
        1. The default is None.

    Returns
    -------
    Detectron2 DefaultPredictor instance
        A Detectron2 DefaultPredictor instance for application to images.
        Weights and settings will have been loaded from the input file paths.

    """
    cfg = smart_D2_cfg_load(inpt_cfg_yaml_path, use_cpu=use_cpu,
                            adj_nms_thresh=adj_nms_thresh,
                            adj_thresh_test=adj_thresh_test)
    cfg.MODEL.WEIGHTS = inpt_weights_path
    return DefaultPredictor(cfg)
