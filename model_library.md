# colab_zirc_dims model library:

Here you can see data on pre-trained, Detectron2-based instance segmentation models that are currently available for use in colab_zirc_dims. This page is inspired by the Detectron2 model zoo page. Models are continually being improved through adjustment of training hyperparameters, so this page (and the models available here) may be subject to change.

## Current 'best' models:

### Summary table:
<table>
<thead>
	<tr>
		<th>Model</th>
		<th>Architecture</th>
		<th>Backbone</th>
		<th>Train/val dataset</th>
		<th>Training iterations</th>
		<th><a href="https://cocodataset.org/#detection-eval" target="_blank" rel="noopener noreferrer">bbox AP</a></th>
		<th><a href="https://cocodataset.org/#detection-eval" target="_blank" rel="noopener noreferrer">mask AP</a></th>
		<th>Links:</th>
	</tr>
</thead>
<tbody>
	<tr>
		<td>M-ST-C</td>
		<td>Mask-RCNN (<a href="https://github.com/facebookresearch/detectron2" target="_blank" rel="noopener noreferrer">Detectron2</a>)</td>
		<td><a href="https://github.com/xiaohu2015/SwinT_detectron2" target="_blank" rel="noopener noreferrer">Swin-T</a></td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#czd_large-dataset" target="_blank" rel="noopener noreferrer">czd_large</a></td>
		<td>7.0k</td>
		<td>75.14</td>
		<td>75.61</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/czd_large_dataset/Swin-T/SwinT_czd_large_v1.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/czd_large_M-ST-C_7.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_large/czd_large_M-ST-C/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
	<tr>
		<td>M-R101-C</td>
		<td>Mask-RCNN (<a href="https://github.com/facebookresearch/detectron2" target="_blank" rel="noopener noreferrer">Detectron2</a>)</td>
		<td>ResNet-101-FPN</td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#czd_large-dataset" target="_blank" rel="noopener noreferrer">czd_large</a></td>
		<td>7.0k</td>
		<td>73.87</td>
		<td>75.92</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/czd_large_dataset/Mask-RCNN/R_101_COCO_czd_large_v1.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/czd_large_M-R101-C_7.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_large/czd_large_M-R101-C/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
	<tr>
		<td>C-V-C</td>
		<td><a href="https://github.com/youngwanLEE/centermask2" target="_blank" rel="noopener noreferrer">Centermask2</a></td>
		<td><a href="https://github.com/youngwanLEE/centermask2" target="_blank" rel="noopener noreferrer">VovNetv2-99</a></td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#czd_large-dataset" target="_blank" rel="noopener noreferrer">czd_large</a></td>
		<td>11.0k</td>
		<td>72.1</td>
		<td>72.15</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/czd_large_dataset/Centermask/Cmask2_czd_large_v1.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/czd_large_C-V-C_11.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_large/czd_large_C-V-C/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
	<tr>
		<td>M-R50-C</td>
		<td>Mask-RCNN (<a href="https://github.com/facebookresearch/detectron2" target="_blank" rel="noopener noreferrer">Detectron2</a>)</td>
		<td>ResNet-50-FPN</td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#czd_large-dataset" target="_blank" rel="noopener noreferrer">czd_large</a></td>
		<td>6.0k</td>
		<td>72.21</td>
		<td>74.31</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/czd_large_dataset/Mask-RCNN/R_50_COCO_czd_large_v1.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/czd_large_M-R50-C_6.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_large/czd_large_M-R50-C/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
</tbody>
</table>

## Legacy models:

These models were trained on the on the relatively small 'czd_orig' dataset. Newer models generally have lower segmentation error rates and we recommend that you use them instead.

### Summary table:
<table>
<thead>
	<tr>
		<th>Model</th>
		<th>Architecture</th>
		<th>Backbone</th>
		<th>Pretraining</th>
		<th>Train/val dataset</th>
		<th>Training images randomly augmented?</th>
		<th>Training iterations</th>
		<th><a href="https://cocodataset.org/#detection-eval" target="_blank" rel="noopener noreferrer">bbox AP</a></th>
		<th><a href="https://cocodataset.org/#detection-eval" target="_blank" rel="noopener noreferrer">mask AP</a></th>
		<th>Links:</th>
	</tr>
</thead>
<tbody>
	<tr>
		<td>101_model_COCO_base</td>
		<td>Mask-RCNN (<a href="https://github.com/facebookresearch/detectron2" target="_blank" rel="noopener noreferrer">Detectron2</a>)</td>
		<td>ResNet-101-FPN</td>
		<td><a href="https://cocodataset.org/#home" target="_blank" rel="noopener noreferrer">COCO</a></td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#legacy-dataset-czd_orig" target="_blank" rel="noopener noreferrer">czd_orig</a></td>
		<td>Yes</td>
		<td>6.0k</td>
		<td>72.57</td>
		<td>67.63</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/orig_dataset/Mask-RCNN/101_model_COCO_base_orig.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/101_model_COCO_base_2_6.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/101_model_COCO_base/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
	<tr>
		<td>centermask2</td>
		<td><a href="https://github.com/youngwanLEE/centermask2" target="_blank" rel="noopener noreferrer">Centermask2</a></td>
		<td><a href="https://github.com/youngwanLEE/centermask2" target="_blank" rel="noopener noreferrer">VovNetv2-99</a></td>
		<td><a href="https://cocodataset.org/#home" target="_blank" rel="noopener noreferrer">COCO</a></td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#legacy-dataset-czd_orig" target="_blank" rel="noopener noreferrer">czd_orig</a></td>
		<td>Yes</td>
		<td>4.0k</td>
		<td>74.37</td>
		<td>67.57</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/orig_dataset/Centermask/Centermask2_orig.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/centermask2_4.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/centermask2/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
	<tr>
		<td>50_model_COCO_base</td>
		<td>Mask-RCNN (<a href="https://github.com/facebookresearch/detectron2" target="_blank" rel="noopener noreferrer">Detectron2</a>)</td>
		<td>ResNet-50-FPN</td>
		<td><a href="https://cocodataset.org/#home" target="_blank" rel="noopener noreferrer">COCO</a></td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#legacy-dataset-czd_orig" target="_blank" rel="noopener noreferrer">czd_orig</a></td>
		<td>Yes</td>
		<td>6.0k</td>
		<td>71.2</td>
		<td>66.21</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/orig_dataset/Mask-RCNN/50_model_COCO_base_orig.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/50_model_COCO_base_2_6.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/50_model_COCO_base/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
	<tr>
		<td>101_from_scratch</td>
		<td>Mask-RCNN (<a href="https://github.com/facebookresearch/detectron2" target="_blank" rel="noopener noreferrer">Detectron2</a>)</td>
		<td>ResNet-101-FPN</td>
		<td>None</td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#legacy-dataset-czd_orig" target="_blank" rel="noopener noreferrer">czd_orig</a></td>
		<td>Yes</td>
		<td>8.0k</td>
		<td>65.84</td>
		<td>63.35</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/orig_dataset/Mask-RCNN/101_from_scratch.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/101_from_scratch_8.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/101_from_scratch/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
	<tr>
		<td>50_from_scratch</td>
		<td>Mask-RCNN (<a href="https://github.com/facebookresearch/detectron2" target="_blank" rel="noopener noreferrer">Detectron2</a>)</td>
		<td>ResNet-50-FPN</td>
		<td>None</td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#legacy-dataset-czd_orig" target="_blank" rel="noopener noreferrer">czd_orig</a></td>
		<td>Yes</td>
		<td>4.0k</td>
		<td>63.4</td>
		<td>61.45</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/orig_dataset/Mask-RCNN/50_from_scratch.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/50_from_scratch_4.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/50_from_scratch/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
	<tr>
		<td>50_from_scratch_no_augs</td>
		<td>Mask-RCNN (<a href="https://github.com/facebookresearch/detectron2" target="_blank" rel="noopener noreferrer">Detectron2</a>)</td>
		<td>ResNet-50-FPN</td>
		<td>None</td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#legacy-dataset-czd_orig" target="_blank" rel="noopener noreferrer">czd_orig</a></td>
		<td>No</td>
		<td>4.0k</td>
		<td>35.99</td>
		<td>35.82</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/orig_dataset/Mask-RCNN/50_from_scratch_no_augs.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/50_from_scratch_no_augs_4.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/50_from_scratch_no_augs/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
	<tr>
		<td>mask_rcnn_swint</td>
		<td>Mask-RCNN (<a href="https://github.com/facebookresearch/detectron2" target="_blank" rel="noopener noreferrer">Detectron2</a>)</td>
		<td><a href="https://github.com/xiaohu2015/SwinT_detectron2" target="_blank" rel="noopener noreferrer">Swin-T</a></td>
		<td><a href="https://cocodataset.org/#home" target="_blank" rel="noopener noreferrer">COCO</a></td>
		<td><a href="https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#legacy-dataset-czd_orig" target="_blank" rel="noopener noreferrer">czd_orig</a></td>
		<td>Yes</td>
		<td>7.0k</td>
		<td>72.42</td>
		<td>67.69</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/orig_dataset/Swin-T/Swin-T_orig.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/mask_rcnn_swint_7.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/mask_rcnn_swint/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
</tbody>
</table>
