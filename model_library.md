# colab_zirc_dims model library:

Here you can see data on pre-trained, Detectron2-based instance segmentation models that are currently available for use in colab_zirc_dims. This page is inspired by the Detectron2 model zoo page. Models are continually being improved through adjustment of training hyperparameters, so this page (and the models available here) may be subject to change.

## Contents:
  * **[Current 'best' models](https://github.com/MCSitar/colab_zirc_dims/blob/main/model_library.md#current-best-models)**
    * **[Training](https://github.com/MCSitar/colab_zirc_dims/blob/main/model_library.md#Training)**
    * **[Deployment checkpoint selection](https://github.com/MCSitar/colab_zirc_dims/blob/main/model_library.md#selection-of-checkpoints-for-deployment)**
    * **[Summary tables](https://github.com/MCSitar/colab_zirc_dims/blob/main/model_library.md#summary-tables)**
    * **[Discussion](https://github.com/MCSitar/colab_zirc_dims/blob/main/model_library.md#discussion)**
   * **[Legacy models](https://github.com/MCSitar/colab_zirc_dims/blob/main/model_library.md#current-best-models)**
     * **[Summary tables](https://github.com/MCSitar/colab_zirc_dims/blob/main/model_library.md#summary-tables-1)**

## Current 'best' models:
**Current as of colab_zirc_dims v1.0.10**

These models were trained on the [czd_large](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#czd_large-dataset) dataset. Models deployed for application by colab_zirc_dims users were chosen using a modified 'early stopping' process: models were trained for a set period of >=12,000 iterations, but only the most performant model checkpoints within the confines of the colab_zirc_dims processing algorithm (i.e., those best able to reproduce the manual measurement results of Leary et al. (2022)) are provided to users. See below for more more information on training and checkpoint selection. You can train your own models using our workflow by [follow the directions here](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#training-new-models-for-colab_zirc_dims).

### Training:

Because training from scratch proved unsuccessful with the 'czd_orig' dataset (see 'Legacy Models' below), all models were fine-tuned after initialization of MS-COCO-pre-trained weights found in their respective repositories. You can find links to these weights in the config file for each model.

We used different, empirically chosen learning rates dependent on model architecture and optimizer: these were respectively 0.000015, 0.00025, 0.00005, and 0.0005 for models M-R101-C (Adam optimizer) and M-R50-C (SGD optimizer), M-ST-C (AdamW optimizer), and C-V-C (SGD optimizer). A warmup period of 1000 iterations was used for all models, followed by stepped learning rate rate drawdowns by a factor of 0.5 (i.e., 'Gamma') starting at 1500 iterations (~2 epochs) and again every further 1500 iteration increment thereafter. This learning rate schedule is somewhat generallized but (in combination with architecture-optimizer dependent base learning rates) seems to yield consistent decreases in validation loss (see 'metrics' data linked in 'Model info' table) where it begins to plateau in the absence of learning rate reduction.

We adopted a fairly aggressive training image augmentation strategy to mitigate overfitting, as shown in the figure below:

[<img align="center" src="https://user-images.githubusercontent.com/74220513/202050575-ca33f6ba-61b4-4fc1-b04f-c76736e709ab.png" width="80%"/>](augfig_with_defocus.png)
<figcaption><b>Random augmentations applied to training images via Detectron2 dataloader. All augmentations besides "defocus" were implemented using default Detectron2 augmentations and transformations. The random (in extent and magnitude) 'defocus blur' augmentation, which is based on a modification of code from the <a href="https://github.com/bethgelab/imagecorruptions" target="_blank" rel="noopener noreferrer">imagecorruptions</a> library, approximates a relatively common tiling-related artefact that appears in LA-ICP-MS mosaic images.</b></figcaption>
<br>
<br>
All models were trained for at least 12,000 total iterations with a batch size of 2. Training loss stabilized by ~2000 iterations for all models (see plot of Mask-RCNN-style 'mask loss', which is a loss component for all trained models, below), and mAP metrics by ~4000 iterations, with largely stochastic variations observed thereafter. Validation loss (metrics vary between model architectures, so 1:1 comparisons are not plottable) did continue to decrease until >= ~8000 iterations for all models, though apparently not at a rate resolvable in mAP metrics.

[<img align="center" src="https://user-images.githubusercontent.com/74220513/202087247-16dc7a32-330f-461c-9acf-3ac6ee18cc9d.png" width="80%"/>](plot_curves_for_github_page.png)
<figcaption><b>Loss and evaluation curves during training: mask loss (average per-ROI binary cross-entropy loss, per He et al. (2022)), MS-COCO bounding box and mask mAP metrics, and approximate grain extent overestimate rates from rapid colab_zirc_dims evaluation of a serialized version of the Leary et al. (2022) grain image-measurement dataset. MS-COCO mAP metrics were evaluated at 200 iteration intervals during training. Evaluations of the serialized dataset were only run where model checkpoints were saved (at ~1000 iteration intervals).</b></figcaption>
<br>
<br>

### Selection of checkpoints for deployment:

Model checkpoints for deployment were selected based on performance in reproducing manual per-grain long and short axis length measurements from Leary et al. (2022) using a fast, streamlined version of the colab_zirc_dims grain measurement algorithm and a serialized version of the Leary et al. (2022) dataset. We narrowed our selection window to **checkpoints at >= 4,000 training iterations** based on the observation that mAP metrics appear to increase up until this point (see curves above). We then selected checkpoints with **minimal proportions of long and/or short axis measurement results that overestimate manual (Leary et al., 2022) measurements by > 20%**.

Performance on the serialized dataset approximates but does differ slighly from results obtainable using conventional colab_zirc_dims processing, apparently due to lossy saving of the per-shot image data when serializing the dataset. Evaluations for the selected model checkpoints were consequently re-run using the conventional colab_zirc_dims process; these results are presented in the 'Evaluation results...' table below.

### Summary tables:
#### Model info:
<!---start_table_ID0--->
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
		<td><b>75.14</b></td>
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
		<td><b>75.92</b></td>
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
<!---end_table_ID0--->

#### Evaluation results on [Leary et al. (2022)](https://doi.org/10.2110/jsr.2021.101) dataset:
<!---start_table_ID1--->
<table>
<thead>
	<tr>
		<th>Model</th>
		<th>Training iterations</th>
		<th>n total</th>
		<th>n successful<sup>a</sup></th>
		<th>failure rate (%)</th>
		<th>avg. abs. long axis error (μm)</th>
		<th>avg. abs. short axis error (μm)</th>
		<th>avg. abs. long axis % error</th>
		<th>avg. abs. short axis % error</th>
		<th>avg. spot segmentation time (s)<sup>b</sup></th>
		<th>Link:</th>
	</tr>
</thead>
<tbody>
	<tr>
		<td>M-ST-C</td>
		<td>7.0k</td>
		<td>5004</td>
		<td>5003</td>
		<td><b>0.02</b></td>
		<td><b>5.66</b></td>
		<td>4.31</td>
		<td><b>7.28</b></td>
		<td>8.57</td>
		<td>0.1142</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_large/czd_large_M-ST-C/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
	<tr>
		<td>M-R101-C</td>
		<td>7.0k</td>
		<td>5004</td>
		<td>4994</td>
		<td>0.1998</td>
		<td>5.76</td>
		<td>4.31</td>
		<td>7.39</td>
		<td>8.59</td>
		<td>0.1205</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_large/czd_large_M-R101-C/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
	<tr>
		<td>C-V-C</td>
		<td>11.0k</td>
		<td>5004</td>
		<td>5000</td>
		<td>0.0799</td>
		<td>5.73</td>
		<td>4.34</td>
		<td>7.35</td>
		<td>8.63</td>
		<td>0.1642</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_large/czd_large_C-V-C/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
	<tr>
		<td>M-R50-C</td>
		<td>6.0k</td>
		<td>5004</td>
		<td>4993</td>
		<td>0.2198</td>
		<td>5.7</td>
		<td><b>4.29</b></td>
		<td>7.32</td>
		<td><b>8.54</b></td>
		<td><b>0.0931</b></td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_large/czd_large_M-R50-C/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
</tbody>
</table>
<sup>a</sup>Segmentation/measurement of a spot is considered to have 'failed' if no grain mask can be obtained in the immediate vicinity of the spot target location
<br>
<sup>b</sup>Please note that this represents only the time taken to obtain a central grain mask from a single spot within colab_zirc_dims processing. Actual per-spot processing speed encompasses measurement of the resulting mask and saving verification data, and will be substantially longer.
<!---end_table_ID1--->

### Discussion:
We recommend model M-ST-C in most cases. This model produces consistently good segmentation results and seems to be robust to image artefacts.

The relatively low bounding box mAP metric for C-V-C belies its accuracy to some degree: our train-validation dataset contains numerous very small grain annotations, which (as noted by Lee and Park (2020)) Centermask struggles with. Though it is thus contraindicated for application to images with many small grains, C-V-C is quite accurate when applied to images with large (relative to image size) grains. It is recommended that users try this model if they find that M-ST-C is failing to identify or producing inaccurate masks when applied to their data.

The aforementioned models rely on code in non-Detectron2 repositories. If users encounter problems related to these dependencies (download and path management doing this is handled automatically within colab_zirc_dims processing notebooks), we recommend that they try the Detectron2 Mask-RCNN models M-R101-C and M-R50-C. These will work with only a basic Detectron2 installation.

## Legacy models:

These models were trained on the on the relatively small '[czd_orig](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#legacy-dataset-czd_orig)' dataset. Newer models generally have lower segmentation error rates and we recommend that you use them instead. Please see our [pre-print manuscript](https://gchron.copernicus.org/preprints/gchron-2022-12/) for details on model training and checkpoint selection.
### Summary tables:
#### Model info:
<!---start_table_ID2--->
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
		<td><b>74.37</b></td>
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
		<td><b>67.69</b></td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/configs/orig_dataset/Swin-T/Swin-T_orig.yaml" target="_blank" rel="noopener noreferrer">config</a> | <a href="https://colabzircdimsmodels.s3.us-west-1.amazonaws.com/mask_rcnn_swint_7.0k.pth" target="_blank" rel="noopener noreferrer">model</a> | <a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/mask_rcnn_swint/training_metrics.json" target="_blank" rel="noopener noreferrer">training metrics</a></td>
	</tr>
</tbody>
</table>
<!---end_table_ID2--->

#### Evaluation results on [Leary et al. (2022)](https://doi.org/10.2110/jsr.2021.101) dataset:
<!---start_table_ID3--->
<table>
<thead>
	<tr>
		<th>Model</th>
		<th>Training iterations</th>
		<th>n total</th>
		<th>n successful<sup>a</sup></th>
		<th>failure rate (%)</th>
		<th>avg. abs. long axis error (μm)</th>
		<th>avg. abs. short axis error (μm)</th>
		<th>avg. abs. long axis % error</th>
		<th>avg. abs. short axis % error</th>
		<th>avg. spot segmentation time (s)<sup>b</sup></th>
		<th>Link:</th>
	</tr>
</thead>
<tbody>
	<tr>
		<td>101_model_COCO_base</td>
		<td>6.0k</td>
		<td>5004</td>
		<td>5003</td>
		<td><b>0.02</b></td>
		<td>6.11</td>
		<td><b>4.39</b></td>
		<td>7.96</td>
		<td><b>8.83</b></td>
		<td>0.1187</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/101_model_COCO_base/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
	<tr>
		<td>centermask2</td>
		<td>4.0k</td>
		<td>5004</td>
		<td>4998</td>
		<td>0.1199</td>
		<td><b>6.02</b></td>
		<td>4.44</td>
		<td>7.85</td>
		<td>9.0</td>
		<td>0.1136</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/centermask2/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
	<tr>
		<td>50_model_COCO_base</td>
		<td>6.0k</td>
		<td>5004</td>
		<td>4992</td>
		<td>0.2398</td>
		<td>6.2</td>
		<td>4.48</td>
		<td>8.04</td>
		<td>8.98</td>
		<td><b>0.0722</b></td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/50_model_COCO_base/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
	<tr>
		<td>101_from_scratch</td>
		<td>8.0k</td>
		<td>5004</td>
		<td>4931</td>
		<td>1.4588</td>
		<td>7.73</td>
		<td>5.65</td>
		<td>9.75</td>
		<td>11.45</td>
		<td>0.1073</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/101_from_scratch/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
	<tr>
		<td>50_from_scratch</td>
		<td>4.0k</td>
		<td>5004</td>
		<td>4988</td>
		<td>0.3197</td>
		<td>7.65</td>
		<td>5.45</td>
		<td>9.46</td>
		<td>10.91</td>
		<td>0.0868</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/50_from_scratch/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
	<tr>
		<td>50_from_scratch_no_augs</td>
		<td>4.0k</td>
		<td>5004</td>
		<td>4749</td>
		<td>5.0959</td>
		<td>13.09</td>
		<td>8.56</td>
		<td>17.18</td>
		<td>18.16</td>
		<td>0.1084</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/50_from_scratch_no_augs/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
	<tr>
		<td>mask_rcnn_swint</td>
		<td>7.0k</td>
		<td>5004</td>
		<td>4993</td>
		<td>0.2198</td>
		<td><b>6.02</b></td>
		<td>4.53</td>
		<td><b>7.71</b></td>
		<td>8.96</td>
		<td>0.1295</td>
		<td><a href="https://raw.githubusercontent.com/MCSitar/colab_zirc_dims/main/model_metrics/czd_orig/mask_rcnn_swint/timed_czd_test_eval.xlsx" target="_blank" rel="noopener noreferrer">data file</a></td>
	</tr>
</tbody>
</table>
<sup>a</sup>Segmentation/measurement of a spot is considered to have 'failed' if no grain mask can be obtained in the immediate vicinity of the spot target location.
<br>
<sup>b</sup>Please note that this represents only the time taken to obtain a central grain mask from a single spot within colab_zirc_dims processing. Actual per-spot processing speed encompasses measurement of the resulting mask and saving verification data, and will be substantially longer.
<!---end_table_ID3--->

## References:

He, K., Gkioxari, G., Dollár, P., and Girshick, R.: Mask R-CNN, arXiv:1703.06870 [cs], 2018.

Leary, R. J., Smith, M. E., and Umhoefer, P.: Grain-Size Control on Detrital Zircon Cycloprovenance in the Late Paleozoic Paradox and Eagle Basins, USA, J. Geophys. Res. Solid Earth, 125, e2019JB019226, https://doi.org/10.1029/2019JB019226, 2020.

Leary, R. J., Smith, M. E., and Umhoefer, P.: Mixed eolian–longshore sediment transport in the late Paleozoic Arizona shelf and Pedregosa basin, U.S.A.: A case study in grain-size analysis of detrital-zircon datasets, Journal of Sedimentary Research, 92, 676–694, https://doi.org/10.2110/jsr.2021.101, 2022.

Lee, Y. and Park, J.: CenterMask : Real-Time Anchor-Free Instance Segmentation, arXiv:1911.06667 [cs], 2020.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B.: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, https://doi.org/10.48550/ARXIV.2103.14030, 2021.

Michaelis, C., Mitzkus, B., Geirhos, R., Rusak, E., Bringmann, O., Ecker, A. S., Bethge, M., and Brendel, W.: Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming, https://doi.org/10.48550/arXiv.1907.07484, 31 March 2020.

Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., and Girshick, R.: Detectron2, 2019.

Ye, H., Yang, Y., and L3str4nge: SwinT_detectron2: v1.2, , https://doi.org/10.5281/ZENODO.6468976, 2021.

