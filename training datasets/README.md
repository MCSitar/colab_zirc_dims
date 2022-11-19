# Contents:
  * **[czd_large dataset](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#czd_large-dataset)**
    * **[Download](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#download-link)**
    * **[Summary table](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#summary-table)**
    * **[Image sources](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#image-sources)**
    * **[Sampling strategy](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#sampling-strategy)**
    * **[Splitting strategy](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#splitting-strategy)**
    * **[Annotation](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#annotation)**
    * **[Detailed breakdown](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#detailed-dataset-breakdown)**
    * **[Training new models](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#training-new-models-for-colab_zirc_dims)**
  * **[Legacy dataset](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#legacy-dataset-czd_orig)**
    * **[Summary table](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#summary-table-1)**
  * **[References](https://github.com/MCSitar/colab_zirc_dims/tree/main/training%20datasets#references)**


# czd_large dataset:
Colab_zirc_dims models are currently trained on the (new) "czd_large" dataset. This is a semi-automatically annotated training-validation dataset that we have created for instance segmentation of mineral grains in reflected light images collected during LA-ICP-MS detrital zircon dating.

## Download link:
At ~500 MB, the zipped dataset is too large for (reasonably paced) serving from GitHub, so we host it on AWS. You can download it here: https://czdtrainingdatasetlarge.s3.amazonaws.com/CZD_train_large.zip

## Summary table:
<table>
<thead>
  <tr>
    <th>Source facility</th>
    <th>Training set images</th>
    <th>Validation set images</th>
    <th>Training set annotations</th>
    <th>Validation set annotations</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>1203</td>
    <td>212</td>
    <td>12923</td>
    <td>2326</td>
  </tr>
  <tr>
    <td><a href="https://www.petrochronology.com/" target="_blank" rel="noopener noreferrer">UCSB</a></td>
    <td>121</td>
    <td>22</td>
    <td>1039</td>
    <td>176</td>
  </tr>
  <tr>
    <td rowspan="2"><b>Total</b></td>
    <td><b>1324</b></td>
    <td><b>234</b></td>
    <td><b>13962</b></td>
    <td><b>2502</b></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>1558</b></td>
    <td align="center" colspan="2"><b>16464</b></td>
  </tr>
</tbody>
</table>

## Image sources:
Constituent images were captured at both the [University of Arizona Laserchron Center](https://sites.google.com/laserchron.org/arizonalaserchroncenter/home) ("ALC"; ~90% of dataset) and the [UC Santa Barbara LA-ICP-MS facility](https://www.petrochronology.com/) ("UCSB"; ~10% of dataset).

ALC images are clipped from pre-ablation, per-sample mosaics of detrital zircon samples described in detail by Leary et al., 2020, with each clipped image centered on an actual LA-ICP-MS shot location. Per-sample clipped sub-image sizes were chosen such that the largest grains in the sample fit within the image. Images vary in quality sample-to-sample, and include a variety of artefacts (e.g., anomalous bright spots, blurry mosaic tiles, etc.). Imaged samples contain a wide range of zircon ages from Proterozoic to late Paleozoic and represent a variety of terrestrial and marine depositional environments including fluvial, delta plain, nearshore, and continental shelf environments, and grains are generally sub-angular to rounded. See Leary et al. (2020a; 2020b) for detailed discussion of these samples.

UCSB images were collected in January 2019 for an as-of-yet unpublished analysis of detrital zircon in rock samples from east-central Nevada, USA; these images are provided as-is (i.e., as individual images centered on LA-ICP-MS shot locations, captured synchronously with ablation). Image quality is generally high but images are occasionally somewhat blurry. Images are of grains derived from samples of Late Mesozoic-Early Cenozoic rocks interpreted to have been deposited by braided stream systems. Dated zircon grains from these samples indicate the presence of mixed populations of Proterozoic grains that likely record long-range fluvial transport (e.g., from the Grenville Orogen to modern day Nevada, USA) and iterative recycling prior to their most recent deposition. These grains are combined in approximately equal proportions with minimally transported Early Cretaceous grains presumably sourced from the ancient Sierran Arc. Images from the UCSB training set consequently include variable mixtures of very well-rounded and relatively fresh, euhedral grains.

## Sampling strategy:
For each sample in our collection of per-shot ALC mosaic sub-image clippings and UCSB per-shot images:
- We used the Python 'random' module to iteratively select random shot/image locations, with the constraint that all images must be completely non-overlapping in both real-world (as per .scancsv and .Align targetting metadata files) and image space.
- We stopped sampling when no new shot locations/images were available that satisfied this condition.
- Finally, we manually removed images apparently lacking clearly visible grains. These included all shots on zircon standards from our UCSB dataset; these standards covered the entire frame of their respective images and were not distinguishable from an epoxy background without a priori knowlege (i.e., filenames). In total, 12 images were removed.

## Splitting strategy:
The dataset was divided into training ("train" subdirectory) and validation ("val" subdirectory) sets via sample-proportionate stratified random selection with a target of an 85%-15% training-validation split. For ALC samples, which include varying (minority) proportions of shot-images targetting zircon standard grains in addition to those targetting detrital grains, training-validation splitting was further stratified such that per-sample proportions of zircon standard images in the validation set approximate those in the training set.

Please refer to the table below for a detailed breakdown of sample-annotation distributions in the training and validation datasets.

## Annotation:
Annotations for both the training and validation sets follow the VGG 2.0 format and can be found in the 'via_region_data.json' file in each subdirectory. All clearly visible reflective mineral grains were segmented into a single "grain" class. These annotations were created using a semi-automated approach: a Detectron2 Mask-RCNN Resnet-101 model (['101_model_COCO_base'](https://github.com/MCSitar/colab_zirc_dims/blob/main/model_library.md#model-info-1)) fine-tuned on [a smaller, manually annotated dataset](https://github.com/MCSitar/colab_zirc_dims/blob/main/training%20datasets/README.md#legacy-dataset-czd_orig) was repurposed to automatically generate per-grain VGG segmentations through polygonization of output detection masks. Auto-generated annotations were manually reviewed for each image and, where necessary, corrected and/or extended using the Via 2.1 image annotation tool (available at https://www.robots.ox.ac.uk/~vgg/software/via).

## Detailed dataset breakdown:
<table>
<thead>
  <tr>
    <th>Sample</th>
    <th>Source facility</th>
    <th>Image size (pixels)</th>
    <th>Image scale (µm/pixel)</th>
    <th>Training set images</th>
    <th>Validation set images</th>
    <th>Training set annotations</th>
    <th>Validation set annotations</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>BC01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>268 x 268</td>
    <td>1.494</td>
    <td>31</td>
    <td>6</td>
    <td>404</td>
    <td>90</td>
  </tr>
  <tr>
    <td>BC02</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>350 x 350</td>
    <td>1.144</td>
    <td>37</td>
    <td>7</td>
    <td>665</td>
    <td>92</td>
  </tr>
  <tr>
    <td>BC05</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>368 x 368</td>
    <td>1.226</td>
    <td>30</td>
    <td>5</td>
    <td>434</td>
    <td>66</td>
  </tr>
  <tr>
    <td>DMR01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>358 x 358</td>
    <td>1.119</td>
    <td>39</td>
    <td>7</td>
    <td>383</td>
    <td>46</td>
  </tr>
  <tr>
    <td>EC01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>292 x 292</td>
    <td>1.025</td>
    <td>46</td>
    <td>8</td>
    <td>801</td>
    <td>194</td>
  </tr>
  <tr>
    <td>GS01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>312 x 312</td>
    <td>1.286</td>
    <td>48</td>
    <td>9</td>
    <td>801</td>
    <td>117</td>
  </tr>
  <tr>
    <td>GS02</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>402 x 402</td>
    <td>0.994</td>
    <td>28</td>
    <td>5</td>
    <td>66</td>
    <td>8</td>
  </tr>
  <tr>
    <td>HAR06</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>324 x 324</td>
    <td>1.236</td>
    <td>37</td>
    <td>7</td>
    <td>304</td>
    <td>48</td>
  </tr>
  <tr>
    <td>HT01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>264 x 264</td>
    <td>1.520</td>
    <td>39</td>
    <td>7</td>
    <td>50</td>
    <td>10</td>
  </tr>
  <tr>
    <td>HT02</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>206 x 206</td>
    <td>1.948</td>
    <td>41</td>
    <td>7</td>
    <td>59</td>
    <td>8</td>
  </tr>
  <tr>
    <td>HT03</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>300 x 300</td>
    <td>1.333</td>
    <td>60</td>
    <td>10</td>
    <td>95</td>
    <td>13</td>
  </tr>
  <tr>
    <td>HT04</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>316 x 316</td>
    <td>1.270</td>
    <td>51</td>
    <td>9</td>
    <td>61</td>
    <td>11</td>
  </tr>
  <tr>
    <td>HT23</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>196 x 196</td>
    <td>2.046</td>
    <td>42</td>
    <td>7</td>
    <td>75</td>
    <td>10</td>
  </tr>
  <tr>
    <td>HT24</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>376 x 376</td>
    <td>1.062</td>
    <td>30</td>
    <td>5</td>
    <td>71</td>
    <td>13</td>
  </tr>
  <tr>
    <td>ILL01</td>
    <td><a href="https://www.petrochronology.com/" target="_blank" rel="noopener noreferrer">UCSB</a></td>
    <td>1280 x 1024</td>
    <td>0.430</td>
    <td>18</td>
    <td>3</td>
    <td>105</td>
    <td>20</td>
  </tr>
  <tr>
    <td>JB01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>332 x 332</td>
    <td>1.207</td>
    <td>31</td>
    <td>6</td>
    <td>610</td>
    <td>101</td>
  </tr>
  <tr>
    <td>KA01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>294 x 294</td>
    <td>1.358</td>
    <td>37</td>
    <td>6</td>
    <td>528</td>
    <td>78</td>
  </tr>
  <tr>
    <td>KA03</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>336 x 336</td>
    <td>1.485</td>
    <td>32</td>
    <td>6</td>
    <td>492</td>
    <td>108</td>
  </tr>
  <tr>
    <td>LC01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>366 x 366</td>
    <td>1.094</td>
    <td>34</td>
    <td>6</td>
    <td>817</td>
    <td>200</td>
  </tr>
  <tr>
    <td>MC04</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>372 x 372</td>
    <td>1.074</td>
    <td>28</td>
    <td>5</td>
    <td>130</td>
    <td>22</td>
  </tr>
  <tr>
    <td>MC05</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>374 x 374</td>
    <td>1.334</td>
    <td>24</td>
    <td>4</td>
    <td>299</td>
    <td>49</td>
  </tr>
  <tr>
    <td>MC17</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>316 x 316</td>
    <td>1.264</td>
    <td>39</td>
    <td>7</td>
    <td>659</td>
    <td>147</td>
  </tr>
  <tr>
    <td>MC18</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>290 x 290</td>
    <td>1.376</td>
    <td>47</td>
    <td>8</td>
    <td>211</td>
    <td>30</td>
  </tr>
  <tr>
    <td>MS06</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>442 x 442</td>
    <td>1.132</td>
    <td>28</td>
    <td>5</td>
    <td>252</td>
    <td>59</td>
  </tr>
  <tr>
    <td>MS07</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>398 x 398</td>
    <td>1.253</td>
    <td>26</td>
    <td>5</td>
    <td>281</td>
    <td>62</td>
  </tr>
  <tr>
    <td>NV01</td>
    <td><a href="https://www.petrochronology.com/" target="_blank" rel="noopener noreferrer">UCSB</a></td>
    <td>1280 x 1024</td>
    <td>0.430</td>
    <td>15</td>
    <td>3</td>
    <td>154</td>
    <td>23</td>
  </tr>
  <tr>
    <td>NV02</td>
    <td><a href="https://www.petrochronology.com/" target="_blank" rel="noopener noreferrer">UCSB</a></td>
    <td>1280 x 1024</td>
    <td>0.430</td>
    <td>14</td>
    <td>2</td>
    <td>89</td>
    <td>13</td>
  </tr>
  <tr>
    <td>NV03</td>
    <td><a href="https://www.petrochronology.com/" target="_blank" rel="noopener noreferrer">UCSB</a></td>
    <td>1280 x 1024</td>
    <td>0.430</td>
    <td>21</td>
    <td>4</td>
    <td>186</td>
    <td>35</td>
  </tr>
  <tr>
    <td>NV07</td>
    <td><a href="https://www.petrochronology.com/" target="_blank" rel="noopener noreferrer">UCSB</a></td>
    <td>1280 x 1024</td>
    <td>0.430</td>
    <td>18</td>
    <td>3</td>
    <td>272</td>
    <td>44</td>
  </tr>
  <tr>
    <td>NV09</td>
    <td><a href="https://www.petrochronology.com/" target="_blank" rel="noopener noreferrer">UCSB</a></td>
    <td>1280 x 1024</td>
    <td>0.430</td>
    <td>20</td>
    <td>4</td>
    <td>123</td>
    <td>21</td>
  </tr>
  <tr>
    <td>NV10</td>
    <td><a href="https://www.petrochronology.com/" target="_blank" rel="noopener noreferrer">UCSB</a></td>
    <td>1280 x 1024</td>
    <td>0.430</td>
    <td>15</td>
    <td>3</td>
    <td>110</td>
    <td>20</td>
  </tr>
  <tr>
    <td>RC33</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>276 x 276</td>
    <td>1.264</td>
    <td>36</td>
    <td>6</td>
    <td>461</td>
    <td>88</td>
  </tr>
  <tr>
    <td>SP01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>414 x 414</td>
    <td>1.209</td>
    <td>25</td>
    <td>4</td>
    <td>177</td>
    <td>25</td>
  </tr>
  <tr>
    <td>TP01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>300 x 300</td>
    <td>1.669</td>
    <td>29</td>
    <td>5</td>
    <td>169</td>
    <td>24</td>
  </tr>
  <tr>
    <td>V01</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>290 x 290</td>
    <td>1.380</td>
    <td>64</td>
    <td>11</td>
    <td>548</td>
    <td>78</td>
  </tr>
  <tr>
    <td>V09</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>296 x 296</td>
    <td>1.353</td>
    <td>50</td>
    <td>9</td>
    <td>619</td>
    <td>126</td>
  </tr>
  <tr>
    <td>V18</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>222 x 222</td>
    <td>1.346</td>
    <td>34</td>
    <td>6</td>
    <td>486</td>
    <td>61</td>
  </tr>
  <tr>
    <td>V21</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>310 x 310</td>
    <td>1.291</td>
    <td>35</td>
    <td>6</td>
    <td>1023</td>
    <td>171</td>
  </tr>
  <tr>
    <td>V26</td>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>312 x 312</td>
    <td>1.284</td>
    <td>45</td>
    <td>8</td>
    <td>892</td>
    <td>171</td>
  </tr>
</tbody>
</table>

## Training new models for colab_zirc_dims:
This directory contains three Jupyter/Colab notebooks (MaskRCNN_Resnet....ipynb, MaskRCNN_SwinT....ipynb, Centermask2....ipynb) which can be used to train, respectively, new Mask-RCNN-ResNet, Mask-RCNN-Swin-T, and Centermask2-VovNet models on the czd_large dataset. These notebooks are runnable as-is after uploading to Google Drive and running in Google Colab (recommended; higher-end GPUs will vastly speed up training), but can also be downloaded and run in a local Anaconda environment with Jupyter installed. Please refer to directions in the training notebooks themselves for using the resulting models with the colab_zirc_dims processing notebooks.

# Legacy dataset: czd_orig:
This was the original dataset used to train colab_zirc_dims models. It was created through iterative, non-sample-proportionate random selection of spots from our collection of per-shot ALC mosaic sub-image clippings and UCSB per-shot images, combined with manual addition and annotation of empirically chosen images to the train > validation sets. Though manual expansion of the training set (i.e., mission creep) did yield increases in accuracy on both the validation set and the (Leary et al., 2022) colab_zirc_dims testing dataset, the validation set is consequently almost certainly less representative than in the case of the 'czd_large'. We recommend using the much larger czd_large dataset for training new models.

## Download link:
This dataset is hosted directly on GitHub. It can be downloaded from the following link: https://github.com/MCSitar/colab_zirc_dims/raw/main/training%20datasets/legacy_dataset_czd_orig/czd_orig.zip

## Summary table:
<table>
<thead>
  <tr>
    <th>Source facility</th>
    <th>Training set images</th>
    <th>Validation set images</th>
    <th>Training set annotations</th>
    <th>Validation set annotations</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://sites.google.com/laserchron.org/arizonalaserchroncenter/home" target="_blank" rel="noopener noreferrer">ALC</a></td>
    <td>94</td>
    <td>19</td>
    <td>1375</td>
    <td>341</td>
  </tr>
  <tr>
    <td><a href="https://www.petrochronology.com/" target="_blank" rel="noopener noreferrer">UCSB</a></td>
    <td>18</td>
    <td>12</td>
    <td>180</td>
    <td>139</td>
  </tr>
  <tr>
    <td rowspan="2"><b>Total</b></td>
    <td><b>112</b></td>
    <td><b>31</b></td>
    <td><b>1555</b></td>
    <td><b>480</b></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>143</b></td>
    <td align="center" colspan="2"><b>2035</b></td>
  </tr>
</tbody>
</table>

## References:
Leary, R. J., Umhoefer, P., Smith, M. E., Smith, T. M., Saylor, J. E., Riggs, N., Burr, G., Lodes, E., Foley, D., Licht, A., Mueller, M. A., and Baird, C.: Provenance of Pennsylvanian–Permian sedimentary rocks associated with the Ancestral Rocky Mountains orogeny in southwestern Laurentia: Implications for continental-scale Laurentian sediment transport systems, Lithosphere, 12, 88–121, https://doi.org/10.1130/L1115.1, 2020.

Leary, R. J., Smith, M. E., and Umhoefer, P.: Grain-Size Control on Detrital Zircon Cycloprovenance in the Late Paleozoic Paradox and Eagle Basins, USA, J. Geophys. Res. Solid Earth, 125, e2019JB019226, https://doi.org/10.1029/2019JB019226, 2020.
