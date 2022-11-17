# Advanced local installation:
The colab_zirc_dims package will work outside of Google Colab, and its processing notebooks can be run as Jupyter notebooks in a local Anaconda environment.
## Local installation:
1.  [Install Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
2.  Open Anaconda
3.  (Optional but recommended) [Create and activate a new virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
4.  [Install an appropriate version of CUDA-equipped Pytorch](https://pytorch.org/). Please refer to Pytorch documentation as to CUDA versions, etc. as these will depend on your device.
5.  Install opencv by running the command ```pip install opencv-python```
6.  [Install Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md). Follow [these directions](https://medium.com/@yogeshkumarpilli/how-to-install-detectron2-on-windows-10-or-11-2021-aug-with-the-latest-build-v0-5-c7333909676f) for Windows installation.
  6.1. If Detectron2 fails to build, try installing ninja (```pip install ninja```), av (```pip install av```) and, for Windows, pywin32 (```conda install -c anaconda pywin32```), then try building it again
7.  Install colab_zirc_dims by running the command ```pip install colab-zirc-dims```
8.  Install jupyter (```conda install -c anaconda jupyter```)
9.  Install ipywidgets (```conda install ipywidgets```)
10.  Activate ipywidgets for Jupyter with the command ```jupyter nbextension enable --py widgetsnbextension```

## Running notebooks locally:
1.  Find and download the desired notebook from [this repository](https://github.com/MCSitar/colab_zirc_dims/tree/main/notebook%20copies).
2.  Move the notebook to a an empty directory somewhere in the path for your Anaconda installation.
3.  Open Jupyter Notebook from the Anaconda Navigator panel, then open and run the notebook as per the instructions therein.

## Running colab_zirc_dims notebooks without an internet connection:
It *should* be possible to run colab_zirc_dims notebooks without an internet connection. This is, however, untested, and does require another, similarly equipped computer with an internet connection. A proposed workflow is as follows:
1.  Set up colab_zirc_dims on a computer with similar hardware and software to the target computer by following the 'Local installation' instructions above.
2.  Following the 'Running notebooks locally' directions, run the desired notebooks with a sample of your data. This will make sure that any non-included dependencies, models, etc. are downloaded.
3.  Follow the directions in [this StackOverflow post](https://stackoverflow.com/a/55103643) to copy your colab_zirc_dims Anaconda environment to the non-connected computer.
4.  Copy the directory with notebooks +/- downloaded data from the internet-connected computer to the target computer. They should now be runnable without an internet connection.
