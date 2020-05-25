# **Deep Lighting Environment Map Estimation from Spherical Panoramas**

[![Paper](http://img.shields.io/badge/paper-arxiv-critical.svg?style=plastic)](https://arxiv.org/abs/2005.08000)
[![Conference](http://img.shields.io/badge/CVPR-2020-blue.svg?style=plastic)](http://cvpr2020.thecvf.com/)
[![Workshop](http://img.shields.io/badge/OmniCV-2020-lightblue.svg?style=plastic)](https://sites.google.com/view/omnicv-cvpr2020/home)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/DeepPanoramaLighting/)

## **Code and Trained Models**

This repository contains inference code and models for the paper Deep Lighting Environment Map Estimation from Spherical Panoramas ([link](https://arxiv.org/abs/2005.08000)).


## Requirements
The code is based on PyTorch and has been tested with Python 3.7 and CUDA 10.0.
We recommend setting up a virtual environment (follow the `virtualenv` documentation) for installing PyTorch and the other necessary Python packages.
Once your environment is set up and activated, install the necessary packages:

`pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html`

## Inference
You can download pre-trained models from [here](https://drive.google.com/open?id=1wr3ljh6EFGRa8VdZ8f2mYVsM2BpSDmqS), which includes pre-trained LDR-to-HDR autoencoder and Lighting Encoder. Please put the extracted files under `models` and run:

`python inference.py`

The following flags specify the required parameters.
- `--input_path`: Specifies the path of the input image.
- `--out_path`: Specifies the file of the output path. 
-  `--deringing`: Enable/disable low pass deringing filter for the predicted SH coefficients.


