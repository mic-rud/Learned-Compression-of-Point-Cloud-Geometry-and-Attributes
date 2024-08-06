# Learned Compression of Point Cloud Geometry and Attributes in a Single Model through Multimodal Rate-Control 

[![arXiv](https://img.shields.io/badge/arXiv-2408.00599-blue.svg)](https://arxiv.org/abs/2408.00599)


## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Installation](#installation)
- [Running the Code](#running-the-code)
- [Dataset](#dataset)
- [Experiments](#experiments)
- [Citation](#citation)

## Overview

This repository accompanies our paper on Learned Compression of Point Cloud Geometry and Attributes in a Single Model through Multimodal Rate-Control. 
The main contribution is the usage of a joint autoencoder to compress the point cloud geometry and attributes in a single autoencoder model. 
As this would result in training an ensemble of models (one for each trade-off between rate, geometry quality and attribute quality), we propose to condition the model on the trade-off between the aformentioned triplet at training time, resulting in a single trained model. 
In contrast to the standard approach for point cloud compression, which requires separate models for geometry and attributes, as well as decoding the geometry for attribute re-projection at the sender, we believe our approach offers more flexibility and reduced encoding latency with a comparable rate-distortion trade-off.

## Methodology

### Multimodal Conditioning

TODO

### Architecture

TODO

## Results

### Pareto-Fronts
Pareto-Fronts are optimized by grid-searching possible combinations between geometry and attribute quality. Thus, they are optimized per-content. 

<p float="left">
  <img src="plot/figures/all/rd-pareto_pcqm_longdress.pdf" width="100" />
  <img src="plot/figures/all/rd-pareto_pcqm_soldier.pdf" width="100" />
  <img src="plot/figures/all/rd-pareto_pcqm_redandblack.pdf" width="100" />
  <img src="plot/figures/all/rd-pareto_pcqm_loot.pdf" width="100" />
</p>

![](plot/resources/rd-pareto_pcqm_longdress.png "longdress")  ![](plot/resources/rd-pareto_pcqm_soldier.png "soldier") ![](plot/resources/rd-pareto_pcqm_redandblack.png "redandblack") ![](plot/resources/rd-pareto_pcqm_loot.png "loot") 

### Generalized Compression (Lossy Geometry, Lossy Attributes)
We compare against G-PCC and V-PCC for the lossy-geometry, lossy-attribute compression use case. 
For this, we select 4 configuration pairs of our model, allowing to consistently outperfom G-PCC and perform on-par with V-PCC.

![](plot/resources/rd-config_pcqm_longdress.png "longdress")  ![](plot/resources/rd-config_pcqm_soldier.png "soldier") ![](plot/resources/rd-config_pcqm_redandblack.png "redandblack") ![](plot/resources/rd-config_pcqm_loot.png "loot") 

![](plot/resources/rd-config_sym_p2p_psnr_longdress.png "longdress")  ![](plot/resources/rd-config_sym_p2p_psnr_soldier.png "soldier") ![](plot/resources/rd-config_sym_p2p_psnr_redandblack.png "redandblack") ![](plot/resources/rd-config_sym_p2p_psnr_loot.png "loot") 

![](plot/resources/rd-config_sym_y_psnr_longdress.png "longdress")  ![](plot/resources/rd-config_sym_y_psnr_soldier.png "soldier") ![](plot/resources/rd-config_sym_y_psnr_redandblack.png "redandblack") ![](plot/resources/rd-config_sym_y_psnr_loot.png "loot") 




### Latency
On a NVIDIA RTX 4090, our model achieves significantly faster compression then G-PCC and V-PCC as well as the learning-based YOGA.
Note that the results for YOGA are obtained from the original paper on different hardware, as it is not yet open-sourced.

### Visual Results



## Usage


### Setup
```
    # Python
    python -m venv .env
    python -m pip install -r requirements.txt

    # Metrics
    git clone https://git.uni-due.de/ncs/research/pointclouds/metrics.git
    git clone https://git.uni-due.de/ncs/research/pointclouds/pointcloud-data.git data

    # Open3D
    sudo apt-get install libosmesa6-dev
    mkdir dependencies & cd dependencies
    git clone https://github.com/isl-org/Open3D

    cd Open3D
    util/install_deps_ubuntu.sh

    mkdir build && cd build

    cmake -DENABLE_HEADLESS_RENDERING=ON \
                    -DBUILD_GUI=OFF \
                    -DBUILD_WEBRTC=OFF \
                    -DUSE_SYSTEM_GLEW=OFF \
                    -DUSE_SYSTEM_GLFW=OFF \
                    ..

    make -j$(nproc)
    make install-pip-package

    # PCQM
    git clone https://github.com/MEPP-team/PCQM.git
    mkdir PCQM/build && cd PCQM/build
    cmake ..
    make
```

G-PCC
```
cd dependencies
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
cd mpeg-pcc-tmc13
mkdir build && cd build
cmake ..
make
```
V-PCC
```
cd dependencies
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc2.git
cd mpeg-pcc-tmc2 && ./build.sh
```

V-PCC is also required for the computation of metrics. 


### Training


### Evaluation


## Citation

If you find our work helpful, please consider citing us in your work:
```
@article{rudolph2024learnedcompressionpointcloud,
      title={Learned Compression of Point Cloud Geometry and Attributes in a Single Model through Multimodal Rate-Control}, 
      author={Michael Rudolph and Aron Riemenschneider and Amr Rizk},
      journal={arXiv preprint arXiv:2408.00599},
      year={2024},
}
```