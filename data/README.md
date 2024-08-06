# PointCloudData

A repository for dataloading and preparation for Point Cloud datasets


## Usage

TODO: How to use the repo
### Downloading raw data
Raw Point Clouds need to be downloaded before preparing the dataset.
We prepared a utility to download raw point clouds from official sources in (download_raw_pointclouds.py)[download_raw_pointclouds.py]. This will fill your disk, as the raw data is approximately ~GB.

To download all data to default locations (./data/raw)
```
    python download_raw_pointclouds.py
```

However, you might configure the downloading to all needs using the following arguments:
```
    python download_raw_pointclouds.py 
        --config <config_path>      # Path to the download_links for all datasets
        --data_path <data_path>     # Path to unpack raw data
        --tmp_path <tmp_path>       # Path to temporaly downlaod files
        --datasets "<datasets>"     # Datasets to download, default: "mvub, 8iVFBv2, uvg-vpc". Comma-separated list
```

### Data Preparation
Dataset preparation can be performed with [prepare_dataset.py](prepare_dataset.py). 
This allows to compile a dataset from

A configuration file is required. A sample can be found in [dataset_dev.yaml](configs/dataset_dev.yaml).
We allow the following syntax:
```
info:
    block_size: [blockX, blockY, blockZ]    #Size of Blocks
train:
    <sequence_name1>:                       # Name of the sequence
        "<frameIndices>"                    # Indices as string
    <sequence_name2>: 
        "<frameIndices>"
    ...
val:
    ...
test:
    ...
```

The following syntax for frameIndexing is allowed
```
"1" #Single frame with index 1
"1, 3, 6, 9"    #Frames with indices 1, 3, 6 and 9
"1:10"          #All frames from 1 to 10
"1:10:2:        #Frames 1, 3, 5, 7, 9 (Frames 1 to 10 with stride 2)
```

To prepare a dataset, call:
```
    python prepare_dataset.py --config_path ./data/configs/dataset_dev.yaml
```

However, you might configure the downloading to all needs using the following arguments:
```
    python prepare_dataset.py 
        --config_path <config_path>     # Path to the dataset configuration
        --dataset_path <data_path>      # Path to place the dataset in
        --raw_config <raw_config>       # Raw configuration file holding info on sequences
        --raw_data_path <path>          # Path to the raw data downloaded with the download utility
```


### Static Point Clouds


[1] Alexiou, E., Viola, I., Borges, T., Fonseca, T., De Queiroz, R., & Ebrahimi, T. (2019). A comprehensive study of the rate-distortion performance in MPEG point cloud compression. APSIPA Transactions on Signal and Information Processing, 8, E27. doi: 10.1017/ATSIP.2019.20
### Dynamic Point Clouds
TODO

## Datasets

### Static Custom Dataset

Static colored dataset as used in [2],[3]

Write how to download the dataset and how the Datastructure looks like @Aron
| Name                  | Link  | Acquired  | Used in   |Structure
| ------                | ------|------     |----       |--------
|8iVFB                  |[1],[4]|yes        |[3]        |4 sequences 1024³ RGB
|MVUB                   |[1]    |yes        |[2], [3]   |10 sequences 1024³ and 512³ RGB  
|MPEG                   |[8]?   |no         |[2], [3]   |
|JPEG                   |       |no         |[2], [3]   |
|8iVSLF                 |[5]    |yes        |           |1 sequence, 6 PC 4096³ RGB
|Owlii                  |[6]    |yes        |           |4 sequences 2048³ mesh (texel sampling?)
|PointXR                |[7]    |no         |[3]        |20 PC 4096³




[1]https://box.nju.edu.cn/f/5ab2aa4dfd9941f5aaae/

[2]A. F. R. Guarda, N. M. M. Rodrigues and F. Pereira, "Adaptive Deep Learning-Based Point Cloud Geometry Coding," in IEEE Journal of Selected Topics in Signal Processing, vol. 15, no. 2, pp. 415-430, Feb. 2021, doi: 10.1109/JSTSP.2020.3047520.
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9309023

[3]https://infoscience.epfl.ch/record/279585

[4]http://plenodb.jpeg.org/pc/8ilabs

[5]https://mpeg-pcc.org/index.php/pcc-content-database/8i-voxelized-surface-light-field-8ivslf-dataset/

[6]https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/

[7]https://www.epfl.ch/labs/mmspg/downloads/pointxr/

[8]http://mpegfs.int-evry.fr/MPEG/PCC/DataSets/pointCloud/CfP/



### 8i Voxelized Dynamic Human Bodies
TODO
