# cryoCARE

This package is a memory efficient implementation of [cryoCARE](https://github.com/juglab/cryoCARE_T2T). 

This setup trains a denoising U-Net for tomographic reconstruction according to the [Noise2Noise](https://arxiv.org/pdf/1803.04189.pdf) training paradigm. 
Therefore the user has to provide two tomograms of the same sample. 
The simplest way to achieve this is with direct-detector movie-frames.

You can use Warp to generate two reconstructed tomograms based on the even/odd frames. Alternatively, the movie-frames can be split in two halves (e.g. with MotionCor2 `-SplitSum 1` or with IMOD `alignframes -debug 10000`) from which two identical, up to random noise, tomograms can be reconstructed. 

These two (even and odd) tomograms can be used as input to this cryoCARE implementation.

## Changelog

### Version 0.2

* `cyroCARE_train` produces a compressed and more portable model. This model can be copied and shared with others without relying on a certain folder structure.
* `cryoCARE_predict` supports to predict multiple tomograms in one run. Streamlined configuration with respect to the changes of `cryoCARE_train`.
* Streamlined installation instructions
* CUDA 11 support
* Minor changes/ fixed couple of bugs:
    * Proper padding of tomograms to avoid black frames in the denoised tomograms
    * Fix computation of validation cut off for small tomograms
    * Fix `cryoCARE_predict` if no tiling happens

## Installation

__Note:__ We assume that you have  [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

First you need to create a conda environment.

### For CUDA 11:
```
conda create -n cryocare_11 python=3.8 cudatoolkit=11.0 cudnn=8.0 -c conda-forge
conda activate cryocare_11
pip install tensorflow==2.4
pip install cryoCARE
```

### For CUDA 10:
```
conda create -n cryocare -c conda-forge -c anaconda python=3 keras-gpu=2.3.1
conda activate cryocare
pip install cryoCARE
```

## Manual
cryoCARE uses `.json` configuration files and is run in three steps:

### 1. Prepare Training Data
To prepare the training data we have to provide all tomograms on which we want to train. 
Create an empty file called `train_data_config.json`, copy-paste the following template and fill it in.
```
{
  "even": [
    "/path/to/even.rec"
  ],
  "odd": [
    "/path/to/odd.rec"
  ],
  "patch_shape": [
    72,
    72,
    72
  ],
  "num_slices": 1200,
  "split": 0.9,
  "tilt_axis": "Y",
  "n_normalization_samples": 500,
  "path": "./"
}
```
#### Parameters:
* `"even"`: List of all even tomograms.
* `"odd"`: List of all odd tomograms. Note the order has to be the same as in `"even"`.
* `"patch_shape"`: Size of the sub-volumes used for training. Should not be smaller than `64, 64, 64`.
* `"num_slices"`: Number of sub-volumes extracted per tomograms. 
* `"tilt_axis"`: Tilt-axis of the tomograms. We split the tomogram along this axis to extract train- and validation data separately.
* `"n_normalization_samples"`: Number of sub-volumes extracted per tomograms, which are used to compute `mean` and `standard deviation` for normalization.
* `"path"`: The training and validation data are saved here.

#### Run Training Data Preparation:
After installation of the package we have access to built in Python-scripts which we can call. 
To run the training data preparation we run the following command:
`cryoCARE_extract_train_data.py --conf train_data_config.json`

### 2. Training
Create an empty file called `train_config.json`, copy-paste the following template and fill it in.
```
{
  "train_data": "./",
  "epochs": 100,
  "steps_per_epoch": 200,
  "batch_size": 16,
  "unet_kern_size": 3,
  "unet_n_depth": 3,
  "unet_n_first": 16,
  "learning_rate": 0.0004,
  "model_name": "model_name",
  "path": "./",
  "gpu_id": 0
}
```

#### Parameters:
* `"train_data"`: Path to the directory containing the train- and validation data. This should be the same as the `"path"` from above.
* `"epochs"`: Number of epochs used to train the network.
* `"steps_per_epoch"`: Number of gradient steps performed per epoch.
* `"batch_size"`: Used training batch size.
* `"unet_kern_size"`: Convolution kernel size of the U-Net. Has to be an odd number.
* `"unet_n_depth"`: Depth of the U-Net.
* `"unet_n_first"`: Number of initial feature channels.
* `"learning_rate"`: Learning rate of the model training.
* `"model_name"`: Name of the model.
* `"path"`: Output path for the model.
* `"gpu_id"`: This is optional. Provide the GPU ID(s) of the GPUs you wish to use.

#### Run Training:
To run the training we run the following command:
`cryoCARE_train.py --conf train_config.json`

You will find a `.tar.gz` file in the directory you specified as `path`. This your model an will be used in the next step.

### 3. Prediction
Create an empty file called `predict_config.json`, copy-paste the following template and fill it in.
```
{
  "path": "path/to/your/model/model_name.tar.gz",
  "even": "/path/to/even.rec",
  "odd": "/path/to/odd.rec",
  "n_tiles": [1,1,1],
  "output": "denoised.rec",
  "overwrite": False,
  "gpu_id": 0
}
```

#### Parameters:
* `"path"`: Path to your model file.
* `"even"`: Path to directory with even tomograms or a specific even tomogram or a list of specific even tomograms.
* `"odd"`: Path to directory with odd tomograms or a specific odd tomogram or a list of specific odd tomograms in the same order as the even tomograms.
* `"n_tiles"`: Initial tiles per dimension. Gets increased if the tiles do not fit on the GPU.
* `"output"`: Path where the denoised tomograms will be written.
* `"overwrite"`: Allow previous files to be overwritten.
* `"gpu_id"`: This is optional. Provide the GPU ID(s) of the GPUs you wish to use.

#### Run Prediction:
To run the training we run the following command:
`cryoCARE_predict.py --conf predict_config.json`

## How to Cite
```
@inproceedings{buchholz2019cryo,
  title={Cryo-CARE: content-aware image restoration for cryo-transmission electron microscopy data},
  author={Buchholz, Tim-Oliver and Jordan, Mareike and Pigino, Gaia and Jug, Florian},
  booktitle={2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019)},
  pages={502--506},
  year={2019},
  organization={IEEE}
}

@article{buchholz2019content,
  title={Content-aware image restoration for electron microscopy.},
  author={Buchholz, Tim-Oliver and Krull, Alexander and Shahidi, R{\'e}za and Pigino, Gaia and J{\'e}kely, G{\'a}sp{\'a}r and Jug, Florian},
  journal={Methods in cell biology},
  volume={152},
  pages={277--289},
  year={2019}
}
```
