## Towards Explainable Multi-modal Motion Prediction using Graph Representations
[![DOI](https://zenodo.org/badge/553454432.svg)](https://zenodo.org/badge/latestdoi/553454432)


This repository contains code for ["Towards Explainable Multi-modal Motion
Prediction using Graph Representations"]() by Sandra Carrasco, Sylwia Majchrowska, Joakim Johnander, Christoffer Petersson and David Fernández LLorca, presented at .. 2022.  

![](https://github.com/sancarlim/Explainable-MP/blob/main/assets/readme.gif)

```bibtex
citation
```
Note: This repository is based on [PGP repository](https://github.com/nachiket92/PGP/tree/main/)
 

## Installation

1. Clone this repository 

2. Set up a new conda environment 
``` shell
conda create --name xscout python=3.7.10
```

3. Install dependencies
```shell
conda activate xscout

# nuScenes devkit
pip install nuscenes-devkit

# Pytorch: The code has been tested with Pytorch 1.7.1, CUDA 10.1, but should work with newer versions
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# Additional utilities
pip install ray
pip install psutil
pip install scipy
pip install positional-encodings
pip install imageio
pip install tensorboard
pip install dgl-cu101
```


## Dataset

1. Download the [nuScenes dataset](https://www.nuscenes.org/download). For this project we just need the following.
    - Metadata for the Trainval split (v1.0)
    - Map expansion pack (v1.3)

2. Organize the nuScenes root directory as follows
```plain
└── nuScenes/
    ├── maps/
    |   ├── basemaps/
    |   ├── expansion/
    |   ├── prediction/
    |   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
    |   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
    |   ├── 53992ee3023e5494b90c316c183be829.png
    |   └── 93406b464a165eaba6d9de76ca09f5da.png
    └── v1.0-trainval
        ├── attribute.json
        ├── calibrated_sensor.json
        ...
        └── visibility.json         
```

3. Run the following script to extract pre-processed data. This speeds up training significantly.
```shell
python preprocess.py -c configs/preprocess_nuscenes.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data
```
You can download the preprocessed data in [this link](https://drive.google.com/file/d/1Ovf4eX4RtejyhX-hji77MjFjOUwTIdbH/view?usp=sharing).


## Evaluation

You can download the trained model weights using [this link](https://drive.google.com/file/d/1i9Afa9UhOPAYbjB9nY6D-En0z8HgoEnl/view?usp=sharing).

To evaluate on the nuScenes val set run the following script. This will generate a text file with evaluation metrics at the specified output directory. The results should match the [benchmark entry](https://eval.ai/web/challenges/challenge-page/591/leaderboard/1659) on Eval.ai. 
```shell
python evaluate.py -c configs/xscout_pgp.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -w path/to/trained/weights
```

## Visualization

To visualize predictions run the following script. This will generate gifs for a set of instance tokens (track ids) from nuScenes val at the specified output directory.  
```shell
python visualize.py -c configs/xscout_pgp.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -w path/to/trained/weights 
``` 
You can indicate the number of modes and future temporal horizon to visualize with ```--num_modes``` and ```--tf``` respectively.


## Training

To train the model from scratch, run
```shell
python train.py -c configs/xscout_pgp.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -n 100
```

The training script will save training checkpoints and tensorboard logs in the output directory. Wandb logger is also supported. You need to specify the entity and project in the ```wandb.init``` function in ```train.py```. If you do not want to log in wandb, please use ```--nowandb``` argument.
 
To launch tensorboard, run
```shell
tensorboard --logdir=path/to/output/directory/tensorboard_logs
```

## Robustness analysis

This repository contains the code to reproduce the robustness analysis (Section IV) presented in ["Towards Trustworthy Multi-Modal Motion Prediction: Evaluation and Interpretability"]() by Sandra Carrasco, Sylwia Majchrowska,Joakim Johnander, Christoffer Petersson and David Fernández LLorca, presented at .. 2022. 

You can download the PGP trained model weights using [this link](https://drive.google.com/file/d/1i9Afa9UhOPAYbjB9nY6D-En0z8HgoEnl/view?usp=sharing).

To evaluate on the nuScenes val set, you can indicate the probability of randomly masking out dynamic objects and/or lanes in ```agent_mask_p_veh```, ```agent_mask_p_ped``` and ```lane_mask_prob``` arguments in the configuration file ```configs/pgp_gatx2_lvm_traversal.yml``` . Indicate a probability of masking out random frames of interacting agents using ```mask_frames_p``` argument. 

```shell
python evaluate.py -c configs/pgp_gatx2_lvm_traversal.yml -r path/to/nuScenes/root/directory -d path/to/directory/with/preprocessed/data -o path/to/output/directory -w path/to/trained/weights
```

