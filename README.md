# DualHead-Network
PyTorch implementation of "Learning Multi-Granular Spatio-Temporal Graph Network for
Skeleton-based Action Recognition" in ACM Multimedia 2021.


## Dependencies

- Python >= 3.6
- PyTorch >= 1.2.0
- PyYAML, tqdm, tensorboardX

## Data Preparation

*Disk usage warning: after preprocessing, the total sizes of datasets are around 38GB, 77GB, 63GB for NTU RGB+D 60, NTU RGB+D 120, and Kinetics 400, respectively. The raw/intermediate sizes may be larger.*

### Download Datasets

There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- Kinetics 400 Skeleton

#### NTU RGB+D 60 and 120

1. Request dataset here: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp

2. Download the skeleton-only datasets:
   - `nturgbd_skeletons_s001_to_s017.zip`  (NTU RGB+D 60)
   - `nturgbd_skeletons_s018_to_s032.zip`  (NTU RGB+D 120, on top of NTU RGB+D 60)
   - Total size should be 5.8GB + 4.5GB.

3. Download missing skeletons lookup files [from the authors' GitHub repo](https://github.com/shahroudy/NTURGB-D#samples-with-missing-skeletons):
   - NTU RGB+D 60 Missing Skeletons:
     `wget https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt`

   - NTU RGB+D 120 Missing Skeletons:
     `wget https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt`

   - Remember to remove the first few lines of text in these files!

#### Kinetics Skeleton 400

1. Download dataset from ST-GCN repo: https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton
2. [This](https://silicondales.com/tutorials/g-suite/how-to-wget-files-from-google-drive/) might be useful if you want to `wget` the dataset from Google Drive

### Data Preprocessing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - kinetics_raw/
    - kinetics_train/
      ...
    - kinetics_val/
      ...
    - kinetics_train_label.json
    - keintics_val_label.json
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
    - NTU_RGBD_samples_with_missing_skeletons.txt
    - NTU_RGBD120_samples_with_missing_skeletons.txt
```

#### Generating Data

1. NTU RGB+D
   - `cd data_gen`
   - `python3 ntu_gendata.py`
   - `python3 ntu120_gendata.py`

[comment]: <> (   - Time estimate is ~ 3hrs to generate NTU 120 on a single core &#40;feel free to parallelize the code :&#41;&#41;)

2. Kinetics
   - `python3 kinetics_gendata.py`

[comment]: <> (   - ~ 70 mins to generate Kinetics data)

3. Generate the bone data with:
   - `python gen_bone_data.py --dataset ntu`
   - `python gen_bone_data.py --dataset ntu120`
   - `python gen_bone_data.py --dataset kinetics`

[comment]: <> (   - )
   
4. Generate the motion data with:
   - `python gen_motion_data.py --dataset ntu`
   - `python gen_motion_data.py --dataset ntu120`
   - `python gen_motion_data.py --dataset kinetics`







## Acknowledgements

This repo is based on
- [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
- [ST-GCN](https://github.com/yysijie/st-gcn)
- [MS-G3D](https://github.com/kenziyuliu/MS-G3D)


Thanks to the original authors for their work!


## Citation

Please cite this work if you find it useful:

```
@inproceedings{chen2021dualhead,
title = {Learning Multi-Granular Spatio-Temporal Graph Network for Skeleton-Based Action Recognition},
author = {Chen, Tailin and Zhou, Desen and Wang, Jian and Wang, Shidong and Guan, Yu and He, Xuming and Ding, Errui},
booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
pages = {4334–4342},
year = {2021},
}
```