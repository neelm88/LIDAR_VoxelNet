# LIDAR VoxelNet

Implementation of VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection in pytorch 2.1.0
Referenced the following repo (essentially kept everything close to the original, but with pytorch instead of tensorflow): https://github.com/gkadusumilli/Voxelnet/tree/master
This repo is also a useful resoruce for debugging: https://github.com/skyhehe123/VoxelNet-pytorch
Link to the paper: https://arxiv.org/abs/1711.06396
## Environment Setup
```
conda env create -f env/conda_env.yml
conda activate lidar_voxelnet
```
I also ran the following to install pytorch with cuda 11.8. (see here for OS specific command: https://pytorch.org/get-started/locally/)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If the above does not work, I placed my pip freeze in the env/requirements.txt file with python version 3.9.18

It may be helpful to run the following command:
```
pip install lightning[extra]
```
Now, run the following:
```
python setup.py build_ext --inplace
```

To train the model run the following command
python train.py

To see the tensorboard, run the following command (replace params accordingly):
```
tensorboard --logdir=params["model_dir"]/params["model_name"]/lightning_logs/
```
If you run into the following issue:
```
ValueError: Duplicate plugins for name projector
```
Run this:
```
pip uninstall tb-nightly tensorboardX tensorboard
pip install tensorboard
```

Make sure the velodyne data is in the following location (or change params["data_root_dir"] accordingly):
```
└── DATA_DIR
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_2
       |   └── velodyne
       └── validation  <--- evaluation data 
       |   ├── image_2
       |   ├── label_2
       |   └── velodyne
       └── LIDAR_VoxelNet
```
To split the training and evaluation data, you can follow these steps: https://github.com/gkadusumilli/Voxelnet/blob/master/VoxelNet_data_creation.ipynb
(Evaluation code has yet to be written.)
