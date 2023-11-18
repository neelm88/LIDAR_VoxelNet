import pytorch_lightning as pl


from model import Model
from config import cfg  # Assuming the configuration is defined in config.py
import torch 

# Define the parameters and device
params = params = {
        "n_epochs":     15,
        "batch_size":   2,  # Set to 2 for the fastest training. (Training time exponentially grows with batch size). Batch Size > 16 causes GPU memory issues for me.
        "small_addon_for_BCE": 1e-6,
        "max_gradient_norm": 5, 
        "alpha_bce":    1.5,
        "beta_bce":     1.0,
        "learning_rate": 0.001,
        "mode":         "train",
        "dump_vis":     "no",
        "data_root_dir": "../",
        "model_dir":    "model",
        "model_name":   "model1",
        "num_threads":  8
    }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_instance = Model(cfg, params, device)



from pytorch_lightning.callbacks import ModelCheckpoint
from train2 import VoxelNetPL


loaded_model = VoxelNetPL.load_from_checkpoint(
    "../LIDAR_VoxelNet/epoch=2-step=300.ckpt",
    model=model_instance
)

# TODO:

#Taking a point cloud/image pair,
# run the model on the point cloud to get the r_map and p_map,
# figure out how to transform that to a bounding box

# also, use for the drawing boxes, utils from skyhehe are good https://github.com/skyhehe123/VoxelNet-pytorch


# steps:
# load checkpoint, call forward on model.py model class to get r_map and p_map. use skyhehe's utils to get bounding boxes
# use the dataloader to get a point cloud and image pair

