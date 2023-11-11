import torch
import time
import lightning.pytorch as pl
import os
from data import create_data_loader
from model import Model
from config import cfg
from sklearn.preprocessing import LabelEncoder
#import matplotlib.pyplot as plt

class VoxelNetPL(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
    
    def training_step(self, data, step):
        start_time = time.time()
        feature_buffer      = data["feature_buffer"]
        coordinate_buffer   = data["coordinate_buffer"]
        targets             = data["targets"]
        pos_equal_one       = data["pos_equal_one"]
        pos_equal_one_reg   = data["pos_equal_one_reg"]
        pos_equal_one_sum   = data["pos_equal_one_sum"]
        neg_equal_one       = data["neg_equal_one"]
        neg_equal_one_sum   = data["neg_equal_one_sum"]

        all_loss = self.model.train_step(feature_buffer, coordinate_buffer, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)
        loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = all_loss
        end_time = time.time()
        self.log("training_loss", loss, batch_size=self.model.params["batch_size"])
        # print(f"Train step {step} with loss {loss} in {end_time - start_time} seconds")
        return loss

    def validation_step(self, data, step):
        start_time = time.time()
        feature_buffer      = data["feature_buffer"]
        coordinate_buffer   = data["coordinate_buffer"]
        targets             = data["targets"]
        pos_equal_one       = data["pos_equal_one"]
        pos_equal_one_reg   = data["pos_equal_one_reg"]
        pos_equal_one_sum   = data["pos_equal_one_sum"]
        neg_equal_one       = data["neg_equal_one"]
        neg_equal_one_sum   = data["neg_equal_one_sum"]

        all_loss = self.model.validate_step(feature_buffer, coordinate_buffer, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)
        loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = all_loss
        end_time = time.time()
        self.log("validation_loss", loss, batch_size=self.model.params["batch_size"])
        # print(f"Validation step {step} with loss {loss} in {end_time - start_time} seconds")
        return loss

    def configure_optimizers(self):
        return [self.model.lr_scheduler.optimizer], [self.model.lr_scheduler]
    


def train_experiment_debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device: {device}")
    params = {
        "n_epochs":     15,
        "batch_size":   2,  # Set to 2 for the fastest training. (Training time exponentially grows with batch size). Batch Size > 16 causes GPU memory issues for me.
        "small_addon_for_BCE": 1e-6,
        "max_gradient_norm": 5, # TODO: See if I can make use of this.
        "alpha_bce":    1.5,
        "beta_bce":     1.0,
        "learning_rate": 0.001,
        "mode":         "train",
        "dump_vis":     "no",
        "data_root_dir": "../",
        "model_dir":    "model",
        "model_name":   "model2",
        "num_threads":  8
    }
    cfg["DATA_DIR"] = params["data_root_dir"]
    cfg["CALIB_DIR"] = os.path.join(cfg["DATA_DIR"], "training/calib")
    label_encoder = LabelEncoder()
    
    train_loader = create_data_loader(cfg, params, 16, "train", is_aug_data=True, label_encoder=label_encoder, create_anchors=True, persistent_workers=True)
    val_batcher = create_data_loader(cfg, params, 16, "eval", is_aug_data=False, label_encoder=label_encoder, create_anchors=False, persistent_workers=True)
    model = Model(cfg, params, device)
    model = VoxelNetPL(model)
    checkpoint_dir = os.path.join("./", params["model_dir"], params["model_name"])
    trainer = pl.Trainer(limit_val_batches=500, max_epochs=params["n_epochs"], devices=1, accelerator=str(device), default_root_dir=checkpoint_dir)
    print("Trainer created, starting training now.")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_batcher)
    #breakpoint()

if __name__ == '__main__':
    train_experiment_debug()