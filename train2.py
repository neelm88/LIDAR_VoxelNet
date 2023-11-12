import torch
import lightning.pytorch as pl
import os
from data import create_data_loader
from model import Model
from config import cfg
from sklearn.preprocessing import LabelEncoder


class VoxelNetPL(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def training_step(self, data, step):
        feature_buffer      = data["feature_buffer"]
        coordinate_buffer   = data["coordinate_buffer"]
        targets             = data["targets"]
        pos_equal_one       = data["pos_equal_one"]
        pos_equal_one_reg   = data["pos_equal_one_reg"]
        pos_equal_one_sum   = data["pos_equal_one_sum"]
        neg_equal_one       = data["neg_equal_one"]
        neg_equal_one_sum   = data["neg_equal_one_sum"]

        p_map, r_map = self.model(feature_buffer, coordinate_buffer)
        loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = self.model.loss_object(
            r_map, p_map, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)

        '''
        preds = [{
            "boxes":    r_map[frame],
            "scores":   p_map[frame].max(2)[0],
            "labels":   p_map[frame].max(2)[1]
        } for frame in range(p_map.shape[0]) ]
        
        tgts = [{ 
            "boxes":    targets[frame],
            "labels":   pos_equal_one[frame]
        } for frame in range(p_map.shape[0]) ]
        for m in self.metrics.values():
            m.update(preds, tgts)
        '''
        self.log("training_loss", loss, batch_size=self.model.params["batch_size"])
        return loss

    def validation_step(self, data, step):
        feature_buffer      = data["feature_buffer"]
        coordinate_buffer   = data["coordinate_buffer"]
        targets             = data["targets"]
        pos_equal_one       = data["pos_equal_one"]
        pos_equal_one_reg   = data["pos_equal_one_reg"]
        pos_equal_one_sum   = data["pos_equal_one_sum"]
        neg_equal_one       = data["neg_equal_one"]
        neg_equal_one_sum   = data["neg_equal_one_sum"]

        p_map, r_map = self.model(feature_buffer, coordinate_buffer)
        loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = self.model.loss_object(
            r_map, p_map, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)
        self.log("validation_loss", loss, batch_size=self.model.params["batch_size"])
       # breakpoint()
        return loss


    def configure_optimizers(self):
        return [self.model.lr_scheduler.optimizer], [self.model.lr_scheduler]
    


def train_experiment_debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        "model_name":   "model1",
        "num_threads":  8
    }

    cfg["DATA_DIR"] = params["data_root_dir"]
    cfg["CALIB_DIR"] = os.path.join(cfg["DATA_DIR"], "training/calib")
    label_encoder = LabelEncoder()
    
    train_loader, val_loader = create_data_loader(cfg, params, 16, "train", is_aug_data=True, label_encoder=label_encoder, create_anchors=True)

    model = Model(cfg, params, device)
    model = VoxelNetPL(model)
    checkpoint_dir = os.path.join("./", params["model_dir"], params["model_name"])
    trainer = pl.Trainer(max_epochs=params["n_epochs"], default_root_dir=checkpoint_dir)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    

if __name__ == '__main__':
    train_experiment_debug()