import torch
import lightning.pytorch as pl
import os
from utils.utils import delta_to_boxes3d, delta_to_boxes3d_hardcoded, cal_anchors_hardcoded
from data import create_data_loader
from model import Model
from config import cfg
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
#from metrics import AveragePrecisionWrapper

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VoxelNetPL(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.anchors = cal_anchors_hardcoded().detach().to(device)

    def training_step(self, data, step):
        feature_buffer      = data["feature_buffer"]
        coordinate_buffer   = data["coordinate_buffer"]
        targets             = data["targets"]
        pos_equal_one       = data["pos_equal_one"]
        pos_equal_one_reg   = data["pos_equal_one_reg"]
        pos_equal_one_sum   = data["pos_equal_one_sum"]
        neg_equal_one       = data["neg_equal_one"]
        neg_equal_one_sum   = data["neg_equal_one_sum"]
        batch_size          = self.model.params["batch_size"]

        p_map, r_map = self.model(feature_buffer, coordinate_buffer)
        loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = self.model.loss_object(
            r_map, p_map, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)
        self.log("training_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True)

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
        batch_size          = self.model.params["batch_size"]

        p_map, r_map = self.model(feature_buffer, coordinate_buffer)
        loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = self.model.loss_object(
            r_map, p_map, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)
        self.log("validation_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True)

      #  r_map         = delta_to_boxes3d(r_map)
      #  targets       = delta_to_boxes3d(targets)
        p_map         = p_map.reshape(        batch_size, -1, 1 * self.model.nclasses)
        pos_equal_one = pos_equal_one.reshape(batch_size, -1, 1 * self.model.nclasses)
       # self.mAP.update(pos_equal_one, p_map)

        for frm in range(batch_size):
            ap = average_precision_score(pos_equal_one[frm].cpu(), p_map[frm].cpu())
            self.log("val_average_precision", ap, batch_size=1, on_step=False, on_epoch=True)

        return loss

    def forward(self, *args, **kwargs):
        p_map, r_map    = self.model(*args)
        p_map           = p_map.reshape(p_map.shape[0], -1, 1)
        r_map           = delta_to_boxes3d_hardcoded(r_map, self.anchors)
        return r_map
#        return r_map[p_map.tile(7) > 0.5] # 50% confidence

    def predict_step(self, data, step):
        return self.forward(data)    

    def configure_optimizers(self):
        return [self.model.lr_scheduler.optimizer], [self.model.lr_scheduler]
    


def train_experiment_debug():
    params = {
        "n_epochs":     200,
        "batch_size":   2,  # Set to 2 for the fastest training. (Training time exponentially grows with batch size). Batch Size > 16 causes GPU memory issues for me.
        "small_addon_for_BCE": 1e-6,
        "max_gradient_norm": 5, # TODO: See if I can make use of this.
        "alpha_bce":    1.5,
        "beta_bce":     1.0,
        "learning_rate": 0.001,
        "mode":         "train",
        "dump_vis":     "no",
        "data_root_dir": "/mnt/d/Datasets/Kitti/",
        "model_dir":    "model",
        "model_name":   "model1",
        "num_threads":  8
    }

    cfg["DATA_DIR"] = params["data_root_dir"]
    cfg["CALIB_DIR"] = os.path.join(cfg["DATA_DIR"], "training/calib")
    label_encoder = LabelEncoder()
    
    train_loader, val_loader = create_data_loader(cfg, params, 16, "train", is_aug_data=True, label_encoder=label_encoder, create_anchors=True, seed=2023)    
    model = Model(cfg, params, device)
    model = VoxelNetPL(model)
    checkpoint_dir = os.path.join("./", params["model_dir"], params["model_name"])
    trainer = pl.Trainer(max_epochs=params["n_epochs"], default_root_dir=checkpoint_dir)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
   
def to_onnx():
    from model_hardcoded import Model as ModelHC
   
    params = {
        "n_epochs":     200,
        "batch_size":   1,  # Set to 2 for the fastest training. (Training time exponentially grows with batch size). Batch Size > 16 causes GPU memory issues for me.
        "small_addon_for_BCE": 1e-6,
        "max_gradient_norm": 5, # TODO: See if I can make use of this.
        "alpha_bce":    1.5,
        "beta_bce":     1.0,
        "learning_rate": 0.001,
        "mode":         "train",
        "dump_vis":     "no",
        "data_root_dir": "/mnt/d/Datasets/Kitti/",
        "model_dir":    "model",
        "model_name":   "model1",
        "num_threads":  8
    }

    cfg["DATA_DIR"] = params["data_root_dir"]
    cfg["CALIB_DIR"] = os.path.join(cfg["DATA_DIR"], "training/calib")
    label_encoder = LabelEncoder()
    
    train_loader, val_loader = create_data_loader(cfg, params, 16, "train", is_aug_data=False, label_encoder=label_encoder, create_anchors=True, seed=2023)    

    #model = Model(cfg, params, device)
    model = ModelHC(requires_grad = False)
    model = VoxelNetPL(model).to(device)

    input_sample = train_loader.dataset[55]
    input_sample = (    
        torch.Tensor(input_sample["feature_buffer"]).to(device).detach().unsqueeze(0), 
        torch.Tensor(input_sample["coordinate_buffer"]).to(torch.int32).unsqueeze(0).to(device).detach()
    )

    model.to_onnx("VoxelNet.onnx", input_sample, export_params=True, training = torch.onnx.TrainingMode.EVAL)

def test_onnx():
    params = {
        "n_epochs":     200,
        "batch_size":   1,  # Set to 2 for the fastest training. (Training time exponentially grows with batch size). Batch Size > 16 causes GPU memory issues for me.
        "small_addon_for_BCE": 1e-6,
        "max_gradient_norm": 5, # TODO: See if I can make use of this.
        "alpha_bce":    1.5,
        "beta_bce":     1.0,
        "learning_rate": 0.001,
        "mode":         "train",
        "dump_vis":     "no",
        "data_root_dir": "/mnt/d/Datasets/Kitti/",
        "model_dir":    "model",
        "model_name":   "model1",
        "num_threads":  8
    }
    cfg["DATA_DIR"] = params["data_root_dir"]
    cfg["CALIB_DIR"] = os.path.join(cfg["DATA_DIR"], "training/calib")
    label_encoder = LabelEncoder()
    train_loader, val_loader = create_data_loader(cfg, params, 16, "train", is_aug_data=False, label_encoder=label_encoder, create_anchors=True, seed=2023)
    input_sample = train_loader.dataset[55]
    input_sample = (    
        torch.Tensor(input_sample["feature_buffer"]).to(device).detach().unsqueeze(0), 
        torch.Tensor(input_sample["coordinate_buffer"]).to(torch.int32).unsqueeze(0).to(device).detach()
    )
    
    import onnxruntime
    ort_session = onnxruntime.InferenceSession("VoxelNet.onnx")
    input_name = ort_session.get_inputs()[0].name
    ort_outs = ort_session.run(None, input_sample)
    breakpoint()

if __name__ == '__main__':
    #train_experiment_debug()
    to_onnx()
    #test_onnx()
    