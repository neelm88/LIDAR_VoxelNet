import torch

# Define your model here

# Initialize Loss and Optimizer

import pickle
import argparse
import os
import time
import torch
from data import create_data_loader
from model import Model
from config import cfg
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import math
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Modifying params here instad of in cli for easy access.
    params = {
        "n_epochs": 15,
        "batch_size": 2,  # Set to 2 for the fastest training. (Training time exponentially grows with batch size). Batch Size > 16 causes GPU memory issues for me.
        "small_addon_for_BCE": 1e-6,
        "max_gradient_norm": 5, # TODO: See if I can make use of this.
        "alpha_bce": 1.5,
        "beta_bce": 1.0,
        "learning_rate": 0.001,
        "mode": "train",
        "dump_vis": "no",
        "data_root_dir": "../",
        "model_dir": "model",
        "model_name": "model1",
        "num_threads": 8
    }
    cfg["DATA_DIR"] = params["data_root_dir"]
    cfg["CALIB_DIR"] = os.path.join(cfg["DATA_DIR"], "training/calib")


    # Datasets creation
    print("Datasets creation (training dataset, sample_test dataset, validation, and dump_test dataset)")
    label_encoder = LabelEncoder()
    train_batcher = create_data_loader(cfg, params, 16, "train", is_aug_data=True, label_encoder=label_encoder, create_anchors=True)
    rand_test_batcher = create_data_loader(cfg, params, 1, "sample_test", is_aug_data=False, label_encoder=label_encoder, create_anchors=False)
    val_batcher = create_data_loader(cfg, params, 16, "eval", is_aug_data=False, label_encoder=label_encoder, create_anchors=False)

    # Model creation
    print("Model creation ...")

    model = Model(cfg, params, device)

    # Checkpoint Manager
    print("Building the checkpoint Manager ...")

    checkpoint_dir = os.path.join(params["model_dir"], params["model_name"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Start training
    print("Start training : ")
    model.to(device)  # Move the model to the desired device (e.g., GPU)
    model.train()
    total_data_points = len(train_batcher.dataset)
    batch_size = train_batcher.batch_size

    # Calculate the number of iterations
    num_iterations = math.ceil(total_data_points / batch_size)
    print(f"Training {params['n_epochs']} epochs with {num_iterations} steps")
    losses = []
    for epoch in range(params["n_epochs"]):
        total_loss = 0
        for step, data in enumerate(train_batcher):
            # Convert your data to PyTorch tensors
            start_time = time.time()
            feature_buffer = data["feature_buffer"].to(device)
            coordinate_buffer = data["coordinate_buffer"].to(device)
            targets = data["targets"].to(device)
            pos_equal_one = data["pos_equal_one"].to(device)
            pos_equal_one_reg = data["pos_equal_one_reg"].to(device)
            pos_equal_one_sum = data["pos_equal_one_sum"].to(device)
            neg_equal_one = data["neg_equal_one"].to(device)
            neg_equal_one_sum = data["neg_equal_one_sum"].to(device)
            if epoch == 0 and step == 0:
                print(f"First Data loaded")            
            
            # Calculate loss
            all_loss = model.train_step(feature_buffer, coordinate_buffer, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)
            print(all_loss)
            if epoch == 0 and step == 0:
                print(f"First Train Step Done")    
            loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss = all_loss
            total_loss += loss
            # Print training information
            end_time = time.time()
            print(
                f"train: {step + 1} @ epoch: {epoch + 1}/{params['n_epochs']} loss: {loss} time: {end_time-start_time} seconds" #  reg_loss: {reg_loss} cls_loss: {cls_loss} cls_pos_loss: {cls_pos_loss} cls_neg_loss: {cls_neg_loss}
            )
            # TODO: Implement Validation
            # TODO: Implement precision metric calculation
            if (step + 1) % num_iterations == 0:
                # Save checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": model.optimizer.state_dict(),
                        "loss": loss,
                    },
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt"),
                )
        model.lr_scheduler.step()
        losses.append(total_loss)

    # Create x-axis values (epochs)
    epochs = list(range(1, len(losses) + 1))

    # Plot the loss curve
    plt.plot(epochs, losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()

