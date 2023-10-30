import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from utils.utils import cal_anchors, process_pointcloud, cal_rpn_target
from aug_data import aug_data

class CustomDataset(Dataset):
    def __init__(self, cfg, params, buffer_size, mode, is_aug_data, label_encoder, create_anchors=False):
        self.cfg = cfg
        self.params = params
        self.mode = mode
        self.buffer_size = buffer_size
        self.is_aug_data = is_aug_data
        self.label_encoder = label_encoder
        data_d = "training" if mode == "train" else "testing" if mode == "test" else "validation"

        if mode != "test":
            label_tags = [os.path.basename(a).split(".")[0] for a in glob.glob(os.path.join(cfg.DATA_DIR, data_d, "label_2/*.txt"))]
        img_tags = [os.path.basename(a).split(".")[0] for a in glob.glob(os.path.join(cfg.DATA_DIR, data_d, "image_2/*.png"))]
        lidar_tags = [os.path.basename(a).split(".")[0] for a in glob.glob(os.path.join(cfg.DATA_DIR, data_d, "velodyne/*.bin"))]

        if mode != "test":
            assert label_tags and img_tags and lidar_tags, "One of the three (label_2, image_2, velodyne) folders is empty, Data folder must not be empty"
            assert not set(label_tags).symmetric_difference(set(img_tags)) and not set(img_tags).symmetric_difference(set(lidar_tags)), "Must have equivalent tags in image_2, label_2, and velodyne dirs, check those files"
        else:
            assert img_tags and lidar_tags, "One of the three (image_2, velodyne) folders is empty, Data folder must not be empty"
            assert not set(img_tags).symmetric_difference(set(lidar_tags)), "Must have equivalent tags in image_2, velodyne dirs, check those files"

        self.tags = lidar_tags
        self.num_examples = len(lidar_tags)
        if create_anchors:
            pass
        self.anchors = cal_anchors(cfg)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        sample = {}  # Create a dictionary to hold data for this item

        data_d = "training" if self.mode == "train" else "testing" if self.mode == "test" else "validation"
        img_dir = os.path.join(self.cfg.DATA_DIR, data_d, "image_2")
        labels_dir = os.path.join(self.cfg.DATA_DIR, data_d, "label_2")
        pc_dir = os.path.join(self.cfg.DATA_DIR, data_d, "velodyne")

        if self.mode in ["train", "sample_test"]:
            random.shuffle(self.tags)
        else:
            self.tags.sort()

        index = self.tags[idx]
        dic = {}

        if self.is_aug_data:
            dic = aug_data(index, os.path.join(self.cfg.DATA_DIR, data_d))
        else:
            pc = np.fromfile(os.path.join(pc_dir, f"{int(index):06d}.bin"), dtype=np.float32).reshape(-1, 4)
            if self.mode == "test":
                dic["lidar"] = pc
                dic["labels"] = []
            elif self.mode == "sample_test" or self.mode == "eval":
                dic["lidar"] = pc
                dic["labels"] = np.array([line.strip() for line in open(os.path.join(labels_dir, f"{int(index):06d}.txt"), 'r').readlines()])
            else:
                dic["lidar"] = 0
                dic["labels"] = np.array([line.strip() for line in open(os.path.join(labels_dir, f"{int(index):06d}.txt"), 'r').readlines()])
            dic["num_points"] = len(pc)

            if self.mode == "train":
                dic["img"] = 0
            else:
                img = read_image(os.path.join(img_dir, f"{int(index):06d}.png"))
                dic["img"] = ToTensor()(img)

            dic["tag"] = f"{int(index):06d}"

            dic.update(process_pointcloud(pc, self.cfg))

        if self.mode in ["train", "eval", "sample_test"]:
            dic["pos_equal_one"], dic["neg_equal_one"], dic["targets"] = cal_rpn_target(dic["labels"][np.newaxis, ...].astype(str), self.cfg.MAP_SHAPE, self.anchors, self.cfg.DETECT_OBJECT, 'lidar')

            dic["pos_equal_one_reg"] = np.concatenate([np.tile(dic["pos_equal_one"][..., [0]], 7), np.tile(dic["pos_equal_one"][..., [1]], 7)], axis=-1)[0]
            dic["pos_equal_one_sum"] = np.clip(np.sum(dic["pos_equal_one"], axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)[0]
            dic["neg_equal_one_sum"] = np.clip(np.sum(dic["neg_equal_one"], axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)[0]

            dic["pos_equal_one"], dic["neg_equal_one"], dic["targets"] = dic["pos_equal_one"][0], dic["neg_equal_one"][0], dic["targets"][0]
        else:
            dic["pos_equal_one"], dic["neg_equal_one"], dic["targets"] = 0, 0, 0
            dic["pos_equal_one_reg"], dic["pos_equal_one_sum"], dic["neg_equal_one_sum"] = 0, 0, 0
        dic["labels"] = self.label_encoder.fit_transform(dic["labels"])
        dic.pop("labels")
        return dic

def create_data_loader(cfg, params, buffer_size, mode, is_aug_data, label_encoder, create_anchors=False):
    custom_dataset = CustomDataset(cfg, params, buffer_size, mode, is_aug_data, label_encoder, create_anchors)
    data_loader = DataLoader(custom_dataset, batch_size=params["batch_size"], shuffle=True if mode == "train" else False, num_workers=params["num_threads"])
    data_loader.num_examples = custom_dataset.num_examples
    return data_loader

# Usage:
# data_loader = create_data_loader(cfg, params, buffer_size, mode, is_aug_data)
# for batch in data_loader:
#     # Process the batch
