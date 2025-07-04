from pathlib import Path
from typing import OrderedDict
import wandb
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
import yaml


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config["common"], config["data"], config["train"], config["model"]


def load_wandb(common_cfg, train_cfg, data_cfg, model_cfg):
    wandb.login(key=common_cfg["wandb_key"])
    config = {"train": train_cfg, "data": data_cfg, "model": model_cfg}
    run = wandb.init(
        project="HybridSign",
        config=config,
        dir=os.path.join(common_cfg["output_dir"], "wandb"),
        mode="disabled" if common_cfg["disable_wandb"] else "online",
    )
    return run


def load_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )


def create_folders(cfg):
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    return videos_dir, checkpoints_dir


def set_seed(cfg):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
