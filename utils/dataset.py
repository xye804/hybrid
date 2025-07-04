import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import pickle
from torch.nn.utils.rnn import pad_sequence


class SignDataset(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        self.data = []
        self._load_data()

    def _load_data(self):
        with open(self.path, "rb") as f:
            loaded_data = pickle.load(f)
            # mean = loaded_data["keypoints"].mean(dim=[0, 1, 2], keepdim=True)
            # std = loaded_data["keypoints"].std(dim=[0, 1, 2], keepdim=True)
            # self.normalizer = Normalizer(mean, std)
            # loaded_data["keypoints"] = normalizer.norm(loaded_data["keypoints"])
            self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    keypoints = [item["keypoints"] for item in batch]
    sentence = [item["sentence"].squeeze(0) for item in batch]
    keypoints = pad_sequence(keypoints, batch_first=True)
    sentence = pad_sequence(sentence, batch_first=True)

    return {"keypoints": keypoints, "sentence": sentence}


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def load_dataloader(cfg, mode):
    assert mode in ["train", "dev", "test"]
    dataset = SignDataset(os.path.join(cfg["root"], f"{mode}.pkl"))

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True if mode == "train" else False,
        drop_last=True if mode == "train" else False,
        num_workers=cfg["num_works"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    dataloader = cycle(dataloader)

    return dataloader, len(dataset)
