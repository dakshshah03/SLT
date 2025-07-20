from lightning import LightningDataModule
from dataset import asl_citizen_dataset
from transformers import VideoMAEImageProcessor
from transforms import VideoMAE_Transform
from torch.utils.data import DataLoader
import os

class ASLCitizenDataModule(LightningDataModule):
    def __init__(self, data_dir, num_classes, train_batch=16, val_batch=16, test_batch=16, model_weights="OpenGVLab/VideoMAEv2-Base"):
        super().__init__()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.model_weights = model_weights

        self.train_batch_size = train_batch
        self.val_batch_size = val_batch
        self.test_batch_size = test_batch

    def setup(self, stage=None):
        self.train_set = asl_citizen_dataset(
            csv_path=os.path.join(self.data_dir, "splits/train.csv"),
            video_dir=os.path.join(self.data_dir, "videos"),
            transform=VideoMAE_Transform(
                VideoMAEImageProcessor.from_pretrained(self.model_weights),
                train=True
            ),
            num_labels=self.num_classes
        )

        self.val_set = asl_citizen_dataset(
            csv_path=os.path.join(self.data_dir, "splits/val.csv"),
            video_dir=os.path.join(self.data_dir, "videos"),
            transform=VideoMAE_Transform(
                VideoMAEImageProcessor.from_pretrained(self.model_weights),
                train=False
            ),
            num_labels=self.num_classes
        )

        self.test_set = asl_citizen_dataset(
            csv_path=os.path.join(self.data_dir, "splits/test.csv"),
            video_dir=os.path.join(self.data_dir, "videos"),
            transform=VideoMAE_Transform(
                VideoMAEImageProcessor.from_pretrained(self.model_weights),
                train=False
            ),
            num_labels=self.num_classes
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False)