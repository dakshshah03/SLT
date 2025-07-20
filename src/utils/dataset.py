import torch
from torch.utils.data import Dataset
import decord
import pandas as pd
import os
import numpy as np
from transforms import VideoMAE_Transform
# from transformers import VideoMAEImageProcessor
import json

class asl_citizen_dataset(Dataset):
    def __init__(self, csv_path, video_dir, transform=None, num_labels=None):
        """
        Takes in ASL citizen formatted CSV file and loads dataset
        
        Args:
            csv_path (string): path to CSV file containing training features
            video_dir (string): path to video directory
            transform (callable, optional): transforms for each set of images. Defaults to None.
        """
        super(asl_citizen_dataset, self).__init__()
        csv_df = pd.read_csv(csv_path)
        self.csv_df = csv_df.drop(columns=['ASL-LEX Code', 'Participant ID'])
        self.video_dir = video_dir
        self.transform = transform
        self.num_labels = num_labels
        
        # list of unique glosses
        # allows training on a subset of training classes
        if num_labels:
            self.glosses = np.sort(self.csv_df['Gloss'].unique())[:num_labels]
            self.csv_df = csv_df.loc[csv_df['Gloss'].isin(self.glosses)]
        else:
            self.glosses = np.sort(self.csv_df['Gloss'].unique())

        self.gloss_to_label = {gloss: i for i, gloss in enumerate(self.glosses)}

    def __len__(self):
        return len(self.csv_df)
    
    def __getitem__(self, idx):
        row = self.csv_df.iloc[idx]
        
        video_filename = row['Video file']
        gloss = row['Gloss']

        video_path = os.path.join(self.video_dir, video_filename)
        video_frames = self.sample_frames(video_path, num_frames=16)
        sample = torch.from_numpy(video_frames).float()

        label = self.gloss_to_label[gloss]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

    def save_label_to_gloss(self, path_to_save):
        """
        Saves the mapping from label (int) to gloss (string) to a json for inference

        Args:
            path_to_save (string): path to save the label to gloss mapping
        """
        # convert self.gloss_to_label to label_to_gloss
        label_to_gloss = {v: k for k, v in self.gloss_to_label.items()}
        file = os.path.join(path_to_save, f"label_to_gloss-{self.num_labels}.json")
        with open(file, 'w') as f:
            json.dump(label_to_gloss, f)

    
    def sample_frames(video_path, num_frames=16):
        """
        Samples <num_frames> frames from a video file

        Args:
            video_path (string): path to video file being sampled
            num_frames (int, optional): number of frames to sample. Defaults to 16.

        Returns:
            video_frames (torch.Tensor): sampled video frames as a tensor of shape (num_frames, 3, H, W)
        """
        video_reader = decord.VideoReader(video_path)
        total_frames = len(video_reader)
        frame_indices = np.sort(np.random.randint(0, total_frames-1, size=num_frames))
        video_frames = video_reader.get_batch(frame_indices).asnumpy().transpose(0, 3, 1, 2)
        video_frames = torch.from_numpy(video_frames.astype(np.float32) / 255.0)
        
        return video_frames
# testing 
# if __name__ == "__main__":
    # dataset = asl_citizen_dataset(
    #     csv_path="data/splits/train.csv",
    #     video_dir="data/ASL-Citizen/videos",
    #     transform=VideoMAE_Transform(VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2"), train=True),
    #     num_labels=100
    # )
    # dataset.save_label_to_gloss(".")
    # video_path = ""
    # vid = asl_citizen_dataset.sample_frames(video_path)


    # processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base")
    # transform = VideoMAE_Transform(processor, train=True)
    # print(processor)
    
    # x = transform(vid)
    # print(x.shape)