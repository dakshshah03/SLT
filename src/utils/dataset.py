import torch
from torch.utils.data import Dataset, DataLoader
import decord
import pandas as pd
import os
import numpy as np
import torchvision.transforms.v2 as transforms
# TODO: Dataloader
# TODO: Dataset
# TODO: Transforms

class asl_citizen_dataset(Dataset):
    def __init__(self, csv_path, video_dir, transform=None):
        """
        Takes in ASL citizen formatted CSV file and loads dataset
        Args:
            csv_path (string): _description_
            video_dir (string): _description_
            transform (callable, optional): _description_. Defaults to None.
        """
        csv_df = pd.read_csv(csv_path)
        self.csv_df = csv_df.drop(columns=['Gloss', 'Participant ID'])
        self.video_dir = video_dir
        self.transform = transform
        
        # list of unique lex codes (labels)
        # using lex codes instead of gloss since this is more standardized
        self.lex_codes = np.sort(self.csv_df['ASL-LEX Code'].unique())
        
        self.lex_code_to_label = {code: i for i, code in enumerate(self.lex_codes)}
        
    def __len__(self):
        return len(self.csv_df)
    
    def __getitem__(self, idx):
        row = self.csv_df.iloc[idx]
        
        video_filename = row['Video file']
        lex_code = row['ASL-LEX Code']
        
        video_path = os.path.join(self.video_dir, video_filename)
        
        video_reader = decord.VideoReader(video_path)
        
        # Loads video frames
        # TODO: sample 16 frames. Maybe preprocessing script for maximal frames?
        video_frames = video_reader.get_batch(range(len(video_reader))).asnumpy()
        sample = torch.from_numpy(video_frames).float()
        
        label = self.lex_code_to_label[lex_code]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
    