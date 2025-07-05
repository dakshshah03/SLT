import torch
from torch.utils.data import Dataset, DataLoader
import decord
import pandas as pd
import os
import numpy as np
import torchvision.transforms.v2 as transforms
# TODO: Transforms

def sample_frames(video_path, num_frames=16):
    # samples 16 random frames from the video. Chose to do this because other approaches would take longer
    # TODO: Look into other preprocessing sampling strategies, in case those work better
    video_reader = decord.VideoReader(video_path)
    total_frames = len(video_reader)
    frame_indices = np.random.randint(1, total_frames, size=num_frames)
    video_frames = video_reader.get_batch(frame_indices)
    
    return video_frames

class asl_citizen_dataset(Dataset):
    def __init__(self, csv_path, video_dir, transform=None, num_labels=None):
        """
        Takes in ASL citizen formatted CSV file and loads dataset
        Args:
            csv_path (string): path to CSV file containing training features
            video_dir (string): path to video directory
            transform (callable, optional): transforms for each set of images. Defaults to None.
        """
        csv_df = pd.read_csv(csv_path)
        self.csv_df = csv_df.drop(columns=['Gloss', 'Participant ID'])
        self.video_dir = video_dir
        self.transform = transform
        
        # list of unique lex codes (labels)
        # using lex codes instead of gloss since this is more standardized
        # allows training on a subset of training classes (for testing)
        if num_labels:
            self.lex_codes = np.sort(self.csv_df['ASL-LEX Code'].unique())[:num_labels]
            self.csv_df = csv_df.loc[csv_df['ASL-LEX Code'] in self.lex_codes]
        else:
            self.lex_codes = np.sort(self.csv_df['ASL-LEX Code'].unique())
        
        self.lex_code_to_label = {code: i for i, code in enumerate(self.lex_codes)}
        
    def __len__(self):
        return len(self.csv_df)
    
    def __getitem__(self, idx):
        row = self.csv_df.iloc[idx]
        
        video_filename = row['Video file']
        lex_code = row['ASL-LEX Code']
        
        video_path = os.path.join(self.video_dir, video_filename)
        video_frames = sample_frames(video_path, num_frames=16)
        sample = torch.from_numpy(video_frames).float()
        
        label = self.lex_code_to_label[lex_code]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

# testing 
# if __name__ == "__main__":
    