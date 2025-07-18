import torch
from torch.utils.data import Dataset
import decord
import pandas as pd
import os
import numpy as np
from transforms import VideoMAE_Transform
from transformers import VideoMAEImageProcessor

# TODO: Clean up code
class asl_citizen_dataset(Dataset):
    def __init__(self, csv_path, video_dir, model="MCG-NJU/videomae-base-finetuned-ssv2", transform=None, num_labels=None):
        """
        Takes in ASL citizen formatted CSV file and loads dataset
        Args:
            csv_path (string): path to CSV file containing training features
            video_dir (string): path to video directory
            transform (callable, optional): transforms for each set of images. Defaults to None.
        """
        super(asl_citizen_dataset, self).__init__()
        csv_df = pd.read_csv(csv_path)
        self.csv_df = csv_df.drop(columns=['Gloss', 'Participant ID'])
        self.video_dir = video_dir
        self.transform = transform
        
        # list of unique lex codes (labels)
        # using lex codes instead of gloss since this is more standardized
        # allows training on a subset of training classes (for testing)
        if num_labels:
            self.lex_codes = np.sort(self.csv_df['ASL-LEX Code'].unique())[:num_labels]
            self.csv_df = csv_df.loc[csv_df['ASL-LEX Code'].isin(self.lex_codes)]
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
        video_frames = self.sample_frames(video_path, num_frames=16)
        sample = torch.from_numpy(video_frames).float()
        
        label = self.lex_code_to_label[lex_code]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
    
    def sample_frames(video_path, num_frames=16):
        video_reader = decord.VideoReader(video_path)
        total_frames = len(video_reader)
        frame_indices = np.sort(np.random.randint(0, total_frames-1, size=num_frames))
        video_frames = video_reader.get_batch(frame_indices).asnumpy().transpose(0, 3, 1, 2)
        video_frames = torch.from_numpy(video_frames.astype(np.float32) / 255.0)
        
        return video_frames

# testing 
if __name__ == "__main__":
    vid = asl_citizen_dataset.sample_frames("test_video.mp4")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
    transform = VideoMAE_Transform(processor, train=True)
    print(type(vid))
    x = transform(vid)
    print(x.shape)