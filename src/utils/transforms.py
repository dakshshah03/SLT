import torchvision.transforms.v2 as T
from transformers import VideoMAEImageProcessor
import torch

class VideoMAE_Transform:
    def __init__(self, image_processor, train=True):
        self.mean = image_processor.image_mean
        self.std = image_processor.image_std
        self.size = image_processor.size["shortest_edge"]
        
        if(train):
            self.transform = T.Compose([
                T.Resize(self.size),
                T.RandomCrop(self.size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomGrayscale(),
                T.Normalize(self.mean, self.std),
                T.ToDtype(torch.float32, scale=True)
            ])
        else:
            self.transform = T.Compose([
                T.Resize(self.size),
                T.Normalize(self.mean, self.std),
                T.ToDtype(torch.float32, scale=True)
            ])
            
    def __call__(self, video_frames):
        """
        video_frames: List[PIL.Image] or List[np.ndarray]
                      or List[Tensor] of shape (T, C, H, W)
        Returns: Tensor of shape (T, C, H, W)
        """
        return self.transform(video_frames)