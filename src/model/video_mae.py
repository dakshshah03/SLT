from transformers import VideoMAEForVideoClassification
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F 

class VideoMAE_ssv2_Finetune(nn.Module):
    """
    VideoMAE model architecture with a 2 layer classifier
    head to support classification for 2700+ gesture classes.
    For use in ASL Gesture recognition.

    Args:
        num_classes (int): Number of classes for output layer
    """
    def __init__(self, num_classes=100):
        super(VideoMAE_ssv2_Finetune, self).__init__()
        
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-ssv2"
        )
        hidden_size = model.config.hidden_size
        
        self.videomae = model.videomae
        self.fc_norm = model.fc_norm
        
        self.output_size = num_classes
        
        for param in self.videomae.parameters():
            param.requires_grad = False
        for param in self.fc_norm.parameters():
            param.requires_grad = False
            
        self.classifier =  nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, pixel_values, labels=None):
        x = self.videomae(pixel_values)
        x = self.fc_norm(x)
        logits = self.classifier(x)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
    
if __name__ == "__main__":
    x = VideoMAE_ssv2_Finetune()
    print(x)