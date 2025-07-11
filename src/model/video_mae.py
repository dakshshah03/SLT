from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, TrainerCallback
import torch.nn as nn
from torch.optim import AdamW

# model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
# print(model)

# TODO: Clean up model, turn it into a pytorch class

def get_videoMAE_ssv2(num_classes):
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-ssv2"
    )
    # freeze backbone
    for param in model.videomae.parameters():
        param.requires_grad = False
    
    # add custom classifier head (2 layers)
    hidden_size = model.config.hidden_size
    model.classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_size // 2, num_classes)
    )
    
    return model

# splitting training into 2-3 parts, with gradual layer unfreezing
def get_videoMAE_adamW(model):
    optimizer_parameters = [
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        {'params': [p for p in model.videomae.parameters() if p.requires_grad], 'lr': 1e-5}
    ]
    return AdamW(optimizer_parameters, weight_decay=0.01)

# print(VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2"))