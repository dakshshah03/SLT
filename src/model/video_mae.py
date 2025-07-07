from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import torch.nn as nn
from torch.optim import AdamW

# model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
# print(model)

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
        nn.Linear(hidden_size // 2, 2700)
    )
    
    return model

# splitting training into 2-3 parts, with gradual layer unfreezing
def get_videoMAE_adamW(model):
    optimizer_parameters = [
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        {'params': [p for p in model.videomae.parameters() if p.requires_grad], 'lr': 1e-5}
    ]
    return AdamW(optimizer_parameters, weight_decay=0.01)


def unfreeze_layers(model, unfreeze_layer_count=2):
    num_encoder_layers = len(model.videomae.encoder.layer)
    
    # unfreezes the last layers specified by unfreeze_layer_count
    for i in range(num_encoder_layers):
        if i > num_encoder_layers - unfreeze_layer_count:
            for param in model.videomae.encoder.layer[i].parameters():
                param.requires_grad = False
        else:
            for param in model.videomae.encoder.layer[i].parameters():
                param.requires_grad = True
    
print(VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2"))