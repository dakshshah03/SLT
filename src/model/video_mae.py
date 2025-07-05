from transformers import VideoMAEForVideoClassification
import torch

model = VideoMAEForVideoClassification.from_pretrained("")

def get_videoMAE_ssv2(num_classes):
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-ssv2"
    )
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    return model