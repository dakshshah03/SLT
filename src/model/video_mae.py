# import torch
from transformers import VideoMAEForVideoClassification

model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

print(model)