# VideoMAE SLT
disclaimer: this readme is in progress, so it will be messy until I start cleaning up the code for actual use.

a deployable ASL SLT model and framework. This is version 1, where I am adapting VideoMAE and finetuning on ASL Citizen.
My current approach is expected to produce crude gloss sequences, that are not necessarily grammatically correct ones.

I am looking into SOTA approaches to produce a coherent sentence:
- CLIP-SLA: Parameter-Efficient CLIP Adaptation for Continuous Sign Language Recognition 
- Multilingual Gloss-free Sign Language Translation: Towards Building a Sign Language Foundation Model
- LLaVA-SLT: Visual Language Tuning for Sign Language Translation

My goal is to progressively increase the complexity of tasks, using different approaches:
1. Build single gesture classification (VideoMAE)
2. Adapt and build on a SOTA approach that can convert a video sequence of gestures into a coherent sequence of glosses
3. Adapt skeleton/landmark approaches and h

require:
decord
transformers
torch
pytorchvideo
pandas
mlflow
numpy


Running:
1. Add relevant information to secrets.conf (template provided)
- https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/connect-environment