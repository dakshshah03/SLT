# VideoMAE SLT (README in progress)
This is an end-to-end ASL sign language translation model designed to be deployable and production-ready for inference

Currently, the project is using:
- Pytorch, Pytorch Lightning, Huggingface transformers
    - model architecture, training loop, initial model weights
    - distributed training (multi-node and multi-gpu)
- Kubernetes/Kubeflow
    - Distributed multi-node training
- MLflow (Databricks hosted)
    - experiment analytics and tracking
    - artifact store
- ONNX and TensorRT
    - Desployment and production-ready for inference
- GitHub Actions, Docker
    - CI/CD triggering on new production weights (marked in MLflow)
    - building inference container

## Setup
TODO: 
- secrets setup instructions
- github actions workflows/secrets
- mlflow instructions
- dataset directory setup instructions
- dockerhub setup instructions
- dockerfile (inference)

## Training
Theres two options for training.
1. Kubernetes (using the Kubeflow operator)
2. Local training

### Kubernetes
TODO:
- kubernetes manifests
    - training
    - pvc

### Local training
#### Requirements
First, you will need to install UV if you haven't already
run the following command to install it:
```
wget -qO- https://astral.sh/uv/install.sh | sh
```
Then, run 
```
uv sync
```
to install the required packages to train the model.
<!-- disclaimer: this readme is in progress, so it will be messy until I start cleaning up the code for actual use.

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

torch lightning


Running:
1. Add relevant information to secrets.conf (template provided)
- https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/connect-environment
2. Set up -->