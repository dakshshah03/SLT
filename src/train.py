import torch
from transformers import TrainingArguments, Trainer, VideoMAEFeatureExtractor

from model import get_videoMAE_ssv2
from utils import asl_citizen_dataset

import mlflow
from mlflow.models import infer_signature



training_args = TrainingArguments(
    output_dir="./model_checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    report_to="mlflow",
    run_name="videomae-asl-run",
)