import torch
from transformers import TrainingArguments, Trainer, VideoMAEFeatureExtractor

from model import get_videoMAE_ssv2, get_videoMAE_adamW, unfreeze_layers
from utils import asl_citizen_dataset

import mlflow

import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(
       "Training script to finetune VideoMAE-ssv2 for ASL Sign Language Translation",
        add_help=False
    )
    
    # TODO: Add more arguments so parameters are less of a pain to set up
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--output_dir', default="./model_artifacts", type=str)
    parser.add_argument('--report_to', default="mlflow", type=str)
    
    args = parser.parse_args()
    
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        report_to=args.report_to,
        run_name="videomae-asl-run",
    )
    
    model = get_videoMAE_ssv2(200)
    optimizer = get_videoMAE_adamW(model)
    
    # TODO: finish training initialization
    trainer_classifier = Trainer(
        model=model,
        args=training_args,
    )
    # TODO: MLFlow Logging