import torch
from torch.optim import AdamW
from transformers import TrainingArguments, Trainer
from transformers import VideoMAEImageProcessor

from model import VideoMAE_ssv2_Finetune, LayerUnfreeze
from utils import asl_citizen_dataset, VideoMAE_Transform

import os

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
    parser.add_argument('--data_dir', default="./../data", type=str)
    parser.add_argument('--num_classes', default=200, type=int,
                        help="Number of classes for ASL Gesture Recognition")

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

    model = VideoMAE_ssv2_Finetune(num_classes=args.num_classes)

    optimizer_parameters = [
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        {'params': [p for p in model.videomae.parameters() if p.requires_grad], 'lr': 1e-5}
    ]
    
    optimizer = AdamW(optimizer_parameters, weight_decay=0.01)
    # TODO: Add LR scheduler(?) unsure if needed. Test empirically
    # TODO: add dataset
    
    train_dataset = asl_citizen_dataset(
        csv_path=os.path.join(args.data_dir, "splits/train.csv"),
        data_dir=os.path.join(args.data_dir, "videos"),
        transform=VideoMAE_Transform(
            VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2"),
            train=True
        ),
        num_labels=args.num_classes
    )
    
    test_dataset = asl_citizen_dataset(
        csv_path=os.path.join(args.data_dir, "splits/test.csv"),
        data_dir=os.path.join(args.data_dir, "videos"),
        transform=VideoMAE_Transform(
            VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2"),
            train=False
        ),
        num_labels=args.num_classes
    )

    trainer_classifier = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=(optimizer, None),   
    )
    # TODO: MLFlow Logging