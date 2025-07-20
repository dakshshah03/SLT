import torch
from torch.optim import AdamW
from transformers import TrainingArguments, Trainer
from transformers import VideoMAEImageProcessor

from model import VideoMAE_ssv2_FinetuneLightning, LayerUnfreezeLightning
from utils import asl_citizen_dataset, VideoMAE_Transform

import os

import lightning as L
from lightning.loggers import MLFlowLogger

import argparse

def train(args, model, train_set, val_set, logger, callbacks):
    model = torch.compile(model)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    train_loader = L.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
    val_loader = L.DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False)
        
    trainer = L.Trainer(
        strategy="ddp",
        devices="auto",
        max_epochs=args.max_epochs,
        precision="16-mixed",
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        default_root_dir=args.output_dir
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    final_model_save_path = os.path.join(args.output_dir, "latest_model.ckpt")
    trainer.save_checkpoint(final_model_save_path)

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

    model = VideoMAE_ssv2_FinetuneLightning(num_classes=args.num_classes)

    train_set = asl_citizen_dataset(
        csv_path=os.path.join(args.data_dir, "splits/train.csv"),
        data_dir=os.path.join(args.data_dir, "videos"),
        transform=VideoMAE_Transform(
            VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2"),
            train=True
        ),
        num_labels=args.num_classes
    )

    val_set = asl_citizen_dataset(
        csv_path=os.path.join(args.data_dir, "splits/val.csv"),
        data_dir=os.path.join(args.data_dir, "videos"),
        transform=VideoMAE_Transform(
            VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2"),
            train=False
        ),
        num_labels=args.num_classes
    )
    
    unfreeze_callback = LayerUnfreezeLightning(
        delay_start=5,  # start unfreezing after 5 epochs
        epoch_step=3,   # unfreeze 1 layer every 2 epochs
        delay_unfreeze_all=15  # unfreeze all layers after 15 epochs
    )
    
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="{epoch}-{val_loss:.4f}", 
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=False,
        verbose=True
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")

    callbacks = [unfreeze_callback, lr_monitor, checkpoint_callback]

    logger = MLFlowLogger(
        experiment_name="ASL-videomae",
        log_model=True,
        run_name="videomae-asl-run",
    )

    
    train(args, model, train_set, val_set, logger, callbacks=callbacks)