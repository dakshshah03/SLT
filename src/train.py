import torch
from torch.optim import AdamW
from transformers import TrainingArguments, Trainer
import lightning as L
from lightning.loggers import MLFlowLogger

from model import VideoMAE_FinetuneLightning, LayerUnfreezeLightning
from utils import ASLCitizenDataModule
import os
import argparse

def train(args, model, logger, callbacks, datamodule):
    model = torch.compile(model)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        
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

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

    final_model_save_path = os.path.join(args.output_dir, "latest_model.ckpt")
    trainer.save_checkpoint(final_model_save_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        "Training script to finetune VideoMAE for ASL Sign Language Translation",
        add_help=False
    )
    
    # TODO: Add more arguments so parameters are less of a pain to set up
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--val_batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--output_dir', default="./model_artifacts", type=str)
    parser.add_argument('--report_to', default="mlflow", type=str)
    parser.add_argument('--data_dir', default="./../data", type=str)
    parser.add_argument('--num_classes', default=200, type=int,
                        help="Number of classes for ASL Gesture Recognition")

    args = parser.parse_args()
    
    # model_weights = "MCG-NJU/videomae-base-finetuned-ssv2"
    model_weights = "OpenGVLab/VideoMAEv2-Base"
    
    model = VideoMAE_FinetuneLightning(num_classes=args.num_classes, model_weights=model_weights)
    
    
    dm = ASLCitizenDataModule(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        train_batch=args.train_batch_size,
        val_batch=args.val_batch_size,
        test_batch=args.test_batch_size,
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

    
    train(args, model, logger, callbacks=callbacks)