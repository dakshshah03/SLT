from transformers import VideoMAEForVideoClassification
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
import lightning as L
from torchmetrics.functional.classification import accuracy

# TODO: Modify model for lightning

# torch native version
class VideoMAE_ssv2_Finetune(nn.Module):
    """
    VideoMAE model architecture with a 2 layer classifier
    head to support classification for 2700+ gesture classes.
    For use in ASL Gesture recognition.

    Args:
        num_classes (int): Number of classes for output layer
    """
    def __init__(self, num_classes=100):
        super(VideoMAE_ssv2_Finetune, self).__init__()
        
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-ssv2"
        )
        hidden_size = model.config.hidden_size
        
        self.videomae = model.videomae
        self.fc_norm = model.fc_norm
        
        self.output_size = num_classes
        
        for param in self.videomae.parameters():
            param.requires_grad = False
        for param in self.fc_norm.parameters():
            param.requires_grad = False
            
        self.classifier =  nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pixel_values, labels=None):
        x = self.videomae(pixel_values)
        x = self.fc_norm(x)
        logits = self.classifier(x)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
    

class VideoMAE_ssv2_FinetuneLightning(L.LightningModule):
    """
    VideoMAE model architecture with a 2 layer classifier
    head to support classification for 2700+ gesture classes.
    For use in ASL Gesture recognition.

    Args:
        num_classes (int): Number of classes for output layer
    """
    def __init__(self, num_classes=100):
        super(VideoMAE_ssv2_FinetuneLightning, self).__init__()
        
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-ssv2"
        )
        self.hidden_size = model.config.hidden_size
        self.output_size = num_classes
        
        self.videomae = model.videomae
        self.fc_norm = model.fc_norm

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size // 2, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        x = self.videomae(pixel_values)
        x = self.fc_norm(x)
        logits = self.classifier(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs, target)
        
        loss = self.loss_fn(outputs, target)
        train_accuracy = accuracy(outputs.argmax(-1), target, num_classes=self.output_size, task='multiclass')
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        
        return {"loss": loss, "accuracy": train_accuracy}
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs, target)
        
        loss = self.loss_fn(outputs, target)
        val_accuracy = accuracy(outputs.argmax(-1), target, num_classes=self.output_size, task='multiclass')
        
        self.log("val_loss", loss, on_epoch=True, logger=True)
        self.log("val_accuracy", val_accuracy, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer_parameters = [
            {'params': filter(lambda p: p.requires_grad, self.classifier.parameters()), 'lr': 1e-4},
            {'params': [self.fc_norm.parameters()], 'lr': 1e-5}
        ]
        
        optimizer = AdamW(optimizer_parameters, weight_decay=0.01)
        
        return {
            "optimizer": optimizer,
            "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", verbose=True),
            "monitor": "val_accuracy",
            "interval": "epoch",
            "frequency": 1
        }
    

if __name__ == "__main__":
    x = VideoMAE_ssv2_FinetuneLightning()
    print(x)