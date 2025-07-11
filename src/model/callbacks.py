from transformers import TrainerCallback


class LayerUnfreeze(TrainerCallback):
    """
    Trainer callback for gradual layer unfreezing. This unfreezes
    layers in the models encoder block. Unfreezes 1 layer every
    <epoch_step> epochs with a delay of <delay_start> epochs before
    unfreezing layers. Will unfreeze all layers after <delay_unfreeze_all>
    epochs.

    Attributes:
        delay_start (int):              Delay on when to start unfreezing layers.
        epoch_step (int):               Number of epochs in between unfreezing operations.
        delay_unfreeze_all (int):       Delay on when to unfreeze all layers.
    """
    def __init__(self, delay_start: int, epoch_step: int, delay_unfreeze_all: int):
        self.delay_start = delay_start
        self.epoch_step = epoch_step
        self.delay_unfreeze_all = delay_unfreeze_all
        
        self.unfrozen_layers = 0 # counts how many unfrozen encoder layers we have
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        model = kwargs['model']
        current_epoch = state.epoch
        num_encoder_layers = len(model.videomae.encoder.layer)
        
        if(self.unfrozen_layers >= num_encoder_layers):
            return 
        
        # starts unfreezing layers after a delay
        # then unfreeze 1 layer per step
        effective_epoch = current_epoch-self.delay_start
        if(self.delay_unfreeze_all >= current_epoch):
            for i in range(1, num_encoder_layers+1):
                if i > num_encoder_layers - self.unfrozen_layers:
                    for param in model.videomae.encoder.layer[i-1].parameters():
                        param.requires_grad = True
        elif(effective_epoch >= 0 and
           (effective_epoch)%self.epoch_step == 0 and 
           self.unfrozen_layers <= num_encoder_layers):
            
            self.unfrozen_layers += 1
            for i in range(1, num_encoder_layers+1):
                if i > num_encoder_layers - self.unfrozen_layers:
                    for param in model.videomae.encoder.layer[i-1].parameters():
                        param.requires_grad = True
                else:
                    for param in model.videomae.encoder.layer[i-1].parameters():
                        param.requires_grad = False