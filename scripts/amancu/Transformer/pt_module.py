import torch 
import pytorch_lightning as pl


from torch import nn
from pt_transformer import pointtransformer_seg_repro
from transformers import get_linear_schedule_with_warmup

class PointTransformerModule(pl.LightningModule):
    def __init__(self, learning_rate = 1e-3, warmup_steps = 2):
        super().__init__()

        self.save_hyperparameters()
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.num_training_steps = 100

        self.loss = nn.CrossEntropyLoss()
        self.model = pointtransformer_seg_repro()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pts, feats, out_pts, out_labels, name, offsets = batch
        y = self.model((pts, feats, offsets))
        print(f'y shape: {y.shape}\nout_labels: {out_labels.shape}')
        train_loss = self.loss(y.unsqueeze(0), out_labels)
        self.log("train_loss", train_loss)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        pts, feats, out_pts, out_labels, name, offsets = batch
        y = self.model((pts, feats, offsets))
        print(f'y shape: {y.shape}\nout_labels: {out_labels.shape}')
        val_loss = self.loss(y.unsqueeze(0), out_labels)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.warmup_steps, self.num_training_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]