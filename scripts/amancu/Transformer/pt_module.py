import torch 
import os
import pytorch_lightning as pl
import numpy as np

from torch import nn
from pt_transformer import pointtransformer_seg_repro
from transformers import get_linear_schedule_with_warmup
from torchmetrics.functional import accuracy
from syconn.proc.meshes import mesh2obj_file_colors

# define colors for visualization
GREY = np.array([180., 180., 180., 255.])
PINK = np.array([10., 255., 10., 255.])

class PointTransformerModule(pl.LightningModule):
    def __init__(self, learning_rate = 1e-3, warmup_steps = 2, logging=False, cloud_path=None):
        super().__init__()
        self.save_hyperparameters()

        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.num_training_steps = 100
        self.logging = logging
        self.cloud_path = cloud_path

        self.loss = nn.CrossEntropyLoss(weight=[])
        self.model = pointtransformer_seg_repro(c=3, k=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pts, feats, out_pts, out_labels, pts_offset, out_offset, name = batch
        pts = pts.squeeze(0)
        feats = feats.squeeze(0)
        out_pts = out_pts.squeeze(0)
        out_labels = out_labels.squeeze(0)
        pts_offset = pts_offset.squeeze(0)
        out_offset = out_offset.squeeze(0)
        y_hat = self.model((pts, feats, pts_offset))
        # print(f'in train batch pts: {pts.shape} and feats shape: {feats.shape} and offset {pts_offset.shape}')
        train_loss = self.loss(y_hat.squeeze(), out_labels.squeeze())
        train_acc = accuracy(torch.argmax(y_hat, dim=1), out_labels)
        self.log("train_loss", train_loss)
        self.log("train_acc", train_acc)
        ind = np.random.randint(0, pts_offset.shape[0])
        return {'loss': train_loss, 'acc':train_acc, 'pts': pts[:pts_offset[ind]],'pred': y_hat[:pts_offset[ind]], 'name': name}
    
    def validation_step(self, batch, batch_idx):
        pts, feats, out_pts, out_labels, pts_offset, out_offset, name = batch
        pts = pts.squeeze(0)
        feats = feats.squeeze(0)
        out_pts = out_pts.squeeze(0)
        out_labels = out_labels.squeeze(0)
        pts_offset = pts_offset.squeeze(0)
        out_offset = out_offset.squeeze(0)
        # print(f'in val batch pts: {pts.shape} and feats shape: {feats.shape} and offset: {pts_offset.shape}')
        y_hat = self.model((pts, feats, pts_offset))
        # print(y.shape)
        val_loss = self.loss(y_hat.squeeze(), out_labels.squeeze())
        val_acc = accuracy(torch.argmax(y_hat, dim=1), out_labels)
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        ind = np.random.randint(0, pts_offset.shape[0])
        return {'loss': val_loss, 'acc': val_acc, 'pts': pts[:pts_offset[ind]],'pred': y_hat[:pts_offset[ind]], 'out_labels': out_labels[:pts_offset[ind]], 'name': name}

    def training_epoch_end(self, training_step_outputs):

        losses, accs, pts, preds, names = [], [], [], [], []

        for out in training_step_outputs:
            loss, acc, pt, pred, name = out['loss'], out['acc'], out['pts'], out['pred'], out['name'] 
            losses.append(loss)
            accs.append(acc)
            pts.append(pts)
            preds.append(pred)
            names.append(name)

        accs = torch.Tensor(accs)
        epoch_accuracy = torch.sum(accs)/len(accs)
        self.log('epoch/train_acc', epoch_accuracy)
        # wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

    def validation_epoch_end(self, validation_step_outputs):
        losses, accs, pts, preds, out_labels, names = [], [], [], [], [], []

        for out in validation_step_outputs:
            loss, acc, pt, pred, out_label, name = out['loss'], out['acc'], out['pts'], out['pred'], out['out_labels'], out['name'] 
            losses.append(loss)
            accs.append(acc)
            pts.append(pt)
            preds.append(pred)
            out_labels.append(out_label)
            names.append(name)

        accs = torch.Tensor(accs)
        epoch_accuracy = torch.sum(accs)/len(accs)

        epoch_accuracy = torch.sum(accs)/len(accs)
        self.log('epoch/val_acc', epoch_accuracy)

        # write val predicted clouds to the specified path for later visualization
        if self.logging:
            # select 3 validation point clouds to save
            indcs = np.random.randint(0,len(pts),size=2)
            for ind in indcs:
                # for prediction
                # print(f'pts shape {preds[ind].shape}')
                colors = np.full(shape=(pts[ind].shape[0], 4,), fill_value=GREY)
                mask = np.where(torch.argmax(preds[ind], dim=1).cpu().detach().numpy() == 1)[0]
                mask = np.array([[x] for x in mask])
                try:
                    np.put_along_axis(colors, mask, PINK, axis=0)
                except:
                    print("No foreground labels in original context.")
                    pass
                mesh2obj_file_colors(os.path.expanduser(
                    self.cloud_path + f'epoch{self.trainer.current_epoch}_{names[ind]}_pred.ply'),
                    [np.array([]), pts[ind].cpu().detach().numpy(), np.array([])], colors)
                
                # for GT
                # print(f'shape {out_labels[ind].unsqueeze(-1).shape}')
                colors = np.full(shape=(pts[ind].shape[0], 4,), fill_value=GREY)
                mask = np.where(out_labels[ind].cpu().detach().numpy() == 1)[0]
                mask = np.array([[x] for x in mask])
                try:
                    np.put_along_axis(colors, mask, PINK, axis=0)
                except:
                    print("No foreground labels in original context.")
                    pass
                mesh2obj_file_colors(os.path.expanduser(
                    self.cloud_path + f'epoch{self.trainer.current_epoch}_{names[ind]}_real.ply'),
                    [np.array([]), pts[ind].cpu().detach().numpy(), np.array([])], colors)

        # wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.warmup_steps, self.num_training_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]