import torch
import morphx.processing.clouds as clouds
import pytorch_lightning as pl

from pt_module import PointTransformerModule
from pytorch_lightning import Trainer, seed_everything

# Dataloader
from pt_dataloader import PtNodeDataModule

# Model
from pt_transformer import PointTransformerSeg

# Loggers and monitors
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
# wandb_logger = WandbLogger(project="merge-error-detection")
lr_monitor = LearningRateMonitor(logging_interval='step')


# Datamodule hyperparams
SCALE_NORM = 5000
BATCH_SIZE = 1
RADIUS = 3000
NPOINTS = 20000
CTX_SIZE = 20000
ROOT = f'/wholebrain/scratch/amancu/mergeError/Nodes/TrainingGT/R{RADIUS}_downsample300/'
LIMIT_SAMPLES = 1
NUM_WORKERS = 16
SHUFFLE = True
TRANSFORMS = {
    'train': clouds.Compose([clouds.Center(), clouds.Normalization(SCALE_NORM)]), #clouds.Compose([clouds.RandomVariation((-30, 30), distr='normal'),  # in nm
                                      #clouds.Center(),
                                      #clouds.Normalization(SCALE_NORM),
                                      #clouds.RandomRotate(apply_flip=True),
                                      #clouds.ElasticTransform(res=(40, 40, 40), sigma=6),
                                      #clouds.RandomScale(distr_scale=0.1, distr='uniform')]),
    'val': clouds.Compose([clouds.Center(), clouds.Normalization(SCALE_NORM)]),

    'test': clouds.Compose([clouds.Center(), clouds.Normalization(SCALE_NORM)])
}


data_module = PtNodeDataModule(BATCH_SIZE, RADIUS, NPOINTS, CTX_SIZE, TRANSFORMS, root=ROOT, 
                                limit_num_samples=LIMIT_SAMPLES, num_workers=NUM_WORKERS, shuffle=SHUFFLE)


train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()


# Model hyperparams
LEARNING_RATE = 1e-3
WARMUP = 1


hyperparams = {
    "learning_rate": LEARNING_RATE,
    "warmup_steps": WARMUP,

}

model = PointTransformerModule(**hyperparams)

# Trainer hyperparams
seed_everything(42, workers=True)
ACCELERATOR = "gpu"
DEVICES = [0]

trainer = Trainer(limit_train_batches=100, max_epochs=10, accelerator=ACCELERATOR, devices=DEVICES)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)