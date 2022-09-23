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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

PROJECT_NAME = '6.5ksampleRun_allMergers_additionalData_noAugm'
TB_PATH = '/cajal/scratch/users/amancu/merge_error/transformer/tensorboard/'
WANDB_PATH = '/cajal/scratch/users/amancu/merge_error/transformer/wandb/'
CLOUD_PATH = f'/cajal/scratch/users/amancu/merge_error/transformer/trainings/{PROJECT_NAME}/'
LOG = True

tensorboard_logger = TensorBoardLogger(save_dir=TB_PATH, name=PROJECT_NAME)
wandb_logger = WandbLogger(save_dir=WANDB_PATH, name=PROJECT_NAME, project="merge-error-detection")
lr_monitor = LearningRateMonitor(logging_interval='step')

# Datamodule hyperparams
SCALE_NORM = 5000
BATCH_SIZE = 8
RADIUS = 3000
NPOINTS = 20000
CTX_SIZE = 20000
# ROOT = f'/wholebrain/scratch/amancu/mergeError/Nodes/TrainingGT/R{RADIUS}_downsample300/'
ROOT = f'/cajal/scratch/users/amancu/merge_error/transformer/GT/training/R{RADIUS}_downsample300/'
LIMIT_SAMPLES = None
NUM_WORKERS = 8
SHUFFLE = True
ADDITIONAL_DATA = True        # Add true negatives to the datset
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
                                limit_num_samples=LIMIT_SAMPLES, num_workers=NUM_WORKERS, shuffle=SHUFFLE, additional_data=ADDITIONAL_DATA)


train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()


# Model hyperparams
LEARNING_RATE = 1e-3
WARMUP = 1

n_gpus = torch.cuda.device_count()
print(f"#gpus available: {n_gpus}")
torch.cuda.set_device(0)

hyperparams = {
    "learning_rate": LEARNING_RATE,
    "warmup_steps": WARMUP,
    "logging": LOG,
    "cloud_path": CLOUD_PATH,
}


model = PointTransformerModule(**hyperparams)

# Trainer hyperparams
seed_everything(42, workers=True)
ACCELERATOR = "gpu"
DEVICE = [0]

wandb_logger.watch(model)

trainer = Trainer(limit_train_batches=100, max_epochs=100, accelerator=ACCELERATOR, devices=DEVICE,
                auto_select_gpus=False, log_every_n_steps=1,) #logger=[wandb_logger, tensorboard_logger], callbacks=[lr_monitor])
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)