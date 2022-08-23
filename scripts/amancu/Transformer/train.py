

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Trainer, seed_everything

import morphx.processing.clouds as clouds


# Dataloader
from pt_dataloader import PtNodeDataModule

# Model
from pt_transformer import PointTransformerSeg

# Loggers
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project="merge-error-detection")

# Hyperparameters
SCALE_NORM = 5000


BATCH_SIZE = 1




transforms = {
    'train': clouds.Compose([clouds.RandomVariation((-30, 30), distr='normal'),  # in nm
                                      clouds.Center(),
                                      clouds.Normalization(SCALE_NORM),
                                      clouds.RandomRotate(apply_flip=True),
                                      clouds.ElasticTransform(res=(40, 40, 40), sigma=6),
                                      clouds.RandomScale(distr_scale=0.1, distr='uniform')]),
    'val': clouds.Compose([clouds.Center(), clouds.Normalization(SCALE_NORM)]),

    'test': clouds.Compose([clouds.Center(), clouds.Normalization(SCALE_NORM)])
}





