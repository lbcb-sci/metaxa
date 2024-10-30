import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from jsonargparse import lazy_instance
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from datasets import MetaDataModule
from schedulers import get_cosine_schedule_with_warmup
from losses import nt_xent_loss

from typing import Optional


class MetaLightningModule(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        lr: float = 3e-4,
        wd: float = 0.0,
        smoothing: float = 0.0,
        warmup: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams['n_classes'] = n_classes

        # KMER model
        # f_in = 5 * kmer_len
        # self.backbone = SimpleKMerModel(f_in, n_classes)

        # CNN model
        # f_in = 4
        # self.backbone = CNNEncodingTransformer(f_in, 512, 8, 8, 2048, n_classes)
        self.backbone = backbone
        # self.head = nn.Sequential(nn.Linear(512, 128))

        self.train_acc = MulticlassAccuracy(n_classes)
        self.train_f1 = MulticlassF1Score(n_classes)
        self.val_acc = MulticlassAccuracy(n_classes)
        self.val_f1 = MulticlassF1Score(n_classes)

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x, _ = self.backbone(x, lens)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.lr,
            weight_decay=self.hparams.wd,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.hparams.warmup, num_training_steps=self.trainer.max_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def training_step(self, batch, batch_idx):
        x, lens, y = batch
        preds, _ = self.backbone(x, lens)  # tf -> transformed
        # cls_tokens = self.backbone(x, lens)  # tf -> transformed
        # cls_tokens = self.head(cls_tokens)

        # preds = preds[: len(y)]
        z_o, z_t = torch.split(preds, len(y))

        """if torch.distributed.get_rank() == 0 and batch_idx % 1000 == 0:
            torch.save(z_o.detach(), f'z_o_{batch_idx}.pt')
            torch.save(z_t.detach(), f'z_t_{batch_idx}.pt')"""

        contrastive_loss = nt_xent_loss(z_t, z_o)
        self.log('train_contrastive_loss_step', contrastive_loss)

        y = torch.cat([y, y], dim=0)
        preds_loss = F.cross_entropy(preds, y, label_smoothing=self.hparams.smoothing)
        self.log('train_preds_loss_step', preds_loss)

        loss = preds_loss + contrastive_loss
        self.log('train_loss_step', loss, prog_bar=True)

        self.train_acc(preds, y)
        self.log('train_acc_step', self.train_acc)

        self.train_f1(preds, y)
        self.log('train_f1_step', self.train_f1)

        return loss

    def validation_step(self, batch, batch_idx):
        x, lens, y = batch
        preds = self(x, lens)

        loss = F.cross_entropy(preds, y)

        self.val_acc(preds, y)
        self.log('val_acc', self.val_acc)

        self.val_f1(preds, y)
        self.log('val_f1', self.val_f1, prog_bar=True)

        self.log('val_loss', loss, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)


class MetaLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, 'mc')
        parser.set_defaults(
            {
                'mc.monitor': 'val_loss',
                'mc.save_top_k': -1,
                'mc.mode': 'min',
                'mc.filename': '{step}-{val_loss:.6f}',
            }
        )

        # LearningRateMonitor
        parser.add_lightning_class_args(LearningRateMonitor, 'lrm')

        # Use wandb logger
        parser.set_defaults(
            {'trainer.logger': lazy_instance(WandbLogger, project='meta_classifier')}
        )

        # parser.link_arguments('data.kmer_len', 'model.kmer_len')
        parser.link_arguments(
            'data.n_classes', 'model.n_classes', apply_on='instantiate'
        )
        parser.link_arguments(
            'data.n_classes',
            'model.backbone.init_args.f_out',
            apply_on='instantiate',
        )


if __name__ == '__main__':
    cli = MetaLightningCLI(
        MetaLightningModule, MetaDataModule, save_config_kwargs={'overwrite': True}
    )
