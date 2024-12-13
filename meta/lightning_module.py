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
import wandb
import wandb.util

from datasets import MetaDataModule
from schedulers import get_cosine_schedule_with_warmup
from losses import nt_xent_loss

from typing import Optional


class MetaLightningModule(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        n_species: int,
        n_genus: int,
        lr: float = 1e-3,
        min_lr: float = 1e-6,
        wd: float = 0.0,
        smoothing: float = 0.0,
        warmup: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams['n_species'] = n_species
        self.hparams['n_genus'] = n_genus

        self.backbone = backbone
        self.fc_species = nn.Linear(backbone.d_model, n_species)
        self.fc_genus = nn.Linear(backbone.d_model, n_genus)

        self.train_acc = MulticlassAccuracy(n_species)
        self.train_f1 = MulticlassF1Score(n_species)
        self.val_acc = MulticlassAccuracy(n_species)
        self.val_f1 = MulticlassF1Score(n_species)

        self.train_acc_genus = MulticlassAccuracy(n_genus)
        self.train_f1_genus = MulticlassF1Score(n_genus)
        self.val_acc_genus = MulticlassAccuracy(n_genus)
        self.val_f1_genus = MulticlassF1Score(n_genus)

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.backbone(x, lens)
        logits_species = self.fc_species(x)
        logits_genus = self.fc_genus(x)
        # logits_genus = None

        return logits_species, logits_genus

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.lr,
            weight_decay=self.hparams.wd,
        )

        min_lambda = self.hparams.lr / self.hparams.min_lr
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.hparams.warmup, self.trainer.max_steps, min_lambda
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def training_step(self, batch, batch_idx):
        x, lens, ys, yg = batch
        logits_species, logits_genus = self(x, lens)

        species_loss = F.cross_entropy(
            logits_species, ys, label_smoothing=self.hparams.smoothing
        )
        self.log('train_loss_step', species_loss)

        genus_loss = F.cross_entropy(
            logits_genus, yg, label_smoothing=self.hparams.smoothing
        )
        self.log('train_genus_loss_step', genus_loss)

        loss = species_loss + genus_loss
        # loss = species_loss
        self.log('train_total_loss_step', loss, prog_bar=True)

        self.train_acc(logits_species, ys)
        self.log('train_acc_step', self.train_acc)

        self.train_f1(logits_species, ys)
        self.log('train_f1_step', self.train_f1)

        self.train_acc_genus(logits_genus, yg)
        self.log('train_acc_genus_step', self.train_acc_genus)

        self.train_f1_genus(logits_genus, yg)
        self.log('train_f1_genus_step', self.train_f1_genus)

        return loss

    def validation_step(self, batch, batch_idx):
        x, lens, ys, yg = batch
        logits_species, logits_genus = self(x, lens)

        species_loss = F.cross_entropy(logits_species, ys)
        genus_loss = F.cross_entropy(logits_genus, yg)

        self.val_acc(logits_species, ys)
        self.log('val_acc', self.val_acc)

        self.val_f1(logits_species, ys)
        self.log('val_f1', self.val_f1, prog_bar=True)

        self.val_acc_genus(logits_genus, yg)
        self.log('val_acc_genus', self.val_acc_genus)

        self.val_f1_genus(logits_genus, yg)
        self.log('val_f1_genus', self.val_f1_genus, prog_bar=True)

        loss = species_loss + genus_loss
        # loss = species_loss
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
            {
                'trainer.logger': lazy_instance(
                    WandbLogger, project='meta_classifier', id=wandb.util.generate_id()
                )
            }
        )

        # parser.link_arguments('data.kmer_len', 'model.kmer_len')
        parser.link_arguments(
            'data.n_species', 'model.n_species', apply_on='instantiate'
        )
        parser.link_arguments('data.n_genus', 'model.n_genus', apply_on='instantiate')


if __name__ == '__main__':
    cli = MetaLightningCLI(
        MetaLightningModule,
        MetaDataModule,
        save_config_kwargs={'overwrite': True},
        seed_everything_default=42,
    )
