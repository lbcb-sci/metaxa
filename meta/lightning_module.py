import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from jsonargparse import lazy_instance
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from models import SimpleKMerModel
from datasets import MetaDataModule
from schedulers import get_cosine_schedule_with_warmup


class MetaLightningModule(L.LightningModule):
    def __init__(
        self,
        kmer_len: int,
        n_classes: int,
        lr: float = 3e-4,
        wd: float = 0.0,
        smoothing: float = 0.0,
        warmup: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams['n_classes'] = n_classes

        f_in = 5 * kmer_len
        self.backbone = SimpleKMerModel(f_in, n_classes)
        # self.backbone = QuartzNet(5, n_classes)

        self.train_acc = MulticlassAccuracy(n_classes)
        self.train_f1 = MulticlassF1Score(n_classes)
        self.val_acc = MulticlassAccuracy(n_classes)
        self.val_f1 = MulticlassF1Score(n_classes)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        return self.backbone(x, attn_mask)

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
        x, attn_mask, y = batch
        preds = self(x, attn_mask)

        loss = F.cross_entropy(preds, y, label_smoothing=self.hparams.smoothing)
        self.log('train_loss_step', loss, prog_bar=True)

        self.train_acc(preds, y)
        self.log('train_acc_step', self.train_acc)

        self.train_f1(preds, y)
        self.log('train_f1_step', self.train_f1)

        return loss

    def validation_step(self, batch, batch_idx):
        x, attn_mask, y = batch
        preds = self(x, attn_mask)

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
        parser.add_lightning_class_args(ModelCheckpoint, 'mc')
        parser.set_defaults(
            {
                'mc.monitor': 'val_loss',
                'mc.save_top_k': -1,
                'mc.mode': 'min',
                'mc.filename': '{step}-{val_loss:.6f}',
            }
        )

        parser.set_defaults(
            {'trainer.logger': lazy_instance(WandbLogger, project='meta_classifier')}
        )

        parser.link_arguments('data.kmer_len', 'model.kmer_len')
        parser.link_arguments(
            'data.n_classes', 'model.n_classes', apply_on='instantiate'
        )


if __name__ == '__main__':
    cli = MetaLightningCLI(
        MetaLightningModule, MetaDataModule, save_config_kwargs={'overwrite': True}
    )
