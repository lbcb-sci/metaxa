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
from flash_attn.ops.triton.layer_norm import RMSNorm

from collections import Counter

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

        self.train_acc = MulticlassAccuracy(n_classes + 1, ignore_index=-1)
        self.train_f1 = MulticlassF1Score(n_classes + 1, ignore_index=-1)
        self.val_acc = MulticlassAccuracy(n_classes + 1, ignore_index=-1)
        self.val_f1 = MulticlassF1Score(n_classes + 1, ignore_index=-1)

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.backbone(x, lens)
        return x

    def configure_optimizers(self):
        decay_params = get_parameter_names(self, [RMSNorm])
        decay_params = [name for name in decay_params if 'bias' not in name]

        grouped_params = [
            {
                'params': [p for n, p in self.named_parameters() if n in decay_params],
                'weight_decay': self.hparams.wd,
            },
            {
                'params': [
                    p for n, p in self.named_parameters() if n not in decay_params
                ],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            grouped_params,
            self.hparams.lr,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.hparams.warmup, num_training_steps=self.trainer.max_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def training_step(self, batch, batch_idx):
        ids, x, lens, y = batch
        preds = self.backbone(x, lens).transpose(1, 2)

        preds_loss = F.cross_entropy(
            preds, y, label_smoothing=self.hparams.smoothing, ignore_index=-1
        )
        self.log('train_preds_loss_step', preds_loss)

        loss = preds_loss
        self.log('train_loss_step', loss, prog_bar=True)

        self.train_acc(preds, y)
        self.log('train_acc_step', self.train_acc)

        self.train_f1(preds, y)
        self.log('train_f1_step', self.train_f1)

        return loss

    def validation_step(self, batch, batch_idx):
        ids, x, lens, y = batch
        preds = self(x, lens).transpose(1, 2)

        loss = F.cross_entropy(preds, y, ignore_index=-1)

        preds = torch.argmax(preds, dim=1).cpu().numpy()
        pred_cls = []
        lens = torch.floor((lens - 31) / 5 + 1).int()
        for p, l in zip(preds, lens):
            c = Counter(p[:l].tolist())
            del c[516]

            if len(c) == 0:
                pred_cls.append(516)
            else:
                pred_cls.append(c.most_common(1)[0][0])

        preds = torch.tensor(pred_cls, device=self.device)
        y = torch.tensor(ids, device=self.device)

        self.val_acc(preds, y)
        self.log('val_acc', self.val_acc)

        self.val_f1(preds, y)
        self.log('val_f1', self.val_f1, prog_bar=True)

        self.log('val_loss', loss, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer. Taken from transformers package.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f'{name}.{n}'
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


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
