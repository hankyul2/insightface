import os
import sys
sys.path.append('.')

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy

from recognition.arcface_torch.backbones import get_model
from recognition.arcface_torch.dataset import MXFaceDataset
from recognition.arcface_torch.eval import verification


class Model(pl.LightningModule):
    def __init__(self, rec, batch_size, num_worker):
        super(Model, self).__init__()
        self.rec = rec
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.backbone = get_model('r100', dropout=0.0, fp16=False, num_features=512)
        self.backbone.load_state_dict(torch.load(os.path.join(rec, "backbone.pth"), map_location='cpu'))
        self.fc = nn.Linear(512, 2)

        metrics = MetricCollection({'top@1':Accuracy(top_k=1)})
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.fc(self.backbone(x))
        loss = F.cross_entropy(y_hat, y)
        output = self.train_metrics(y_hat, y)
        self.log_dict({'train_loss': loss})
        self.log_dict(output)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.fc(self.backbone(x))
        loss = F.cross_entropy(y_hat, y)
        output = self.valid_metrics(y_hat, y)
        self.log_dict({'valid_loss': loss})
        self.log_dict(output, add_dataloader_idx=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([
            {'params': self.fc.parameters(), 'lr': 0.001}
        ], momentum=0.9, weight_decay=5e-4)
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_set = MXFaceDataset(root_dir=self.rec)
        train_set.imgidx = np.array(range(1, 4000001))
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_worker)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl_list = []
        for ds_name in ['mfr1_valid']:
            (xs, _), _ = verification.load_bin(os.path.join(self.rec, f'{ds_name}.bin'), (112, 112))
            half = len(xs)//2
            ys = torch.cat([torch.ones(half), torch.zeros(half)]).long()
            ds = [(((x / 255) - 0.5) / 0.5, y) for x, y in zip(xs, ys)]
            dl = DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_worker)
            dl_list.append(dl)
        return dl_list

if __name__ == '__main__':
    model = Model('/home/hankyul/hdd_ext/face/classifier(n=2)', 128, 2)
    trainer = pl.Trainer(gpus='2,3', accelerator='ddp', max_epochs=5, val_check_interval=0.25)
    trainer.fit(model)
