import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, AUROC
import timm


def build_model(args):
    """Build model from PyTorch Image Models (timm)"""

    assert args.arch in timm.list_models(), "Model not in timm."

    num_classes = len(args.tasks)
    model = timm.create_model(args.arch, pretrained=True, 
        num_classes=num_classes, global_pool=args.global_pool)

    return model


class Classifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.model = build_model(args)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        # metrics
        self.eval_acc = Accuracy(num_classes=args.num_classes)
        self.eval_auroc = AUROC(num_classes=args.num_classes, average=None)

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)
        self.log("train/loss", loss)

        return loss 
    
    def eval_step(self, batch, batch_idx, prefix: str):
        input, target = batch
        output = self.forward(input)
        loss_val = F.binary_cross_entropy_with_logits(output, target)
        self.log(f"{prefix}/loss", loss_val)

        # update metrics
        pred = torch.sigmoid(output)
        self.eval_acc(pred, target.int())
        self.eval_auroc(pred, target.int())
        return {f'{prefix}_loss': loss_val}

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")  

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def validation_epoch_end(self, outputs):
        mean_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = self.eval_acc.compute()
        auc = self.eval_auroc.compute()
        auc_mean = auc.mean()

        self.log('val/accuracy', acc)
        self.log('val/loss', mean_val_loss)
        for i in range(self.args.num_classes):
            self.log(f'val/auc_{i}', auc[i])
        self.log(f'val/auc_mean', auc_mean)
        return {'val_acc': acc, 'val_loss': mean_val_loss,'val_auc':auc}

    def test_epoch_end(self, outputs):
        mean_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        acc = self.eval_acc.compute()
        auc = self.eval_auroc.compute()
        auc_mean = auc.mean()

        self.log('test/accuracy', acc)
        self.log('test/loss', mean_test_loss)
        for i in range(self.args.num_classes):
            task = self.args.tasks[i]
            self.log(f'test/{task}_auc', auc[i])
        self.log(f'test/auc_mean', auc_mean)
        return {'test_acc': acc, 'test_loss': mean_test_loss,'test_auc':auc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                        self.parameters(),
                        betas=(0.9, 0.999),
                        lr=self.args.lr, 
                        weight_decay=self.args.weight_decay
                        )

        scheduler = ReduceLROnPlateau(optimizer,patience=2, mode='min',threshold=0.0001,min_lr=1e-7,verbose=True)
        scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val/loss"}
       
        return [optimizer], [scheduler]
