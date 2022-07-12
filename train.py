import argparse
import yaml
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from util.dataset import CheXpertDM
from models import Classifier

def get_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                        help="Path to the config file in yaml format")
    return parser.parse_args() 

def main(args):
    
    with open(args.cfg_path) as file:
        args = edict(yaml.safe_load(file))

    data_module = CheXpertDM(args)
    model = Classifier(args)

    checkpoint = ModelCheckpoint(monitor='val/loss', mode="min", save_weights_only=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(accelerator="gpu", devices=1,
                        num_nodes=1, 
                        precision=16,
                        max_epochs=args.epochs,
                        callbacks=[checkpoint,lr_monitor],
                        val_check_interval=0.25)

    trainer.fit(model, data_module)

if __name__ == "__main__":
    args = get_args()
    main(args)