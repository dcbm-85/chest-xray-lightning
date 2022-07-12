import argparse
import yaml
from easydict import EasyDict as edict
import pandas as pd

from pytorch_lightning import Trainer

from util.dataset import CheXpertDM
from models import Classifier

# args
parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")

def main():
 
    args = parser.parse_args()
    with open(args.cfg_path) as file:
        args = edict(yaml.safe_load(file))
    
    data_module = CheXpertDM(args)

    model = Classifier(args).load_from_checkpoint(checkpoint_path=args.ckpt_path)
    trainer = Trainer(logger=False, accelerator='gpu', devices=1)
    
    # list of length 1
    test_results = trainer.test(model, data_module.test_dataloader())
    
    # saveing test results to csv
    metrics = pd.Series(test_results[0])
    metrics.to_csv(args.output_path)

if __name__ == '__main__':
    main()