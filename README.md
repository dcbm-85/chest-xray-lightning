# CheXpert classification with Pytorch Lightning

## Introduction 

**CheXpert classification with Pytorch Lightning** is a tool for multi-label classification of chest X-Ray images with deep neural networks. It is based on PyTorch Lightning and [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) and was designed using the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset.

## Data
The CheXpert dataset contains 224,316 chest radiographs of 65,240 unique patients. The high-resolution (~439Gb) or downsampled version (~11 Gb) and can be downloaded after registration at this [link](https://stanfordmlgroup.github.io/competitions/chexpert/). 

There are 200 X-Ray studies from 200 patients for validation and the official test set is not public. Due to the nature of our investigation, in this study we split the official training set into a new training (70%), validation (10%) and test sets (20%) with no patient overlap between the sets. We also use the U-Ones approach, replacing uncertainty labels (-1) by 1.

## Using the repository

### Installation using Anaconda

```
conda create --name chexpert_pl python=3
conda activate chexpert_pl
git clone https://github.com/dcbm-85/chest-xray-lightning
cd chest-xray-lightning
pip install -r requirements.txt
```

### Data preparation

Download and unzip the downsampled version version of the CheXpert dataset in `./data/`. Then make the new splits of the dataset using default parameters:

```shell
$ python util/split.py 
```
### Traning models

You can train models using the script

```shell
$ python train.py configs/train.yaml 
```

With ```timm``` you can choose the pretrained model of your choice for fine-tuning and making the corresponding change in the ```configs/config_train.yaml```. As our default model we use ```densenet121```, following the CheXpert paper, batch size of 64, Adam optmizer with learning rate 0.001 and train the model for 5 epochs, saving the model with the lowest loss values. The checkpoint and metrics will be save in the default Lighting logger folder ```lightning_logs```.

### Results

For testing your model you can run

```shell
$ python test.py configs/test.yaml 
```

For the deafult traning and the test split we obtain the following AUCROC for the competition tasks:

|Cardiomegaly|Edema|Consolidation|Atelectasis|Pleural_Effusion| Mean|
|---------|-----|---|----|-----|-----|
|0.8433|0.8563|0.6897|0.7170|0.8748|0.7962|

Using the 200 patients official validation dataset:

|Cardiomegaly|Edema|Consolidation|Atelectasis|Pleural_Effusion| Mean|
|---------|-----|---|----|-----|-----|
|0.7918|0.9218|0.8637|0.8421|0.9279|0.8695|

The model used for these results is saved on `util/chexpert_model.ckpt`.
## Reference <a name="reference"></a>

<b id="foot1">1.</b>  Irvin et al. [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://arxiv.org/abs/1901.07031), AAAI 2019