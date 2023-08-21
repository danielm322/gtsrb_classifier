# GTSRB Classifier

Repository for the _German Traffic Sign Recognition Benchmark_ (GTSRB) dataset using ResNet-18 DNN architecture.

## GTSRB Dataset Download
A pre-processed GTSRB dataset is available in this [Talkspirit link](https://cea.talkspirit.com/#/l/permalink/drive/6443269a12d91f3cab7e05ef)

The preprocessed dataset has images of size: (32x32)

## Dataset Shift Detection
The Dataset shift detection experiments use images of size: (128x128). To this end, the GTSRB datamodule uses image resize in the image transformations. 

## Requirements

Overview of required deep learning libraries:

```txt
albumentations==1.2.1
dropblock==0.3.0
lightning-bolts==0.5.0
pytorch-lightning==1.7.2
rich==12.5.1
tensorboard==2.10.0
torch==1.12.1
torchmetrics==0.10.0rc0
torchvision==0.13.1
tqdm==4.64.0
hydra-core==1.3.2
mlflow==1.30.0
```

For a detailed list of requirements, check and use the following files:

*Update:* For factory AI use, the `requirements-pip-local-FAI.txt` file works fine (Use instructions below).

- `requirements-pip-local-FAI.txt` to install requirements on a local PC with pip and GPU or on Factory AI
- `requirements-conda-FAI.txt` to install requirements on Factory-AI with conda.
- `requirements-cpu-only.txt` to install on local pc with only cpu.

## Factory AI setup instructions
As tested on 28 of april 2023, the following instructions have worked succesfully to prepare an 
environment in Factory AI:

* Upload code if you haven't done it yet. For uploading the code you can copy it with `scp` following instructions from factory AI. 
Another option is to use a mirror repository from where you can clone and synchronize code.
* Upload dataset if you haven't done it yet with `scp` following instructions from factory AI.

* Load conda module: 
```bash
module load anaconda/4.9.3
```
* Load CUDA module:
```bash
module load cuda/11.6
```

* Create and activate conda environment with python 3.7:
```bash
conda create -n gtsrb_env python=3.7
conda activate gtsrb_env
```

* Install requirements with pip:
```bash
pip install -r requirements-pip-local-FAI.txt
```
* Launch the training script with 2 GPUs has proven to be launched immediately (no waiting time). So edit your 
slurm script accordingly. See `gtsrb_resnet_daniel.slurm` and `gtsrb_resnet.slurm` for details. 
Be sure to write your own mail, your own path to the environment, the correct number of gpus, etc.

**Note:** Since the integration with hydra and mlflow, you can edit the hyperparameters directly on the configuration files,
or pass them in the command line (see `gtsrb_resnet_daniel.slurm` to see the hydra syntax for command line parameters). 
And you do not have to worry about logging these hyperparameters as they will be automatically logged by both of these libraries.

## Use in local pc
After creating your environment either for cpu or gpu, run training script locally on a PC or laptop:

```bash
>$ python3 train_gtsrb_classifier.py
```
