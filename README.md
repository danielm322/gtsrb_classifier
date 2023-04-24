# GTSRB Classifier

Repository for the _German Traffic Sign Recognition Benchmark_ (GTSRB) dataset using ResNet-18 DNN architecture.

## GTSRB Dataset Download
A pre-processed GTSRB dataset is available in this [Talkspirit link](https://cea.talkspirit.com/#/l/permalink/drive/6443269a12d91f3cab7e05ef)

The preprocessed dataset has images of size: (32x32)

## Requirements

Overview of required deep learning libraries:

```txt
albumentations==1.2.1
dropblock==0.3.0
lightning-bolts==0.5.0
pytorch-lightning==1.7.2
tensorboard==2.10.0
torch==1.12.1
torchmetrics==0.10.0rc0
torchvision==0.13.1
tqdm==4.64.0
```

For a detailed list of requirements, check and use the followinf files:

- `requirements-pip-local.txt` to install requirements on a local PC with pip.
- `requirements-conda-FAI.txt` to install requirements on Factory-AI with conda.


## Usage

Run training script locally on a PC or laptop with GPU:

```bash
>$ python3 train_gtsrb_classifier.py
```

Run training script (slurm script) in HPC (Factory-AI) with 4 GPUs:

```bash
>$ sbatch gtsrb_resnet.slurm
```

- For more details about the employed HPC resources, check the `gtsrb_resnet.slurm` script.