# GTSRB Classifier

Repository for the _German Traffic Sign Recognition Benchmark_ (GTSRB) dataset using ResNet-18 DNN architecture.

## GTSRB Dataset Download
A pre-processed GTSRB dataset is available in this [Talkspirit link](https://cea.talkspirit.com/#/l/permalink/drive/6443269a12d91f3cab7e05ef)

The preprocessed dataset has images with size: (32x32)

## Usage

Run training script locally on a PC pr laptop with GPU:

```bash
>$ python3 train_gtsrb_classifier.py
```

Run training script (slurm script) in HPC (Factory-AI) with 4 GPUs:

```bash
>$ sbatch gtsrb_resnet.slurm
```

- For more details about the employed HPC resources, check the `gtsrb_resnet.slurm` script.