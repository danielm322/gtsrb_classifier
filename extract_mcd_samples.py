# from collections import namedtuple
import numpy as np
# import matplotlib.pyplot as plt
# import random
# import pandas as pd
# import seaborn as sns
# from PIL import Image
from icecream import ic
import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms as transform_lib
# from pytorch_lightning.callbacks import TQDMProgressBar
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
from datasets import GtsrbModule
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules import STL10DataModule
from models import ResnetModule
from dropblock import DropBlock2D
from ls_ood_detect_cea.uncertainty_estimation import Hook
from ls_ood_detect_cea.uncertainty_estimation import get_latent_represent_mcd_samples
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z
from ls_ood_detect_cea.ood_detection_dataset import build_ood_detection_ds
from ls_ood_detect_cea.dimensionality_reduction import plot_samples_pacmap
from ls_ood_detect_cea.detectors import KDEClassifier
from ls_ood_detect_cea.metrics import get_ood_detector_results, plot_roc_ood_detector, get_hz_detector_results
from ls_ood_detect_cea.detectors import DetectorKDE
from ls_ood_detect_cea import get_hz_scores

# Datasets paths
dataset_path = "./gtsrb-data/"
cifar10_data_dir = "./ood_datasets/cifar10_data/"
stl10_data_dir = "./ood_datasets/stl10-data/"
# Trained model path
gtsrb_model_path = "./lightning_logs/2023-09-04_09-20-29_vanilla/epoch=206-step=115299.ckpt"  # Daniel
MC_SAMPLES = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ################################################################################################
    ################################################################################################
    #                                 LOAD DATASETS                                   ##############
    ################################################################################################
    ################################################################################################

    ##################################################################
    # GTSRB NORMAL DATASET
    ###################################################################
    gtsrb_normal_dm = GtsrbModule(img_size=(128, 128), data_path=dataset_path, batch_size=1, shuffle=False)

    gtsrb_normal_dm.setup(stage='fit')
    gtsrb_normal_dm.setup(stage='validate')
    gtsrb_normal_dm.setup(stage='test')

    gtsrb_normal_train_loader = gtsrb_normal_dm.train_dataloader()
    gtsrb_normal_valid_loader = gtsrb_normal_dm.val_dataloader()
    gtsrb_normal_test_loader = gtsrb_normal_dm.test_dataloader()

    #####################################################################
    # GTSRB ANOMALIES DATASET
    #####################################################################
    gtsrb_anomal_dm = GtsrbModule(
        img_size=(128, 128),
        data_path=dataset_path,
        batch_size=1,
        anomaly_transforms=True,
        shuffle=True
    )

    gtsrb_anomal_dm.setup(stage='fit')
    gtsrb_anomal_dm.setup(stage='validate')
    gtsrb_anomal_dm.setup(stage='test')

    gtsrb_anomal_train_loader = gtsrb_anomal_dm.train_dataloader()
    gtsrb_anomal_valid_loader = gtsrb_anomal_dm.val_dataloader()
    gtsrb_anomal_test_loader = gtsrb_anomal_dm.test_dataloader()

    ######################################################################
    # CIFAR10 OOD DATASET
    ######################################################################
    cifar10_dm = CIFAR10DataModule(data_dir=cifar10_data_dir,
                                   val_split=0.2,
                                   num_workers=10,
                                   normalize=True,
                                   batch_size=1,
                                   seed=10,
                                   drop_last=True,
                                   shuffle=True)

    # cifar10_transforms = transform_lib.Compose([
    #     transform_lib.Resize((64, 64)),
    #     transform_lib.ToTensor(),
    #     transform_lib.Normalize(
    #         mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    #     )
    # ])
    cifar10_transforms = transform_lib.Compose([
        transform_lib.Resize((128, 128)),
        transform_lib.ToTensor(),
        transform_lib.Normalize(
            mean=[0.3337, 0.3064, 0.3171],
            std=[0.2672, 0.2564, 0.2629]
        )
    ])

    cifar10_dm.train_transforms = cifar10_transforms
    cifar10_dm.test_transforms = cifar10_transforms
    cifar10_dm.val_transforms = cifar10_transforms

    cifar10_dm.prepare_data()
    cifar10_dm.setup(stage='fit')
    cifar10_dm.setup(stage='test')

    cifar10_train_loader = cifar10_dm.train_dataloader()
    cifar10_valid_loader = cifar10_dm.val_dataloader()
    cifar10_test_loader = cifar10_dm.test_dataloader()
    # ic(len(cifar10_train_loader));
    # ic(len(cifar10_valid_loader));
    # ic(len(cifar10_test_loader));

    ##########################################################
    # STL-10 OoD
    #########################################################
    stl10_dm = STL10DataModule(data_dir=stl10_data_dir,
                               train_val_split=3000,
                               num_workers=10,
                               batch_size=1,
                               seed=10,
                               drop_last=True,
                               shuffle=True)

    # stl10_transforms = transform_lib.Compose([
    #     transform_lib.Resize((32, 32)),
    #     transform_lib.ToTensor(),
    #     transform_lib.Normalize(
    #         mean=(0.43, 0.42, 0.39),
    #         std=(0.27, 0.26, 0.27)
    #     )
    # ])

    stl10_transforms = transform_lib.Compose([
        transform_lib.Resize((128, 128)),
        transform_lib.ToTensor(),
        transform_lib.Normalize(
            mean=(0.3337, 0.3064, 0.3171),
            std=(0.2672, 0.2564, 0.2629)
        )
    ])

    stl10_dm.train_transforms = stl10_transforms
    stl10_dm.test_transforms = stl10_transforms
    stl10_dm.val_transforms = stl10_transforms

    stl10_dm.prepare_data()
    stl10_train_loader = stl10_dm.train_dataloader_labeled()
    stl10_valid_loader = stl10_dm.val_dataloader_labeled()
    stl10_test_loader = stl10_dm.test_dataloader()

    ####################################################################
    # Load trained model
    ####################################################################
    gtsrb_model = ResnetModule.load_from_checkpoint(checkpoint_path=gtsrb_model_path)

    gtsrb_model.eval();

    # Add Hooks
    gtsrb_model_dropblock2d_layer_hook = Hook(gtsrb_model.model.dropblock2d_layer)

    # Monte Carlo Dropout - Enable Dropout @ Test Time!
    def resnet18_enable_dropblock2d_test(m):
        if type(m) == DropBlock2D:
            m.train()

    gtsrb_model.to(device);
    gtsrb_model.eval();
    gtsrb_model.apply(resnet18_enable_dropblock2d_test); # enable dropout

    ####################################################################################################################
    ####################################################################################################################
    #########################################################################
    # Extract MCDO latent samples
    #########################################################################
    gtsrb_resnet_gtsrb_normal_train_16mc_samples = get_latent_represent_mcd_samples(gtsrb_model.model,
                                                                                    gtsrb_normal_train_loader,
                                                                                    MC_SAMPLES,
                                                                                    gtsrb_model_dropblock2d_layer_hook,
                                                                                    get_2d_rep_mean=True)

    ic(gtsrb_resnet_gtsrb_normal_train_16mc_samples.shape)
    gtsrb_resnet_gtsrb_normal_valid_16mc_samples = get_latent_represent_mcd_samples(gtsrb_model.model,
                                                                                    gtsrb_normal_valid_loader,
                                                                                    MC_SAMPLES,
                                                                                    gtsrb_model_dropblock2d_layer_hook,
                                                                                    get_2d_rep_mean=True)

    gtsrb_resnet_gtsrb_normal_test_16mc_samples = get_latent_represent_mcd_samples(gtsrb_model.model,
                                                                                   gtsrb_normal_test_loader,
                                                                                   MC_SAMPLES,
                                                                                   gtsrb_model_dropblock2d_layer_hook,
                                                                                   get_2d_rep_mean=True)

    gtsrb_resnet_gtsrb_anomal_valid_16mc_samples = get_latent_represent_mcd_samples(gtsrb_model.model,
                                                                                    gtsrb_anomal_valid_loader,
                                                                                    MC_SAMPLES,
                                                                                    gtsrb_model_dropblock2d_layer_hook,
                                                                                    get_2d_rep_mean=True)

    gtsrb_resnet_gtsrb_anomal_test_16mc_samples = get_latent_represent_mcd_samples(gtsrb_model.model,
                                                                                   gtsrb_anomal_test_loader,
                                                                                   MC_SAMPLES,
                                                                                   gtsrb_model_dropblock2d_layer_hook,
                                                                                   get_2d_rep_mean=True)

    gtsrb_resnet_cifar10_valid_16mc_samples = get_latent_represent_mcd_samples(gtsrb_model.model,
                                                                               cifar10_valid_loader,
                                                                               MC_SAMPLES,
                                                                               gtsrb_model_dropblock2d_layer_hook,
                                                                               get_2d_rep_mean=True)

    gtsrb_resnet_cifar10_test_16mc_samples = get_latent_represent_mcd_samples(gtsrb_model.model,
                                                                              cifar10_test_loader,
                                                                              MC_SAMPLES,
                                                                              gtsrb_model_dropblock2d_layer_hook,
                                                                              get_2d_rep_mean=True)

    gtsrb_resnet_stl10_valid_16mc_samples = get_latent_represent_mcd_samples(gtsrb_model.model,
                                                                             stl10_valid_loader,
                                                                             MC_SAMPLES,
                                                                             gtsrb_model_dropblock2d_layer_hook,
                                                                             get_2d_rep_mean=True)

    gtsrb_resnet_stl10_test_16mc_samples = get_latent_represent_mcd_samples(gtsrb_model.model,
                                                                            stl10_test_loader,
                                                                            MC_SAMPLES,
                                                                            gtsrb_model_dropblock2d_layer_hook,
                                                                            get_2d_rep_mean=True)

    # Calculate entropies
    _, gtsrb_rn18_h_z_gtsrb_normal_train_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_normal_train_16mc_samples,
                                                                 mcd_samples_nro=MC_SAMPLES)
    _, gtsrb_rn18_h_z_gtsrb_normal_valid_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_normal_valid_16mc_samples,
                                                                 mcd_samples_nro=MC_SAMPLES)
    _, gtsrb_rn18_h_z_gtsrb_normal_test_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_normal_test_16mc_samples,
                                                                mcd_samples_nro=MC_SAMPLES)
    _, gtsrb_rn18_h_z_gtsrb_anomal_valid_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_anomal_valid_16mc_samples,
                                                                 mcd_samples_nro=MC_SAMPLES)
    _, gtsrb_rn18_h_z_gtsrb_anomal_test_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_anomal_test_16mc_samples,
                                                                mcd_samples_nro=MC_SAMPLES)
    _, gtsrb_rn18_h_z_cifar10_valid_samples_np = get_dl_h_z(gtsrb_resnet_cifar10_valid_16mc_samples,
                                                            mcd_samples_nro=MC_SAMPLES)
    _, gtsrb_rn18_h_z_cifar10_test_samples_np = get_dl_h_z(gtsrb_resnet_cifar10_test_16mc_samples,
                                                           mcd_samples_nro=MC_SAMPLES)
    _, gtsrb_rn18_h_z_stl10_valid_samples_np = get_dl_h_z(gtsrb_resnet_stl10_valid_16mc_samples,
                                                          mcd_samples_nro=MC_SAMPLES)
    _, gtsrb_rn18_h_z_stl10_test_samples_np = get_dl_h_z(gtsrb_resnet_stl10_test_16mc_samples,
                                                         mcd_samples_nro=MC_SAMPLES)

    # Concatenate valid and test sets
    gtsrb_h_z = np.concatenate(
        (gtsrb_rn18_h_z_gtsrb_normal_valid_samples_np, gtsrb_rn18_h_z_gtsrb_normal_test_samples_np)
    )
    gtsrb_anomal_h_z = np.concatenate(
        (gtsrb_rn18_h_z_gtsrb_anomal_valid_samples_np, gtsrb_rn18_h_z_gtsrb_anomal_test_samples_np)
    )
    cifar10_h_z = np.concatenate(
        (gtsrb_rn18_h_z_cifar10_valid_samples_np, gtsrb_rn18_h_z_cifar10_test_samples_np)
    )
    stl10_h_z = np.concatenate(
        (gtsrb_rn18_h_z_stl10_valid_samples_np, gtsrb_rn18_h_z_stl10_test_samples_np)
    )
    ######################################################
    # Evaluate OoD detection methods
    ######################################################
    # Build KDE detector
    gtsrb_ds_shift_detector = DetectorKDE(train_embeddings=gtsrb_rn18_h_z_gtsrb_normal_train_samples_np)

    scores_gtsrb = get_hz_scores(gtsrb_ds_shift_detector, gtsrb_h_z)

    scores_gtsrb_anomal = get_hz_scores(gtsrb_ds_shift_detector, gtsrb_anomal_h_z)

    scores_cifar10 = get_hz_scores(gtsrb_ds_shift_detector, cifar10_h_z)

    scores_stl10 = get_hz_scores(gtsrb_ds_shift_detector, stl10_h_z)

    # Results
    # GTSRB vs GTSRB nomalies
    print("Test InD shape", scores_gtsrb.shape);
    print("Test OoD shape", scores_gtsrb_anomal.shape);
    print("Experiment: gtsrb vs gtsrb-anomal");

    results_gtsrb_anomal_validation = get_hz_detector_results(detect_exp_name="gtsrb vs. gtsrb-anomal",
                                                              ind_samples_scores=scores_gtsrb,
                                                              ood_samples_scores=scores_gtsrb_anomal)
    print("Test InD shape", scores_gtsrb.shape);
    print("Test OoD shape", scores_cifar10.shape);
    print("Experiment: gtsrb vs cifar10");

    results_cifar10_validation = get_hz_detector_results(detect_exp_name="gtsrb vs. cifar10",
                                                         ind_samples_scores=scores_gtsrb,
                                                         ood_samples_scores=scores_cifar10)
    print("Test InD shape", scores_gtsrb.shape);
    print("Test OoD shape", scores_stl10.shape);
    print("Experiment: gtsrb vs stl10");

    results_stl10_validation = get_hz_detector_results(detect_exp_name="gtsrb vs. stl10",
                                                       ind_samples_scores=scores_gtsrb,
                                                       ood_samples_scores=scores_stl10)


if __name__ == '__main__':
    main()
