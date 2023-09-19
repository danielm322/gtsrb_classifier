# from collections import namedtuple
import numpy as np
# import matplotlib.pyplot as plt
import random
# import pandas as pd
# import seaborn as sns
# from PIL import Image
from icecream import ic
import torch
import hydra
import mlflow
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms as transform_lib
from omegaconf import DictConfig
# from pytorch_lightning.callbacks import TQDMProgressBar
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import GtsrbModule
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules import STL10DataModule
from helper_functions import log_params_from_omegaconf_dict
from models import ResnetModule
from dropblock import DropBlock2D
from ls_ood_detect_cea.uncertainty_estimation import Hook, MCDSamplesExtractor
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z
# from ls_ood_detect_cea.ood_detection_dataset import build_ood_detection_ds
# from ls_ood_detect_cea.dimensionality_reduction import plot_samples_pacmap
# from ls_ood_detect_cea.detectors import KDEClassifier
from ls_ood_detect_cea.metrics import get_hz_detector_results, \
    save_roc_ood_detector, save_scores_plots
from ls_ood_detect_cea.detectors import DetectorKDE
from ls_ood_detect_cea import get_hz_scores

# Datasets paths
dataset_path = "./gtsrb-data/"
cifar10_data_dir = "./ood_datasets/cifar10_data/"
stl10_data_dir = "./ood_datasets/stl10-data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If both next two flags are false, mlflow will create a local tracking uri for the experiment
# Upload analysis to the TDL server
UPLOAD_FROM_LOCAL_TO_SERVER = True
# Upload analysis ran on the TDL server
UPLOAD_FROM_SERVER_TO_SERVER = False
assert UPLOAD_FROM_SERVER_TO_SERVER + UPLOAD_FROM_LOCAL_TO_SERVER <= 1


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    ################################################################################################
    #                                 LOAD DATASETS                                   ##############
    ################################################################################################

    ##################################################################
    # GTSRB NORMAL DATASET
    ###################################################################
    gtsrb_normal_dm = GtsrbModule(img_size=(cfg.datamodule.image_width, cfg.datamodule.image_height),
                                  data_path=dataset_path,
                                  batch_size=1,
                                  shuffle=False)

    gtsrb_normal_dm.setup(stage='fit')
    gtsrb_normal_dm.setup(stage='validate')
    gtsrb_normal_dm.setup(stage='test')

    # Subset train data loader to speed up OoD detection calculations
    gtsrb_ds_len = len(gtsrb_normal_dm.ds_gtsrb_train)
    indices_train_dl = list(range(gtsrb_ds_len))

    random.seed(cfg.seed)
    random.shuffle(indices_train_dl)

    split = int(np.floor(gtsrb_ds_len * cfg.train_subsamples_size))
    samples_idx = indices_train_dl[:split]
    # ic(len(samples_idx));

    train_sampler = SubsetRandomSampler(samples_idx)

    gtsrb_normal_dm.shuffle = False
    gtsrb_normal_dm.ds_gtsrb_train_sampler = train_sampler

    gtsrb_normal_train_loader = gtsrb_normal_dm.train_dataloader()
    gtsrb_normal_valid_loader = gtsrb_normal_dm.val_dataloader()
    gtsrb_normal_test_loader = gtsrb_normal_dm.test_dataloader()

    #####################################################################
    # GTSRB ANOMALIES DATASET
    #####################################################################
    gtsrb_anomal_dm = GtsrbModule(
        img_size=(cfg.datamodule.image_width, cfg.datamodule.image_height),
        data_path=dataset_path,
        batch_size=1,
        anomaly_transforms=True,
        shuffle=True
    )

    gtsrb_anomal_dm.setup(stage='fit')
    gtsrb_anomal_dm.setup(stage='validate')
    gtsrb_anomal_dm.setup(stage='test')

    # gtsrb_anomal_train_loader = gtsrb_anomal_dm.train_dataloader()
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
        transform_lib.Resize((cfg.datamodule.image_width, cfg.datamodule.image_height)),
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

    # cifar10_train_loader = cifar10_dm.train_dataloader()
    cifar10_valid_loader = cifar10_dm.val_dataloader()
    cifar10_test_loader = cifar10_dm.test_dataloader()
    # ic(len(cifar10_train_loader));
    ic(len(cifar10_valid_loader));
    ic(len(cifar10_test_loader));

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
        transform_lib.Resize((cfg.datamodule.image_width, cfg.datamodule.image_height)),
        transform_lib.ToTensor(),
        transform_lib.Normalize(
            mean=(0.3337, 0.3064, 0.3171),
            std=(0.2672, 0.2564, 0.2629)
        )
    ])

    # stl10_dm.train_transforms = stl10_transforms
    stl10_dm.test_transforms = stl10_transforms
    stl10_dm.val_transforms = stl10_transforms

    stl10_dm.prepare_data()
    # stl10_train_loader = stl10_dm.train_dataloader_labeled()
    stl10_valid_loader = stl10_dm.val_dataloader_labeled()
    stl10_test_loader = stl10_dm.test_dataloader()

    ####################################################################
    # Load trained model
    ####################################################################
    gtsrb_model = ResnetModule.load_from_checkpoint(checkpoint_path=cfg.gtsrb_model_path)

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
    # Extract MCD samples
    mcd_extractor = MCDSamplesExtractor(
        model=gtsrb_model.model,
        mcd_nro_samples=cfg.mcd_n_samples,
        hook_dropout_layer=gtsrb_model_dropblock2d_layer_hook,
        layer_type=cfg.layer_type,
        device=device,
        architecture=cfg.architecture,
        location=cfg.hook_location,
        reduction_method=cfg.reduction_method,
        input_size=cfg.datamodule.image_width,
        original_resnet_architecture=cfg.original_resnet_architecture
    )
    # InD train set
    gtsrb_resnet_gtsrb_normal_train_16mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(gtsrb_normal_train_loader)
    del gtsrb_normal_train_loader
    # InD valid set
    gtsrb_resnet_gtsrb_normal_valid_16mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(gtsrb_normal_valid_loader)
    del gtsrb_normal_valid_loader
    # InD test set
    gtsrb_resnet_gtsrb_normal_test_16mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(gtsrb_normal_test_loader)
    del gtsrb_normal_test_loader
    del gtsrb_normal_dm
    # Anomalies valid set
    gtsrb_resnet_gtsrb_anomal_valid_16mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(gtsrb_anomal_valid_loader)
    del gtsrb_anomal_valid_loader
    # Anomalies test set
    gtsrb_resnet_gtsrb_anomal_test_16mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(gtsrb_anomal_test_loader)
    del gtsrb_anomal_test_loader
    del gtsrb_anomal_dm

    # Cifar
    gtsrb_resnet_cifar10_valid_16mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(cifar10_valid_loader)
    del cifar10_valid_loader
    gtsrb_resnet_cifar10_test_16mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(cifar10_test_loader)
    del cifar10_test_loader
    del cifar10_dm

    # STL
    gtsrb_resnet_stl10_valid_16mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(stl10_valid_loader)
    del stl10_valid_loader
    gtsrb_resnet_stl10_test_16mc_samples = mcd_extractor.get_ls_mcd_samples_baselines(stl10_test_loader)
    del stl10_test_loader
    del stl10_dm

    # Calculate entropies
    _, gtsrb_rn18_h_z_gtsrb_normal_train_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_normal_train_16mc_samples,
                                                                 mcd_samples_nro=cfg.mcd_n_samples,
                                                                 parallel_run=True)
    del gtsrb_resnet_gtsrb_normal_train_16mc_samples
    _, gtsrb_rn18_h_z_gtsrb_normal_valid_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_normal_valid_16mc_samples,
                                                                 mcd_samples_nro=cfg.mcd_n_samples,
                                                                 parallel_run=True)
    del gtsrb_resnet_gtsrb_normal_valid_16mc_samples
    _, gtsrb_rn18_h_z_gtsrb_normal_test_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_normal_test_16mc_samples,
                                                                mcd_samples_nro=cfg.mcd_n_samples,
                                                                 parallel_run=True)
    del gtsrb_resnet_gtsrb_normal_test_16mc_samples
    _, gtsrb_rn18_h_z_gtsrb_anomal_valid_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_anomal_valid_16mc_samples,
                                                                 mcd_samples_nro=cfg.mcd_n_samples,
                                                                 parallel_run=True)
    del gtsrb_resnet_gtsrb_anomal_valid_16mc_samples
    _, gtsrb_rn18_h_z_gtsrb_anomal_test_samples_np = get_dl_h_z(gtsrb_resnet_gtsrb_anomal_test_16mc_samples,
                                                                mcd_samples_nro=cfg.mcd_n_samples,
                                                                 parallel_run=True)
    del gtsrb_resnet_gtsrb_anomal_test_16mc_samples
    _, gtsrb_rn18_h_z_cifar10_valid_samples_np = get_dl_h_z(gtsrb_resnet_cifar10_valid_16mc_samples,
                                                            mcd_samples_nro=cfg.mcd_n_samples,
                                                                     parallel_run=True)
    del gtsrb_resnet_cifar10_valid_16mc_samples
    _, gtsrb_rn18_h_z_cifar10_test_samples_np = get_dl_h_z(gtsrb_resnet_cifar10_test_16mc_samples,
                                                           mcd_samples_nro=cfg.mcd_n_samples,
                                                                     parallel_run=True)
    del gtsrb_resnet_cifar10_test_16mc_samples
    _, gtsrb_rn18_h_z_stl10_valid_samples_np = get_dl_h_z(gtsrb_resnet_stl10_valid_16mc_samples,
                                                          mcd_samples_nro=cfg.mcd_n_samples,
                                                                     parallel_run=True)
    del gtsrb_resnet_stl10_valid_16mc_samples
    _, gtsrb_rn18_h_z_stl10_test_samples_np = get_dl_h_z(gtsrb_resnet_stl10_test_16mc_samples,
                                                         mcd_samples_nro=cfg.mcd_n_samples,
                                                                     parallel_run=True)
    del gtsrb_resnet_stl10_test_16mc_samples

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

    #######################################################################
    # Setup MLFLow
    #######################################################################
    # Setup MLFlow for experiment tracking
    # MlFlow configuration
    experiment_name = cfg.logger.mlflow.experiment_name
    if UPLOAD_FROM_LOCAL_TO_SERVER:
        mlflow.set_tracking_uri("http://10.8.33.50:5050")
    elif UPLOAD_FROM_SERVER_TO_SERVER:
        mlflow.set_tracking_uri("http://127.0.0.1:5051")
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(
            name=experiment_name,
        )
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

    ############################################################################################################
    ############################################################################################################
    ##########################################################################
    # Start the evaluation run
    ##########################################################################
    # Define mlflow run to log metrics and parameters
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # Log parameters with mlflow
        log_params_from_omegaconf_dict(cfg)

        ######################################################
        # Evaluate OoD detection method
        ######################################################
        # Build KDE detector
        gtsrb_ds_shift_detector = DetectorKDE(train_embeddings=gtsrb_rn18_h_z_gtsrb_normal_train_samples_np)

        # Extract Density scores
        scores_gtsrb = get_hz_scores(gtsrb_ds_shift_detector, gtsrb_h_z)
        scores_gtsrb_anomal = get_hz_scores(gtsrb_ds_shift_detector, gtsrb_anomal_h_z)
        scores_cifar10 = get_hz_scores(gtsrb_ds_shift_detector, cifar10_h_z)
        scores_stl10 = get_hz_scores(gtsrb_ds_shift_detector, stl10_h_z)

        # Results
        # GTSRB vs GTSRB nomalies
        print("Experiment: gtsrb vs gtsrb-anomal");
        r_anomal_v, r_anomal_v_mlflow = get_hz_detector_results(detect_exp_name="gtsrb vs gtsrb-anomal",
                                                                  ind_samples_scores=scores_gtsrb,
                                                                  ood_samples_scores=scores_gtsrb_anomal,
                                                                  return_results_for_mlflow=True)
        # Add OoD dataset to metrics name
        r_anomal_v_mlflow = dict([("gtsrb-anomal_" + k, v) for k, v in r_anomal_v_mlflow.items()])
        mlflow.log_metrics(r_anomal_v_mlflow)
        # Plot ROC curve
        roc_curve_anomal = save_roc_ood_detector(
            results_table=r_anomal_v,
            plot_title=f"ROC gtsrb vs gtsrb-anomal {cfg.layer_type} layer"
        )
        # Log the plot with mlflow
        mlflow.log_figure(figure=roc_curve_anomal,
                          artifact_file="figs/roc_curve_anomal.png")

        # GTSRB vs CIFAR10
        print("Experiment: gtsrb vs cifar10");
        r_cifar_v, r_cifar_v_mlflow = get_hz_detector_results(detect_exp_name="gtsrb vs cifar10",
                                                                ind_samples_scores=scores_gtsrb,
                                                                ood_samples_scores=scores_cifar10,
                                                                return_results_for_mlflow=True)
        # Add OoD dataset to metrics name
        r_cifar_v_mlflow = dict([("cifar10_" + k, v) for k, v in r_cifar_v_mlflow.items()])
        mlflow.log_metrics(r_cifar_v_mlflow)
        # Plot ROC curve
        roc_curve_cifar = save_roc_ood_detector(
            results_table=r_cifar_v,
            plot_title=f"ROC gtsrb vs cifar10 {cfg.layer_type} layer"
        )
        # Log the plot with mlflow
        mlflow.log_figure(figure=roc_curve_cifar,
                          artifact_file="figs/roc_curve_cifar.png")
        # gtsrb vs stl10
        print("Experiment: gtsrb vs stl10");
        r_stl_v, r_stl_v_mlflow = get_hz_detector_results(detect_exp_name="gtsrb vs stl10",
                                                              ind_samples_scores=scores_gtsrb,
                                                              ood_samples_scores=scores_stl10,
                                                              return_results_for_mlflow=True)
        # Add OoD dataset to metrics name
        r_stl_v_mlflow = dict([("stl10_" + k, v) for k, v in r_stl_v_mlflow.items()])
        mlflow.log_metrics(r_stl_v_mlflow)
        # Plot ROC curve
        roc_curve_stl = save_roc_ood_detector(
            results_table=r_stl_v,
            plot_title=f"ROC gtsrb vs stl10 {cfg.layer_type} layer"
        )
        # Log the plot with mlflow
        mlflow.log_figure(figure=roc_curve_stl,
                          artifact_file="figs/roc_curve_stl.png")

        # Plots comparison of densities
        gsc, gga, gc, gs = save_scores_plots(scores_gtsrb,
                                             scores_gtsrb_anomal,
                                             scores_stl10,
                                             scores_cifar10)
        mlflow.log_figure(figure=gga.figure,
                          artifact_file="figs/gga.png")
        mlflow.log_figure(figure=gsc.figure,
                          artifact_file="figs/gsc.png")
        mlflow.log_figure(figure=gc.figure,
                          artifact_file="figs/gc.png")
        mlflow.log_figure(figure=gs.figure,
                          artifact_file="figs/gs.png")

        mlflow.end_run()


if __name__ == '__main__':
    main()


"""
Experiment: gtsrb vs gtsrb-anomal
AUROC: 0.8000
FPR95: 0.7141
AUPR: 0.7845
Test InD shape (4882,)
Test OoD shape (20000,)
Experiment: gtsrb vs cifar10
AUROC: 0.9723
FPR95: 0.1849
AUPR: 0.9388
Test InD shape (4882,)
Test OoD shape (11000,)
Experiment: gtsrb vs stl10
AUROC: 0.9893
FPR95: 0.0525
AUPR: 0.9822"""