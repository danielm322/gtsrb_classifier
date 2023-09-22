import numpy as np
import random
import pandas as pd
from icecream import ic
import torch
import hydra
import mlflow
from torchvision import transforms as transform_lib
from omegaconf import DictConfig
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import GtsrbModule
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules import STL10DataModule
from helper_functions import log_params_from_omegaconf_dict
from models import ResnetModule
from dropblock import DropBlock2D
from ls_ood_detect_cea.uncertainty_estimation import Hook, MCDSamplesExtractor, get_predictive_uncertainty_score, \
    get_msp_score, get_energy_score, MDSPostprocessor, KNNPostprocessor
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z
from ls_ood_detect_cea.metrics import get_hz_detector_results, \
    save_roc_ood_detector, save_scores_plots, get_pred_scores_plots_gtsrb
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
    # Get date-time to save df later
    current_date = cfg.log_dir.split("/")[-1]
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
    gtsrb_model.apply(resnet18_enable_dropblock2d_test);  # enable dropout
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
        original_resnet_architecture=cfg.original_resnet_architecture,
        return_raw_predictions=True
    )
    # InD train set
    gtsrb_resnet_gtsrb_normal_train_16mc_samples, ind_train_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        gtsrb_normal_train_loader
    )
    del ind_train_preds
    # InD valid set
    gtsrb_resnet_gtsrb_normal_valid_16mc_samples, ind_valid_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        gtsrb_normal_valid_loader
    )
    # InD test set
    gtsrb_resnet_gtsrb_normal_test_16mc_samples, ind_test_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        gtsrb_normal_test_loader
    )
    # Anomalies valid set
    gtsrb_resnet_gtsrb_anomal_valid_16mc_samples, anomal_valid_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        gtsrb_anomal_valid_loader
    )
    # Anomalies test set
    gtsrb_resnet_gtsrb_anomal_test_16mc_samples, anomal_test_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        gtsrb_anomal_test_loader
    )

    # Cifar
    gtsrb_resnet_cifar10_valid_16mc_samples, cifar_valid_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        cifar10_valid_loader
    )
    gtsrb_resnet_cifar10_test_16mc_samples, cifar_test_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        cifar10_test_loader
    )

    # STL
    gtsrb_resnet_stl10_valid_16mc_samples, stl_valid_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        stl10_valid_loader
    )
    gtsrb_resnet_stl10_test_16mc_samples, stl_test_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        stl10_test_loader
    )

    #######################
    # Maximum softmax probability score calculations
    gtsrb_model.eval()  # No MCD needed here
    ind_gtsrb_test_pred_msp_score = get_msp_score(dnn_model=gtsrb_model.model,
                                                  input_dataloader=gtsrb_normal_test_loader)
    ood_gtsrb_anomal_test_pred_msp_score = get_msp_score(dnn_model=gtsrb_model.model,
                                                         input_dataloader=gtsrb_anomal_test_loader)
    ood_cifar10_test_pred_msp_score = get_msp_score(dnn_model=gtsrb_model.model,
                                                    input_dataloader=cifar10_test_loader)
    ood_stl10_test_pred_msp_score = get_msp_score(dnn_model=gtsrb_model.model,
                                                  input_dataloader=stl10_test_loader)
    # %%
    ind_gtsrb_valid_pred_msp_score = get_msp_score(dnn_model=gtsrb_model.model,
                                                   input_dataloader=gtsrb_normal_valid_loader)
    ood_gtsrb_anomal_valid_pred_msp_score = get_msp_score(dnn_model=gtsrb_model.model,
                                                          input_dataloader=gtsrb_anomal_valid_loader)
    ood_cifar10_valid_pred_msp_score = get_msp_score(dnn_model=gtsrb_model.model,
                                                     input_dataloader=cifar10_valid_loader)
    ood_stl10_valid_pred_msp_score = get_msp_score(dnn_model=gtsrb_model.model,
                                                   input_dataloader=stl10_valid_loader)
    # Concatenate
    ind_gtsrb_pred_msp_score = np.concatenate((ind_gtsrb_valid_pred_msp_score, ind_gtsrb_test_pred_msp_score))
    ood_gtsrb_anomal_pred_msp_score = np.concatenate(
        (ood_gtsrb_anomal_valid_pred_msp_score, ood_gtsrb_anomal_test_pred_msp_score))
    ood_cifar10_pred_msp_score = np.concatenate((ood_cifar10_valid_pred_msp_score, ood_cifar10_test_pred_msp_score))
    ood_stl10_pred_msp_score = np.concatenate((ood_stl10_valid_pred_msp_score, ood_stl10_test_pred_msp_score))

    ##############
    # Get energy scores
    ind_gtsrb_test_pred_energy_score = get_energy_score(dnn_model=gtsrb_model.model,
                                                        input_dataloader=gtsrb_normal_test_loader)
    ood_gtsrb_anomal_test_pred_energy_score = get_energy_score(dnn_model=gtsrb_model.model,
                                                               input_dataloader=gtsrb_anomal_test_loader)
    ood_cifar10_test_pred_energy_score = get_energy_score(dnn_model=gtsrb_model.model,
                                                          input_dataloader=cifar10_test_loader)
    ood_stl10_test_pred_energy_score = get_energy_score(dnn_model=gtsrb_model.model,
                                                        input_dataloader=stl10_test_loader)
    # %%
    ind_gtsrb_valid_pred_energy_score = get_energy_score(dnn_model=gtsrb_model.model,
                                                         input_dataloader=gtsrb_normal_valid_loader)
    ood_gtsrb_anomal_valid_pred_energy_score = get_energy_score(dnn_model=gtsrb_model.model,
                                                                input_dataloader=gtsrb_anomal_valid_loader)
    ood_cifar10_valid_pred_energy_score = get_energy_score(dnn_model=gtsrb_model.model,
                                                           input_dataloader=cifar10_valid_loader)
    ood_stl10_valid_pred_energy_score = get_energy_score(dnn_model=gtsrb_model.model,
                                                         input_dataloader=stl10_valid_loader)
    # %%
    ind_gtsrb_pred_energy_score = np.concatenate((ind_gtsrb_valid_pred_energy_score, ind_gtsrb_test_pred_energy_score))
    ood_gtsrb_anomal_pred_energy_score = np.concatenate(
        (ood_gtsrb_anomal_valid_pred_energy_score, ood_gtsrb_anomal_test_pred_energy_score))
    ood_cifar10_pred_energy_score = np.concatenate(
        (ood_cifar10_valid_pred_energy_score, ood_cifar10_test_pred_energy_score))
    ood_stl10_pred_energy_score = np.concatenate((ood_stl10_valid_pred_energy_score, ood_stl10_test_pred_energy_score))

    ################
    # Get Mahalanobis distance scores
    gtsrb_model_avgpool_layer_hook = Hook(gtsrb_model.model.avgpool)
    # %%
    m_dist_gtsrb = MDSPostprocessor(num_classes=43, setup_flag=False)
    # %%
    m_dist_gtsrb.setup(gtsrb_model.model,
                       gtsrb_normal_train_loader,
                       layer_hook=gtsrb_model_avgpool_layer_hook)
    _, ind_gtsrb_valid_m_dist_score = m_dist_gtsrb.postprocess(gtsrb_model.model,
                                                               gtsrb_normal_valid_loader,
                                                               gtsrb_model_avgpool_layer_hook)

    _, ind_gtsrb_test_m_dist_score = m_dist_gtsrb.postprocess(gtsrb_model.model,
                                                              gtsrb_normal_test_loader,
                                                              gtsrb_model_avgpool_layer_hook)

    _, ood_gtsrb_anomal_valid_m_dist_score = m_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                      gtsrb_anomal_valid_loader,
                                                                      gtsrb_model_avgpool_layer_hook)

    _, ood_gtsrb_anomal_test_m_dist_score = m_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                     gtsrb_anomal_test_loader,
                                                                     gtsrb_model_avgpool_layer_hook)

    _, ood_cifar10_valid_m_dist_score = m_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                 cifar10_valid_loader,
                                                                 gtsrb_model_avgpool_layer_hook)

    _, ood_cifar10_test_m_dist_score = m_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                cifar10_test_loader,
                                                                gtsrb_model_avgpool_layer_hook)

    _, ood_stl10_valid_m_dist_score = m_dist_gtsrb.postprocess(gtsrb_model.model,
                                                               stl10_valid_loader,
                                                               gtsrb_model_avgpool_layer_hook)

    _, ood_stl10_test_m_dist_score = m_dist_gtsrb.postprocess(gtsrb_model.model,
                                                              stl10_test_loader,
                                                              gtsrb_model_avgpool_layer_hook)

    ind_gtsrb_m_dist_score = np.concatenate((ind_gtsrb_valid_m_dist_score, ind_gtsrb_test_m_dist_score))
    ood_gtsrb_anomal_m_dist_score = np.concatenate(
        (ood_gtsrb_anomal_valid_m_dist_score, ood_gtsrb_anomal_test_m_dist_score))
    ood_cifar10_m_dist_score = np.concatenate((ood_cifar10_valid_m_dist_score, ood_cifar10_test_m_dist_score))
    ood_stl10_m_dist_score = np.concatenate((ood_stl10_valid_m_dist_score, ood_stl10_test_m_dist_score))

    ####################
    # KNN detector
    knn_dist_gtsrb = KNNPostprocessor(K=50, setup_flag=False)
    knn_dist_gtsrb.setup(gtsrb_model.model,
                         gtsrb_normal_train_loader,
                         layer_hook=gtsrb_model_avgpool_layer_hook)

    _, ind_gtsrb_valid_kth_dist_score = knn_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                   gtsrb_normal_valid_loader,
                                                                   gtsrb_model_avgpool_layer_hook)

    _, ind_gtsrb_test_kth_dist_score = knn_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                  gtsrb_normal_test_loader,
                                                                  gtsrb_model_avgpool_layer_hook)

    _, ood_gtsrb_anomal_valid_kth_dist_score = knn_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                          gtsrb_anomal_valid_loader,
                                                                          gtsrb_model_avgpool_layer_hook)

    _, ood_gtsrb_anomal_test_kth_dist_score = knn_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                         gtsrb_anomal_test_loader,
                                                                         gtsrb_model_avgpool_layer_hook)

    _, ood_cifar10_valid_kth_dist_score = knn_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                     cifar10_valid_loader,
                                                                     gtsrb_model_avgpool_layer_hook)

    _, ood_cifar10_test_kth_dist_score = knn_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                    cifar10_test_loader,
                                                                    gtsrb_model_avgpool_layer_hook)

    _, ood_stl10_valid_kth_dist_score = knn_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                   stl10_valid_loader,
                                                                   gtsrb_model_avgpool_layer_hook)

    _, ood_stl10_test_kth_dist_score = knn_dist_gtsrb.postprocess(gtsrb_model.model,
                                                                  stl10_test_loader,
                                                                  gtsrb_model_avgpool_layer_hook)

    ind_gtsrb_kth_dist_score = np.concatenate((ind_gtsrb_valid_kth_dist_score, ind_gtsrb_test_kth_dist_score))
    ood_gtsrb_anomal_kth_dist_score = np.concatenate(
        (ood_gtsrb_anomal_valid_kth_dist_score, ood_gtsrb_anomal_test_kth_dist_score))
    ood_cifar10_kth_dist_score = np.concatenate((ood_cifar10_valid_kth_dist_score, ood_cifar10_test_kth_dist_score))
    ood_stl10_kth_dist_score = np.concatenate((ood_stl10_valid_kth_dist_score, ood_stl10_test_kth_dist_score))

    # Clean memory
    del gtsrb_normal_train_loader
    del gtsrb_normal_valid_loader
    del gtsrb_normal_test_loader
    del gtsrb_anomal_valid_loader
    del gtsrb_anomal_test_loader
    del cifar10_valid_loader
    del cifar10_test_loader
    del stl10_valid_loader
    del stl10_test_loader

    del gtsrb_normal_dm
    del gtsrb_anomal_dm
    del cifar10_dm
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
        ##########################################################################################
        ########################################################
        # Evaluate baselines
        ########################################################
        ########################
        # Predictive uncertainty - mutual information
        # InD set
        dl_gtsrb_pred_h, dl_gtsrb_mi = get_predictive_uncertainty_score(
            input_samples=torch.cat((ind_valid_preds, ind_test_preds), dim=0),
            mcd_nro_samples=cfg.mcd_n_samples
        )
        # Anomalies set
        dl_gtsrb_anomal_pred_h, dl_gtsrb_anomal_mi = get_predictive_uncertainty_score(
            input_samples=torch.cat((anomal_valid_preds, anomal_test_preds), dim=0),
            mcd_nro_samples=cfg.mcd_n_samples
        )
        # Cifar10
        dl_cifar10_pred_h, dl_cifar10_mi = get_predictive_uncertainty_score(
            input_samples=torch.cat((cifar_valid_preds, cifar_test_preds), dim=0),
            mcd_nro_samples=cfg.mcd_n_samples
        )
        # STL10
        dl_stl10_pred_h, dl_stl10_mi = get_predictive_uncertainty_score(
            input_samples=torch.cat((stl_valid_preds, stl_test_preds), dim=0),
            mcd_nro_samples=cfg.mcd_n_samples
        )
        # Pass to numpy
        ind_gtsrb_pred_h_score = dl_gtsrb_pred_h.cpu().numpy()
        ind_gtsrb_pred_mi_score = dl_gtsrb_mi.cpu().numpy()
        ood_gtsrb_anomal_pred_h_score = dl_gtsrb_anomal_pred_h.cpu().numpy()
        ood_gtsrb_anomal_pred_mi_score = dl_gtsrb_anomal_mi.cpu().numpy()
        ood_cifar10_pred_h_score = dl_cifar10_pred_h.cpu().numpy()
        ood_cifar10_pred_mi_score = dl_cifar10_mi.cpu().numpy()
        ood_stl10_pred_h_score = dl_stl10_pred_h.cpu().numpy()
        ood_stl10_pred_mi_score = dl_stl10_mi.cpu().numpy()
        # Dictionary that defines experiments names, InD and OoD datasets
        # We use some negative uncertainty scores to align with the convention that positive
        # (in-distribution) samples have higher scores (see plots)
        baselines_experiments = {
            "anomal pred h": {
                "InD": -ind_gtsrb_pred_h_score,
                "OoD": -ood_gtsrb_anomal_pred_h_score
            },
            "cifar10 pred h": {
                "InD": -ind_gtsrb_pred_h_score,
                "OoD": -ood_cifar10_pred_h_score
            },
            "stl10 pred h": {
                "InD": -ind_gtsrb_pred_h_score,
                "OoD": -ood_stl10_pred_h_score
            },
            "anomal mi": {
                "InD": -ind_gtsrb_pred_mi_score,
                "OoD": -ood_gtsrb_anomal_pred_mi_score
            },
            "cifar10 mi": {
                "InD": -ind_gtsrb_pred_mi_score,
                "OoD": -ood_cifar10_pred_mi_score
            },
            "stl10 mi": {
                "InD": -ind_gtsrb_pred_mi_score,
                "OoD": -ood_stl10_pred_mi_score
            },
            "anomal msp": {
                "InD": ind_gtsrb_pred_msp_score,
                "OoD": ood_gtsrb_anomal_pred_msp_score
            },
            "cifar10 msp": {
                "InD": ind_gtsrb_pred_msp_score,
                "OoD": ood_cifar10_pred_msp_score
            },
            "stl10 msp": {
                "InD": ind_gtsrb_pred_msp_score,
                "OoD": ood_stl10_pred_msp_score
            },
            "anomal energy": {
                "InD": ind_gtsrb_pred_energy_score,
                "OoD": ood_gtsrb_anomal_pred_energy_score
            },
            "cifar10 energy": {
                "InD": ind_gtsrb_pred_energy_score,
                "OoD": ood_cifar10_pred_energy_score
            },
            "stl10 energy": {
                "InD": ind_gtsrb_pred_energy_score,
                "OoD": ood_stl10_pred_energy_score
            },
            "anomal mdist": {
                "InD": ind_gtsrb_m_dist_score,
                "OoD": ood_gtsrb_anomal_m_dist_score
            },
            "cifar10 mdist": {
                "InD": ind_gtsrb_m_dist_score,
                "OoD": ood_cifar10_m_dist_score
            },
            "stl10 mdist": {
                "InD": ind_gtsrb_m_dist_score,
                "OoD": ood_stl10_m_dist_score
            },
            "anomal knn": {
                "InD": ind_gtsrb_kth_dist_score,
                "OoD": ood_gtsrb_anomal_kth_dist_score
            },
            "stl10 knn": {
                "InD": ind_gtsrb_kth_dist_score,
                "OoD": ood_stl10_kth_dist_score
            },
            "cifar10 knn": {
                "InD": ind_gtsrb_kth_dist_score,
                "OoD": ood_cifar10_kth_dist_score
            }
        }
        baselines_plots = {
            "Predictive H distribution": {
                "InD": ind_gtsrb_pred_h_score,
                "anomal": ood_gtsrb_anomal_pred_h_score,
                "stl10": ood_stl10_pred_h_score,
                "cifar10": ood_cifar10_pred_h_score,
                "x_axis": "Predictive H score",
                "plot_name": "pred_h"
            },
            "Predictive MI distribution": {
                "InD": ind_gtsrb_pred_mi_score,
                "anomal": ood_gtsrb_anomal_pred_mi_score,
                "stl10": ood_stl10_pred_mi_score,
                "cifar10": ood_cifar10_pred_mi_score,
                "x_axis": "Predictive MI score",
                "plot_name": "pred_mi"
            },
            "Predictive MSP distribution": {
                "InD": ind_gtsrb_pred_msp_score,
                "anomal": ood_gtsrb_anomal_pred_msp_score,
                "stl10": ood_stl10_pred_msp_score,
                "cifar10": ood_cifar10_pred_msp_score,
                "x_axis": "Predictive MSP score",
                "plot_name": "pred_msp"
            },
            "Predictive energy score distribution": {
                "InD": ind_gtsrb_pred_energy_score,
                "anomal": ood_gtsrb_anomal_pred_energy_score,
                "stl10": ood_stl10_pred_energy_score,
                "cifar10": ood_cifar10_pred_energy_score,
                "x_axis": "Predictive energy score",
                "plot_name": "pred_energy"
            },
            "Mahalanobis Distance distribution": {
                "InD": ind_gtsrb_m_dist_score,
                "anomal": ood_gtsrb_anomal_m_dist_score,
                "stl10": ood_stl10_m_dist_score,
                "cifar10": ood_cifar10_m_dist_score,
                "x_axis": "Mahalanobis Distance score",
                "plot_name": "pred_mdist"
            },
            "kNN distance distribution": {
                "InD": ind_gtsrb_kth_dist_score,
                "anomal": ood_gtsrb_anomal_kth_dist_score,
                "stl10": ood_stl10_kth_dist_score,
                "cifar10": ood_cifar10_kth_dist_score,
                "x_axis": "kNN distance score",
                "plot_name": "pred_knn"
            }
        }
        # Make all plots
        for plot_title, experiment in baselines_plots.items():
            # Plot score values predictive entropy
            pred_score_plot = get_pred_scores_plots_gtsrb(ind_gtsrb_pred_score=experiment["InD"],
                                                          gtsrb_anomal_pred_score=experiment["anomal"],
                                                          stl10_pred_score=experiment["stl10"],
                                                          cifar10_pred_score=experiment["cifar10"],
                                                          x_axis_name=experiment["x_axis"],
                                                          title=plot_title)
            mlflow.log_figure(figure=pred_score_plot.figure,
                              artifact_file=f"figs/{experiment['plot_name']}.png")

        # Initialize df to store all the results
        overall_metrics_df = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                                   'fpr', 'tpr', 'roc_thresholds',
                                                   'precision', 'recall', 'pr_thresholds'])
        # Log all baselines experiments
        for experiment_name, experiment in baselines_experiments.items():
            r_df, r_mlflow = get_hz_detector_results(detect_exp_name=experiment_name,
                                                     ind_samples_scores=experiment["InD"],
                                                     ood_samples_scores=experiment["OoD"],
                                                     return_results_for_mlflow=True)
            r_mlflow = dict([(f"{experiment_name}_{k}", v) for k, v in r_mlflow.items()])
            mlflow.log_metrics(r_mlflow)
            roc_curve = save_roc_ood_detector(
                results_table=r_df,
                plot_title=f"ROC gtsrb vs {experiment_name} {cfg.layer_type} layer"
            )
            mlflow.log_figure(figure=roc_curve,
                              artifact_file=f"figs/roc_{experiment_name}.png")
            overall_metrics_df.append(r_df)

        # Clean memory
        del baselines_plots
        del baselines_experiments

        del ind_gtsrb_pred_h_score
        del ind_gtsrb_pred_mi_score
        del ood_gtsrb_anomal_pred_h_score
        del ood_cifar10_pred_h_score
        del ood_stl10_pred_h_score
        del ood_gtsrb_anomal_pred_mi_score
        del ood_cifar10_pred_mi_score
        del ood_stl10_pred_mi_score

        del ind_gtsrb_pred_msp_score
        del ood_gtsrb_anomal_pred_msp_score
        del ood_cifar10_pred_msp_score
        del ood_stl10_pred_msp_score

        del ind_gtsrb_pred_energy_score
        del ood_gtsrb_anomal_pred_energy_score
        del ood_cifar10_pred_energy_score
        del ood_stl10_pred_energy_score

        del ind_gtsrb_m_dist_score
        del ood_gtsrb_anomal_m_dist_score
        del ood_cifar10_m_dist_score
        del ood_stl10_m_dist_score

        del ind_gtsrb_kth_dist_score
        del ood_gtsrb_anomal_kth_dist_score
        del ood_stl10_kth_dist_score
        del ood_cifar10_kth_dist_score


        ######################################################
        # Evaluate OoD detection method LaRED
        ######################################################
        # Build KDE detector
        gtsrb_ds_shift_detector = DetectorKDE(train_embeddings=gtsrb_rn18_h_z_gtsrb_normal_train_samples_np)

        # Extract Density scores
        scores_gtsrb = get_hz_scores(gtsrb_ds_shift_detector, gtsrb_h_z)
        scores_gtsrb_anomal = get_hz_scores(gtsrb_ds_shift_detector, gtsrb_anomal_h_z)
        scores_cifar10 = get_hz_scores(gtsrb_ds_shift_detector, cifar10_h_z)
        scores_stl10 = get_hz_scores(gtsrb_ds_shift_detector, stl10_h_z)

        la_red_experiments = {
            "anomal LaRED": {
                "InD": scores_gtsrb,
                "OoD": scores_gtsrb_anomal
            },
            "cifar10 LaRED": {
                "InD": scores_gtsrb,
                "OoD": scores_cifar10
            },
            "stl10 LaRED": {
                "InD": scores_gtsrb,
                "OoD": scores_stl10
            }
        }
        # Log Results
        for experiment_name, experiment in la_red_experiments.items():
            print("Experiment: gtsrb vs gtsrb-anomal");
            r_df, r_mlflow = get_hz_detector_results(detect_exp_name=experiment_name,
                                                     ind_samples_scores=experiment["InD"],
                                                     ood_samples_scores=experiment["OoD"],
                                                     return_results_for_mlflow=True)
            # Add OoD dataset to metrics name
            r_mlflow = dict([(f"{experiment_name}_{k}", v) for k, v in r_mlflow.items()])
            mlflow.log_metrics(r_mlflow)
            # Plot ROC curve
            roc_curve = save_roc_ood_detector(
                results_table=r_df,
                plot_title=f"ROC gtsrb vs {experiment_name} {cfg.layer_type} layer"
            )
            # Log the plot with mlflow
            mlflow.log_figure(figure=roc_curve,
                              artifact_file=f"figs/roc_{experiment_name}.png")
            overall_metrics_df.append(r_df)
        overall_metrics_df_name = f"./results_csvs/{current_date}_experiment.csv"
        overall_metrics_df.to_csv(path_or_buf=overall_metrics_df_name)
        mlflow.log_artifact(overall_metrics_df_name)

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
