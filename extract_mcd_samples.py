import numpy as np
import random
import os
from icecream import ic
import torch
import hydra
from torchvision import transforms as transform_lib
from omegaconf import DictConfig
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import GtsrbModule
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules import STL10DataModule
from models import ResnetModule
from dropblock import DropBlock2D
from ls_ood_detect_cea.uncertainty_estimation import Hook, MCDSamplesExtractor, \
    get_msp_score, get_energy_score, MDSPostprocessor, KNNPostprocessor
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z

# Datasets paths
dataset_path = "./gtsrb-data/"
cifar10_data_dir = "./ood_datasets/cifar10_data/"
stl10_data_dir = "./ood_datasets/stl10-data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    # Create data folder with the name of the model if it doesn't exist
    mcd_samples_folder = "./Mcd_samples/"
    os.makedirs(mcd_samples_folder, exist_ok=True)
    save_dir = f"{mcd_samples_folder}{cfg.gtsrb_model_path.split('/')[2]}/{cfg.layer_type}"
    assert not os.path.exists(save_dir), "Folder already exists!"
    os.mkdir(save_dir)
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
    torch.save(
        gtsrb_resnet_gtsrb_normal_train_16mc_samples,
        f"{save_dir}/gtrsb_train_{gtsrb_resnet_gtsrb_normal_train_16mc_samples.shape[0]}_{gtsrb_resnet_gtsrb_normal_train_16mc_samples.shape[1]}_mcd_samples.pt",
    )
    # InD valid set
    gtsrb_resnet_gtsrb_normal_valid_16mc_samples, ind_valid_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        gtsrb_normal_valid_loader
    )
    torch.save(
        gtsrb_resnet_gtsrb_normal_valid_16mc_samples,
        f"{save_dir}/gtrsb_valid_{gtsrb_resnet_gtsrb_normal_valid_16mc_samples.shape[0]}_{gtsrb_resnet_gtsrb_normal_valid_16mc_samples.shape[1]}_mcd_samples.pt",
    )
    torch.save(
        ind_valid_preds,
        f"{save_dir}/gtrsb_valid_{ind_valid_preds.shape[0]}_{ind_valid_preds.shape[1]}_mcd_preds.pt",
    )
    # InD test set
    gtsrb_resnet_gtsrb_normal_test_16mc_samples, ind_test_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        gtsrb_normal_test_loader
    )
    torch.save(
        gtsrb_resnet_gtsrb_normal_test_16mc_samples,
        f"{save_dir}/gtrsb_test_{gtsrb_resnet_gtsrb_normal_test_16mc_samples.shape[0]}_{gtsrb_resnet_gtsrb_normal_test_16mc_samples.shape[1]}_mcd_samples.pt",
    )
    torch.save(
        ind_test_preds,
        f"{save_dir}/gtrsb_test_{ind_test_preds.shape[0]}_{ind_test_preds.shape[1]}_mcd_preds.pt",
    )
    # Anomalies valid set
    gtsrb_resnet_gtsrb_anomal_valid_16mc_samples, anomal_valid_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        gtsrb_anomal_valid_loader
    )
    torch.save(
        gtsrb_resnet_gtsrb_anomal_valid_16mc_samples,
        f"{save_dir}/gtrsb_anomal_valid_{gtsrb_resnet_gtsrb_anomal_valid_16mc_samples.shape[0]}_{gtsrb_resnet_gtsrb_anomal_valid_16mc_samples.shape[1]}_mcd_samples.pt",
    )
    torch.save(
        anomal_valid_preds,
        f"{save_dir}/gtrsb_anomal_valid_{anomal_valid_preds.shape[0]}_{anomal_valid_preds.shape[1]}_mcd_preds.pt",
    )
    # Anomalies test set
    gtsrb_resnet_gtsrb_anomal_test_16mc_samples, anomal_test_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        gtsrb_anomal_test_loader
    )
    torch.save(
        gtsrb_resnet_gtsrb_anomal_test_16mc_samples,
        f"{save_dir}/gtrsb_anomal_test_{gtsrb_resnet_gtsrb_anomal_test_16mc_samples.shape[0]}_{gtsrb_resnet_gtsrb_anomal_test_16mc_samples.shape[1]}_mcd_samples.pt",
    )
    torch.save(
        anomal_test_preds,
        f"{save_dir}/gtrsb_anomal_test_{anomal_test_preds.shape[0]}_{anomal_test_preds.shape[1]}_mcd_preds.pt",
    )

    # Cifar
    # Valid
    gtsrb_resnet_cifar10_valid_16mc_samples, cifar_valid_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        cifar10_valid_loader
    )
    torch.save(
        gtsrb_resnet_cifar10_valid_16mc_samples,
        f"{save_dir}/cifar_valid_{gtsrb_resnet_cifar10_valid_16mc_samples.shape[0]}_{gtsrb_resnet_cifar10_valid_16mc_samples.shape[1]}_mcd_samples.pt",
    )
    torch.save(
        cifar_valid_preds,
        f"{save_dir}/cifar_valid_{cifar_valid_preds.shape[0]}_{cifar_valid_preds.shape[1]}_mcd_preds.pt",
    )
    # Test
    gtsrb_resnet_cifar10_test_16mc_samples, cifar_test_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        cifar10_test_loader
    )
    torch.save(
        gtsrb_resnet_cifar10_test_16mc_samples,
        f"{save_dir}/cifar_test_{gtsrb_resnet_cifar10_test_16mc_samples.shape[0]}_{gtsrb_resnet_cifar10_test_16mc_samples.shape[1]}_mcd_samples.pt",
    )
    torch.save(
        cifar_test_preds,
        f"{save_dir}/cifar_test_{cifar_test_preds.shape[0]}_{cifar_test_preds.shape[1]}_mcd_preds.pt",
    )

    # STL
    # Valid
    gtsrb_resnet_stl10_valid_16mc_samples, stl_valid_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        stl10_valid_loader
    )
    torch.save(
        gtsrb_resnet_stl10_valid_16mc_samples,
        f"{save_dir}/stl_valid_{gtsrb_resnet_stl10_valid_16mc_samples.shape[0]}_{gtsrb_resnet_stl10_valid_16mc_samples.shape[1]}_mcd_samples.pt",
    )
    torch.save(
        stl_valid_preds,
        f"{save_dir}/stl_valid_{stl_valid_preds.shape[0]}_{stl_valid_preds.shape[1]}_mcd_preds.pt",
    )
    # Test
    gtsrb_resnet_stl10_test_16mc_samples, stl_test_preds = mcd_extractor.get_ls_mcd_samples_baselines(
        stl10_test_loader
    )
    torch.save(
        gtsrb_resnet_stl10_test_16mc_samples,
        f"{save_dir}/stl_test_{gtsrb_resnet_stl10_test_16mc_samples.shape[0]}_{gtsrb_resnet_stl10_test_16mc_samples.shape[1]}_mcd_samples.pt",
    )
    torch.save(
        stl_test_preds,
        f"{save_dir}/stl_test_{stl_test_preds.shape[0]}_{stl_test_preds.shape[1]}_mcd_preds.pt",
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
    np.save(
        f"{save_dir}/gtsrb_msp", ind_gtsrb_pred_msp_score,
    )
    np.save(
        f"{save_dir}/gtsrb_anomal_msp", ood_gtsrb_anomal_pred_msp_score,
    )
    np.save(
        f"{save_dir}/cifar_msp", ood_cifar10_pred_msp_score,
    )
    np.save(
        f"{save_dir}/stl_msp", ood_stl10_pred_msp_score,
    )

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
    np.save(
        f"{save_dir}/gtsrb_energy", ind_gtsrb_pred_energy_score,
    )
    np.save(
        f"{save_dir}/gtsrb_anomal_energy", ood_gtsrb_anomal_pred_energy_score,
    )
    np.save(
        f"{save_dir}/cifar_energy", ood_cifar10_pred_energy_score,
    )
    np.save(
        f"{save_dir}/stl_energy", ood_stl10_pred_energy_score,
    )
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
    np.save(
        f"{save_dir}/gtsrb_mdist", ind_gtsrb_m_dist_score,
    )
    np.save(
        f"{save_dir}/gtsrb_anomal_mdist", ood_gtsrb_anomal_m_dist_score,
    )
    np.save(
        f"{save_dir}/cifar_mdist", ood_cifar10_m_dist_score,
    )
    np.save(
        f"{save_dir}/stl_mdist", ood_stl10_m_dist_score,
    )
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
    np.save(
        f"{save_dir}/gtsrb_knn", ind_gtsrb_kth_dist_score,
    )
    np.save(
        f"{save_dir}/gtsrb_anomal_knn", ood_gtsrb_anomal_kth_dist_score,
    )
    np.save(
        f"{save_dir}/cifar_knn", ood_cifar10_kth_dist_score,
    )
    np.save(
        f"{save_dir}/stl_knn", ood_stl10_kth_dist_score,
    )
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
    np.save(
        f"{save_dir}/gtsrb_h_z_train", gtsrb_rn18_h_z_gtsrb_normal_train_samples_np,
    )
    np.save(
        f"{save_dir}/gtsrb_h_z", gtsrb_h_z,
    )
    np.save(
        f"{save_dir}/gtsrb_anomal_h_z", gtsrb_anomal_h_z,
    )
    np.save(
        f"{save_dir}/cifar_h_z", cifar10_h_z,
    )
    np.save(
        f"{save_dir}/stl_h_z", stl10_h_z,
    )


if __name__ == '__main__':
    main()
