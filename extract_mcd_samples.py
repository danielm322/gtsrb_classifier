import numpy as np
import random
import os
import torchvision
from icecream import ic
import torch
import hydra
from torch.utils.data import DataLoader
from torchvision import transforms as transform_lib
from omegaconf import DictConfig
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import GtsrbModule
from pl_bolts.datamodules import CIFAR10DataModule, FashionMNISTDataModule
from pl_bolts.datamodules import STL10DataModule
from datasets.cifar10 import get_cifar10_input_transformations, fmnist_to_cifar_format
from models import ResnetModule
from dropblock import DropBlock2D
from ls_ood_detect_cea.uncertainty_estimation import Hook, MCDSamplesExtractor, \
    get_msp_score, get_energy_score, MDSPostprocessor, KNNPostprocessor
from ls_ood_detect_cea.uncertainty_estimation import get_dl_h_z

# Datasets paths
gtsrb_path = "./data/gtsrb-data/"
cifar10_data_dir = "./data/cifar10-data/"
stl10_data_dir = "./data/stl10-data/"
fmnist_data_dir = "./data/fmnist-data"
svhn_data_dir = './data/svhn-data/'
places_data_dir = "./data/places-data"
textures_data_dir = "./data/textures-data"
lsun_data_dir = "./data/lsun-data"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_WORKERS = os.cpu_count() - 4 if os.cpu_count() >= 8 else os.cpu_count() - 2
CIFAR10_TEST_SIZE = 0.5  # From test size 10000 default:0.5
FMNIST_TEST_SIZE = 0.5  # From test size 10000 default:0.5
SVHN_TEST_SIZE = 0.2  # From test size 26000 default:0.2
PLACES_TEST_SIZE = 0.2  # From test size 36500 default:0.14
EXTRACT_MCD_SAMPLES_AND_ENTROPIES = True  # set False for debugging purposes


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def extract_and_save_mcd_samples(cfg: DictConfig) -> None:
    ################################################################################################
    #                                 LOAD DATASETS                                   ##############
    ################################################################################################

    ##################################################################
    # GTSRB NORMAL DATASET
    ###################################################################
    ood_datasets_dict = {}
    if cfg.ind_dataset == "gtsrb":
        ic("gtsrb as InD")
        gtsrb_normal_dm = GtsrbModule(img_size=(cfg.datamodule.image_width, cfg.datamodule.image_height),
                                      data_path=gtsrb_path,
                                      batch_size=1,
                                      shuffle=False, )

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

        ind_dataset_dict = {
            "train": gtsrb_normal_dm.train_dataloader(),
            "valid": gtsrb_normal_dm.val_dataloader(),
            "test": gtsrb_normal_dm.test_dataloader()
        }

    elif cfg.ind_dataset == "cifar10" and "gtsrb" in cfg.ood_datasets:
        ic("gtsrb as OoD")
        train_transforms, test_transforms = get_cifar10_input_transformations(
            cifar10_normalize_inputs=cfg.datamodule.cifar10_normalize_inputs,
            img_size=cfg.datamodule.image_width,
            data_augmentations="none",
            anomalies=False
        )
        gtsrb_normal_dm = GtsrbModule(img_size=(cfg.datamodule.image_width, cfg.datamodule.image_height),
                                      data_path=gtsrb_path,
                                      batch_size=1,
                                      shuffle=False,
                                      train_transforms=train_transforms,
                                      valid_transforms=test_transforms,
                                      test_transforms=test_transforms,
                                      )
        gtsrb_normal_dm.setup(stage='fit')
        gtsrb_normal_dm.setup(stage='validate')
        gtsrb_normal_dm.setup(stage='test')
        # Add to ood datasets dict
        ood_datasets_dict["gtsrb"] = {
            "valid": gtsrb_normal_dm.val_dataloader(),
            "test": gtsrb_normal_dm.test_dataloader()
        }

    #####################################################################
    # GTSRB ANOMALIES DATASET
    #####################################################################
    if cfg.ind_dataset == "gtsrb" and "anomalies" in cfg.ood_datsets:
        ic("gtsrb anomalies as OoD")
        gtsrb_anomal_dm = GtsrbModule(
            img_size=(cfg.datamodule.image_width, cfg.datamodule.image_height),
            data_path=gtsrb_path,
            batch_size=1,
            anomaly_transforms=True,
            shuffle=True,
        )

        gtsrb_anomal_dm.setup(stage='fit')
        gtsrb_anomal_dm.setup(stage='validate')
        gtsrb_anomal_dm.setup(stage='test')

        # Add to ood datasets dict
        ood_datasets_dict["gtsrb_anomal"] = {
            "valid": gtsrb_anomal_dm.val_dataloader(),
            "test": gtsrb_anomal_dm.test_dataloader()
        }

    ######################################################################
    # CIFAR10 DATASET
    ######################################################################
    if cfg.ind_dataset == "cifar10":
        ic("cifar10 as InD")
        train_transforms, test_transforms = get_cifar10_input_transformations(
            cifar10_normalize_inputs=cfg.datamodule.cifar10_normalize_inputs,
            img_size=cfg.datamodule.image_width,
            data_augmentations="none",
            anomalies=False
        )
        cifar10_dm = CIFAR10DataModule(data_dir=cifar10_data_dir,
                                       batch_size=1,
                                       train_transforms=train_transforms,
                                       test_transforms=test_transforms,
                                       val_transforms=test_transforms,
                                       )
        cifar10_dm.prepare_data()
        cifar10_dm.setup(stage='fit')
        cifar10_dm.setup(stage='test')
        # Subset train dataset
        subset_ds_len = int(len(cifar10_dm.dataset_train) * cfg.train_subsamples_size)
        cifar10_train_subset = torch.utils.data.random_split(
            cifar10_dm.dataset_train,
            [subset_ds_len, len(cifar10_dm.dataset_train) - subset_ds_len],
            torch.Generator().manual_seed(cfg.seed)
        )[0]
        # Subset the test dataset
        # Here, for several datasets a double split is needed since the script extracts by default from a
        # validation and test set
        cifar10_test_subset = torch.utils.data.random_split(
            cifar10_dm.dataset_test,
            [int(len(cifar10_dm.dataset_test) * CIFAR10_TEST_SIZE),
             int(len(cifar10_dm.dataset_test) * (1.0 - CIFAR10_TEST_SIZE))],
            torch.Generator().manual_seed(cfg.seed)
        )[0]
        cifar10_test_subset, cifar10_valid_subset = torch.utils.data.random_split(
            cifar10_test_subset,
            [int(len(cifar10_test_subset) * 0.5), int(len(cifar10_test_subset) * 0.5)],
            torch.Generator().manual_seed(cfg.seed)
        )

        cifar10_dm.shuffle = False
        ind_dataset_dict = {
            "train": DataLoader(cifar10_train_subset, batch_size=1, shuffle=True),
            "valid": DataLoader(cifar10_valid_subset, batch_size=1, shuffle=True),
            "test": DataLoader(cifar10_test_subset, batch_size=1, shuffle=True)
        }

    elif cfg.ind_dataset == "gtsrb" and "cifar10" in cfg.ood_datasets:
        ic("cifar10 as OoD")
        cifar10_dm = CIFAR10DataModule(data_dir=cifar10_data_dir,
                                       val_split=0.2,
                                       normalize=False,
                                       batch_size=1,
                                       seed=cfg.seed,
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
        ood_datasets_dict["cifar10"] = {
            "valid": cifar10_dm.val_dataloader(),
            "test": cifar10_dm.test_dataloader()
        }
    ######################################################################
    # CIFAR10 ANOMALIES DATASET
    ######################################################################
    if cfg.ind_dataset == "cifar10" and "anomalies" in cfg.ood_datasets:
        ic("cifar10 anomalies as OoD")
        anomal_train_transforms, anomal_test_transforms = get_cifar10_input_transformations(
            cifar10_normalize_inputs=cfg.datamodule.cifar10_normalize_inputs,
            img_size=cfg.datamodule.image_width,
            data_augmentations="none",
            anomalies=True,
        )
        cifar10_anomal_dm = CIFAR10DataModule(data_dir=cifar10_data_dir,
                                              batch_size=1,
                                              train_transforms=anomal_train_transforms,
                                              test_transforms=anomal_test_transforms,
                                              val_transforms=anomal_test_transforms)
        cifar10_anomal_dm.prepare_data()
        cifar10_anomal_dm.setup(stage='fit')
        cifar10_anomal_dm.setup(stage='test')
        # Subset the test dataset
        cifar10_anomal_test_subset = torch.utils.data.random_split(
            cifar10_anomal_dm.dataset_test,
            [int(len(cifar10_anomal_dm.dataset_test) * CIFAR10_TEST_SIZE),
             int(len(cifar10_anomal_dm.dataset_test) * (1.0 - CIFAR10_TEST_SIZE))],
            torch.Generator().manual_seed(cfg.seed)
        )[0]
        cifar10_anomal_test_subset, cifar10_anomal_valid_subset = torch.utils.data.random_split(
            cifar10_anomal_test_subset,
            [int(len(cifar10_anomal_test_subset) * 0.5), int(len(cifar10_anomal_test_subset) * 0.5)],
            torch.Generator().manual_seed(cfg.seed)
        )
        ood_datasets_dict["cifar10_anomal"] = {
            "valid": DataLoader(cifar10_anomal_valid_subset, batch_size=1, shuffle=True),
            "test": DataLoader(cifar10_anomal_test_subset, batch_size=1, shuffle=True),
        }
    ##########################################################
    # STL-10 OoD
    #########################################################
    if "stl10" in cfg.ood_datasets:
        ic("stl10 as OoD")
        stl10_dm = STL10DataModule(data_dir=stl10_data_dir,
                                   train_val_split=3000,
                                   num_workers=N_WORKERS,
                                   batch_size=1,
                                   seed=cfg.seed,
                                   drop_last=True,
                                   shuffle=True)

        if cfg.ind_dataset == "gtsrb":
            stl10_transforms = transform_lib.Compose([
                transform_lib.Resize((cfg.datamodule.image_width, cfg.datamodule.image_height)),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=(0.3337, 0.3064, 0.3171),
                    std=(0.2672, 0.2564, 0.2629)
                )
            ])
        # Cifar as InD
        else:
            stl10_transforms = test_transforms

        stl10_dm.test_transforms = stl10_transforms
        stl10_dm.val_transforms = stl10_transforms

        stl10_dm.prepare_data()
        ood_datasets_dict["stl10"] = {
            "valid": stl10_dm.val_dataloader_labeled(),
            "test": stl10_dm.test_dataloader()
        }

    ##########################################################
    # Fashion MNIST OoD
    ##########################################################
    if "fmnist" in cfg.ood_datasets and cfg.ind_dataset == "cifar10":
        ic("fmnist as OoD")
        fmnist_dm = FashionMNISTDataModule(data_dir=fmnist_data_dir,
                                           val_split=0.2,
                                           num_workers=N_WORKERS,
                                           normalize=False,
                                           batch_size=1,
                                           seed=cfg.seed,
                                           shuffle=True,
                                           drop_last=True
                                           )
        fmnist_transforms = fmnist_to_cifar_format(img_size=cfg.datamodule.image_width,
                                                   cifar_normalize=cfg.datamodule.cifar10_normalize_inputs)

        fmnist_dm.test_transforms = fmnist_transforms
        fmnist_dm.val_transforms = fmnist_transforms
        fmnist_dm.prepare_data()
        fmnist_dm.setup(stage='fit')
        fmnist_dm.setup(stage='test')
        # Subset test dataset
        fmnist_test_size = int((FMNIST_TEST_SIZE / 2) * len(fmnist_dm.dataset_test))
        fmnist_valid_size = int((FMNIST_TEST_SIZE / 2) * len(fmnist_dm.dataset_test))
        fmnist_test = torch.utils.data.random_split(
            fmnist_dm.dataset_test,
            [len(fmnist_dm.dataset_test) - fmnist_test_size - fmnist_valid_size,
             fmnist_test_size + fmnist_valid_size],
            torch.Generator().manual_seed(cfg.seed)
        )[1]
        fmnist_valid, fmnist_test = torch.utils.data.random_split(
            fmnist_test,
            [fmnist_valid_size, fmnist_test_size],
            torch.Generator().manual_seed(cfg.seed)
        )
        ood_datasets_dict["fmnist"] = {
            "valid": DataLoader(fmnist_valid, batch_size=1, shuffle=True),
            "test": DataLoader(fmnist_test, batch_size=1, shuffle=True),
        }
        del fmnist_dm
    ##########################################################
    # SVHN OoD
    ##########################################################
    if "svhn" in cfg.ood_datasets and cfg.ind_dataset == "cifar10":
        ic("svhn as OoD")
        svhn_init_valid = torchvision.datasets.SVHN(
            root=svhn_data_dir,
            split="test",
            download=True,
            transform=test_transforms
        )
        svhn_test_size = int((SVHN_TEST_SIZE / 2) * len(svhn_init_valid))
        svhn_valid_size = int((SVHN_TEST_SIZE / 2) * len(svhn_init_valid))
        svhn_test = torch.utils.data.random_split(
            svhn_init_valid,
            [len(svhn_init_valid) - svhn_valid_size - svhn_test_size, svhn_test_size + svhn_valid_size],
            torch.Generator().manual_seed(cfg.seed)
        )[1]
        svhn_valid, svhn_test = torch.utils.data.random_split(
            svhn_test,
            [svhn_valid_size, svhn_test_size],
            torch.Generator().manual_seed(cfg.seed)
        )
        # MNIST test set loader
        ood_datasets_dict["svhn"] = {
            "valid": DataLoader(svhn_valid, batch_size=1, shuffle=True),
            "test": DataLoader(svhn_test, batch_size=1, shuffle=True)
        }
        del svhn_init_valid
    ##########################################################
    # Places 365 OoD
    ##########################################################
    if "places" in cfg.ood_datasets and cfg.ind_dataset == "cifar10":
        ic("places as OoD")
        places_init_valid = torchvision.datasets.Places365(places_data_dir,
                                                           split="val",
                                                           small=True,
                                                           download=False,
                                                           transform=test_transforms, )
        places_test_size = int((PLACES_TEST_SIZE / 2) * len(places_init_valid))
        places_test = torch.utils.data.random_split(
            places_init_valid,
            [len(places_init_valid) - 2 * places_test_size, 2 * places_test_size],
            torch.Generator().manual_seed(cfg.seed)
        )[1]
        places_valid, places_test = torch.utils.data.random_split(
            places_test,
            [places_test_size, places_test_size],
            torch.Generator().manual_seed(cfg.seed)
        )
        # MNIST test set loader
        ood_datasets_dict["places"] = {
            "valid": DataLoader(places_valid, batch_size=1, shuffle=True),
            "test": DataLoader(places_test, batch_size=1, shuffle=True)
        }
        del places_init_valid

    ##########################################################
    # Textures OoD
    ##########################################################
    if "textures" in cfg.ood_datasets and cfg.ind_dataset == "cifar10":
        ic("textures as OoD")
        textures_init_train = torchvision.datasets.DTD(textures_data_dir,
                                                       split="train",
                                                       download=True,
                                                       transform=test_transforms, )
        textures_init_val = torchvision.datasets.DTD(textures_data_dir,
                                                     split="val",
                                                     download=True,
                                                     transform=test_transforms, )
        textures_test = torchvision.datasets.DTD(textures_data_dir,
                                                 split="test",
                                                 download=True,
                                                 transform=test_transforms, )
        textures_val = torch.utils.data.ConcatDataset([textures_init_train, textures_init_val])
        # MNIST test set loader
        ood_datasets_dict["textures"] = {
            "valid": DataLoader(textures_val, batch_size=1, shuffle=True),
            "test": DataLoader(textures_test, batch_size=1, shuffle=True)
        }
        # del textures_init_valid

    ####################################################################
    # Load trained model
    ####################################################################
    rn_model = ResnetModule.load_from_checkpoint(checkpoint_path=cfg.model_path)
    rn_model.eval()

    # Add Hooks
    if cfg.layer_type == "FC":
        hooked_layer = Hook(rn_model.model.dropout_layer)
    # Conv (dropblock)
    else:
        hooked_layer = Hook(rn_model.model.dropblock2d_layer)

    # Monte Carlo Dropout - Enable Dropout @ Test Time!
    def resnet18_enable_dropblock2d_test(m):
        if type(m) == DropBlock2D or type(m) == torch.nn.Dropout:
            m.train()

    rn_model.to(device)
    rn_model.eval()
    rn_model.apply(resnet18_enable_dropblock2d_test)  # enable dropout

    # Create data folder with the name of the model if it doesn't exist
    mcd_samples_folder = f"./Mcd_samples/ind_{cfg.ind_dataset}/"
    os.makedirs(mcd_samples_folder, exist_ok=True)
    save_dir = f"{mcd_samples_folder}{cfg.model_path.split('/')[2]}/{cfg.layer_type}"
    assert not os.path.exists(save_dir), "Folder already exists!"
    os.makedirs(save_dir)
    ####################################################################################################################
    ####################################################################################################################
    if EXTRACT_MCD_SAMPLES_AND_ENTROPIES:
        #########################################################################
        # Extract MCDO latent samples
        #########################################################################
        # Extract MCD samples
        mcd_extractor = MCDSamplesExtractor(
            model=rn_model.model,
            mcd_nro_samples=cfg.mcd_n_samples,
            hook_dropout_layer=hooked_layer,
            layer_type=cfg.layer_type,
            device=device,
            architecture=cfg.architecture,
            location=cfg.hook_location,
            reduction_method=cfg.reduction_method,
            input_size=cfg.datamodule.image_width,
            original_resnet_architecture=cfg.original_resnet_architecture,
            return_raw_predictions=True
        )

        # Extract and save InD samples and entropies
        ind_valid_test_entropies = []
        for split, data_loader in ind_dataset_dict.items():
            print(f"\nExtracting InD {cfg.ind_dataset} {split}")
            mcd_samples, mcd_preds = mcd_extractor.get_ls_mcd_samples_baselines(data_loader)
            torch.save(
                mcd_samples,
                f"{save_dir}/{cfg.ind_dataset}_{split}_{mcd_samples.shape[0]}_{mcd_samples.shape[1]}_mcd_samples.pt",
            )
            if not split == "train":
                torch.save(
                    mcd_preds,
                    f"{save_dir}/{cfg.ind_dataset}_{split}_mcd_preds.pt",
                )
            ind_h_z_samples_np = get_dl_h_z(mcd_samples,
                                            mcd_samples_nro=cfg.mcd_n_samples,
                                            parallel_run=True)[1]
            if split == "train":
                np.save(
                    f"{save_dir}/{cfg.ind_dataset}_h_z_train", ind_h_z_samples_np,
                )
            else:
                ind_valid_test_entropies.append(ind_h_z_samples_np)

        ind_h_z = np.concatenate(
            (ind_valid_test_entropies[0], ind_valid_test_entropies[1])
        )
        np.save(
            f"{save_dir}/{cfg.ind_dataset}_h_z", ind_h_z,
        )
        del mcd_samples
        del mcd_preds
        del ind_h_z
        del ind_valid_test_entropies
        del ind_h_z_samples_np

        # Extract and save OoD samples and entropies
        for dataset_name, data_loaders in ood_datasets_dict.items():
            print(f"Saving samples and entropies from {dataset_name}")
            mcd_samples_ood_v, mcd_preds_ood_v = mcd_extractor.get_ls_mcd_samples_baselines(data_loaders["valid"])
            torch.save(
                mcd_samples_ood_v,
                f"{save_dir}/{dataset_name}_valid_{mcd_samples_ood_v.shape[0]}_{mcd_samples_ood_v.shape[1]}_mcd_samples.pt",
            )
            torch.save(mcd_preds_ood_v, f"{save_dir}/{dataset_name}_valid_mcd_preds.pt")
            mcd_samples_ood_t, mcd_preds_ood_t = mcd_extractor.get_ls_mcd_samples_baselines(data_loaders["test"])
            torch.save(
                mcd_samples_ood_t,
                f"{save_dir}/{dataset_name}_test_{mcd_samples_ood_t.shape[0]}_{mcd_samples_ood_t.shape[1]}_mcd_samples.pt",
            )
            torch.save(mcd_preds_ood_t, f"{save_dir}/{dataset_name}_test_mcd_preds.pt")
            h_z_valid_np = get_dl_h_z(mcd_samples_ood_v,
                                      mcd_samples_nro=cfg.mcd_n_samples,
                                      parallel_run=True)[1]
            h_z_test_np = get_dl_h_z(mcd_samples_ood_t,
                                     mcd_samples_nro=cfg.mcd_n_samples,
                                     parallel_run=True)[1]
            ood_h_z = np.concatenate((h_z_valid_np, h_z_test_np))
            np.save(f"{save_dir}/{dataset_name}_h_z", ood_h_z)

        del mcd_preds_ood_v
        del mcd_preds_ood_t
        del ood_h_z
        del h_z_test_np
        del h_z_valid_np
        del mcd_samples_ood_t
        del mcd_samples_ood_v

    #######################
    # Maximum softmax probability and energy scores calculations
    rn_model.eval()  # No MCD needed here
    # InD data
    ind_valid_test_msp = []
    ind_valid_test_energy = []
    for split, data_loader in ind_dataset_dict.items():
        if not split == "train":
            print(f"\nMsp and energy from InD {split}")
            if "msp" in cfg.baselines:
                ind_valid_test_msp.append(
                    get_msp_score(dnn_model=rn_model.model, input_dataloader=data_loader)
                )
            if "energy" in cfg.baselines:
                ind_valid_test_energy.append(
                    get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loader)
                )

    # Concatenate
    if "msp" in cfg.baselines:
        ind_msp_score = np.concatenate((ind_valid_test_msp[0], ind_valid_test_msp[1]))
        np.save(f"{save_dir}/{cfg.ind_dataset}_msp", ind_msp_score)
    if "energy" in cfg.baselines:
        ind_energy_score = np.concatenate((ind_valid_test_energy[0], ind_valid_test_energy[1]))
        np.save(f"{save_dir}/{cfg.ind_dataset}_energy", ind_energy_score)

    for dataset_name, data_loaders in ood_datasets_dict.items():
        print(f"\nmsp and energy from OoD {dataset_name}")
        if "msp" in cfg.baselines:
            ood_valid_msp_score = get_msp_score(dnn_model=rn_model.model, input_dataloader=data_loaders["valid"])
            ood_test_msp_score = get_msp_score(dnn_model=rn_model.model, input_dataloader=data_loaders["test"])
            ood_msp_score = np.concatenate((ood_valid_msp_score, ood_test_msp_score))
            np.save(f"{save_dir}/{dataset_name}_msp", ood_msp_score)
        if "energy" in cfg.baselines:
            ood_test_energy_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["test"])
            ood_valid_energy_score = get_energy_score(dnn_model=rn_model.model, input_dataloader=data_loaders["valid"])
            ood_energy_score = np.concatenate((ood_test_energy_score, ood_valid_energy_score))
            np.save(f"{save_dir}/{dataset_name}_energy", ood_energy_score)

    #############################
    # Prepare Mahalanobis distance and kNN scores estimators
    # Hook now the average pool layer
    gtsrb_model_avgpool_layer_hook = Hook(rn_model.model.avgpool)
    if "mdist" in cfg.baselines:
        # Instantiate and setup estimator
        m_dist_estimator = MDSPostprocessor(num_classes=10 if cfg.ind_dataset == "cifar10" else 43, setup_flag=False)
        m_dist_estimator.setup(rn_model.model, ind_dataset_dict["train"], layer_hook=gtsrb_model_avgpool_layer_hook)
    if "knn" in cfg.baselines:
        # Instantiate kNN postprocessor
        knn_dist_estimator = KNNPostprocessor(K=50, setup_flag=False)
        knn_dist_estimator.setup(rn_model.model, ind_dataset_dict["train"], layer_hook=gtsrb_model_avgpool_layer_hook)

    # Get results from Mahalanobis and kNN estimators
    # InD samples
    ind_valid_test_mdist = []
    ind_valid_test_knn = []
    for split, data_loader in ind_dataset_dict.items():
        if not split == "train":
            print(f"\nMdist and kNN from InD {split}")
            if "mdist" in cfg.baselines:
                ind_valid_test_mdist.append(
                    m_dist_estimator.postprocess(rn_model.model, data_loader, gtsrb_model_avgpool_layer_hook)[1]
                )
            if "knn" in cfg.baselines:
                ind_valid_test_knn.append(
                    knn_dist_estimator.postprocess(rn_model.model, data_loader, gtsrb_model_avgpool_layer_hook)[1]
                )
    if "mdist" in cfg.baselines:
        # Concatenate ind samples
        ind_mdist_score = np.concatenate((ind_valid_test_mdist[0], ind_valid_test_mdist[1]))
        np.save(f"{save_dir}/{cfg.ind_dataset}_mdist", ind_mdist_score)
    if "knn" in cfg.baselines:
        ind_knn_score = np.concatenate((ind_valid_test_knn[0], ind_valid_test_knn[1]))
        np.save(f"{save_dir}/{cfg.ind_dataset}_knn", ind_knn_score)
    # OoD samples
    for dataset_name, data_loaders in ood_datasets_dict.items():
        print(f"\nMdist and kNN from OoD {dataset_name}")
        if "mdist" in cfg.baselines:
            # Mdist
            ood_valid_m_dist_score = m_dist_estimator.postprocess(rn_model.model,
                                                                  data_loaders["valid"],
                                                                  gtsrb_model_avgpool_layer_hook)[1]
            ood_test_m_dist_score = m_dist_estimator.postprocess(rn_model.model,
                                                                 data_loaders["test"],
                                                                 gtsrb_model_avgpool_layer_hook)[1]
            ood_m_dist_score = np.concatenate((ood_valid_m_dist_score, ood_test_m_dist_score))
            np.save(f"{save_dir}/{dataset_name}_mdist", ood_m_dist_score)
        if "knn" in cfg.baselines:
            # kNN
            ood_valid_kth_dist_score = knn_dist_estimator.postprocess(rn_model.model,
                                                                      data_loaders["valid"],
                                                                      gtsrb_model_avgpool_layer_hook)[1]

            ood_test_kth_dist_score = knn_dist_estimator.postprocess(rn_model.model,
                                                                     data_loaders["test"],
                                                                     gtsrb_model_avgpool_layer_hook)[1]
            ood_kth_dist_score = np.concatenate((ood_valid_kth_dist_score, ood_test_kth_dist_score))
            np.save(f"{save_dir}/{dataset_name}_knn", ood_kth_dist_score)

    print("Done!")


if __name__ == '__main__':
    extract_and_save_mcd_samples()
