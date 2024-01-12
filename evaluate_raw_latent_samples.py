import mlflow
import numpy as np
import pandas as pd
import torch
import os
import hydra
from ls_ood_detect_cea import Hook, log_evaluate_lared_larem, save_scores_plots, apply_pca_ds_split, \
    apply_pca_transform, save_roc_ood_detector, select_and_log_best_lared_larem
from omegaconf import DictConfig
from tqdm import tqdm

from helper_functions import log_params_from_omegaconf_dict
from models import ResnetModule
from datasets.custom_dataloaders import get_data_loaders_image_classification
from ls_ood_detect_cea.dev import RawLatentSamplesExtractor


# Datasets paths
datasets_paths_dict = {
    "gtsrb": "./data/gtsrb-data/",
    "cifar10": "./data/cifar10-data/",
    "stl10": "./data/stl10-data/",
    "fmnist": "./data/fmnist-data",
    "svhn": './data/svhn-data/',
    "places": "./data/places-data",
    "textures": "./data/textures-data",
    "lsun_c": "./data/LSUN",
    "lsun_r": "./data/LSUN_resize",
    "isun": "./data/iSUN"
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_WORKERS = os.cpu_count() - 4 if os.cpu_count() >= 8 else os.cpu_count() - 2
# N_WORKERS = int(np.floor(os.cpu_count() / torch.cuda.device_count() * 0.95))
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
    ind_dataset_dict, ood_datasets_dict = get_data_loaders_image_classification(
        cfg=cfg,
        datasets_paths=datasets_paths_dict,
        n_workers=N_WORKERS
    )
    ####################################################################
    # Load trained model
    ####################################################################
    rn_model = ResnetModule.load_from_checkpoint(checkpoint_path=cfg.model_path)
    rn_model.eval()

    # Add Hooks
    if cfg.layer_type == "FC":
        hooked_layer = Hook(rn_model.model.avgpool)
    # Conv (dropblock)
    else:
        if cfg.hook_location == 1:
            hooked_layer = Hook(rn_model.model.layer1)
        elif cfg.hook_location == 2:
            hooked_layer = Hook(rn_model.model.layer2)
        elif cfg.hook_location == 3:
            hooked_layer = Hook(rn_model.model.layer3)
        # Layer 4
        else:
            hooked_layer = Hook(rn_model.model.layer4)

    rn_model.to(device)
    rn_model.eval()
    # rn_model.apply(resnet18_enable_dropblock2d_test)  # enable dropout
    # mcd_samples_folder = f"./Mcd_samples/ind_{cfg.ind_dataset}/"
    # save_dir = f"{mcd_samples_folder}{cfg.model_path.split('/')[2]}/{cfg.layer_type}"

    #########################################################################
    # Extract MCDO latent samples
    #########################################################################
    # Conv Layers
    mcd_extractor = RawLatentSamplesExtractor(
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
        return_raw_predictions=False
    )
    # Store latent conv activations
    ind_latent_activations = {}
    for split, data_loader in ind_dataset_dict.items():
        print(f"\nExtracting InD {cfg.ind_dataset} {split}")
        ind_latent_activations[split] = mcd_extractor.get_ls_mcd_samples(data_loader).cpu().numpy()
    ind_latent_activations["test_valid"] = np.concatenate(
        (
            ind_latent_activations["test"],
            ind_latent_activations["valid"]
        )
    )
    # Extract and store OoD samples and entropies
    ood_latent_activations = {}
    for dataset_name, data_loaders in ood_datasets_dict.items():
        print(f"Saving samples from {dataset_name}")

        ood_latent_activations[dataset_name] = torch.cat(
            (
                mcd_extractor.get_ls_mcd_samples(data_loaders["valid"]),
                mcd_extractor.get_ls_mcd_samples(data_loaders["test"])
            )
        ).cpu().numpy()

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

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
        # Log parameters with mlflow
        log_params_from_omegaconf_dict(cfg)
        ######################################################
        # Evaluate OoD detection method LaRED & LaREM
        ######################################################
        # Initialize df to store all the results
        overall_metrics_df = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                                   'fpr', 'tpr', 'roc_thresholds',
                                                   'precision', 'recall', 'pr_thresholds'])
        print("LaRED & LaREM running...")
        # Perform evaluation with the complete vector of latent representations
        r_df, ind_lared_scores, ood_lared_scores_dict = log_evaluate_lared_larem(
            ind_train_h_z=ind_latent_activations["train"],
            ind_test_h_z=ind_latent_activations["test_valid"],
            ood_h_z_dict=ood_latent_activations,
            experiment_name_extension="",
            return_density_scores=True,
            mlflow_logging=True
        )
        # Add results to df
        overall_metrics_df = overall_metrics_df.append(r_df)
        # Plots comparison of densities
        lared_scores_plots_dict = save_scores_plots(ind_lared_scores,
                                                    ood_lared_scores_dict,
                                                    cfg.ood_datasets,
                                                    cfg.ind_dataset)
        for plot_name, plot in lared_scores_plots_dict.items():
            mlflow.log_figure(figure=plot.figure,
                              artifact_file=f"figs/{plot_name}.png")

        # Perform evaluation with PCA reduced vectors
        for n_components in tqdm(cfg.n_pca_components, desc="Evaluating PCA"):
            # Perform PCA dimension reduction
            pca_h_z_ind_train, pca_transformation = apply_pca_ds_split(
                samples=ind_latent_activations["train"],
                nro_components=n_components
            )
            pca_h_z_ind_test = apply_pca_transform(ind_latent_activations["test_valid"], pca_transformation)
            ood_pca_dict = {}
            for ood_dataset in cfg.ood_datasets:
                ood_pca_dict[ood_dataset] = apply_pca_transform(ood_latent_activations[ood_dataset],
                                                                pca_transformation)

            r_df = log_evaluate_lared_larem(
                ind_train_h_z=pca_h_z_ind_train,
                ind_test_h_z=pca_h_z_ind_test,
                ood_h_z_dict=ood_pca_dict,
                experiment_name_extension=f" PCA {n_components}",
                return_density_scores=False,
                log_step=n_components,
                mlflow_logging=True
            )
            # Add results to df
            overall_metrics_df = overall_metrics_df.append(r_df)

        # overall_metrics_df_name = f"./results_csvs/{current_date}_experiment.csv.gz"
        # overall_metrics_df.to_csv(path_or_buf=overall_metrics_df_name, compression="gzip")
        # mlflow.log_artifact(overall_metrics_df_name)
        if cfg.layer_type == "Conv":
            hook_layer_type = "Conv"
        else:
            hook_layer_type = "FC"

        # Plot Roc curves together, by OoD dataset
        for ood_dataset in cfg.ood_datasets:
            temp_df = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                            'fpr', 'tpr', 'roc_thresholds',
                                            'precision', 'recall', 'pr_thresholds'])
            temp_df_pca_lared = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                                      'fpr', 'tpr', 'roc_thresholds',
                                                      'precision', 'recall', 'pr_thresholds'])
            temp_df_pca_larem = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                                      'fpr', 'tpr', 'roc_thresholds',
                                                      'precision', 'recall', 'pr_thresholds'])
            for row_name in overall_metrics_df.index:
                if ood_dataset in row_name and "PCA" not in row_name:
                    temp_df = temp_df.append(overall_metrics_df.loc[row_name])
                    temp_df.rename(index={row_name: row_name.split(ood_dataset)[1]}, inplace=True)
                elif ood_dataset in row_name and "PCA" in row_name and "LaREM" in row_name:
                    temp_df_pca_larem = temp_df_pca_larem.append(overall_metrics_df.loc[row_name])
                    temp_df_pca_larem.rename(index={row_name: row_name.split(ood_dataset)[1]},
                                             inplace=True)
                elif ood_dataset in row_name and "PCA" in row_name and "LaRED" in row_name:
                    temp_df_pca_lared = temp_df_pca_lared.append(overall_metrics_df.loc[row_name])
                    temp_df_pca_lared.rename(index={row_name: row_name.split(ood_dataset)[1]},
                                             inplace=True)
            # Plot ROC curve
            roc_curve = save_roc_ood_detector(
                results_table=temp_df,
                plot_title=f"ROC {cfg.ind_dataset} vs {ood_dataset} {hook_layer_type} layer"
            )
            # Log the plot with mlflow
            mlflow.log_figure(figure=roc_curve,
                              artifact_file=f"figs/roc_{ood_dataset}.png")
            roc_curve_pca_larem = save_roc_ood_detector(
                results_table=temp_df_pca_larem,
                plot_title=f"ROC {cfg.ind_dataset} vs {ood_dataset} LaREM PCA {hook_layer_type} layer"
            )
            # Log the plot with mlflow
            mlflow.log_figure(figure=roc_curve_pca_larem,
                              artifact_file=f"figs/roc_{ood_dataset}_pca_larem.png")
            roc_curve_pca_lared = save_roc_ood_detector(
                results_table=temp_df_pca_lared,
                plot_title=f"ROC {cfg.ind_dataset} vs {ood_dataset} LaRED PCA {hook_layer_type} layer"
            )
            # Log the plot with mlflow
            mlflow.log_figure(figure=roc_curve_pca_lared,
                              artifact_file=f"figs/roc_{ood_dataset}_pca_lared.png")
        # Extract mean for LaRED & LaREM across datasets
        # LaRED
        auroc_lared, aupr_lared, fpr_lared, best_comp_lared = select_and_log_best_lared_larem(
            overall_metrics_df, cfg.n_pca_components, technique="LaRED", log_mlflow=True
        )
        # LaREM
        auroc_larem, aupr_larem, fpr_larem, best_comp_larem = select_and_log_best_lared_larem(
            overall_metrics_df, cfg.n_pca_components, technique="LaREM", log_mlflow=True
        )
        print(f"LaREM best comp {best_comp_larem}, auroc {auroc_larem}, aupr {aupr_larem}, "
              f"fpr {fpr_larem}")
        print(f"LaRED best comp {best_comp_lared}, auroc {auroc_lared}, aupr {aupr_lared}, "
              f"fpr {fpr_lared}")


if __name__ == '__main__':
    main()
