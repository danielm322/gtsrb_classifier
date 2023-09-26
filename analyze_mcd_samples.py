import numpy as np
import pandas as pd
import torch
import hydra
import mlflow
from omegaconf import DictConfig
from os.path import join as op_join
from tqdm import tqdm
from helper_functions import log_params_from_omegaconf_dict
from ls_ood_detect_cea.uncertainty_estimation import get_predictive_uncertainty_score
from ls_ood_detect_cea.metrics import get_hz_detector_results, \
    save_roc_ood_detector, save_scores_plots, get_pred_scores_plots_gtsrb
from ls_ood_detect_cea.detectors import DetectorKDE
from ls_ood_detect_cea import get_hz_scores

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
    # Get sampels folder
    mcd_samples_folder = "./Mcd_samples/"
    save_dir = f"{mcd_samples_folder}{cfg.gtsrb_model_path.split('/')[2]}/{cfg.layer_type}"

    ######################################################################
    # Load all data
    ######################################################################
    # Raw predictions
    ind_valid_preds = torch.load(f=op_join(save_dir, "gtrsb_valid_mcd_preds.pt"), map_location=device)
    ind_test_preds = torch.load(f=op_join(save_dir, "gtrsb_test_mcd_preds.pt"), map_location=device)
    anomal_valid_preds = torch.load(f=op_join(save_dir, "gtrsb_anomal_valid_mcd_preds.pt"), map_location=device)
    anomal_test_preds = torch.load(f=op_join(save_dir, "gtrsb_anomal_test_mcd_preds.pt"), map_location=device)
    cifar_valid_preds = torch.load(f=op_join(save_dir, "cifar_valid_mcd_preds.pt"), map_location=device)
    cifar_test_preds = torch.load(f=op_join(save_dir, "cifar_test_mcd_preds.pt"), map_location=device)
    stl_valid_preds = torch.load(f=op_join(save_dir, "stl_valid_mcd_preds.pt"), map_location=device)
    stl_test_preds = torch.load(f=op_join(save_dir, "stl_test_mcd_preds.pt"), map_location=device)
    # MSP scores
    ind_gtsrb_pred_msp_score = np.load(file=op_join(save_dir, "gtsrb_msp.npy"))
    ood_gtsrb_anomal_pred_msp_score = np.load(file=op_join(save_dir, "gtsrb_anomal_msp.npy"))
    ood_cifar10_pred_msp_score = np.load(file=op_join(save_dir, "cifar_msp.npy"))
    ood_stl10_pred_msp_score = np.load(file=op_join(save_dir, "stl_msp.npy"))
    # Energy scores
    ind_gtsrb_pred_energy_score = np.load(file=op_join(save_dir, "gtsrb_energy.npy"))
    ood_gtsrb_anomal_pred_energy_score = np.load(file=op_join(save_dir, "gtsrb_anomal_energy.npy"))
    ood_cifar10_pred_energy_score = np.load(file=op_join(save_dir, "cifar_energy.npy"))
    ood_stl10_pred_energy_score = np.load(file=op_join(save_dir, "stl_energy.npy"))
    # Mahalanobis distance scores
    ind_gtsrb_m_dist_score = np.load(file=op_join(save_dir, "gtsrb_mdist.npy"))
    ood_gtsrb_anomal_m_dist_score = np.load(file=op_join(save_dir, "gtsrb_anomal_mdist.npy"))
    ood_cifar10_m_dist_score = np.load(file=op_join(save_dir, "cifar_mdist.npy"))
    ood_stl10_m_dist_score = np.load(file=op_join(save_dir, "stl_mdist.npy"))
    # kNN score
    ind_gtsrb_kth_dist_score = np.load(file=op_join(save_dir, "gtsrb_knn.npy"))
    ood_gtsrb_anomal_kth_dist_score = np.load(file=op_join(save_dir, "gtsrb_anomal_knn.npy"))
    ood_cifar10_kth_dist_score = np.load(file=op_join(save_dir, "cifar_knn.npy"))
    ood_stl10_kth_dist_score = np.load(file=op_join(save_dir, "stl_knn.npy"))
    # Entropies
    gtsrb_rn18_h_z_gtsrb_normal_train_samples_np = np.load(file=op_join(save_dir, "gtsrb_h_z_train.npy"))
    gtsrb_h_z = np.load(file=op_join(save_dir, "gtsrb_h_z.npy"))
    gtsrb_anomal_h_z = np.load(file=op_join(save_dir, "gtsrb_anomal_h_z.npy"))
    cifar10_h_z = np.load(file=op_join(save_dir, "cifar_h_z.npy"))
    stl10_h_z = np.load(file=op_join(save_dir, "stl_h_z.npy"))

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
        # Make all baselines plots
        for plot_title, experiment in tqdm(baselines_plots.items(), desc="Plotting baselines"):
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
        for experiment_name, experiment in tqdm(baselines_experiments.items(), desc="Logging baselines"):
            r_df, r_mlflow = get_hz_detector_results(detect_exp_name=experiment_name,
                                                     ind_samples_scores=experiment["InD"],
                                                     ood_samples_scores=experiment["OoD"],
                                                     return_results_for_mlflow=True)
            r_mlflow = dict([(f"{experiment_name}_{k}", v) for k, v in r_mlflow.items()])
            mlflow.log_metrics(r_mlflow)
            # roc_curve = save_roc_ood_detector(
            #     results_table=r_df,
            #     plot_title=f"ROC gtsrb vs {experiment_name} {cfg.layer_type} layer"
            # )
            # mlflow.log_figure(figure=roc_curve,
            #                   artifact_file=f"figs/roc_{experiment_name}.png")
            overall_metrics_df = overall_metrics_df.append(r_df)

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
        print("LaRED running...")
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
        for experiment_name, experiment in tqdm(la_red_experiments.items(), desc="Logging LaRED"):
            r_df, r_mlflow = get_hz_detector_results(detect_exp_name=experiment_name,
                                                     ind_samples_scores=experiment["InD"],
                                                     ood_samples_scores=experiment["OoD"],
                                                     return_results_for_mlflow=True)
            # Add OoD dataset to metrics name
            r_mlflow = dict([(f"{experiment_name}_{k}", v) for k, v in r_mlflow.items()])
            mlflow.log_metrics(r_mlflow)
            # Plot ROC curve
            # roc_curve = save_roc_ood_detector(
            #     results_table=r_df,
            #     plot_title=f"ROC gtsrb vs {experiment_name} {cfg.layer_type} layer"
            # )
            # # Log the plot with mlflow
            # mlflow.log_figure(figure=roc_curve,
            #                   artifact_file=f"figs/roc_{experiment_name}.png")
            overall_metrics_df = overall_metrics_df.append(r_df)
        overall_metrics_df_name = f"./results_csvs/{current_date}_experiment.csv"
        overall_metrics_df.to_csv(path_or_buf=overall_metrics_df_name)
        mlflow.log_artifact(overall_metrics_df_name)

        # Plot Roc curves together, by OoD dataset
        for experiment in ("anomal", "cifar10", "stl10"):
            temp_df = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                            'fpr', 'tpr', 'roc_thresholds',
                                            'precision', 'recall', 'pr_thresholds'])
            for row_name in overall_metrics_df.index:
                if experiment in row_name:
                    temp_df = temp_df.append(overall_metrics_df.loc[row_name])
                    temp_df.rename(index={row_name: row_name.split(experiment)[1]}, inplace=True)

            # Plot ROC curve
            roc_curve = save_roc_ood_detector(
                results_table=temp_df,
                plot_title=f"ROC gtsrb vs {experiment} {cfg.layer_type} layer"
            )
            # Log the plot with mlflow
            mlflow.log_figure(figure=roc_curve,
                              artifact_file=f"figs/roc_{experiment}.png")

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
