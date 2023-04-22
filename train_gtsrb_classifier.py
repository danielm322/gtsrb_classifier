from argument_parser import argpument_parser
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.callbacks import TrainingDataMonitor
from pytorch_lightning.plugins.environments import SLURMEnvironment
from models import ResnetModule
from datasets import GtsrbModule
from icecream import ic


def main(args):
    #####################
    #      Get Args     #
    #####################
    model_type = args.model
    max_nro_epochs = args.epochs
    batch_size = args.batch_size
    random_seed_everything = args.random_seed
    dataset_path = args.dataset_path
    loss_type = args.loss_type
    rich_progbar = args.rich_progbar
    slurm_training = args.slurm_training
    gpus_nro = args.gpus

    print(' ')
    print('=' * 60)
    ic(model_type)
    ic(max_nro_epochs)
    ic(batch_size)
    ic(loss_type)
    ic(random_seed_everything)
    ic(slurm_training)
    ic(gpus_nro)
    print('=' * 60)
    print(' ')
    
    ############################
    #      Seed Everything     #
    ############################
    torch.cuda.empty_cache()
    pl.seed_everything(random_seed_everything)
    #######################################
    #      Training Monitor/Callbacks     #
    #######################################
    checkpoint_callback = ModelCheckpoint(monitor="Validation loss",
                                          mode='min',
                                          every_n_epochs=1,
                                          save_top_k=2,
                                          save_last=True,
                                          save_on_train_epoch_end=False)
    
    monitor = TrainingDataMonitor(log_every_n_steps=20)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    if rich_progbar:  # fancy aesthetic progress bar
        progress_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow",
                                                            progress_bar="green1",
                                                            progress_bar_finished="green1",
                                                            batch_progress="green_yellow",
                                                            time="grey82",
                                                            processing_speed="grey82",
                                                            metrics="grey82"))
    else:  # normal aesthetic progress bar
        progress_bar = TQDMProgressBar(refresh_rate=10)     
        
    ###############################
    #      Get Dataset Module     #
    ###############################
    data_module = GtsrbModule(data_path=dataset_path,
                              img_size=(32, 32),
                              batch_size=batch_size,
                              shuffle=True)

    data_module.setup(stage="fit")
    data_module.setup(stage="validate")
    data_module.setup(stage="test")
    
    num_classes = len(data_module.ds_gtsrb_train.classes)
    ic(num_classes)
    #############################
    #      Get Model Module     #
    #############################
    model_module =  ResnetModule(arch_name=model_type,
                                 input_channels=3,
                                 num_classes=num_classes,
                                 dropblock=True,
                                 dropblock_prob=0.5,
                                 dropout=True,
                                 dropout_prob=0.3,
                                 loss_fn=loss_type,
                                 optimizer_lr=1e-4,
                                 optimizer_weight_decay=1e-4,
                                 max_nro_epochs=max_nro_epochs)
    
    ########################################
    #      Start Module/Model Training     #
    ########################################
    if slurm_training:  # slurm training on HPC with multiple GPUs
        ic(slurm_training)
        trainer = pl.Trainer(accelerator='gpu',
                             devices=gpus_nro,
                             num_nodes=1,
                             strategy='ddp',
                             max_epochs=max_nro_epochs,
                             callbacks=[progress_bar,
                                        checkpoint_callback,
                                        monitor,
                                        lr_monitor],
                             plugins=[SLURMEnvironment(auto_requeue=False)])

    else:  # training locally in computer with GPU
        ic(slurm_training)
        trainer = pl.Trainer(accelerator='gpu',
                             devices=gpus_nro,
                             max_epochs=max_nro_epochs,
                             callbacks=[progress_bar,
                                        checkpoint_callback,
                                        monitor,
                                        lr_monitor])

    # Fit Trainer
    trainer.fit(model=model_module, datamodule=data_module)  # fit a model!


if __name__ == "__main__":
    args = argpument_parser()
    main(args)

