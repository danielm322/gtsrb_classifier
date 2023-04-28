import argparse


def argpument_parser():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '-e',
        '--epochs',
        dest='epochs',
        default=20,
        type=int,
        help='Max number of epochs for Training'
    )
    
    argparser.add_argument(
        '-m',
        '--model',
        dest='model',
        default='resnet18',
        choices=["resnet18","resnet34", "resnet50", "resnet101", "resnet152"],
        type=str,
        help='Resnet architecture model'
    )
    
    argparser.add_argument(
        '-b',
        '--batchsize',
        dest='batch_size',
        default=16,
        type=int,
        help='Batch size'
    )    

    argparser.add_argument(
        "--loss_type",
        dest="loss_type",
        type=str,
        default='cross_entropy',
        choices=["nll", "cross_entropy", "focal"],
        help="Loss type (default: cross_entropy)")

    argparser.add_argument(
        '-s',
        '--seed',
        dest='random_seed',
        default=9290,
        type=int,
        help='Random Seed Everything'
    )
    
    argparser.add_argument(
        '-p',
        '--datapath',
        dest='dataset_path',
        default='./gtsrb-data/',
        type=str,
        help='Dataset path'
    )
    
    argparser.add_argument(
        '--slurm',
        dest='slurm_training',
        action='store_true',
        help='slurm training on HPC')
    argparser.set_defaults(slurm_training=False)

    argparser.add_argument(
        '-g',
        '--gpus',
        dest='gpus',
        default=-1,
        type=int,
        help='Number of GPUs'
    )

    argparser.add_argument(
        '--rich_progbar',
        dest='rich_progbar',
        action='store_true',
        help='TQMD Progress bar')
    argparser.set_defaults(rich_progbar=False)

    ####################################################################################
    # Model hyper-parameters added here to improve traceability
    ####################################################################################
    argparser.add_argument(
        '--input-channels',
        dest='input_channels',
        default=3,
        type=int,
        help='Input channels'
    )

    argparser.add_argument(
        '--drop-block',
        dest='drop_block',
        action='store_true',
        help='Dropblock module')
    argparser.set_defaults(drop_block=True)

    argparser.add_argument(
        '--dropblock-prob',
        dest='dropblock_prob',
        default=0.5,
        type=float,
        help='Dropblock probability'
    )

    argparser.add_argument(
        '--dropout',
        dest='dropout',
        action='store_true',
        help='use dropout')
    argparser.set_defaults(dropout=True)

    argparser.add_argument(
        '--dropout-prob',
        dest='dropout_prob',
        default=0.3,
        type=float,
        help='Dropout probability'
    )

    argparser.add_argument(
        '-lr',
        '--learning-rate',
        dest='learning_rate',
        default=1e-4,
        type=float,
        help='Learning rate'
    )

    argparser.add_argument(
        '--weight-decay',
        dest='optimizer_weight_decay',
        default=1e-4,
        type=float,
        help='Optimizer weight decay'
    )

    argparser.add_argument(
        '--image-width',
        dest='image_width',
        default=32,
        type=int,
        help='Image width'
    )
    argparser.add_argument(
        '--image-height',
        dest='image_height',
        default=32,
        type=int,
        help='Image height'
    )

    argparser.add_argument(
        '--shuffle',
        dest='shuffle',
        action='store_true',
        help='Shuffle dataset')
    argparser.set_defaults(shuffle=True)
    # parse all the arguments
    arguments = argparser.parse_args()
    
    return arguments