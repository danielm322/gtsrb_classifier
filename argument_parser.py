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
    
    # parse all the arguments
    arguments = argparser.parse_args()
    
    return arguments