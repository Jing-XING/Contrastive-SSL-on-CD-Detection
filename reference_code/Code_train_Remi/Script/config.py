import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg



# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--data_dir', type=str, default='../Data/CrohnConsensus2C/unsplited',
                      help='Directory in which data is stored')
data_arg.add_argument('--dir_output', type=str, default='csv_pred_truth',
                      help='csv_output dir in test')
data_arg.add_argument('--csv', type=str, default= '../Data/Crohn2020_consensus/consensus_full.csv',
                      help='CSV for dataloader')
data_arg.add_argument('--shuffle_mode', type=str, default= 'KFold',
                      help='Shuffle mode Kfold or ShuffleSplit')
data_arg.add_argument('--valid_size', type=float, default=0.1,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of training set used for testing')
data_arg.add_argument('--batch_size', type=int, default=16,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_crossVal', type=int, default=1,
                      help='number of cross-validation')
data_arg.add_argument('--num_workers', type=int, default=0,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')
data_arg.add_argument('--nb_classes', type=int, default=2,
                      help='number of classes')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')

train_arg.add_argument('--epochs', type=int, default=700,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=3e-4,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=30,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=50,
                       help='Number of epochs to wait before stopping train')


# other params
misc_arg = add_argument_group('Misc.')

misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--test', type=str2bool, default=False,
                      help='Whether it is a final training or just a test')
misc_arg.add_argument('--random_seed', type=int, default=0,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--add_seed', type=str2bool, default=True,
                      help="Whether add the seed value to the model name")
misc_arg.add_argument('--ckpt_dir', type=str, default='../ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')






def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
