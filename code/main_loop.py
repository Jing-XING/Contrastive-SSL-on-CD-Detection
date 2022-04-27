import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
from typing import Any, Mapping, Text, Type, Union

from byol import byol_experiment
from byol import eval_experiment
from byol.configs import byol as byol_config
from byol.configs import eval as eval_config
flags.DEFINE_string('experiment_mode','pretrain','the experiment mode ,pretrain or eval')
flags.DEFINE_string('worker_mode','train','mode, train or eval')
flags.DEFINE_integer('pretrain_epochs', 1000, 'Number of pre-training epochs')
flags.DEFINE_integer('batch_size', 4096, 'Total batch size')
flags.DEFINE_string('checkpoint_root', '/tmp/byol',
                    'The directory to save checkpoints to.')# todo need mkdir?
flags.DEFINE_integer('log_tensors_interval', 60, 'Log tensors every n seconds.')# todo ?
FLAGS = flags.FLAGS
# Experiment = Union[
#     Type[byol_experiment.ByolExperiment],
#     Type[eval_experiment.EvalExperiment]]# todo ?
def train_loop(experiment_class, config):
    '''
    The main training loop.
    Periodically save a checkpoint to be evaluated in the eval_loop.
    Args:
        experiment_class: the constructor for the experiment (either byol_experiment
        or eval_experiment).
    config: the experiment config.
    '''
    experiment = experiment_class(**config)
def main():
    if FLAGS.experiment_mode == 'pretrain':
        experiment_class = byol_experiment.ByolExperiment
        config = byol_config.get_config(FLAGS.pretrain_epochs, FLAGS.batch_size)
    elif FLAGS.experiment_mode == 'Linear-eval':
        experiment_class = eval_experiment.EvalExperiment
        config = eval_config.get_config(f'{FLAGS.checkpoint_root}/pretrain.pkl,',
                                        FLAGS.bath_size)
    else:
        raise ValueError(f'Unknown experiment mode: {FLAGS.experiment_mode}')
    config['checkpointing_config']['checkpoint_dir'] = FLAGS.checkpoint_root
    if FLAGS.worker_mode == 'train':
        train_loop(experiment_class, config)
    elif FLAGS.worker_mode == 'eval':
        eval_loop(experiment_class, config)

if __name__ == '__main__':
    main()


