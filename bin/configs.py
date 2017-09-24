"""
embedder
__author__ = 'josh (heythisischo@gamil.com)'
__copyright__ = 'No copyright. Just copyleft!'
"""

#pylint: disable=no-member


import argparse
import pprint
import logging
import sys
from pathlib import Path
from torch import optim


#############
# Directory #
#############

PROJECT_DIR = Path(__file__).resolve().parent.parent # grace directory
LIB_DIR = PROJECT_DIR.joinpath('lib') # './lib
sys.path.append(LIB_DIR) # Add PYTHONPATH
RSC_DIR = PROJECT_DIR.joinpath('rsc') # './rsc
DATA_DIR = PROJECT_DIR.joinpath('data') # ./data
LOG_DIR = PROJECT_DIR.joinpath('logdir') # ./logdir


OPTIMIZER_DICT = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}


class Config(object):
    """
    Configuration Class: set kwargs as class attributes with setattr
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = OPTIMIZER_DICT[value]
                setattr(self, key, value)

        self.context_len = self.window * 2 + 1

        self.model_id = self.get_model_id()

        # Where to save model checkpoints / log tensorboard summary
        self.model_dir = self.log_dir.joinpath(self.model_id)
        self.ckpt_path = self.model_dir.joinpath('ckpt')

        if self.is_train:
            if self.clean:
                self.model_dir.mkdir(exist_ok=True)
            else:
                self.model_dir.mkdir(exist_ok=False)


    def get_model_id(self):
        """Configuration summary string"""
        model_ids = [self.model_name, ]
        model_ids.append('cut%d' % self.cutoff)
        model_ids.append('pho' if self.phoneme else 'chr')
        model_ids.append('w%d' % self.window)
        model_ids.append('char%d' % self.char_dim)
        model_ids.append('word%d' % self.word_dim)
        model_ids.append('pos%d' % self.pos_dim)
        model_ids.append('gzte' if self.gazet_embed else 'gzt1')
        model_ids.append('pe%d' % (1 if self.pos_enc else 0))
        model_ids.append('re%d' % self.rvt_epoch)
        model_ids.append('rt%d' % self.rvt_term)
        model_ids.append('bs%d' % self.batch_size)
        return '.'.join(model_ids)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

###################
# Default Options #
###################
GPU_NUM = 0
WINDOW = 10
CHAR_DIM = 50
WORD_DIM = 50
RVT_EPOCH = 2
RVT_TERM = 10
BATCH_SIZE = 100


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser(description='Get configuration for NER model')

    #================ Mode ==============#
    parser.add_argument('--is-train', action='store_true', default=False)
    parser.add_argument('--gpu-num', help='GPU number to use <default: 0>',
                        metavar='INT', type=int, default=0)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    parser.add_argument('--clean', help='clean all existing checkpoints',
                        action='store_true', default=False)

    #================ Path ==============#
    parser.add_argument('-r', '--rsc-dir', help='resource directory',
                        metavar='DIR', default=RSC_DIR)
    parser.add_argument('-d', '--data-dir', help='data directory',
                        metavar='DIR', default=DATA_DIR)
    parser.add_argument('-p', '--in-pfx', help='input data prefix',
                        metavar='NAME', type=str, default='v2')
    parser.add_argument('--log-dir', help='tensorboard log dir'
                        '<default: ./logdir>',
                        metavar='DIR', default=LOG_DIR)

    #================ Data ==============#
    parser.add_argument('--cutoff', help='cutoff', action='store',
                        type=int, metavar='NUM', default=5)
    parser.add_argument('--n-tags', help='number of NER tags',
                        type=int, metavar='NUM', default=11)

    #================ Train ==============#
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--batch-size',
                        help='batch size <default: 100>', metavar='INT',
                        type=int, default=100)
    parser.add_argument('--rvt-epoch',
                        help='최대치 파라미터로 되돌아갈 epoch 횟수'
                        '<default: 2>', type=int, metavar='NUM', default=2)
    parser.add_argument('--rvt-term',
                        help='파라미터로 되돌아갈 최대 횟수 <default: 10>',
                        type=int, metavar='NUM', default=10)

    #================ Model ==============#
    parser.add_argument('-m', '--model-name', help='model name',
                        metavar='NAME', default='Cnn8')
    parser.add_argument('--rnn', type=str, default='gru')

    parser.add_argument('--jaso-dim', type=int, metavar='INT', default=50)
    parser.add_argument('--char-dim', type=int, metavar='INT', default=50)
    parser.add_argument('--word-dim', type=int, metavar='INT', default=50)
    parser.add_argument('--pos-dim', type=int, metavar='INT', default=20)

    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--num-layers', help='number of rnn layers', default=2)

    parser.add_argument('--window',
                        help='left/right character window length <default: 10>',
                        metavar='INT', type=int, default=10)
    parser.add_argument('--phoneme',
                        help='expand phonemes context', action='store_true')
    parser.add_argument('--disable-pos-enc', dest='pos_enc',
                        help='disable positional encoding',
                        action='store_false', default=True)
    parser.add_argument('--gazet-embed', help='gazetteer type',
                        action='store_true', default=False)

    #=============== Parse Arguments===============#
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]


    if kwargs.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Return Config class
    kwargs = vars(kwargs) # Namespace => Dictionary
    kwargs.update(optional_kwargs) # update optional keyword arguments
    return Config(**kwargs)


# for debugging
#if __name__ == '__main__':
#    config = get_config()
#    import ipdb; ipdb.set_trace()
