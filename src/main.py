import argparse
import os
import sys

from src.trainer import Trainer


def get_args():
    # define parser
    parser = argparse.ArgumentParser()

    # set args
    parser.add_argument('--data_dir', type=str, default='./data',
                        action='store', dest='data_dir', help='data directory')
    parser.add_argument('--out_dir', type=str, default='./out',
                        action='store', dest='out_dir', help='output directory')
    parser.add_argument('--train_file', type=str, default='train_data.csv',
                        action='store', dest='train_file', help='train file name')
    parser.add_argument('--val_file', type=str, default='val_data.csv',
                        action='store', dest='val_file', help='val file name')
    parser.add_argument('--test_file', type=str, default='test_data.csv',
                        action='store', dest='test_file', help='test file name')

    parser.add_argument('--poly_order', type=int, default=3,
                        action='store', dest='poly_order', help='highest polynomial order in sindy library')
    parser.add_argument('--include_constant', type=bool, default=True,
                        action='store', dest='include_constant', help='include constant function in sindy library')
    parser.add_argument('--use_sine', type=bool, default=True,
                        action='store', dest='use_sine', help='use sine function in sindy library')

    parser.add_argument('--input_var_dim', type=int, default=17,
                        action='store', dest='input_var_dim', help='dimension of input variable')
    parser.add_argument('--state_var_dim', type=int, default=2,
                        action='store', dest='state_var_dim', help='dimension of state variable')

    parser.add_argument('--device', type=str, default='cpu',
                        action='store', dest='device', help='device to run operations on')

    parser.add_argument('--learning_rate', type=float, default=.00001,
                        action='store', dest='learning_rate', help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=.000001,
                        action='store', dest='weight_decay', help='weight decay for training')
    parser.add_argument('--epochs', type=int, default=100,
                        action='store', dest='epochs', help='epochs for training')

    parser.add_argument('--window_length', type=int, default=1,
                        action='store', dest='window_length', help='batch window size')

    # return parsed arguments
    return parser.parse_args()


def main():
    os.makedirs(ARGS.out_dir, exist_ok=True)
    os.makedirs(os.path.join(ARGS.out_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(ARGS.out_dir, 'plots'), exist_ok=True)

    trainer = Trainer(**vars(ARGS))
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    # get args
    ARGS = get_args()

    # exec
    main()

    # exit without errors
    sys.exit()
