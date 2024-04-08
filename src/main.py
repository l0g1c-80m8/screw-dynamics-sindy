import argparse
import sys

from src.trainer import Trainer


def get_args():
    # define parser
    parser = argparse.ArgumentParser()

    # set args
    parser.add_argument('--data_dir', type=str, default='./data',
                        action='store', dest='data_dir', help='data directory')
    parser.add_argument('--sensor_file', type=str, default='sensor_data.csv',
                        action='store', dest='sensor_file', help='sensor file name')
    parser.add_argument('--observation_file', type=str, default='observation_data.csv',
                        action='store', dest='observation_file', help='observation file name')

    # return parsed arguments
    return parser.parse_args()


def main():
    trainer = Trainer()
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    # get args
    ARGS = get_args()

    # exec
    main()

    # exit without errors
    sys.exit()
