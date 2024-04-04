import argparse
import sys

from src.dataloader import ScrewdrivingDataset


def get_args():
    # define parser
    parser = argparse.ArgumentParser()

    # set args
    parser.add_argument('--data_dir', type=str, default='./data',
                        action='store', dest='data_dir', help='data directory')
    parser.add_argument('--file_ext', type=str, default='csv',
                        action='store', dest='file_ext', help='data directory')

    # return parsed arguments
    return parser.parse_args()


def main():
    ScrewdrivingDataset(**vars(ARGS))


if __name__ == '__main__':
    # get args
    ARGS = get_args()

    # exec
    main()

    # exit without errors
    sys.exit()
