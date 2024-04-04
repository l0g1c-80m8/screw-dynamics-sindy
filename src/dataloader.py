import os
import pandas as pd
from torch.utils.data import Dataset


class ScrewdrivingDataset(Dataset):

    def __init__(self, data_dir, file_ext):
        super().__init__()

        self._X = []
        self._y = []
        self._data_dir = data_dir
        self._file_ext = file_ext

        # read data
        self._init_data()

    def _init_data(self):
        for dir_item in os.listdir(self._data_dir):
            dir_item_path = os.path.join(self._data_dir, dir_item)

            if os.path.isdir(dir_item_path):
                continue
            if not dir_item_path.endswith(self._file_ext):
                continue

            sensor_data = pd.read_csv(dir_item_path)

            print(sensor_data)

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return None
