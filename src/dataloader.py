import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ScrewdrivingDataset(Dataset):
    _WINDOW_LENGTH = 200
    _STIFFNESS_NORM = [3000, 3000, 3000, 300, 300, 300]

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

            # read in sensor data in a pd data frame and process it into chunks of _WINDOW_LENGTH
            sensor_data_df = pd.read_csv(dir_item_path)

            if sensor_data_df.shape[0] < self._WINDOW_LENGTH:
                continue

            for idx in range(sensor_data_df.shape[0] // self._WINDOW_LENGTH):
                s_idx = idx * self._WINDOW_LENGTH
                e_idx = s_idx + self._WINDOW_LENGTH

                # TODO: extract X, Y, Z data from camera and read it here

                orientation_data = np.column_stack((
                    np.asarray(sensor_data_df['A'][s_idx:e_idx], dtype=np.float32) / np.pi,
                    np.asarray(sensor_data_df['B'][s_idx:e_idx], dtype=np.float32) / np.pi,
                    np.asarray(sensor_data_df['C'][s_idx:e_idx], dtype=np.float32) / np.pi
                ))

                stiffness_data = np.column_stack((
                    np.asarray(sensor_data_df['Kx'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[0],
                    np.asarray(sensor_data_df['Ky'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[1],
                    np.asarray(sensor_data_df['Kz'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[2],
                    np.asarray(sensor_data_df['Rot_Kx'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[3],
                    np.asarray(sensor_data_df['Rot_Ky'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[4],
                    np.asarray(sensor_data_df['Rot_Kz'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[5]
                ))

                damping_data = np.column_stack((
                    np.asarray(sensor_data_df['Cx'][s_idx:e_idx], dtype=np.float32),
                    np.asarray(sensor_data_df['Cy'][s_idx:e_idx], dtype=np.float32),
                    np.asarray(sensor_data_df['Cz'][s_idx:e_idx], dtype=np.float32),
                    np.asarray(sensor_data_df['Rot_Cx'][s_idx:e_idx], dtype=np.float32),
                    np.asarray(sensor_data_df['Rot_Cy'][s_idx:e_idx], dtype=np.float32),
                    np.asarray(sensor_data_df['Rot_Cz'][s_idx:e_idx], dtype=np.float32)
                ))

                self._X.append(np.column_stack((
                    # TODO: append the position data here
                    orientation_data,
                    stiffness_data,
                    damping_data
                )))
                # TODO: append the derivate of the position data to y here

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        return [self._X[idx], self._y[idx]]
