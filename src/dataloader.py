import os
import numpy as np
import pandas as pd
from argparse import Namespace
from torch.utils.data import Dataset


class ScrewdrivingDataset(Dataset):
    _WINDOW_LENGTH = 200
    _STIFFNESS_NORM = [3000, 3000, 3000, 300, 300, 300]

    def __init__(self, data_dir, sensor_file, observation_file, mode='train', **kwargs):
        super().__init__()

        self._X = []
        self._y = []
        self._data_dir = os.path.join(data_dir, kwargs.get('{}_dataset_subdir'.format(mode), ''))
        self._sensor_file = sensor_file
        self._observation_file = observation_file

        # read data
        self._init_data()

    def _init_data(self):
        for dir_item in os.listdir(self._data_dir):
            subdir_path = os.path.join(self._data_dir, dir_item)

            if not os.path.isdir(subdir_path):
                continue

            sensor_file = os.path.join(subdir_path, self._sensor_file)
            observation_file = os.path.join(subdir_path, self._observation_file)

            if not os.path.exists(sensor_file) or not os.path.exists(observation_file):
                continue

            # read in sensor data in a pd data frame and process it into chunks of _WINDOW_LENGTH
            sensor_data_df = pd.read_csv(sensor_file)
            observation_data_df = pd.read_csv(observation_file)

            if sensor_data_df.shape[0] < self._WINDOW_LENGTH or observation_data_df.shape[0] < self._WINDOW_LENGTH:
                continue

            for idx in range(sensor_data_df.shape[0] // self._WINDOW_LENGTH):
                s_idx = idx * self._WINDOW_LENGTH
                e_idx = s_idx + self._WINDOW_LENGTH

                state_data = np.column_stack((
                    np.asarray(observation_data_df[s_idx:e_idx]['X'], dtype=np.float32),
                    np.asarray(observation_data_df[s_idx:e_idx]['Y'], dtype=np.float32),
                    np.asarray(observation_data_df[s_idx:e_idx]['Z'], dtype=np.float32),
                ))

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
                    orientation_data,  # input variable (X)
                    stiffness_data,  # control variable (U)
                    damping_data  # control variable (U)
                )))
                self._y.append(state_data)  # state variable

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        return [self._X[idx], self._y[idx]]
