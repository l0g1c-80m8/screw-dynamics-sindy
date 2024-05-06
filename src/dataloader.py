import os
import numpy as np
import pandas as pd
from derivative import dxdt
from torch.utils.data import Dataset


class ScrewdrivingDataset(Dataset):
    _WINDOW_LENGTH = 200
    _STIFFNESS_NORM = [3000, 3000, 3000, 300, 300, 300]

    def __init__(self, mode='train', **kwargs):
        super().__init__()

        self._X = []
        self._y = []
        self._data_file = os.path.join(kwargs['data_dir'], kwargs['{}_file'.format(mode)])

        # read data
        self._init_data()

    def _init_data(self):
        df = pd.read_csv(self._data_file)

        for idx in range(df.shape[0] // self._WINDOW_LENGTH):
            s_idx = idx * self._WINDOW_LENGTH
            e_idx = s_idx + self._WINDOW_LENGTH

            timestamps = np.asarray(df[s_idx:e_idx]['time'], dtype=np.float64)

            velocity_data = np.column_stack((
                np.asarray(
                    dxdt(np.asarray(df[s_idx:e_idx]['X'], dtype=np.float32), timestamps, kind='finite_difference', k=1),
                    dtype=np.float32
                ),
                np.asarray(
                    dxdt(np.asarray(df[s_idx:e_idx]['Y'], dtype=np.float32), timestamps, kind='finite_difference', k=1),
                    dtype=np.float32),
                np.asarray(
                    dxdt(np.asarray(df[s_idx:e_idx]['Z'], dtype=np.float32), timestamps, kind='finite_difference', k=1),
                    dtype=np.float32),
            ))

            position_data = np.column_stack((
                np.asarray(df[s_idx:e_idx]['X'], dtype=np.float32),
                np.asarray(df[s_idx:e_idx]['Y'], dtype=np.float32),
                np.asarray(df[s_idx:e_idx]['Z'], dtype=np.float32),
            ))

            orientation_data = np.column_stack((
                np.asarray(df['A'][s_idx:e_idx], dtype=np.float32) / np.pi,
                np.asarray(df['B'][s_idx:e_idx], dtype=np.float32) / np.pi,
                np.asarray(df['C'][s_idx:e_idx], dtype=np.float32) / np.pi
            ))

            stiffness_data = np.column_stack((
                np.asarray(df['Kx'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[0],
                np.asarray(df['Ky'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[1],
                np.asarray(df['Kz'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[2],
                np.asarray(df['Rot_Kx'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[3],
                np.asarray(df['Rot_Ky'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[4],
                np.asarray(df['Rot_Kz'][s_idx:e_idx], dtype=np.float32) / self._STIFFNESS_NORM[5]
            ))

            damping_data = np.column_stack((
                np.asarray(df['Cx'][s_idx:e_idx], dtype=np.float32),
                np.asarray(df['Cy'][s_idx:e_idx], dtype=np.float32),
                np.asarray(df['Cz'][s_idx:e_idx], dtype=np.float32),
                np.asarray(df['Rot_Cx'][s_idx:e_idx], dtype=np.float32),
                np.asarray(df['Rot_Cy'][s_idx:e_idx], dtype=np.float32),
                np.asarray(df['Rot_Cz'][s_idx:e_idx], dtype=np.float32)
            ))

            self._X.append(np.column_stack((
                position_data,  # input variable (X)
                orientation_data,  # input variable (X)
                stiffness_data,  # control variable (U)
                damping_data  # control variable (U)
            )))
            self._y.append(velocity_data)  # state variable

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        return [self._X[idx], self._y[idx]]
