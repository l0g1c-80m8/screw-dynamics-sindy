import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataloader import ScrewdrivingDataset
from src.model import SindyModel


class Trainer:
    def __init__(self, **kwargs):
        self._params = Namespace(**kwargs)

        self._model = SindyModel(**kwargs)

        self._device = torch.device(self._params.device)
        self._model.to(self._device)

        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(),
            lr=self._params.learning_rate,
            weight_decay=self._params.weight_decay
        )
        self._loss = torch.nn.MSELoss(reduction='mean')

        self._train_dataset_obj = ScrewdrivingDataset(mode='train', **kwargs)
        self._test_dataset_obj = ScrewdrivingDataset(mode='test', **kwargs)
        self._val_dataset_obj = ScrewdrivingDataset(mode='val', **kwargs)

        self._writer = SummaryWriter()

    def train(self):
        train_data_loader = DataLoader(self._train_dataset_obj)
        val_data_loader = DataLoader(self._val_dataset_obj)

        self._model.train()

        for epoch in range(self._params.epochs):
            print('Training epoch: {}'.format(epoch + 1))

            current_train_loss, current_val_loss = 0., 0.

            for x, x_dot in train_data_loader:
                x, x_dot = x.to(self._device), x_dot.to(self._device)

                self._optimizer.zero_grad()
                pred_x_dot = self._model(x)
                train_loss = self._loss(pred_x_dot, x_dot)
                train_loss.backward()
                self._optimizer.step()

                current_train_loss += train_loss.item()

            train_loss = current_train_loss / len(train_data_loader)
            print('Training loss: {}'.format(train_loss))
            self._writer.add_scalar("Loss/train", train_loss, epoch)

            with torch.no_grad():
                self._model.eval()
                for x, x_dot in val_data_loader:
                    x, x_dot = x.to(self._device), x_dot.to(self._device)
                    pred_x_dot = self._model(x)
                    val_loss = self._loss(pred_x_dot, x_dot)
                    current_val_loss += val_loss.item()

                val_loss = current_val_loss / len(val_data_loader)
                print('Validation loss: {}'.format(val_loss))
                self._writer.add_scalar("Loss/val", val_loss, epoch)

            self._model.train()

        self._writer.close()

    def evaluate(self):
        test_data_loader = DataLoader(self._test_dataset_obj)
        current_test_loss = 0.

        d_x, d_x_dot, d_pred_x_dot = [], [], []

        self._model.eval()
        with torch.no_grad():
            for x, x_dot in test_data_loader:
                x, x_dot = x.to(self._device), x_dot.to(self._device)
                pred_x_dot = self._model(x)

                d_x.extend(x.cpu().numpy())
                d_x_dot.extend(x_dot.cpu().numpy())
                d_pred_x_dot.extend(pred_x_dot.cpu().numpy())

                test_loss = self._loss(pred_x_dot, x_dot)
                current_test_loss += test_loss.item()

        d_x = np.array(d_x).reshape(self._params.window_length, self._params.input_var_dim)
        d_x_dot = np.array(d_x_dot).reshape(self._params.window_length, self._params.state_var_dim)
        d_pred_x_dot = np.array(d_pred_x_dot).reshape(self._params.window_length, self._params.state_var_dim)

        print('Test loss: {}'.format(current_test_loss))
        print('Coefficients: {}'.format(self._model.coefficients))

        timestamps = np.arange(0, self._params.window_length)

        plt.figure(figsize=(50, 30))

        for idx in range(d_x_dot.shape[1]):
            plt.subplot(3, d_x_dot.shape[1], idx + 1)
            plt.plot(timestamps, d_pred_x_dot[:, idx], 'b', label='Predicted')
            plt.plot(timestamps, d_x_dot[:, idx], 'g', label='Actual')
            plt.title(f'Variable {idx + 1} (0: Vx, 1: Vy)')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self._params.out_dir, 'test_loss_compare.png'))
        plt.show()
