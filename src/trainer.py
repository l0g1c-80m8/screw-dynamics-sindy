import torch
from argparse import Namespace
from torch.utils.data import DataLoader

from src.dataloader import ScrewdrivingDataset
from src.model import SindyModel


class Trainer:
    def __init__(self, **kwargs):
        self._params = Namespace(**kwargs)

        self._model = SindyModel(**kwargs)

        torch.device(self._params.device)

        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(),
            lr=self._params.learning_rate,
            weight_decay=self._params.weight_decay
        )
        self._loss = torch.nn.MSELoss()

        self._dataset_obj = ScrewdrivingDataset(**kwargs)

    def train(self):
        train_data_loader = DataLoader(self._dataset_obj)

        self._model.to(torch.device(self._params.device))
        self._model.get_latent_params = True

        for epoch in range(self._params.epochs):
            print('training epoch: {}'.format(epoch + 1))

            self._model.train()

            current_train_loss = 0.
            current_val_loss = 0.

            for batch, (x, x_dot) in enumerate(train_data_loader):
                self._optimizer.zero_grad()
                pred_x_dot = self._model(x)

                x_dot = x_dot.to(torch.device(self._params.device))
                train_loss = self._loss(x_dot, pred_x_dot)

                self._model.zero_grad()
                train_loss.backward()

                self._optimizer.step()
                current_train_loss += train_loss.item()

            print('loss: {}'.format(current_train_loss / len(train_data_loader.dataset)))

            self._model.eval()

    def evaluate(self):
        pass
