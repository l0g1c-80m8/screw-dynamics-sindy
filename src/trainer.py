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

        torch.device(self._params.device)

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

            train_loss = current_train_loss / len(train_data_loader.dataset)
            print('training loss: {}'.format(train_loss))
            self._writer.add_scalar("Loss/train", train_loss, epoch)

            for batch, (x, x_dot) in enumerate(val_data_loader):
                self._optimizer.zero_grad()
                pred_x_dot = self._model(x)

                x_dot = x_dot.to(torch.device(self._params.device))
                train_loss = self._loss(x_dot, pred_x_dot)
                current_val_loss += train_loss.item()

            val_loss = current_val_loss / len(val_data_loader.dataset)
            print('validation loss: {}'.format(val_loss))
            self._writer.add_scalar("Loss/val", val_loss, epoch)

            self._model.eval()

        self._writer.close()

    def evaluate(self):
        test_data_loader = DataLoader(self._test_dataset_obj)
        x, x_dot = next(iter(test_data_loader))
        self._model.eval()
        print('Coefficients: {}'.format(self._model.coefficients))

        pred_x_dot = self._model(x)

        # TODO: do something with the predicted values
