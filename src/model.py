import torch
from torch import nn
from argparse import Namespace
from scipy.special import binom
from sklearn.preprocessing import PolynomialFeatures


class SindyModel(nn.Module):
    _COFF_KEY = 'sindy_coefficients'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._params = Namespace(**kwargs)

        self._model_params = torch.nn.ParameterDict({
            self._COFF_KEY: torch.nn.Parameter(torch.empty(self._params.library_dim, self._params.state_var_dim))
        })
        torch.nn.init.xavier_uniform_(self._model_params[self._COFF_KEY])

        self._coefficient_mask = torch.ones(
            (self._params.library_dim, self._params.state_var_dim),
            device=self._params.device,
            dtype=torch.float32
        )

        self._Theta = PolynomialFeatures(self._params.poly_order)

        self._library_dim = None

    @property
    def library_dim(self):
        if self._library_dim is None:
            library_dim = 0
            for k in range(self._params.poly_order + 1):
                library_dim += int(binom(self._params.input_var_dim + k - 1, k))
            if self._params.use_sine:
                library_dim += self._params.input_var_dim
            if not self._params.include_constant:
                library_dim -= 1
            self._library_dim = library_dim

        return self._library_dim

    def forward(self, x):
        x_dot_batch = torch.Tensor(size=(*x.shape, self._params.state_var_dim))

        for idx in range(x_dot_batch.shape[0]):
            x_idx = x[idx].detach().cpu().numpy()
            theta_idx = torch.from_numpy(self._Theta.fit_transform(x_idx))
            theta_idx = theta_idx.to(torch.device(self._params.device))
            if self._params.include_sine:
                theta_idx = torch.hstack((theta_idx, torch.sin(x[idx])))
            x_dot_batch[idx] = torch.matmul(theta_idx, self.coefficient_mask * self._model_params[self._COFF_KEY])

        return x_dot_batch.to(self._params.device)
