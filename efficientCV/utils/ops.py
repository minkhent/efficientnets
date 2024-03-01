import torch
from torch import nn


class DropConnect(nn.Module):

    "Implementation of Drop connect module also known as Stochatic depth."

    def __init__(self, p: float, device: str) -> None:
        super().__init__()
        self.p = p
        self.device = device

    def drop_connect(self, x: torch.Tensor, drop_connect_prob: float) -> torch.Tensor:
        keep_prob = 1 - drop_connect_prob
        batch_size = x.shape[0]
        if self.device == "cuda":
            random_tensor = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
        else:
            random_tensor = torch.FloatTensor(batch_size, 1, 1, 1).uniform_()
        binary_mask = keep_prob > random_tensor
        x = (x / keep_prob) * binary_mask

        return x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.drop_connect(input, self.p)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p})"
        return s
