import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape (batch,)
        returns: shape (batch, embedding_dim)
        """
        half_dim = self.embedding_dim // 2
        device = t.device

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half_dim, device=device).float()
            / half_dim
        )

        args = t[:, None] * frequencies[None, :]

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class MLPVectorField(nn.Module):
    def __init__(
        self,
        dim: int,
        time_embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_hidden_layers: int = 5,
    ):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)

        layers = []

        input_dim = dim + time_embedding_dim

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        inp = torch.cat([z_t, t_emb], dim=-1)
        return self.net(inp)