import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.datasets import make_moons


class FlowDataset(Dataset):
    def __init__(self, n_samples):
        x, _ = make_moons(
            n_samples=n_samples,
            noise=0.08
        )
        self.data = torch.tensor(
            x,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3,64), nn.ELU(),
            nn.Linear(64,64), nn.ELU(),
            nn.Linear(64,64), nn.ELU(),
            nn.Linear(64, 2)
        )

    def forward(self,x,t):
        return self.network(torch.cat([x,t], dim=-1))

    def step(self, x_t, t_start, t_end):
      t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
      delta = t_end - t_start
      k_1 = self(x_t, t_start)
      x_mid = x_t + k_1 * delta / 2
      t_mid = t_start + delta / 2
      k_2 = self(x_mid, t_mid)
      return x_t + delta * k_2
