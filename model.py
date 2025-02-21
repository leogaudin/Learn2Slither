import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
        self,
        n_observations,
        n_actions,
    ):
        """ Initialize Deep Q Network
        """
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 42, dtype=torch.float32)
        self.layer2 = nn.Linear(42, 42, dtype=torch.float32)
        self.layer3 = nn.Linear(42, n_actions, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def save(self, filename):
        """ Save model
        """
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        """ Load model
        """
        self.load_state_dict(torch.load(filename))

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """ Call model (redefinition for type hinting)
        """
        return super().__call__(*args, **kwargs)
