"""Identity layer (MATLAB nsoltIdentityLayer)."""

import torch.nn as nn

__all__ = ["NsoltIdentityLayer"]


class NsoltIdentityLayer(nn.Module):
    """Equivalent to MATLAB ``nsoltIdentityLayer``."""

    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self.description = "Identity"

    def forward(self, X):
        return X
