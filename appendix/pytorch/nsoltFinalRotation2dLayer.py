import torch
import torch.nn as nn

class NsoltFinalRotation2dLayer(nn.Module):

    def __init__(self):
        super(NsoltFinalRotation2dLayer, self).__init__()

    def forward(self,x):
        return x