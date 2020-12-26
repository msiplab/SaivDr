import torch
import torch.nn as nn

class NsoltIntermediateRotation2dLayer(nn.Module):

    def __init__(self):
        super(NsoltIntermediateRotation2dLayer, self).__init__()

    def forward(self,x):
        return x