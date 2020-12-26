import torch
import torch.nn as nn

class NsoltSubbandSerialization2dLayer(nn.Module):

    def __init__(self):
        super(NsoltSubbandSerialization2dLayer, self).__init__()

    def forward(self,x):
        return x