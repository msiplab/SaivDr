import torch
import torch.nn as nn

class NsoltSubbandDeserialization2dLayer(nn.Module):

    def __init__(self):
        super(NsoltSubbandDeserialization2dLayer, self).__init__()

    def forward(self,x):
        return x