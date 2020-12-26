import torch
import torch.nn as nn

class NsoltChannelSeparation2dLayer(nn.Module):

    def __init__(self):
        super(NsoltChannelSeparation2dLayer, self).__init__()

    def forward(self,x):
        return x