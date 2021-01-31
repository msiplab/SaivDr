import torch
import torch.nn as nn

class NsoltIntermediateRotation2dLayer(nn.Module):

    def __init__(self,
        number_of_channels=[],
        mode='Synthesis',
        name=''):
        super(NsoltIntermediateRotation2dLayer, self).__init__()
        self.name = name
        self.number_of_channels = number_of_channels
        self.mode = mode
        self.description = self.mode \
                + " NSOLT intermediate rotation " \
                + "(ps,pa) = (" \
                + str(self.number_of_channels[0]) + "," \
                + str(self.number_of_channels[1]) + ")"

    def forward(self,x):
        return x