import torch
import torch.nn as nn

class NsoltAtomExtension2dLayer(nn.Module):

    def __init__(self,
            name='',
            number_of_channels=[],
            direction='',
            target_channels=''):
        super(NsoltAtomExtension2dLayer, self).__init__()
        self.number_of_channels = number_of_channels
        self.name = name
        self.direction = direction
        self.target_channels = target_channels
        self.description = self.direction \
            + " shift " \
            + self.target_channels \
            + " Coefs. " \
            + "(ps,pa) = (" \
            + str(self.number_of_channels[0]) + "," \
            + str(self.number_of_channels[1]) + ")"
        
        self.type = ''        

    def forward(self,x):
        return x

