import torch
import torch.nn as nn

class NsoltAtomExtension2dLayer(nn.Module):

    def __init__(self,
            name='',
            numberOfChannels=[],
            direction='',
            targetChannels=''):
        super(NsoltAtomExtension2dLayer, self).__init__()
        self.numberOfChannels = numberOfChannels
        self.name = name
        self.direction = direction
        self.targetChannels = targetChannels
        self.description = self.direction \
            + " shift " \
            + self.targetChannels \
            + " Coefs. " \
            + "(ps,pa) = (" \
            + str(self.numberOfChannels[0]) + "," \
            + str(self.numberOfChannels[1]) + ")"
        
        self.type = ''        

    def forward(self,x):
        return x

