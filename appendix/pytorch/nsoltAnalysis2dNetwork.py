import torch
import torch.nn as nn
from nsoltBlockDct2dLayer import NsoltBlockDct2dLayer 
from nsoltInitialRotation2dLayer import NsoltInitialRotation2dLayer 
from nsoltLayerExceptions import InvalidNumberOfChannels, InvalidPolyPhaseOrder, InvalidNumberOfVanishingMoments

class NsoltAnalysis2dNetwork(nn.Module):
    """
    NSOLTANALYSIS2DNETWORK
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2021, Yasas Dulanjaya, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/ 
    """
    def __init__(self,
        number_of_channels=[],
        decimation_factor=[],
        polyphase_order=[0,0],
        number_of_vanishing_moments=1):
        super(NsoltAnalysis2dNetwork, self).__init__()

        # Check and set parameters
        # # of channels
        if number_of_channels[0] != number_of_channels[1]:
            raise InvalidNumberOfChannels(
            '[%d, %d] : Currently, Type-I NSOLT is only suported, where the symmetric and antisymmetric channel numbers should be the same.'\
            % (number_of_channels[0], number_of_channels[1]))
        self.number_of_channels = number_of_channels

        # Decimaton factor
        self.decimation_factor = decimation_factor

        # Polyphase order
        if any(torch.tensor(polyphase_order)%2):
            raise InvalidPolyPhaseOrder(
            '%d + %d : Currently, even polyphase orders are only supported.'\
            % (polyphase_order[0], polyphase_order[1]))
        self.polyphase_order = polyphase_order

        # # of vanishing moments
        if number_of_vanishing_moments < 0 \
            or number_of_vanishing_moments > 1:
            raise InvalidNumberOfVanishingMoments(
            '%d : The number of vanishing moment must be either of 0 or 1.'\
            % number_of_vanishing_moments)
        self.number_of_vanishing_moments = number_of_vanishing_moments

        # Instantiation of layers
        self.layerE0 = NsoltBlockDct2dLayer(
            decimation_factor=decimation_factor,
            name='E0'
        )
        self.layerV0 = NsoltInitialRotation2dLayer(
            number_of_channels=number_of_channels,
            decimation_factor=decimation_factor,
            name='V0'
        )

    def forward(self,x):
        u = self.layerE0.forward(x)
        y = self.layerV0.forward(u)
        return y
