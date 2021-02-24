import torch
import torch.nn as nn
from nsoltBlockDct2dLayer import NsoltBlockDct2dLayer 
from nsoltInitialRotation2dLayer import NsoltInitialRotation2dLayer 
from nsoltAtomExtension2dLayer import NsoltAtomExtension2dLayer
from nsoltIntermediateRotation2dLayer import NsoltIntermediateRotation2dLayer
from nsoltLayerExceptions import InvalidNumberOfChannels, InvalidPolyPhaseOrder, InvalidNumberOfVanishingMoments
from nsoltUtility import Direction

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
            % (polyphase_order[Direction.VERTICAL], polyphase_order[Direction.HORIZONTAL]))
        self.polyphase_order = polyphase_order

        # # of vanishing moments
        if number_of_vanishing_moments < 0 \
            or number_of_vanishing_moments > 1:
            raise InvalidNumberOfVanishingMoments(
            '%d : The number of vanishing moment must be either of 0 or 1.'\
            % number_of_vanishing_moments)
        self.number_of_vanishing_moments = number_of_vanishing_moments

        # Instantiation of layers
        self.layers = nn.Sequential()
        self.layers.add_module('E0',NsoltBlockDct2dLayer(
            decimation_factor=decimation_factor))
        self.layers.add_module('V0',NsoltInitialRotation2dLayer(
            number_of_channels=number_of_channels,
            decimation_factor=decimation_factor))
        # Horizontal extension
        for iOrderH in range(2,polyphase_order[Direction.HORIZONTAL]+1,2):
            self.layers.add_module('Qh%drd'%(iOrderH-1),NsoltAtomExtension2dLayer(
                number_of_channels=number_of_channels,
                direction='Right',
                target_channels='Difference'))
            self.layers.add_module('Vh%d'%(iOrderH-1),NsoltIntermediateRotation2dLayer(
                number_of_channels=number_of_channels,
                mode='Analysis',
                mus=-1))
            self.layers.add_module('Qh%dls'%iOrderH,NsoltAtomExtension2dLayer(
                number_of_channels=number_of_channels,
                direction='Left',
                target_channels='Sum'))
            self.layers.add_module('Vh%d'%iOrderH,NsoltIntermediateRotation2dLayer(
                number_of_channels=number_of_channels,
                mode='Analysis',
                mus=-1))
        # Vertical extension
        for iOrderV in range(2,polyphase_order[Direction.VERTICAL]+1,2):            
            self.layers.add_module('Qv%ddd'%(iOrderV-1),NsoltAtomExtension2dLayer(
                number_of_channels=number_of_channels,
                direction='Down',
                target_channels='Difference'))                
            self.layers.add_module('Vv%d'%(iOrderV-1),NsoltIntermediateRotation2dLayer(
                number_of_channels=number_of_channels,
                mode='Analysis',
                mus=-1))
            self.layers.add_module('Qv%dus'%iOrderV,NsoltAtomExtension2dLayer(
                number_of_channels=number_of_channels,
                direction='Up',
                target_channels='Sum'))                
            self.layers.add_module('Vv%d'%iOrderV,NsoltIntermediateRotation2dLayer(
                number_of_channels=number_of_channels,
                mode='Analysis',
                mus=-1))
        
    def forward(self,x):
        return self.layers.forward(x)