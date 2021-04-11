import torch
import torch.nn as nn
from nsoltBlockDct2dLayer import NsoltBlockDct2dLayer 
from nsoltInitialRotation2dLayer import NsoltInitialRotation2dLayer 
from nsoltAtomExtension2dLayer import NsoltAtomExtension2dLayer
from nsoltIntermediateRotation2dLayer import NsoltIntermediateRotation2dLayer
from nsoltChannelSeparation2dLayer import NsoltChannelSeparation2dLayer
from nsoltLayerExceptions import InvalidNumberOfChannels, InvalidPolyPhaseOrder, InvalidNumberOfVanishingMoments, InvalidNumberOfLevels
from nsoltUtility import Direction

class NsoltAnalysis2dNetwork(nn.Module):
    """
    NSOLTANALYSIS2DNETWORK
    
    Requirements: Python 3.7.x, PyTorch 1.7.x/1.8.x
    
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
        number_of_vanishing_moments=1,
        number_of_levels=0):
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

        # # of levels
        if not isinstance(number_of_levels, int):
            raise InvalidNumberOfLevels(
            '%f : The number of levels must be integer.'\
            % number_of_levels)   
        if number_of_levels < 0:
            raise InvalidNumberOfLevels(
            '%d : The number of levels must be greater than or equal to 0.'\
            % number_of_levels)
        self.number_of_levels = number_of_levels

        # Instantiation of layers
        if self.number_of_levels == 0:
            nlevels = 1
        else:
            nlevels = self.number_of_levels
        stages = [ nn.Sequential() for iStage in range(nlevels) ]
        for iStage in range(len(stages)):
            iLevel = iStage+1
            strLv = 'Lv%0d_'%iLevel

            # Initial blocks
            stages[iStage].add_module(strLv+'E0',NsoltBlockDct2dLayer(
                decimation_factor=decimation_factor))
            stages[iStage].add_module(strLv+'V0',NsoltInitialRotation2dLayer(
                number_of_channels=number_of_channels,
                decimation_factor=decimation_factor,
                no_dc_leakage=(self.number_of_vanishing_moments==1)))

            # Horizontal extension
            for iOrderH in range(2,polyphase_order[Direction.HORIZONTAL]+1,2):
                stages[iStage].add_module(strLv+'Qh%drd'%(iOrderH-1),NsoltAtomExtension2dLayer(
                    number_of_channels=number_of_channels,
                    direction='Right',
                    target_channels='Difference'))
                stages[iStage].add_module(strLv+'Vh%d'%(iOrderH-1),NsoltIntermediateRotation2dLayer(
                    number_of_channels=number_of_channels,
                    mode='Analysis',
                    mus=-1))
                stages[iStage].add_module(strLv+'Qh%dls'%iOrderH,NsoltAtomExtension2dLayer(
                    number_of_channels=number_of_channels,
                    direction='Left',
                    target_channels='Sum'))
                stages[iStage].add_module(strLv+'Vh%d'%iOrderH,NsoltIntermediateRotation2dLayer(
                    number_of_channels=number_of_channels,
                    mode='Analysis',
                    mus=-1))
                
            # Vertical extension
            for iOrderV in range(2,polyphase_order[Direction.VERTICAL]+1,2):            
                stages[iStage].add_module(strLv+'Qv%ddd'%(iOrderV-1),NsoltAtomExtension2dLayer(
                    number_of_channels=number_of_channels,
                    direction='Down',
                    target_channels='Difference'))                
                stages[iStage].add_module(strLv+'Vv%d'%(iOrderV-1),NsoltIntermediateRotation2dLayer(
                    number_of_channels=number_of_channels,
                    mode='Analysis',
                    mus=-1))
                stages[iStage].add_module(strLv+'Qv%dus'%iOrderV,NsoltAtomExtension2dLayer(
                    number_of_channels=number_of_channels,
                    direction='Up',
                    target_channels='Sum'))                
                stages[iStage].add_module(strLv+'Vv%d'%iOrderV,NsoltIntermediateRotation2dLayer(
                    number_of_channels=number_of_channels,
                    mode='Analysis',
                    mus=-1))
            # Channel Separation for intermediate stages
            if self.number_of_levels > 0:
                stages[iStage].add_module(strLv+'Sp',NsoltChannelSeparation2dLayer())

        # Stack modules as a list
        self.layers = nn.ModuleList(stages)
        
    def forward(self,x):
        if self.number_of_levels == 0: # Flat structure
            for m in self.layers:
                y = m.forward(x) 
            return y
        else: # Tree structure
            stride = self.decimation_factor
            nSamples = x.size(0)
            nComponents = x.size(1)
            nrows = int(x.size(2)/stride[Direction.VERTICAL])
            ncols = int(x.size(3)/stride[Direction.HORIZONTAL])
            y = []
            iLevel = 1                   
            for m in self.layers:
                yac, ydc = m.forward(x)
                y.insert(0,yac)
                if iLevel < self.number_of_levels:
                    x = ydc.view(nSamples,nComponents,nrows,ncols)
                    nrows = int(nrows/stride[Direction.VERTICAL])
                    ncols = int(ncols/stride[Direction.HORIZONTAL])     
                    iLevel += 1             
                else:
                    y.insert(0,ydc)
        
            return tuple(y)
    
    @property
    def T(self):
        from nsoltSynthesis2dNetwork import NsoltSynthesis2dNetwork
        import re

        # Create synthesizer as the adjoint of SELF
        synthesizer = NsoltSynthesis2dNetwork(
            number_of_channels=self.number_of_channels,
            decimation_factor=self.decimation_factor,
            polyphase_order=self.polyphase_order,
            number_of_vanishing_moments=self.number_of_vanishing_moments,
            number_of_levels=self.number_of_levels            
        )

        if self.number_of_levels == 0:
            nlevels = 1
        else:
            nlevels = self.number_of_levels

        # Copy state dictionary
        ana_state_dict = self.state_dict()
        syn_state_dict = synthesizer.state_dict()
        for key in syn_state_dict.keys():
            istage_ana = int(re.sub('^layers\.|\.Lv\d_.+$','',key))
            istage_syn = (nlevels-1)-istage_ana
            angs = ana_state_dict[key\
                .replace('layers.%d'%istage_ana,'layers.%d'%istage_syn)\
                .replace('~','')\
                .replace('T.angles','.angles') ] 
            syn_state_dict[key] = angs
        
        # Load state dictionary
        synthesizer.load_state_dict(syn_state_dict)

        # Return adjoint
        return synthesizer.to(angs.device)
