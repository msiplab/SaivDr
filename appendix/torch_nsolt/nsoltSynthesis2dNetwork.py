import torch
import torch.nn as nn
from nsoltBlockIdct2dLayer import NsoltBlockIdct2dLayer 
from nsoltFinalRotation2dLayer import NsoltFinalRotation2dLayer 
from nsoltAtomExtension2dLayer import NsoltAtomExtension2dLayer
from nsoltIntermediateRotation2dLayer import NsoltIntermediateRotation2dLayer
from nsoltChannelConcatenation2dLayer import NsoltChannelConcatenation2dLayer
from nsoltLayerExceptions import InvalidNumberOfChannels, InvalidPolyPhaseOrder, InvalidNumberOfVanishingMoments, InvalidNumberOfLevels
from nsoltUtility import Direction

class NsoltSynthesis2dNetwork(nn.Module):
    """
    NSOLTSYNTHESIS2DNETWORK
    
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
        super(NsoltSynthesis2dNetwork, self).__init__()
        
        # Check and set parameters
        # # of channels
        if number_of_channels[0] != number_of_channels[1]:
            raise InvalidNumberOfChannels(
                    '[%d, %d] : Currently, Type-I NSOLT is only suported, where the symmetric and antisymmetric channel numbers should be the same.'\
                    %(number_of_channels[0],number_of_channels[1]))
        self.number_of_channels = number_of_channels
        
        # Decimaton factor
        self.decimation_factor = decimation_factor
        
        # Polyphase order
        if any(torch.tensor(polyphase_order)%2):
            raise InvalidPolyPhaseOrder(
                    '%d + %d : Currently, even polyphase orders are only supported.'\
                    %(polyphase_order[0],polyphase_order[1]))
        self.polyphase_order = polyphase_order
        
        # # of vanishing moments
        if number_of_vanishing_moments < 0 \
            or number_of_vanishing_moments > 1:
                raise InvalidNumberOfVanishingMoments(
                        '%d : The number of vanishing moment must be either of 0 or 1.'\
                        %(number_of_vanishing_moments))
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
            iLevel = nlevels - iStage
            strLv = 'Lv%0d_'%iLevel
            
            # Channel Concatanation 
            if self.number_of_levels > 0:
                stages[iStage].add_module(strLv+'Cc',NsoltChannelConcatenation2dLayer())
            
            # Vertical concatenation
            for iOrderV in range(polyphase_order[Direction.VERTICAL],1,-2):            
                stages[iStage].add_module(strLv+'Vv~%d'%(iOrderV),NsoltIntermediateRotation2dLayer(
                    number_of_channels=number_of_channels,
                    mode='Synthesis',
                    mus=-1))
                stages[iStage].add_module(strLv+'Qv~%dus'%(iOrderV),NsoltAtomExtension2dLayer(
                    number_of_channels=number_of_channels,
                    direction='Down',
                    target_channels='Sum'))
                stages[iStage].add_module(strLv+'Vv~%d'%(iOrderV-1),NsoltIntermediateRotation2dLayer(
                    number_of_channels=number_of_channels,
                    mode='Synthesis',
                    mus=-1))
                stages[iStage].add_module(strLv+'Qv~%ddd'%(iOrderV-1),NsoltAtomExtension2dLayer(
                    number_of_channels=number_of_channels,
                    direction='Up',
                    target_channels='Difference'))
            
            # Horizontal concatenation
            for iOrderH in range(polyphase_order[Direction.HORIZONTAL],1,-2):
                stages[iStage].add_module(strLv+'Vh~%d'%(iOrderH),NsoltIntermediateRotation2dLayer(
                    number_of_channels=number_of_channels,
                    mode='Synthesis',
                    mus=-1))
                stages[iStage].add_module(strLv+'Qh~%dls'%(iOrderH),NsoltAtomExtension2dLayer(
                    number_of_channels=number_of_channels,
                    direction='Right',
                    target_channels='Sum'))
                stages[iStage].add_module(strLv+'Vh~%d'%(iOrderH-1),NsoltIntermediateRotation2dLayer(
                    number_of_channels=number_of_channels,
                    mode='Synthesis',
                    mus=-1))
                stages[iStage].add_module(strLv+'Qh~%drd'%(iOrderH-1),NsoltAtomExtension2dLayer(
                    number_of_channels=number_of_channels,
                    direction='Left',
                    target_channels='Difference'))
                
            stages[iStage].add_module(strLv+'V0~',NsoltFinalRotation2dLayer(
                number_of_channels=number_of_channels,
                decimation_factor=decimation_factor,
                no_dc_leakage=(self.number_of_vanishing_moments==1)))
            stages[iStage].add_module(strLv+'E0~',NsoltBlockIdct2dLayer(
                decimation_factor=decimation_factor))    
        
        # Stack modules as a list
        self.layers = nn.ModuleList(stages)
            
    def forward(self,x):
        if self.number_of_levels == 0: # Flat structure
            for m in self.layers:
                xdc = m.forward(x)
            return xdc
        else: # tree structure
            stride = self.decimation_factor
            nSamples = x[0].size(0)
            nrows = x[0].size(1)
            ncols = x[0].size(2)
            iLevel = self.number_of_levels
            for m in self.layers:
                if iLevel == self.number_of_levels:
                    xdc = x[0]
                xac = x[self.number_of_levels-iLevel+1]
                y = m[0].forward(xac,xdc)
                y = m[1::].forward(y)
                nrows *= stride[Direction.VERTICAL]
                ncols *= stride[Direction.HORIZONTAL]
                xdc = y.reshape(nSamples,nrows,ncols,1)             
                iLevel -= 1
        return xdc.view(nSamples,1,nrows,ncols)

    @property
    def T(self):
        from nsoltAnalysis2dNetwork import NsoltAnalysis2dNetwork
        import re

        # Create analyzer as the adjoint of SELF
        analyzer = NsoltAnalysis2dNetwork(
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
        syn_state_dict = self.state_dict()
        ana_state_dict = analyzer.state_dict()
        for key in syn_state_dict.keys():
            istage_ana = int(re.sub('^layers\.|\.Lv\d_.+$','',key))            
            istage_syn = (nlevels-1)-istage_ana
            angs = syn_state_dict[key]
            ana_state_dict[key\
                .replace('layers.%d'%istage_ana,'layers.%d'%istage_syn)\
                .replace('~','')\
                .replace('T.angles','.angles') ] = angs
        
        # Load state dictionary
        analyzer.load_state_dict(ana_state_dict)

        # Return adjoint
        return analyzer.to(angs.device)


