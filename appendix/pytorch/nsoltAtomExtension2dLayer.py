import torch
import torch.nn as nn
from nsoltLayerExceptions import InvalidDirection, InvalidTargetChannels

class NsoltAtomExtension2dLayer(nn.Module):
    """
    NSOLTATOMEXTENSION2DLAYER
        コンポーネント別に入力(nComponents=1のみサポート):
            nSamples x nChsTotal x nRows x nCols 
    
        コンポーネント別に出力(nComponents=1のみサポート):
            nSamples x nChsTotal x nRows x nCols 
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2020, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
                    Faculty of Engineering, Niigata University,
                    8050 2-no-cho Ikarashi, Nishi-ku,
                    Niigata, 950-2181, JAPAN
    
    http://msiplab.eng.niigata-u.ac.jp/
    """
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
        nchs = self.number_of_channels
        target = self.target_channels        
        dir = self.direction
        #
        if dir=='Right':
            shift = ( 0, 0, 0, 1 )
        elif dir=='Left':
            shift = ( 0, 0, 0, -1 )
        elif dir=='Down':
            shift = ( 0, 0, 1, 0 )
        elif dir=='Up':
            shift = ( 0, 0, -1, 0 )
        else:
            raise InvalidDirection(
                '%s : Direction should be either of Right, Left, Down or Up'\
                % self.direction
            )

        return self.atomext_(x,shift)

    def atomext_(self,X,shift):
        ps, pa = self.number_of_channels #[0], self.number_of_channels[1]
        target = self.target_channels
        #
        Y = X
        # Block butterfly
        Ys = Y[:,:ps,:,:]
        Ya = Y[:,ps:,:,:]
        Y = torch.cat((Ys+Ya,Ys-Ya),dim=1)
        # Block circular shift
        if target=='Lower':
            Y[:,ps:,:,:] = torch.roll(Y[:,ps:,:,:],shifts=shift,dims=(0,1,2,3))
        elif target=='Upper':
            Y[:,:ps,:,:] = torch.roll(Y[:,:ps,:,:],shifts=shift,dims=(0,1,2,3))
        else:
            raise InvalidTargetChannels(
                '%s : TaregetChannels should be either of Lower or Upper'\
                % self.target_channels
            )
                
        # Block butterfly
        Ys = Y[:,:ps,:,:]
        Ya = Y[:,ps:,:,:]
        Y = torch.cat((Ys+Ya, Ys-Ya),dim=1)

        # Output
        return Y/2.0