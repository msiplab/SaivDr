import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import numpy as np 
from nsoltAtomExtension2dLayer import NsoltAtomExtension2dLayer

nchs = [ [3,3], [4,4] ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]
dir = [ 'Right', 'Left', 'Up', 'Down' ]

class NsoltAtomExtention2dLayerself(unittest.TestCase):
    """
    NSOLTATOMEXTENSION2DLAYERself
    
        コンポーネント別に入力(nComponents=1のみサポート):
            nChsTotal x nRows x nCols x nSamples
    
        コンポーネント別に出力(nComponents=1のみサポート):
            nChsTotal x nRows x nCols x nSamples
    
    Requirements: Python 3.x, PyTorch 1.7.x

    Copyright (c) 2020, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/
    """
    @parameterized.expand(
        list(itertools.product(nchs))
    )
    def testConstructor(self,nchs):

        # Expctd values
        expctdName = 'Qn'
        expctdDirection = 'Right'
        expctdTargetChannels = 'Lower'
        expctdDescription = "Right shift Lower Coefs. " \
            + "(ps,pa) = (" + str(nchs[0]) + "," + str(nchs[1]) + ")"
        
        # Instantiation of target class
        layer = NsoltAtomExtension2dLayer(
            number_of_channels=nchs,
            name=expctdName,
            direction=expctdDirection,
            target_channels=expctdTargetChannels
        )

        # Actual values
        actualName = layer.name 
        actualDirection = layer.direction 
        actualTargetChannels = layer.target_channels 
        actualDescription = layer.description 

        # Evaluation
        self.assertTrue(isinstance(layer, nn.Module))
        self.assertEqual(actualName,expctdName)
        self.assertEqual(actualDirection,expctdDirection)
        self.assertEqual(actualTargetChannels,expctdTargetChannels)
        self.assertEqual(actualDescription,expctdDescription)


    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,dir,datatype))
    )
    def testPredictGrayscaleShiftLowerCoefs(self, \
            nchs, nrows, ncols, dir, datatype):
        atol= 1e-6
            
        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        target = 'Lower'
        # nChsTotal x nRows x nCols x nSamples
        X = torch.randn(nChsTotal,nrows,ncols,nSamples,dtype=datatype)
        
        # Expected values
        if dir=='Right':
            shift = ( 0, 0, 1, 0 )
        elif dir=='Left':
            shift = ( 0, 0, -1, 0 )
        elif dir=='Down':
            shift = ( 0, 1, 0, 0 )
        elif dir=='Up':
            shift = ( 0, -1, 0, 0 )
        else:
            shift = ( 0, 0, 0, 0 )
    
        # nRows x nCols x nChsTotal x nSamples
        ps = nchs[0]
        pa = nchs[1]
        Y = X 
        # Block butterfly
        Ys = Y[:ps,:,:,:]
        Ya = Y[ps:,:,:,:]
        Y =  torch.cat((Ys+Ya, Ys-Ya),dim=0)/np.sqrt(2)
        # Block circular shift
        Y[ps:,:,:,:] = torch.roll(Y[ps:,:,:,:],shifts=shift,dims=(0,1,2,3))
        # Block butterfly
        Ys = Y[:ps,:,:,:]
        Ya = Y[ps:,:,:,:]
        Y =  torch.cat((Ys+Ya ,Ys-Ya),dim=0)/np.sqrt(2)
        # Output
        expctdZ = Y 

        # Instantiation of target class
        layer = NsoltAtomExtension2dLayer( \
            number_of_channels=nchs, \
            name='Qn~', \
            direction=dir, \
            target_channels=target)

        # Actual values
        actualZ = layer.forward(X)
        
        # Evaluation
        self.assertEqual(actualZ.dtype,datatype) 
        self.assertTrue(torch.isclose(actualZ,expctdZ,rtol=0.,atol=atol).all())
    
if __name__ == '__main__':
    unittest.main()