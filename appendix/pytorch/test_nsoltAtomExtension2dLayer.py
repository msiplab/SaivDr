import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
from nsoltAtomExtension2dLayer import NsoltAtomExtension2dLayer

nchs = [ [3,3], [4,4] ]
datatype = [ 'single', 'double' ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]
dir = [ 'Right', 'Left', 'Up', 'Down' ]

class NsoltAtomExtention2dLayerTestCase(unittest.TestCase):
    """
    NSOLTATOMEXTENSION2DLAYERTESTCASE
    
        コンポーネント別に入力(nComponents=1のみサポート):
            nChsTotal x nRows x nCols x nSamples
    
        コンポーネント別に出力(nComponents=1のみサポート):
            nChsTotal x nRows x nCols x nSamples
    
    Requirements: Python 3.x

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

    """
    def testForward(self):
        target = NsoltAtomExtension2dLayer()
        x = 0.0
        valueExpctd = 0.0
        valueActual = target.forward(x)
        self.assertEqual(valueExpctd, valueActual)
    """
    
if __name__ == '__main__':
    unittest.main()