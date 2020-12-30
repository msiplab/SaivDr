import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import numpy as np
from nsoltBlockDct2dLayer import NsoltBlockDct2dLayer

stride = [ [1, 1], [2, 2], [2, 4], [4, 1], [4, 4] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
    
class NsoltBlockDct2dLayerTestCase(unittest.TestCase):
    """
    NSOLTBLOCKDCT2DLAYERTESTCASE
    
       ベクトル配列をブロック配列を入力:
          (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
    
       コンポーネント別に出力:
          nDecs x nRows x nCols x nLays x nSamples
    
    Requirements: Rython 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2020, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/
    """
    @parameterized.expand(
        list(itertools.product(stride))
    )
    def testConstructor(self,stride):
        # Expected values
        expctdName = 'E0'
        expctdDescription = "Block DCT of size " \
            + str(stride[0]) + "x" + str(stride[1])

        # Instantiation of target class
        layer = NsoltBlockDct2dLayer(
                decimation_factor=stride,
                name=expctdName
            )

        # Actual values
        actualName = layer.name
        actualDescription = layer.description

        # Evaluation
        self.assertTrue(isinstance(layer, nn.Module))        
        self.assertEqual(actualName,expctdName)
        self.assertEqual(actualDescription,expctdDescription)

if __name__ == '__main__':
    unittest.main()