import itertools
import unittest
from parameterized import parameterized
import math
import torch
import torch.nn as nn

from nsoltFinalRotation2dLayer import NsoltFinalRotation2dLayer
from nsoltUtility import Direction

nchs = [ [2, 2], [3, 3], [4, 4] ]
stride = [ [1, 1], [1, 2], [2, 2] ]
mus = [ -1, 1 ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]

class NsoltFinalRotation2dLayerTestCase(unittest.TestCase):
    """
    NSOLTFINALROTATION2DLAYERTESTCASE 
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nChs
    
       コンポーネント別に出力(nComponents):
          nSamples x nRows x nCols x nDecs
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2020-2021, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/ 
    """

    @parameterized.expand(
        list(itertools.product(nchs,stride))
    )
    def testConstructor(self,
        nchs, stride):
        
        # Expected values
        expctdName = 'V0~'
        expctdDescription = "NSOLT final rotation " \
                + "(ps,pa) = (" \
                + str(nchs[0]) + "," + str(nchs[1]) + "), " \
                + "(mv,mh) = (" \
                + str(stride[Direction.VERTICAL]) + "," + str(stride[Direction.HORIZONTAL]) + ")"

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
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

    @parameterized.expand(
        list(itertools.product(nchs,stride,nrows,ncols,datatype))
    )
    def testPredictGrayscale(self,
        nchs, stride, nrows, ncols, datatype):
        rtol,atol=1e-5,1e-8

        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,sum(nchs),dtype=datatype,requires_grad=True)
        
        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps, pa = nchs
        W0T = torch.eye(ps,dtype=datatype)
        U0T = torch.eye(pa,dtype=datatype)
        Y = X
        Ys = Y[:,:,:,:ps].view(-1,ps).T
        Ya = Y[:,:,:,ps:].view(-1,pa).T
        Zsa = torch.cat(
                ( W0T[:int(math.ceil(nDecs/2.)),:].mm(Ys), 
                  U0T[:int(math.floor(nDecs/2.)),:].mm(Ya) ),dim=0)
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                name='V0~'
            )

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

if __name__ == '__main__':
    unittest.main()
