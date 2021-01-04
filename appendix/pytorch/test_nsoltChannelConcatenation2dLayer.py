import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import numpy as np
from nsoltChannelConcatenation2dLayer import NsoltChannelConcatenation2dLayer

nchs = [ [3, 3], [4, 4] ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]

class NsoltAtomExtention2dLayerTestCase(unittest.TestCase):
    """
    NSOLTCHANNELCONCATENATION2DLAYERTESTCASE
    
       ２コンポーネント入力(nComponents=2のみサポート):
          nSamples x nRows x nCols x (nChsTotal-1) 
          nSamples x nRows x nCols
    
       １コンポーネント出力(nComponents=1のみサポート):
          nSamples x nRows x nCols x nChsTotal
    
     Requirements: Python 3.7.x, PyTorch 1.7.x
    
     Copyright (c) 2020-2021, Shogo MURAMATSU
    
     All rights reserved.
    
     Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/
    """

    def testConstructor(self):
        #target = NsoltChannelConcatenation2dLayer()
        #self.assertTrue(isinstance(target, nn.Module))
        
        # Expected values
        expctdName = 'Cn'
        expctdDescription = "Channel concatenation"
            
        # Instantiation of target class
        layer = NsoltChannelConcatenation2dLayer(
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
        list(itertools.product(nchs,nrows,ncols,datatype))
    )
    def testPredict(self,
        nchs,nrows,ncols,datatype):
        atol=1e-6

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x (nChsTotal-1)
        Xac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,requires_grad=True)
        # nSamples x nRows x nCols 
        Xdc = torch.randn(nSamples,nrows,ncols,dtype=datatype,requires_grad=True)

        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        expctdZ = torch.cat((Xdc.unsqueeze(dim=3),Xac),dim=3)

        # Instantiation of target class
        layer = NsoltChannelConcatenation2dLayer(
                name='Cn'
            )

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(Xac=Xac,Xdc=Xdc)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.isclose(actualZ,expctdZ,rtol=0.,atol=atol).all())
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,datatype))
    )
    def testBackward(self,
        nchs,nrows,ncols,datatype):
        atol=1e-6
    
        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x (nChsTotal-1)
        Xac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,requires_grad=True)
        # nSamples x nRows x nCols 
        Xdc = torch.randn(nSamples,nrows,ncols,dtype=datatype,requires_grad=True)
        # nSamples x nRows x nCols x nChsTotal
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
    
        # Expected values
        # nSamples x nRows x nCols x (nChsTotal-1) 
        expctddLdXac = dLdZ[:,:,:,1:]
        # nSamples x nRows x nCols x 1  
        expctddLdXdc = dLdZ[:,:,:,0]

        # Instantiation of target class
        layer = NsoltChannelConcatenation2dLayer(
                name='Cn'
            )

        # Actual values
        Z = layer.forward(Xac=Xac,Xdc=Xdc)
        Z.backward(dLdZ)
        actualdLdXac = Xac.grad
        actualdLdXdc = Xdc.grad

        # Evaluation
        self.assertEqual(actualdLdXdc.dtype,datatype)
        self.assertEqual(actualdLdXac.dtype,datatype)    
        self.assertTrue(torch.isclose(actualdLdXdc,expctddLdXdc,rtol=0.,atol=atol).all())
        self.assertTrue(torch.isclose(actualdLdXac,expctddLdXac,rtol=0.,atol=atol).all())        
        self.assertTrue(Z.requires_grad)

if __name__ == '__main__':
    unittest.main()
