import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
from nsoltChannelConcatenation2dLayer import NsoltChannelConcatenation2dLayer

nchs = [ [3, 3], [4, 4] ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]

class NsoltChannelConcatenation2dLayerTestCase(unittest.TestCase):
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
        rtol,atol=1e-5,1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x (nChsTotal-1)
        Xac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols 
        Xdc = torch.randn(nSamples,nrows,ncols,dtype=datatype,device=device,requires_grad=True)

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
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,datatype))
    )
    def testPredictUnsqueezedXdc(self,
        nchs,nrows,ncols,datatype):
        rtol,atol=1e-5,1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x (nChsTotal-1)
        Xac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols x 1
        Xdc = torch.randn(nSamples,nrows,ncols,1,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        expctdZ = torch.cat((Xdc,Xac),dim=3)

        # Instantiation of target class 
        layer = NsoltChannelConcatenation2dLayer(
                name='Cn'
            )

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(Xac=Xac,Xdc=Xdc)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)    

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,datatype))
    )
    def testBackward(self,
        nchs,nrows,ncols,datatype):
        rtol,atol=1e-5,1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                
    
        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x (nChsTotal-1)
        Xac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols 
        Xdc = torch.randn(nSamples,nrows,ncols,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols x nChsTotal
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)
    
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
        self.assertTrue(torch.allclose(actualdLdXdc,expctddLdXdc,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdXac,expctddLdXac,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,datatype))
    )
    def testBackwardUnsqueezedXdc(self,
        nchs,nrows,ncols,datatype):
        rtol,atol=1e-5,1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                
    
        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x (nChsTotal-1)
        Xac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols x 1
        Xdc = torch.randn(nSamples,nrows,ncols,1,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols x nChsTotal
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)
    
        # Expected values
        # nSamples x nRows x nCols x (nChsTotal-1) 
        expctddLdXac = dLdZ[:,:,:,1:]
        # nSamples x nRows x nCols x 1  
        expctddLdXdc = dLdZ[:,:,:,0].unsqueeze(dim=3)

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
        self.assertTrue(torch.allclose(actualdLdXdc,expctddLdXdc,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdXac,expctddLdXac,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)


if __name__ == '__main__':
    unittest.main()
