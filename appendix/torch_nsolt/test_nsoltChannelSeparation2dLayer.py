import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
from nsoltChannelSeparation2dLayer import NsoltChannelSeparation2dLayer

nchs = [ [3, 3], [4, 4] ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]

class NsoltChannelSeparation2dLayerTestCase(unittest.TestCase):
    """
    NSOLTCHANNELSEPARATION2DLAYERTESTCASE
    
       １コンポーネント入力(nComponents=1のみサポート):
          nSamples x nRows x nCols x nChsTotal 
    
       ２コンポーネント出力(nComponents=2のみサポート):
          nSamples x nRows x nCols x (nChsTotal-1) 
          nSamples x nRows x nCols 
    
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
        expctdName = 'Sp'
        expctdDescription = "Channel separation"
            
        # Instantiation of target class
        layer = NsoltChannelSeparation2dLayer(
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
        
        # nSamples x nRows x nCols x nChsTotal 
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        # nSamples x nRows x nCols x (nChsTotal-1)  
        expctdZac = X[:,:,:,1:]
        # nSamples x nRows x nCols  
        expctdZdc = X[:,:,:,0]

        # Instantiation of target class
        layer = NsoltChannelSeparation2dLayer(
                name='Sp'
            )

        # Actual values
        with torch.no_grad():
            actualZac, actualZdc = layer.forward(X)
                        
        # Evaluation
        self.assertEqual(actualZac.dtype,datatype)
        self.assertEqual(actualZdc.dtype,datatype)
        self.assertTrue(torch.allclose(actualZac,expctdZac,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualZdc,expctdZdc,rtol=rtol,atol=atol))
        #self.assertFalse(actualZac.requires_grad)
        #self.assertFalse(actualZdc.requires_grad)

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
        # nSamplesx nRows x nCols x nChsTotal 
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols x (nChsTotal-1)
        dLdZac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype)
        dLdZac = dLdZac.to(device)
        # nSamples x nRows x nCols
        dLdZdc = torch.randn(nSamples,nrows,ncols,dtype=datatype)
        dLdZdc = dLdZdc.to(device)
            
        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        expctddLdX = torch.cat((dLdZdc.unsqueeze(dim=3),dLdZac),dim=3)

        # Instantiation of target class
        layer = NsoltChannelSeparation2dLayer(
                name='Sp'
            )

        # Actual values
        Zac,Zdc = layer.forward(X)
        Zac.backward(dLdZac,retain_graph=True)
        Zdc.backward(dLdZdc,retain_graph=False)
        actualdLdX = X.grad        

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(Zac.requires_grad)
        self.assertTrue(Zdc.requires_grad)

if __name__ == '__main__':
    unittest.main()
