import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
#import torch_dct as dct
import scipy.fftpack as fftpack
import math
from nsoltBlockIdct2dLayer import NsoltBlockIdct2dLayer
from nsoltUtility import Direction

stride = [ [1, 1], [2, 2], [2, 4], [4, 1], [4, 4] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
isdevicetest = True

class NsoltBlockIdct2dLayerTestCase(unittest.TestCase):
    """
    NSOLTBLOCKIDCT2DLAYERTESTCASE  
    
       コンポーネント別に入力:
          nSamples x nRows x nCols x nDecs 
    
       ベクトル配列をブロック配列にして出力:
          nSamples x nComponents x (Stride(1)xnRows) x (Stride(2)xnCols) 
    
    Requirements: Python 3.7.x, PyTorch 1.7.x/1.8.x
    
    Copyright (c) 2020-2021, Shogo MURAMATSU
    
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
        expctdName = 'E0~'
        expctdDescription = "Block IDCT of size " \
            + str(stride[Direction.VERTICAL])+ "x" \
            + str(stride[Direction.HORIZONTAL])
            
        # Instantiation of target class
        layer = NsoltBlockIdct2dLayer(
            decimation_factor=stride,
            name=expctdName)
            
        # Actual values
        actualName = layer.name
        actualDescription = layer.description
            
        # Evaluation
        self.assertTrue(isinstance(layer, nn.Module))
        self.assertEqual(actualName,expctdName)
        self.assertEqual(actualDescription,expctdDescription)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testPredictGrayScale(self,
        stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu") 

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nComponents = 1
        # nSamples x nRows x nCols x nDecs         
        X = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        A = permuteIdctCoefs_(X,stride)
        #Y = dct.idct_2d(A,norm='ortho')
        Y = torch.tensor(fftpack.idct(fftpack.idct(A.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Y = Y.to(device)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)

        # Instantiation of target class
        layer = NsoltBlockIdct2dLayer(
               decimation_factor=stride,
                name='E0~'
            )

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testForwardGrayScale(self,
        stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")            

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nComponents = 1
        # nSamples x nRows x nCols x nDecs         
        X = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        A = permuteIdctCoefs_(X,stride)
        #Y = dct.idct_2d(A,norm='ortho')
        Y = torch.tensor(fftpack.idct(fftpack.idct(A.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Y = Y.to(device)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)

        # Instantiation of target class
        layer = NsoltBlockIdct2dLayer(
               decimation_factor=stride,
                name='E0~'
            )

        # Actual values
        actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertTrue(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testPredictRgbColor(self,
        stride, height, width, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")      

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nComponents = 3 # RGB
        # nSamples x nRows x nCols x nDecs         
        Xr = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        Xg = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        Xb = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        Ar = permuteIdctCoefs_(Xr,stride)
        Ag = permuteIdctCoefs_(Xg,stride)
        Ab = permuteIdctCoefs_(Xb,stride)                
        #Yr = dct.idct_2d(Ar,norm='ortho')
        Yr = torch.tensor(fftpack.idct(fftpack.idct(Ar.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Yr = Yr.to(device)
        #Yg = dct.idct_2d(Ag,norm='ortho')
        Yg = torch.tensor(fftpack.idct(fftpack.idct(Ag.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Yg = Yg.to(device)        
        #Yb = dct.idct_2d(Ab,norm='ortho')
        Yb = torch.tensor(fftpack.idct(fftpack.idct(Ab.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Yb = Yb.to(device)                
        expctdZ = torch.cat((
            Yr.reshape(nSamples,1,height,width),
            Yg.reshape(nSamples,1,height,width),
            Yb.reshape(nSamples,1,height,width)),dim=1)

            
        # Instantiation of target class
        layer = NsoltBlockIdct2dLayer(
                decimation_factor=stride,
                number_of_components=nComponents,
                name='E0~'
            )

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(Xr,Xg,Xb)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testForwardRgbColor(self,
        stride, height, width, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nComponents = 3 # RGB
        # nSamples x nRows x nCols x nDecs         
        Xr = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        Xg = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        Xb = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        Ar = permuteIdctCoefs_(Xr,stride)
        Ag = permuteIdctCoefs_(Xg,stride)
        Ab = permuteIdctCoefs_(Xb,stride)                
        #Yr = dct.idct_2d(Ar,norm='ortho')
        Yr = torch.tensor(fftpack.idct(fftpack.idct(Ar.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Yr = Yr.to(device)
        #Yg = dct.idct_2d(Ag,norm='ortho')
        Yg = torch.tensor(fftpack.idct(fftpack.idct(Ag.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Yg = Yg.to(device)        
        #Yb = dct.idct_2d(Ab,norm='ortho')
        Yb = torch.tensor(fftpack.idct(fftpack.idct(Ab.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Yb = Yb.to(device)                
        expctdZ = torch.cat((
            Yr.reshape(nSamples,1,height,width),
            Yg.reshape(nSamples,1,height,width),
            Yb.reshape(nSamples,1,height,width)),dim=1)
            
        # Instantiation of target class
        layer = NsoltBlockIdct2dLayer(
                decimation_factor=stride,
                number_of_components=nComponents,
                name='E0~'
            )

        # Actual values
        actualZ = layer.forward(Xr,Xg,Xb)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertTrue(actualZ.requires_grad)    

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testBackwardGrayScale(self,
        stride, height, width, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")     

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nComponents = 1
        # Source (nSamples x nRows x nCols x nDecs)
        X = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)  
        # nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols)
        dLdZ = torch.rand(nSamples,nComponents,height,width,dtype=datatype)
        dLdZ = dLdZ.to(device)
    
        # Expected values
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        #Y = dct.dct_2d(dLdZ.view(arrayshape),norm='ortho')
        Y = torch.tensor(fftpack.dct(fftpack.dct(dLdZ.cpu().view(arrayshape).detach().numpy(),axis=2,type=2,norm='ortho'),axis=1,type=2,norm='ortho'),dtype=datatype)
        Y = Y.to(device)
        
        A = permuteDctCoefs_(Y)
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        expctddLdX = A.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        layer = NsoltBlockIdct2dLayer(
                decimation_factor=stride,
                name='E0~'
            )

        # Actual values
        Z = layer.forward(X)
        Z.backward(dLdZ)
        actualdLdX = X.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testBackwardRgbColor(self,
        stride, height, width, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")    

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nComponents = 3 # RGB
        # Source (nSamples x nRows x nCols x nDecs)
        Xr = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)     
        Xg = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        Xb = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)          
        # nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols)
        dLdZ = torch.rand(nSamples,nComponents,height,width,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        #Y = dct.dct_2d(dLdZ.view(arrayshape),norm='ortho')
        Y = torch.tensor(fftpack.dct(fftpack.dct(dLdZ.cpu().view(arrayshape).detach().numpy(),axis=2,type=2,norm='ortho'),axis=1,type=2,norm='ortho'),dtype=datatype)
        Y = Y.to(device)
        
        A = permuteDctCoefs_(Y)
        # Rearrange the DCT Coefs. (nSamples x nRows x nCols x nDecs)
        Z = A.view(nSamples,nComponents,nrows,ncols,nDecs) 
        expctddLdXr,expctddLdXg,expctddLdXb = map(lambda x: torch.squeeze(x,dim=1),torch.chunk(Z,nComponents,dim=1))

        # Instantiation of target class
        layer = NsoltBlockIdct2dLayer(
                decimation_factor=stride,
                number_of_components=nComponents,
                name='E0~'
            )

        # Actual values
        Z = layer.forward(Xr,Xg,Xb)
        Z.backward(dLdZ)
        actualdLdXr = Xr.grad
        actualdLdXg = Xg.grad
        actualdLdXb = Xb.grad

        # Evaluation
        self.assertEqual(actualdLdXr.dtype,datatype)
        self.assertEqual(actualdLdXg.dtype,datatype)
        self.assertEqual(actualdLdXb.dtype,datatype)                
        self.assertTrue(torch.allclose(actualdLdXr,expctddLdXr,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdXg,expctddLdXg,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdXb,expctddLdXb,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

def permuteDctCoefs_(x):
    cee = x[:,0::2,0::2].reshape(x.size(0),-1)
    coo = x[:,1::2,1::2].reshape(x.size(0),-1)
    coe = x[:,1::2,0::2].reshape(x.size(0),-1)
    ceo = x[:,0::2,1::2].reshape(x.size(0),-1)
    return torch.cat((cee,coo,coe,ceo),dim=-1)

def permuteIdctCoefs_(x,block_size):
    coefs = x.view(-1,block_size[Direction.VERTICAL]*block_size[Direction.HORIZONTAL]) # x.view(-1,math.prod(block_size))
    decY_ = block_size[Direction.VERTICAL]
    decX_ = block_size[Direction.HORIZONTAL]
    chDecY = int(math.ceil(decY_/2.))
    chDecX = int(math.ceil(decX_/2.))
    fhDecY = int(math.floor(decY_/2.))
    fhDecX = int(math.floor(decX_/2.))
    nQDecsee = chDecY*chDecX
    nQDecsoo = fhDecY*fhDecX
    nQDecsoe = fhDecY*chDecX
    cee = coefs[:,:nQDecsee]
    coo = coefs[:,nQDecsee:nQDecsee+nQDecsoo]
    coe = coefs[:,nQDecsee+nQDecsoo:nQDecsee+nQDecsoo+nQDecsoe]
    ceo = coefs[:,nQDecsee+nQDecsoo+nQDecsoe:]
    nBlocks = coefs.size(0)
    value = torch.zeros(nBlocks,decY_,decX_,dtype=x.dtype)
    value[:,0::2,0::2] = cee.view(nBlocks,chDecY,chDecX)
    value[:,1::2,1::2] = coo.view(nBlocks,fhDecY,fhDecX)
    value[:,1::2,0::2] = coe.view(nBlocks,fhDecY,chDecX)
    value[:,0::2,1::2] = ceo.view(nBlocks,chDecY,fhDecX)
    return value

if __name__ == '__main__':
    unittest.main()