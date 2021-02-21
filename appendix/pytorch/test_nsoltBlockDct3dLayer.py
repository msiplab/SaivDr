import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import torch_dct as dct
import math
from nsoltBlockDct3dLayer import NsoltBlockDct3dLayer
from nsoltUtility import Direction

stride = [ [1, 1, 1], [2, 2, 2], [1, 2, 4], [4, 2, 1], [2, 4, 1] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
depth = [ 8, 16, 32 ]
    
class NsoltBlockDct3dLayerTestCase(unittest.TestCase):
    """
    NsoltBlockDct3dLayerTESTCASE
    
       ベクトル配列をブロック配列を入力(nComponents=1のみサポート):
          nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) x (Stride[2]xnLays)
    
       コンポーネント別に出力(nComponents=1のみサポート):
          nSamples x nRows x nCols x nLays x nDecs
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2020-2021, Yuya Kodama, Shogo MURAMATSU 
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU
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
            + str(stride[Direction.VERTICAL]) + "x" \
            + str(stride[Direction.HORIZONTAL]) + "x" \
            + str(stride[Direction.DEPTH])\

        # Instantiation of target class
        layer = NsoltBlockDct3dLayer(
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
        list(itertools.product(stride,height,width,depth,datatype))
    )
    def testPredict(self,
            stride, height, width, depth, datatype):
        rtol,atol = 1e-5,1e-8

        # Parameters
        nSamples = 8
        nComponents = 1
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols)) x (Stride[2]xnLays))
        X = torch.rand(nSamples,nComponents,height,width,depth,dtype=datatype,requires_grad=True)

        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        nlays = int(math.ceil(depth/stride[Direction.DEPTH])) #.astype(int)
        ndecs = stride[0]*stride[1]*stride[2] # math.prod(stride)
        # Block DCT (nSamples x nComponents x nrows x ncols x nlays) x decV x decH x decD
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct.dct_3d(X.view(arrayshape),norm='ortho')
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols x nlays) x (decV x decH x decD)
        A = permuteDctCoefs_(Y)
        expctdZ = A.view(nSamples,nrows,ncols,nlays,ndecs)

        # Instantiation of target class
        layer = NsoltBlockDct3dLayer(
                decimation_factor=stride,
                name='E0'
            )
            
        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)         
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,depth,datatype))
    )
    def testForward(self,
            stride, height, width, depth, datatype):
        rtol,atol=1e-5,1e-8
            
        # Parameters
        nSamples = 8
        nComponents = 1
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols)) x (Stride[2]xnLays))
        X = torch.rand(nSamples,nComponents,height,width,depth,dtype=datatype,requires_grad=True)

        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        nlays = int(math.ceil(depth/stride[Direction.DEPTH])) #.astype(int)
        ndecs = stride[0]*stride[1]*stride[2] # math.prod(stride)
        # Block DCT (nSamples x nComponents x nrows x ncols x nlays) x decV x decH x decD
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct.dct_3d(X.view(arrayshape),norm='ortho')
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols x nlays) x (decV x decH x decD)
        A = permuteDctCoefs_(Y)
        expctdZ = A.view(nSamples,nrows,ncols,nlays,ndecs)

        # Instantiation of target class
        layer = NsoltBlockDct3dLayer(
                decimation_factor=stride,
                name='E0'
            )
            
        # Actual values
        actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)         
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertTrue(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,depth,datatype))
    )
    def testBackward(self,
            stride, height, width, depth, datatype):
        rtol,atol = 1e-3,1e-6

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        nlays = int(math.ceil(depth/stride[Direction.DEPTH])) #.astype(int)
        ndecs = stride[0]*stride[1]*stride[2] # math.prod(stride)
        nComponents = 1

        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols)) x (Stride[2]xnLays))
        X = torch.rand(nSamples,nComponents,height,width,depth,dtype=datatype,requires_grad=True)        
        # nSamples x nRows x nCols x nLays x nDecs
        dLdZ = torch.rand(nSamples,nrows,ncols,nlays,ndecs,dtype=datatype)

        # Expected values
        A = permuteIdctCoefs_(dLdZ,stride)
        Y = dct.idct_3d(A,norm='ortho')
        expctddLdX = Y.reshape(nSamples,nComponents,height,width,depth)
        
        # Instantiation of target class
        layer = NsoltBlockDct3dLayer(
                decimation_factor=stride,
                name='E0'
            )
            
        # Actual values
        Z = layer.forward(X)
        Z.backward(dLdZ)
        actualdLdX = X.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)


def permuteDctCoefs_(x):
    ceee = x[:,0::2,0::2,0::2].reshape(x.size(0),-1)
    ceeo = x[:,0::2,0::2,1::2].reshape(x.size(0),-1)
    ceoe = x[:,0::2,1::2,0::2].reshape(x.size(0),-1)
    ceoo = x[:,0::2,1::2,1::2].reshape(x.size(0),-1)
    coee = x[:,1::2,0::2,0::2].reshape(x.size(0),-1)
    coeo = x[:,1::2,0::2,1::2].reshape(x.size(0),-1)
    cooe = x[:,1::2,1::2,0::2].reshape(x.size(0),-1)
    cooo = x[:,1::2,1::2,1::2].reshape(x.size(0),-1)
    return torch.cat((ceee,ceeo,ceoe,ceoo,coee,coeo,cooe,cooo),dim=-1)

def permuteIdctCoefs_(x,block_size):
    coefs = x.view(-1,block_size[Direction.VERTICAL]*block_size[Direction.HORIZONTAL]*block_size[Direction.DEPTH]) # x.view(-1,math.prod(block_size)) 
    decY_ = block_size[Direction.VERTICAL]
    decX_ = block_size[Direction.HORIZONTAL]
    decZ_ = block_size[Direction.DEPTH]
    chDecY = int(math.ceil(decY_/2.)) #.astype(int)
    chDecX = int(math.ceil(decX_/2.)) #.astype(int)
    chDecZ = int(math.ceil(decZ_/2.)) #.astype(int)
    fhDecY = int(math.floor(decY_/2.)) #.astype(int)
    fhDecX = int(math.floor(decX_/2.)) #.astype(int)
    fhDecZ = int(math.floor(decZ_/2.)) #.astype(int)

    nQDecseee = chDecY*chDecX*chDecZ
    nQDecseeo = chDecY*chDecX*fhDecZ
    nQDecseoe = chDecY*fhDecX*chDecZ
    nQDecseoo = chDecY*fhDecX*fhDecZ
    nQDecsoee = fhDecY*chDecX*chDecZ
    nQDecsoeo = fhDecY*chDecX*fhDecZ
    nQDecsooe = fhDecY*fhDecX*chDecZ
    nQDecsooo = fhDecY*fhDecX*fhDecZ

    start_idx = 0
    ceee, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecseee)
    ceeo, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecseeo)
    ceoe, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecseoe)
    ceoo, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecseoo)
    coee, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecsoee)
    coeo, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecsoeo)
    cooe, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecsooe)
    cooo, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecsooo)

    nBlocks = coefs.size(0)

    value = torch.zeros(nBlocks,decY_,decX_,decZ_,dtype=x.dtype)

    value[:,0::2,0::2,0::2] = ceee.view(nBlocks,chDecY,chDecX,chDecZ)
    value[:,0::2,0::2,1::2] = ceeo.view(nBlocks,chDecY,chDecX,fhDecZ)
    value[:,0::2,1::2,0::2] = ceoe.view(nBlocks,chDecY,fhDecX,chDecZ)
    value[:,0::2,1::2,1::2] = ceoo.view(nBlocks,chDecY,fhDecX,fhDecZ)
    value[:,1::2,0::2,0::2] = coee.view(nBlocks,fhDecY,chDecX,chDecZ)
    value[:,1::2,0::2,1::2] = coeo.view(nBlocks,fhDecY,chDecX,fhDecZ)
    value[:,1::2,1::2,0::2] = cooe.view(nBlocks,fhDecY,fhDecX,chDecZ)
    value[:,1::2,1::2,1::2] = cooo.view(nBlocks,fhDecY,fhDecX,fhDecZ)
    
    return value

def coefs_align(coefs,start_idx,end_idx):
    output = coefs[:,start_idx:end_idx]
    return output, end_idx


if __name__ == '__main__':
    unittest.main()