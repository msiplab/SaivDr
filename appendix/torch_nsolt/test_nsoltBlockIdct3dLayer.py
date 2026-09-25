import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import torch_dct as dct
import math
from nsoltBlockIdct3dLayer import NsoltBlockIdct3dLayer
from nsoltUtility import Direction

# stride = [ [4, 2, 1] ]
# stride = [ [1, 1, 1], [2, 2, 2], [1, 2, 4], [4, 2, 1] ]
stride = [ [1, 1, 1], [2, 2, 2], [1, 2, 4], [4, 2, 1], [2, 4, 1], [4, 4, 2], [2, 4, 4] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
depth = [ 8, 16, 32 ]

class NsoltBlockIdct3dLayerTestCase(unittest.TestCase):
    """
    NSOLTBLOCKIDCT3DLAYERTESTCASE  
    
       コンポーネント別に入力:
          nSamples x nRows x nCols x nLays x nDecs 
    
       ベクトル配列をブロック配列にして出力:
          nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) x (Stride[2]xnLays)
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2020-2021, Yuya Kodama, Shogo MURAMATSU
    
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
            + str(stride[Direction.VERTICAL]) + "x" \
            + str(stride[Direction.HORIZONTAL]) + "x" \
            + str(stride[Direction.DEPTH])
            
        # Instantiation of target class
        layer = NsoltBlockIdct3dLayer(
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
        list(itertools.product(stride,height,width,depth,datatype))
    )
    def testPredictGrayScale(self,
        stride, height, width, depth, datatype):
        rtol,atol = 1e-5,1e-6 # atol as AbsoluteTolerance(1e-6) in the MATLAB test

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nlays = int(math.ceil(depth/stride[Direction.DEPTH]))
        nDecs = stride[0]*stride[1]*stride[2] # math.prod(stride)
        nComponents = 1
        # nSamples x nRows x nCols x nLays x nDecs         
        X = torch.rand(nSamples,nrows,ncols,nlays,nDecs,dtype=datatype,requires_grad=True)

        # Expected values
        A = permuteIdctCoefs_(X,stride)
        Y = dct.idct_3d(A,norm='ortho')
        expctdZ = block_merge_(Y,nSamples,nComponents,height,width,depth)

        # Instantiation of target class
        layer = NsoltBlockIdct3dLayer(
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
        list(itertools.product(stride,height,width,depth,datatype))
    )
    def testForwardGrayScale(self,
        stride, height, width, depth, datatype):
        rtol,atol = 1e-5,1e-6 # atol as AbsoluteTolerance(1e-6) in the MATLAB test

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nlays = int(math.ceil(depth/stride[Direction.DEPTH]))
        nDecs = stride[0]*stride[1]*stride[2] # math.prod(stride)
        nComponents = 1
        # nSamples x nRows x nCols x nLays x nDecs         
        X = torch.rand(nSamples,nrows,ncols,nlays,nDecs,dtype=datatype,requires_grad=True)

        # Expected values
        A = permuteIdctCoefs_(X,stride)
        Y = dct.idct_3d(A,norm='ortho')
        expctdZ = block_merge_(Y,nSamples,nComponents,height,width,depth)

        # Instantiation of target class
        layer = NsoltBlockIdct3dLayer(
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
        list(itertools.product(stride,height,width,depth,datatype))
    )
    def testBackwardGrayScale(self,
        stride, height, width, depth, datatype):
        rtol,atol=1e-3,1e-6

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nlays = int(math.ceil(depth/stride[Direction.DEPTH]))
        nDecs = stride[0]*stride[1]*stride[2] # math.prod(stride)
        nComponents = 1
        # Source (nSamples x nRows x nCols x nLays x nDecs)
        X = torch.rand(nSamples,nrows,ncols,nlays,nDecs,dtype=datatype,requires_grad=True)        
        # nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) x (Stride[2]xnLays)
        dLdZ = torch.rand(nSamples,nComponents,height,width,depth,dtype=datatype)
    
        # Expected values
        Y = dct.dct_3d(block_split_(dLdZ,stride),norm='ortho')
        A = permuteDctCoefs_(Y)
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols x nlays) x (decV x decH x decD)
        expctddLdX = A.view(nSamples,nrows,ncols,nlays,nDecs)

        # Instantiation of target class
        layer = NsoltBlockIdct3dLayer(
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



def permuteDctCoefs_(x):
    """
    The same order as Cvhd in MATLAB nsoltBlockDct3dLayer, i.e.,
    [ eee, eoo, ooe, oeo, eeo, eoe, ooo, oee ] (yxz),
    where the coefficients in each group are ordered as in getMatrixE0_ of
    the MATLAB test case and saivdr.dictionary.nsoltx (depth fastest,
    vertical slowest)
    """
    n = x.size(0)
    vec = lambda c: c.reshape(n,-1)
    ceee = vec(x[:,0::2,0::2,0::2])
    ceoo = vec(x[:,0::2,1::2,1::2])
    cooe = vec(x[:,1::2,1::2,0::2])
    coeo = vec(x[:,1::2,0::2,1::2])
    ceeo = vec(x[:,0::2,0::2,1::2])
    ceoe = vec(x[:,0::2,1::2,0::2])
    cooo = vec(x[:,1::2,1::2,1::2])
    coee = vec(x[:,1::2,0::2,0::2])
    return torch.cat((ceee,ceoo,cooe,coeo,ceeo,ceoe,cooo,coee),dim=-1)

def permuteIdctCoefs_(x,block_size):
    """
    Inverse of permuteDctCoefs_
    """
    decY_ = block_size[Direction.VERTICAL]
    decX_ = block_size[Direction.HORIZONTAL]
    decZ_ = block_size[Direction.DEPTH]
    coefs = x.reshape(-1,decY_*decX_*decZ_)
    nBlocks = coefs.size(0)
    value = torch.zeros(nBlocks,decY_,decX_,decZ_,dtype=x.dtype,device=x.device)
    start_idx = 0
    for py,px,pz in [ (0,0,0), (0,1,1), (1,1,0), (1,0,1), (0,0,1), (0,1,0), (1,1,1), (1,0,0) ]:
        ny = len(range(py,decY_,2))
        nx = len(range(px,decX_,2))
        nz = len(range(pz,decZ_,2))
        c, start_idx = coefs_align(coefs,start_idx,start_idx+ny*nx*nz)
        value[:,py::2,px::2,pz::2] = c.reshape(nBlocks,ny,nx,nz)
    return value

def coefs_align(coefs,start_idx,end_idx):
    output = coefs[:,start_idx:end_idx]
    return output, end_idx

def block_split_(x,block_size):
    """
    Split volumes into blocks as MATLAB vol2col_ in the test case does
      (nSamples x nComponents x (decV x nRows) x (decH x nCols) x (decD x nLays))
       -> (nSamples x nComponents x nRows x nCols x nLays) x decV x decH x decD
    """
    decV = block_size[Direction.VERTICAL]
    decH = block_size[Direction.HORIZONTAL]
    decD = block_size[Direction.DEPTH]
    nSamples, nComponents, height, width, depth = x.size()
    return x.reshape(nSamples,nComponents,height//decV,decV,width//decH,decH,depth//decD,decD)\
        .permute(0,1,2,4,6,3,5,7).reshape(-1,decV,decH,decD)

def block_merge_(y,nSamples,nComponents,height,width,depth):
    """
    Merge blocks into volumes (inverse of block_split_)
    """
    decV = y.size(1)
    decH = y.size(2)
    decD = y.size(3)
    return y.reshape(nSamples,nComponents,height//decV,width//decH,depth//decD,decV,decH,decD)\
        .permute(0,1,2,5,3,6,4,7).reshape(nSamples,nComponents,height,width,depth)

if __name__ == '__main__':
    unittest.main()