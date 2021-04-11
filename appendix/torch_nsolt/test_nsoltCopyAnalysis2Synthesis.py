import itertools
import unittest
from parameterized import parameterized
import random
import torch
from nsoltSynthesis2dNetwork import NsoltSynthesis2dNetwork
from nsoltAnalysis2dNetwork import NsoltAnalysis2dNetwork
from orthonormalTransform import OrthonormalTransform

nchs = [ [2, 2], [3, 3], [4, 4] ]
stride = [ [1, 1], [1, 2], [2, 1], [2, 2] ]
ppord = [ [0, 0], [0, 2], [2, 0], [2, 2], [4, 4] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
nvm = [ 0, 1 ]
nlevels = [ 1, 2, 3 ]
isdevicetest = True

class NsoltCopyAnalysis2SynthesisTestCase(unittest.TestCase):
    """
    NSOLTCOPYANALYSIS2SYNTHESISTESTCASE Test cases for Nsolt{Analysis,Synthesis}2dNetwork
    
    Requirements: Python 3.7.x/3.8.x, PyTorch 1.7.x/1.8.x
    
    Copyright (c) 2021, Yasas Dulanjaya and Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/
    """

    @parameterized.expand(
        list(itertools.product(height,width,nchs,stride,ppord,nlevels,nvm,datatype))
    )
    def testAdjointOfAnalysis2dNetwork(self,
        height,width,nchs,stride,ppord,nlevels,nvm,datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")                

        # Initialization function of angle parameters
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.zeros_(m.angles)

        # Parameters
        nSamples = 8
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        expctdY = X.detach()

        # Instantiation of target class
        analyzer = NsoltAnalysis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            polyphase_order=ppord,
            number_of_vanishing_moments=nvm,
            number_of_levels=nlevels                        
        ).to(device)        

        # Initialization of angle parameters
        analyzer.apply(init_angles)

        # Adjoint 
        synthesizer = analyzer.T

        # Actual values
        with torch.no_grad():
            actualY = synthesizer.forward(analyzer.forward(X))

        # Evaluation
        self.assertTrue(isinstance(synthesizer, NsoltSynthesis2dNetwork))
        self.assertTrue(torch.allclose(actualY,expctdY,rtol=rtol,atol=atol))
        self.assertFalse(actualY.requires_grad)

    @parameterized.expand(
        list(itertools.product(height,width,nchs,stride,ppord,nlevels,nvm,datatype))
    )
    def testAdjointOfAnalysis2dNetworkRandomInitialization(self,
        height,width,nchs,stride,ppord,nlevels,nvm,datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")          

        # Initialization function of angle parameters
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.normal_(m.angles)

        # Parameters
        nSamples = 8
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        expctdY = X.detach()

        # Instantiation of target class
        analyzer = NsoltAnalysis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            polyphase_order=ppord,
            number_of_vanishing_moments=nvm,
            number_of_levels=nlevels                 
        ).to(device)

        # Initialization of angle parameters
        analyzer.apply(init_angles)

        # Adjoint 
        synthesizer = analyzer.T

        # Actual values
        with torch.no_grad():
            actualY = synthesizer.forward(analyzer.forward(X))

        # Evaluation
        self.assertTrue(isinstance(synthesizer, NsoltSynthesis2dNetwork))
        self.assertTrue(torch.allclose(actualY,expctdY,rtol=rtol,atol=atol))
        self.assertFalse(actualY.requires_grad)  

if __name__ == '__main__':
    unittest.main()