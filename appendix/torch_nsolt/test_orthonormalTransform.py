import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import math
from random import *
from orthonormalTransform import OrthonormalTransform
from nsoltLayerExceptions import InvalidMode, InvalidMus
from nsoltUtility import OrthonormalMatrixGenerationSystem

datatype = [ torch.float, torch.double ]
ncols = [ 1, 2, 4 ]
npoints = [ 1, 2, 3, 4, 5, 6 ]
mode = [ 'Analysis', 'Synthesis' ]
isdevicetest = True

class OrthonormalTransformTestCase(unittest.TestCase):
    """
    ORTHONORMALTRANSFORMTESTCASE
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2021, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/    
    """

    @parameterized.expand(
        list(itertools.product(datatype,ncols))
    )
    def testConstructor(self,datatype,ncols):
        rtol,atol = 1e-5,1e-8 
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")        

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)
        X = X.to(device)

        expctdZ = X
        expctdNParams = 1
        expctdMode = 'Analysis'

        # Instantiation of target class
        target = OrthonormalTransform(device=device)

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)
        actualNParams = len(target.parameters().__next__())
        actualMode = target.mode
        
        # Evaluation
        self.assertTrue(isinstance(target,nn.Module))        
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))    
        self.assertEqual(actualNParams,expctdNParams)
        self.assertEqual(actualMode,expctdMode)

    @parameterized.expand(
        list(itertools.product(datatype,ncols))
    )
    def testConstructorToDevice(self,datatype,ncols):
        rtol,atol = 1e-5,1e-8 
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")        

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)
        X = X.to(device)

        expctdZ = X
        expctdNParams = 1
        expctdMode = 'Analysis'

        # Instantiation of target class
        target = OrthonormalTransform()
        target = target.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)
        actualNParams = len(target.parameters().__next__())
        actualMode = target.mode
        
        # Evaluation
        self.assertTrue(isinstance(target,nn.Module))        
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))    
        self.assertEqual(actualNParams,expctdNParams)
        self.assertEqual(actualMode,expctdMode)

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def testCallWithAngles(self,datatype,ncols,mode):
        rtol,atol = 1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")        

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)   
        X = X.to(device)   
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Instantiation of target class
        target = OrthonormalTransform(mode=mode,device=device)
        #target.angles.data = torch.tensor([math.pi/4])
        target.angles = nn.init.constant_(target.angles,val=math.pi/4)

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))        

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def testCallWithAnglesToDevice(self,datatype,ncols,mode):
        rtol,atol = 1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")        

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)   
        X = X.to(device)   
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Instantiation of target class
        target = OrthonormalTransform(mode=mode)
        target = target.to(device)
        #target.angles.data = torch.tensor([math.pi/4])
        target.angles = nn.init.constant_(target.angles,val=math.pi/4)

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))        

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def testCallWithAnglesAndMus(self,datatype,ncols,mode):
        rtol,atol = 1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")        

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)   
        X = X.to(device)
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ -math.sin(math.pi/4), -math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Instantiation of target class
        target = OrthonormalTransform(mode=mode,device=device)
        target.angles = nn.init.constant_(target.angles,val=math.pi/4)
        target.mus = torch.tensor([1, -1])        

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))


    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def testCallWithAnglesAndMusToDevice(self,datatype,ncols,mode):
        rtol,atol = 1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")        

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)   
        X = X.to(device)
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ -math.sin(math.pi/4), -math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Instantiation of target class
        target = OrthonormalTransform(mode=mode)
        target = target.to(device)
        target.angles = nn.init.constant_(target.angles,val=math.pi/4)
        target.mus = torch.tensor([1, -1])        

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def testSetAngles(self,datatype,ncols,mode):
        rtol,atol = 1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")        

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)  
        X = X.to(device)
        R = torch.eye(2,dtype=datatype)
        R = R.to(device)
        expctdZ = R @ X

        # Instantiation of target class
        target = OrthonormalTransform(mode=mode,device=device)

        # Actual values
        with torch.no_grad():        
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

        # Expcted values
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4), math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Actual values
        target.angles.data = torch.tensor([math.pi/4]).to(device)
        actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def testSetAnglesToDevice(self,datatype,ncols,mode):
        rtol,atol = 1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")        

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)  
        X = X.to(device)
        R = torch.eye(2,dtype=datatype)
        R = R.to(device)
        expctdZ = R @ X

        # Instantiation of target class
        target = OrthonormalTransform(mode=mode)
        target = target.to(device)

        # Actual values
        with torch.no_grad():        
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

        # Expcted values
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4), math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Actual values
        target.angles.data = torch.tensor([math.pi/4])
        target = target.to(device)
        actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def test4x4(self,datatype,ncols,mode):
        rtol,atol = 1e-5,1e-8 

        # Expected values
        expctdNorm = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = OrthonormalTransform(n=4,mode=mode)
        #target.angles.data = torch.randn(6,dtype=datatype)
        target.angles = nn.init.normal_(target.angles)

        # Actual values
        unitvec = torch.randn(4,ncols,dtype=datatype)
        unitvec /= unitvec.norm()
        with torch.no_grad():        
            actualNorm = target.forward(unitvec).norm() #.numpy()

        # Evaluation
        message = "actualNorm=%s differs from %s" % ( str(actualNorm), str(expctdNorm) )
        #self.assertTrue(np.isclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)        
        self.assertTrue(torch.isclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def test8x8(self,datatype,ncols,mode):
        rtol,atol = 1e-5,1e-8 
        
        # Expected values
        expctdNorm = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = OrthonormalTransform(n=8,mode=mode)
        target.angles.data = torch.randn(28,dtype=datatype)

        # Actual values
        unitvec = torch.randn(8,ncols,dtype=datatype)
        unitvec /= unitvec.norm()
        with torch.no_grad():        
            actualNorm = target.forward(unitvec).norm() #.numpy()

        # Evaluation
        message = "actualNorm=%s differs from %s" % ( str(actualNorm), str(expctdNorm) )
        #self.assertTrue(np.isclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)        
        self.assertTrue(torch.isclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)                


    @parameterized.expand(
        list(itertools.product(datatype,ncols,npoints,mode))
    )
    def testNxN(self,datatype,ncols,npoints,mode):
        rtol,atol = 1e-5,1e-8 
        
        # Expected values
        expctdNorm = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        nAngles = int(npoints*(npoints-1)/2)
        target = OrthonormalTransform(n=npoints,mode=mode)
        target.angles.data = torch.randn(nAngles,dtype=datatype)

        # Actual values
        unitvec = torch.randn(npoints,ncols,dtype=datatype)
        unitvec /= unitvec.norm()
        with torch.no_grad():        
            actualNorm = target.forward(unitvec).norm() #.numpy()

        # Evaluation
        message = "actualNorm=%s differs from %s" % ( str(actualNorm), str(expctdNorm) )
        #self.assertTrue(np.isclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)        
        self.assertTrue(torch.isclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)                

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def test4x4red(self,datatype,ncols,mode):
        rtol,atol = 1e-5,1e-8 
        
        # Configuration
        nPoints = 4
        nAngles = int(nPoints*(nPoints-1)/2)

        # Expected values
        expctdLeftTop = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,mode=mode)
        #target.angles.data = 2*math.pi*torch.rand(nAngles,dtype=datatype)
        target.angles = nn.init.uniform_(target.angles,a=0.0,b=2*math.pi)
        target.angles.data[:nPoints-1] = torch.zeros(nPoints-1)

        # Actual values
        with torch.no_grad():       
            matrix = target.forward(torch.eye(nPoints,dtype=datatype))
        actualLeftTop = matrix[0,0] #.numpy()
        
        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )        
        #self.assertTrue(np.isclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)        
        self.assertTrue(torch.isclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)                

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def test8x8red(self,datatype,ncols,mode):
        rtol,atol = 1e-5,1e-8 
        
        # Configuration
        nPoints = 8
        nAngles = int(nPoints*(nPoints-1)/2)

        # Expected values
        expctdLeftTop = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,mode=mode)
        #target.angles.data = 2*math.pi*torch.rand(nAngles,dtype=datatype)
        target.angles = nn.init.uniform_(target.angles,a=0.0,b=2*math.pi)
        target.angles.data[:nPoints-1] = torch.zeros(nPoints-1)

        # Actual values
        with torch.no_grad():       
            matrix = target.forward(torch.eye(nPoints,dtype=datatype))
        actualLeftTop = matrix[0,0] #.numpy()
        
        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )        
        #self.assertTrue(np.isclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)
        self.assertTrue(torch.isclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype,ncols,npoints,mode))
    )
    def testNxNred(self,datatype,ncols,npoints,mode):
        rtol,atol = 1e-5,1e-8 
        
        # Configuration
        nAngles = int(npoints*(npoints-1)/2)

        # Expected values
        expctdLeftTop = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = OrthonormalTransform(n=npoints,mode=mode)
        target.angles = nn.init.uniform_(target.angles,a=0.0,b=2*math.pi)
        target.angles.data[:npoints-1] = torch.zeros(npoints-1)

        # Actual values
        with torch.no_grad():       
            matrix = target.forward(torch.eye(npoints,dtype=datatype))
        actualLeftTop = matrix[0,0] #.numpy()
        
        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )        
        #self.assertTrue(np.isclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)        
        self.assertTrue(torch.isclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)                
    

    @parameterized.expand(
        list(itertools.product(datatype,mode))
    )
    def testBackward(self,datatype,mode):
        rtol,atol = 1e-5,1e-8 
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")        

        # Configuration
        ncols = 1
        nPoints = 2

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)     
        #X = X.to(device)   
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)
        dLdZ = dLdZ.to(device)
        R = torch.eye(nPoints,dtype=datatype)
        R = R.to(device)
        dRdW = torch.tensor([
            [ 0., -1.],
            [ 1., 0.] ],
            dtype=datatype)
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = dLdZ.T @ dRdW @ X 
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = dLdZ.T @ dRdW.T @ X             

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,device=device,mode=mode)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad
    
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))
        
    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def testBackwardMultiColumns(self,datatype,ncols,mode):
        rtol,atol = 1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        nPoints = 2

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)     
        #X = X.to(device)   
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)
        dLdZ = dLdZ.to(device)
        R = torch.eye(nPoints,dtype=datatype)
        R = R.to(device)
        dRdW = torch.tensor([
            [ 0., -1.],
            [ 1., 0.] ],
            dtype=datatype)
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,device=device,mode=mode)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad
    
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def testBackwardMultiColumnsAngs(self,datatype,ncols,mode):
        rtol,atol = 1e-3,1e-6 
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        nPoints = 2

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)    
        #X = X.to(device)    
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)
        dLdZ = dLdZ.to(device)
        #angle = 2.*math.pi*randn(1)
        angle = 2.*math.pi*gauss(mu=0.,sigma=1.) #randn(1)
        R = torch.tensor([[ math.cos(angle), -math.sin(angle) ],
            [ math.sin(angle), math.cos(angle)]], 
            dtype=datatype) #.squeeze()
        R = R.to(device)
        dRdW = torch.tensor([[ -math.sin(angle), -math.cos(angle) ],
            [ math.cos(angle), -math.sin(angle)]], 
            dtype=datatype) #.squeeze()
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))            

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,device=device,mode=mode)
        #target.angles.data = torch.tensor(angle,dtype=datatype)
        target.angles = nn.init.constant_(target.angles,val=angle)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad
    
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,mode,ncols))
    )
    def testBackwardAngsAndMus(self,datatype,mode,ncols):
        rtol,atol = 1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Analysis'
        nPoints = 2
        #ncols = 1
        mus = [1,-1]

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)   
        #X = X.to(device)     
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)
        dLdZ = dLdZ.to(device)
        # angle = 2.*math.pi*randn(1)
        angle = 2.*math.pi*gauss(mu=0.,sigma=1.) #randn(1)        
        R = torch.tensor([[ math.cos(angle), -math.sin(angle) ],
            [ -math.sin(angle), -math.cos(angle)]], 
            dtype=datatype) #.squeeze()
        R = R.to(device)
        dRdW = torch.tensor([[ -math.sin(angle), -math.cos(angle) ],
            [ -math.cos(angle), math.sin(angle)]], 
            dtype=datatype) #.squeeze()
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))            

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,device=device,mode=mode)
        target.angles = nn.init.constant_(target.angles,val=angle)
        target.mus = torch.tensor(mus,dtype=datatype)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad 
  
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    def testInstantiationWithInvalidMode(self):
        mode = 'Invalid'

        # Instantiation of target class
        with self.assertRaises(InvalidMode):
            target = OrthonormalTransform(mode=mode)
        
    def testSetInvalidMode(self):
        mode = 'Invalid'        
        with self.assertRaises(InvalidMode):
            target = OrthonormalTransform()
            target.mode = 'InvalidMode'

    def testInstantiationWithInvalidMus(self):
        mus = 2
        with self.assertRaises(InvalidMus):
            target = OrthonormalTransform(mus=mus)
        
    def testSetInvalidMus(self):        
        mus = 2        
        with self.assertRaises(InvalidMus):
            target = OrthonormalTransform()
            target.mus = mus

    @parameterized.expand(
        list(itertools.product(datatype,mode,ncols))
    )
    def testBackwardSetAngles(self,datatype,mode,ncols):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode='Synthesis'
        nPoints=2
        #ncols=1

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)  
        #X = X.to(device) 
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)     
        dLdZ = dLdZ.to(device)   
        R = torch.eye(nPoints,dtype=datatype)
        R = R.to(device)
        dRdW = torch.tensor([
            [ 0., -1.],
            [ 1., 0.] ],
            dtype=datatype)
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))            

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,device=device,mode=mode)
        target.angles = nn.init.zeros_(target.angles)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad 

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

        # Expected values
        X = X.detach()
        X.requires_grad = True
        angle = 2.*math.pi*gauss(mu=0.,sigma=1.)
        R = torch.tensor([[ math.cos(angle), -math.sin(angle) ],
            [ math.sin(angle), math.cos(angle)]], 
            dtype=datatype) #.squeeze()
        R = R.to(device)
        dRdW = torch.tensor([[ -math.sin(angle), -math.cos(angle) ],
            [ math.cos(angle), -math.sin(angle)]], 
            dtype=datatype) #.squeeze()
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))            

        # Set angles
        target.angles = nn.init.constant_(target.angles,val=angle)
                 
        # Actual values
        torch.autograd.set_detect_anomaly(True)   
        Z = target.forward(X)
        target.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad 

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,mode,ncols))
    )
    def testForward4x4RandAngs(self,datatype,mode,ncols):
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 4
        #ncols = 2
        angs = 2.*math.pi*torch.randn(6,dtype=datatype).to(device)        
        mus = [ -1, 1, -1, 1 ]

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype)
        X = X.to(device)
        R = torch.as_tensor(
            torch.tensor(mus).view(-1,1) * \
            torch.tensor(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0, 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            ) @ torch.tensor(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            ),dtype=datatype)
        R = R.to(device)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles.data = angs
        target.mus = mus

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,mode,ncols))
    )
    def testForward4x4RandAngsToDevice(self,datatype,mode,ncols):
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 4
        #ncols = 2
        mus = [ -1, 1, -1, 1 ]
        angs = 2.*math.pi*torch.randn(6)

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype)
        X = X.to(device)
        R = torch.as_tensor(
            torch.tensor(mus).view(-1,1) * \
            torch.tensor(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0, 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            ) @ torch.tensor(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            ),dtype=datatype)
        R = R.to(device)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,mode=mode)
        target.angles.data = angs
        target.mus = mus
        target = target.to(device,dtype=datatype)        

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,mode,ncols))
    )
    def testBackward4x4RandAngPdAng2(self,datatype,mode,ncols):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 4
        #ncols = 2
        mus = [ -1, 1, -1, 1 ]
        angs = 2.*math.pi*torch.randn(6,dtype=datatype,device=device)
        pdAng = 2

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)    
        dLdZ = dLdZ.to(device)    
        R = torch.as_tensor(
            torch.tensor(mus).view(-1,1) * \
            torch.tensor(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0, 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            ) @ torch.tensor(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            ),dtype=datatype)
        R = R.to(device)
        dRdW = torch.as_tensor(
            torch.tensor(mus).view(-1,1) * \
            torch.tensor(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0, 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            ) @ torch.tensor( 
                [ [1, 0, 0, 0 ], 
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor( # Partial Diff. pdAng = 2
                [ [ math.cos(angs[2]+math.pi/2.), 0, 0, -math.sin(angs[2]+math.pi/2.) ],
                 [0, 0, 0, 0 ],
                 [0, 0, 0, 0 ],
                 [ math.sin(angs[2]+math.pi/2.), 0, 0, math.cos(angs[2]+math.pi/2.) ] ]
            ) @ torch.tensor(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            ),dtype=datatype)
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X)) 
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))    

         # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles.data = angs
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad[pdAng]
        
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))
    
    @parameterized.expand(
        list(itertools.product(datatype,mode,ncols))
    )
    def testBackward4x4RandAngPdAng5(self,datatype,mode,ncols):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 4
        #ncols = 2
        mus = [ 1, 1, -1, -1 ]
        angs = 2.*math.pi*torch.randn(6,dtype=datatype,device=device)
        pdAng = 5

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype) 
        dLdZ = dLdZ.to(device)       
        R = torch.as_tensor(
            torch.tensor(mus).view(-1,1) * \
            torch.tensor(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0, 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            ) @ torch.tensor(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            ),dtype=datatype)
        R = R.to(device)
        dRdW = torch.as_tensor(
            torch.tensor(mus).view(-1,1) * \
            torch.tensor( # Partial Diff. pdAng = 5
                [ [0, 0, 0, 0. ],
                 [0, 0, 0., 0. ],
                 [0, 0, math.cos(angs[5]+math.pi/2), -math.sin(angs[5]+math.pi/2) ],
                 [0., 0, math.sin(angs[5]+math.pi/2), math.cos(angs[5]+math.pi/2) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            ) @ torch.tensor( 
                [ [1, 0, 0, 0 ], 
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor( 
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            ) @ torch.tensor(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            ),dtype=datatype)
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X)) 
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))    

         # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles.data = angs
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad[pdAng]
        
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))
    
    @parameterized.expand(
        list(itertools.product(mode,ncols))
    )
    def testBackward4x4RandAngPdAng1(self,mode,ncols):
        datatype=torch.double
        rtol,atol = 1e-2,1e-5
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 4
        #ncols = 2
        mus = [ -1, -1, -1, -1 ]
        angs = 2.*math.pi*torch.randn(6,dtype=datatype,device=device)
        pdAng = 1
        delta = 1e-4

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)        
        dLdZ = dLdZ.to(device)
        R = torch.as_tensor(
            torch.tensor(mus).view(-1,1) * \
            torch.tensor(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0, 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            ) @ torch.tensor(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            ),dtype=datatype)
        R = R.to(device)
        dRdW = torch.as_tensor(
            (1./delta) * torch.tensor(mus).view(-1,1) * \
            torch.tensor( 
                [ [1, 0, 0, 0. ],
                 [0, 1, 0., 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0., 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            ) @ torch.tensor(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            ) @ torch.tensor( 
                [ [1, 0, 0, 0 ], 
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            ) @ torch.tensor( 
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            ) @ ( 
                torch.tensor( 
               [ [math.cos(angs[1]+delta/2.), 0, -math.sin(angs[1]+delta/2.), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]+delta/2.), 0, math.cos(angs[1]+delta/2.), 0 ],
                 [0, 0, 0, 1 ] ] ) - \
                torch.tensor( 
               [ [math.cos(angs[1]-delta/2.), 0, -math.sin(angs[1]-delta/2.), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]-delta/2.), 0, math.cos(angs[1]-delta/2.), 0 ],
                 [0, 0, 0, 1 ] ] )
            ) @ torch.tensor(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            ),dtype=datatype)
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X)) 
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))    

         # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles.data = angs
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad[pdAng]
        
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))
    
    @parameterized.expand(
        list(itertools.product(mode,ncols))
    )
    def testBackward8x8RandAngPdAng4(self,mode,ncols):
        datatype=torch.double
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 8
        #ncols = 2
        angs0 = 2*math.pi*torch.rand(28,dtype=datatype,device=device)
        angs1 = angs0.clone()
        angs2 = angs0.clone()
        pdAng = 4
        delta = 1e-4
        angs1[pdAng] = angs0[pdAng]-delta/2.
        angs2[pdAng] = angs0[pdAng]+delta/2.

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)   
        dLdZ = dLdZ.to(device)     
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=False
            )
        R = omgs(angles=angs0,mus=1)
        R = R.to(device)
        dRdW = ( omgs(angles=angs2,mus=1) - omgs(angles=angs1,mus=1) )/delta
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X)) 
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))    

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,device=device,mode=mode)
        target.angles.data = angs0

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad[pdAng]
        
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(mode,ncols))
    )
    def testBackward8x8RandAngMusPdAng13(self,mode,ncols):
        datatype = torch.double
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 8
        #ncols = 2
        mus = [ 1,1,1,1,-1,-1,-1,-1 ]
        angs0 = 2*math.pi*torch.rand(28,dtype=datatype,device=device)
        angs1 = angs0.clone()
        angs2 = angs0.clone()
        pdAng = 13
        delta = 1e-4
        angs1[pdAng] = angs0[pdAng]-delta/2.
        angs2[pdAng] = angs0[pdAng]+delta/2.

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)        
        dLdZ = dLdZ.to(device)
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=False
            )
        R = omgs(angles=angs0,mus=mus)
        R = R.to(device)
        dRdW = ( omgs(angles=angs2,mus=mus) - omgs(angles=angs1,mus=mus) )/delta
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X)) 
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))    

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,device=device,mode=mode)
        target.angles.data = angs0
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad[pdAng]
        
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(mode,ncols))
    )
    def testBackword8x8RandAngMusPdAng7(self,mode,ncols):
        datatype = torch.double
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 8
        #ncols = 2
        mus = [ 1,-1,1,-1,1,-1,1,-1 ]
        angs0 = 2*math.pi*torch.rand(28,dtype=datatype).to(device)
        angs1 = angs0.clone()
        angs2 = angs0.clone()
        pdAng = 7
        delta = 1e-4
        angs1[pdAng] = angs0[pdAng]-delta/2.
        angs2[pdAng] = angs0[pdAng]+delta/2.

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=False)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)  
        dLdZ = dLdZ.to(device) 

        # Instantiation of target class
        target0 = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target0.angles.data = angs0
        target0.mus = mus
        target1 = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target1.angles.data = angs1
        target1.mus = mus
        target2 = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target2.angles.data = angs2
        target2.mus = mus    

        # Expctd values
        if mode=='Analysis':
            bwmode='Synthesis'
        else:
            bwmode='Analysis'
        backprop = OrthonormalTransform(n=nPoints,dtype=datatype,mode=bwmode)
        backprop.angles.data = angs0
        backprop.mus = mus
        torch.autograd.set_detect_anomaly(True)                
        dZdW = (target2.forward(X) - target1.forward(X))/delta # ~ d(R*X)/dW
        expctddLdW = torch.sum(dLdZ * dZdW) # ~ dLdW

        # Actual values
        X.detach()
        X.requires_grad = True
        Z = target0.forward(X)
        target0.zero_grad()
        #print(torch.autograd.gradcheck(target0,(X,angs0)))
        Z.backward(dLdZ)
        actualdLdW = target0.angles.grad[pdAng]

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(mode,ncols,npoints))
    )
    def testGradCheckNxNRandAngMus(self,mode,ncols,npoints):
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
            
        # Configuration
        datatype = torch.double
        nPoints = npoints
        nAngs = int(nPoints*(nPoints-1)/2.)
        angs = 2.*math.pi*torch.randn(nAngs,dtype=datatype).to(device)
        mus = (-1)**torch.randint(high=2,size=(nPoints,))

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)   
        dLdZ = dLdZ.to(device)

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles.data = angs
        target.mus = mus
        torch.autograd.set_detect_anomaly(True)                
        Z = target.forward(X)
        target.zero_grad()

        # Evaluation        
        self.assertTrue(torch.autograd.gradcheck(target,(X,)))

    @parameterized.expand(
        list(itertools.product(mode,ncols,npoints))
    )
    def testGradCheckNxNRandAngMusToDevice(self,mode,ncols,npoints):
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
            
        # Configuration
        datatype = torch.double
        nPoints = npoints
        nAngs = int(nPoints*(nPoints-1)/2.)
        mus = (-1)**torch.randint(high=2,size=(nPoints,))
        angs = 2.*math.pi*torch.randn(nAngs,dtype=datatype)

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)   
        dLdZ = dLdZ.to(device)

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles.data = angs
        target.mus = mus
        target = target.to(device)
        torch.autograd.set_detect_anomaly(True)                
        Z = target.forward(X)
        target.zero_grad()

        # Evaluation        
        self.assertTrue(torch.autograd.gradcheck(target,(X,)))

if __name__ == '__main__':
    unittest.main()