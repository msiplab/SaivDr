import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import math
from random import *
from orthonormalTransform import OrthonormalTransform
from nsoltLayerExceptions import InvalidMode, InvalidMus

datatype = [ torch.float, torch.double ]
ncols = [ 1, 2, 4 ]
npoints = [ 1, 2, 3, 4, 5, 6 ]
mode = [ 'Analysis', 'Synthesis' ]

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
        rtol,atol = 1e-05,1e-08 

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)
        expctdZ = X
        expctdNParams = 1
        expctdMode = 'Analysis'

        # Instantiation of target class
        target = OrthonormalTransform()

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
        rtol,atol = 1e-05,1e-08 

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)      
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ],
            dtype=datatype)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Instantiation of target class
        target = OrthonormalTransform(mode=mode)
        #target.angles.data = torch.tensor([math.pi/4])
        target.angles = nn.init.constant_(target.angles,math.pi/4)

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))        

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def testCallWithAnglesAndMus(self,datatype,ncols,mode):
        rtol,atol = 1e-05,1e-08 

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)   
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ -math.sin(math.pi/4), -math.cos(math.pi/4) ] ],
            dtype=datatype)
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Instantiation of target class
        target = OrthonormalTransform(mode=mode)
        target.angles = nn.init.constant_(target.angles,math.pi/4)
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
        rtol,atol = 1e-05,1e-08 
    
        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)  
        R = torch.eye(2,dtype=datatype)
        expctdZ = R @ X

        # Instantiation of target class
        target = OrthonormalTransform(mode=mode)

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
        if mode!='Synthesis':
            expctdZ = R @ X
        else:
            expctdZ = R.T @ X

        # Actual values
        target.angles.data = torch.tensor([math.pi/4])
        actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,ncols,mode))
    )
    def test4x4(self,datatype,ncols,mode):
        rtol,atol = 1e-05,1e-08 
        
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
        rtol,atol = 1e-05,1e-08 
        
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
        rtol,atol = 1e-05,1e-08 
        
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
        rtol,atol = 1e-05,1e-08 
        
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
        rtol,atol = 1e-05,1e-08 
        
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
        rtol,atol = 1e-05,1e-08 
        
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
        rtol,atol = 1e-05,1e-08 

        # Configuration
        ncols = 1
        nPoints = 2

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,requires_grad=True)        
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)
        R = torch.eye(nPoints,dtype=datatype)
        dRdW = torch.tensor([
            [ 0., -1.],
            [ 1., 0.] ],
            dtype=datatype)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = expctddLdX.T @ dRdW @ X 
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = expctddLdX.T @ dRdW.T @ X             

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
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
        rtol,atol = 1e-05,1e-08 

        # Configuration
        nPoints = 2

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,requires_grad=True)        
        dLdZ = (1/ncols)*torch.randn(nPoints,ncols,dtype=datatype)
        R = torch.eye(nPoints,dtype=datatype)
        dRdW = torch.tensor([
            [ 0., -1.],
            [ 1., 0.] ],
            dtype=datatype)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW.T @ X))

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
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
        rtol,atol = 1e-04,1e-07 

        # Configuration
        nPoints = 2

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,requires_grad=True)        
        dLdZ = (1/ncols)*torch.randn(nPoints,ncols,dtype=datatype)
        #angle = 2.*math.pi*randn(1)
        angle = 2.*math.pi*gauss(mu=0.,sigma=1.) #randn(1)
        R = torch.tensor([[ math.cos(angle), -math.sin(angle) ],
            [ math.sin(angle), math.cos(angle)]], 
            dtype=datatype) #.squeeze()
        dRdW = torch.tensor([[ -math.sin(angle), -math.cos(angle) ],
            [ math.cos(angle), -math.sin(angle)]], 
            dtype=datatype) #.squeeze()
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW.T @ X))            

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        #target.angles.data = torch.tensor(angle,dtype=datatype)
        target.angles = nn.init.constant_(target.angles,angle)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
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
        rtol,atol = 1e-04,1e-07 

        # Configuration
        #mode = 'Analysis'
        nPoints = 2
        #ncols = 1
        mus = [1,-1]

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,requires_grad=True)        
        dLdZ = (1/ncols)*torch.randn(nPoints,ncols,dtype=datatype)
        # angle = 2.*math.pi*randn(1)
        angle = 2.*math.pi*gauss(mu=0.,sigma=1.) #randn(1)        
        R = torch.tensor([[ math.cos(angle), -math.sin(angle) ],
            [ -math.sin(angle), -math.cos(angle)]], 
            dtype=datatype) #.squeeze()
        dRdW = torch.tensor([[ -math.sin(angle), -math.cos(angle) ],
            [ -math.cos(angle), math.sin(angle)]], 
            dtype=datatype) #.squeeze()
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW.T @ X))            

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles = nn.init.constant_(target.angles,angle)
        target.mus = torch.tensor(mus,dtype=datatype)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
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

    """
    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testBackwardSetAngles(self,datatype):
        rtol,atol = 1e-05,1e-08 

        # Configuration
        mode='Analysis'
        nPoints=2
        ncols=1

        # Expected values
        X = torch.randn(nPoints,ncols,dtype=datatype,requires_grad=True)        
        dLdZ = (1/ncols)*torch.randn(nPoints,ncols,dtype=datatype)        
        R = torch.eye(nPoints,dtype=datatype)
        dRdW = torch.tensor([
            [ 0., -1.],
            [ 1., 0.] ],
            dtype=datatype)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW.T @ X))            

         # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles = nn.init.zeros_(target.angles)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad 

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

        # Expected values
        #angle = 2.*math.pi*randn(1)
        angle = 2.*math.pi*gauss(mu=0.,sigma=1.)
        R = torch.tensor([[ math.cos(angle), -math.sin(angle) ],
            [ -math.sin(angle), -math.cos(angle)]], 
            dtype=datatype) #.squeeze()
        dRdW = torch.tensor([[ -math.sin(angle), -math.cos(angle) ],
            [ -math.cos(angle), math.sin(angle)]], 
            dtype=datatype) #.squeeze()
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW @ X))
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(expctddLdX * (dRdW.T @ X))            

        # Set angles
        target.angles.data = torch.tensor(angle,dtype=datatype)
        
        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target.forward(X)
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad 

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))
        """

    """
    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test4x4RandAngs(self,datatype):
        rtol,atol = 1e-05,1e-08 

        # Expcted values
        mus = [ -1, 1, -1, 1 ]
        angs = 2*math.pi*rand(6)

        expctdM = torch.tensor(
            np.diag(mus).dot(np.array(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0, 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            )).dot(np.array(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            )),dtype=datatype)
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype
            )

        # Actual values
        actualM = omgs(angles=angs,mus=mus)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=rtol,atol=atol).all())

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testBackward4x4RandAngPdAng2(self,datatype):
        rtol,atol = 1e-05,1e-08 

        # Expcted values
        mus = [ -1, 1, -1, 1 ]
        angs = 2*math.pi*rand(6)
        pdAng = 2

        expctdM = torch.tensor(
            np.diag(mus).dot(np.array(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0, 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            )).dot(np.array( 
                [ [1, 0, 0, 0 ], 
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array( # Partial Diff.
                [ [ math.cos(angs[2]+math.pi/2), 0, 0, -math.sin(angs[2]+math.pi/2) ],
                 [0, 0, 0, 0 ],
                 [0, 0, 0, 0 ],
                 [ math.sin(angs[2]+math.pi/2), 0, 0, math.cos(angs[2]+math.pi/2) ] ]
            )).dot(np.array(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            )),dtype=datatype)
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=rtol,atol=atol).all())

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testBackward4x4RandAngPdAng5(self,datatype):
        rtol,atol = 1e-05,1e-08 

        # Expcted values
        mus = [ 1, 1, -1, -1 ]
        angs = 2*math.pi*rand(6)
        pdAng = 5

        expctdM = torch.tensor(
            np.diag(mus).dot(np.array( # Partial Diff.
                [ [0, 0, 0, 0. ],
                 [0, 0, 0., 0. ],
                 [0, 0, math.cos(angs[5]+math.pi/2), -math.sin(angs[5]+math.pi/2) ],
                 [0., 0, math.sin(angs[5]+math.pi/2), math.cos(angs[5]+math.pi/2) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            )).dot(np.array( 
                [ [1, 0, 0, 0 ], 
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array( 
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            )).dot(np.array(
               [ [math.cos(angs[1]), 0, -math.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]), 0, math.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            )),dtype=datatype)
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=rtol,atol=atol).all())    

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testBackward4x4RandAngPdAng1(self,datatype):
        rtol,atol = 1e-05,1e-08 

        # Expcted values
        mus = [ -1, -1, -1, -1 ]
        angs = 2*math.pi*rand(6)
        pdAng = 1
        delta = 1e-10

        expctdM = torch.tensor(
            (1/delta)*np.diag(mus).dot(np.array( 
                [ [1, 0, 0, 0. ],
                 [0, 1, 0., 0. ],
                 [0, 0, math.cos(angs[5]), -math.sin(angs[5]) ],
                 [0., 0, math.sin(angs[5]), math.cos(angs[5]) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, math.cos(angs[4]), 0, -math.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, math.sin(angs[4]), 0, math.cos(angs[4]) ] ]
            )).dot(np.array( 
                [ [1, 0, 0, 0 ], 
                 [0, math.cos(angs[3]), -math.sin(angs[3]), 0 ],
                 [0, math.sin(angs[3]), math.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array( 
                [ [ math.cos(angs[2]), 0, 0, -math.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ math.sin(angs[2]), 0, 0, math.cos(angs[2]) ] ]
            )).dot(np.array( 
               [ [math.cos(angs[1]+delta/2), 0, -math.sin(angs[1]+delta/2), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]+delta/2), 0, math.cos(angs[1]+delta/2), 0 ],
                 [0, 0, 0, 1 ] ]
            ) - np.array( 
               [ [math.cos(angs[1]-delta/2), 0, -math.sin(angs[1]-delta/2), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]-delta/2), 0, math.cos(angs[1]-delta/2), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            )),dtype=datatype)
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=rtol,atol=atol).all())    

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testBackward8x8RandAngPdAng13(self,datatype):
        atol=1e-2

        # Expcted values
        pdAng = 13
        delta = 1e-5
        angs0 = 2*math.pi*rand(28)
        angs1 = angs0.copy()
        angs2 = angs0.copy()
        angs1[pdAng] = angs0[pdAng]-delta/2
        angs2[pdAng] = angs0[pdAng]+delta/2

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=False
            )
        expctdM = ( omgs(angles=angs2,mus=1) - omgs(angles=angs1,mus=1) )/delta
        
        # Instantiation of target class
        omgs.partial_difference=True
        actualM = omgs(angles=angs0,mus=1,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=rtol,atol=atol).all())    
    """

if __name__ == '__main__':
    unittest.main()