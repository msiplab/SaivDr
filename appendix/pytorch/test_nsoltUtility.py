import itertools
import unittest
from parameterized import parameterized
import torch
#import torch.nn as nn
import numpy as np
from numpy.random import rand
#from numpy.linalg import norm
from nsoltUtility import OrthonormalMatrixGenerationSystem

datatype = [ torch.float, torch.double ]

class OrthonormalMatrixGenerationSystemTestCase(unittest.TestCase):
    """
    ORTHONORMALMATRIXGENERATIONSYSTEMTESTCASE
    
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
        list(itertools.product(datatype))
    )
    def testConstructor(self,datatype):
        atol=1e-10

        # Expected values
        expctdM = torch.eye(2,dtype=datatype)

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        angles = 0
        mus = 1
        actualM = omgs(angles=angles,mus=mus)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testCallWithAngles(self,datatype):
        atol=1e-10

        # Expected values
        expctdM = torch.tensor([
            [np.cos(np.pi/4), -np.sin(np.pi/4)],
            [np.sin(np.pi/4),  np.cos(np.pi/4)] ],
            dtype=datatype)
            
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )
            
        # Actual values
        actualM = omgs(angles=np.pi/4,mus=1)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testCallWithAnglesAndMus(self,datatype):
        atol=1e-10

        # Expected values
        expctdM = torch.tensor([
            [ np.cos(np.pi/4), -np.sin(np.pi/4)],
            [-np.sin(np.pi/4), -np.cos(np.pi/4)] ],
            dtype=datatype)
            
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )
            
        # Actual values
        actualM = omgs(angles=np.pi/4, mus=[ 1, -1] )

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testSetAngles(self,datatype):
        atol=1e-10
        
        # Expected values
        expctdM = torch.eye(2,dtype=datatype)

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        actualM = omgs(angles=0,mus=1)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())        

        # Expected values
        expctdM = torch.tensor([
            [np.cos(np.pi/4), -np.sin(np.pi/4)],
            [np.sin(np.pi/4),  np.cos(np.pi/4)] ],
            dtype=datatype)

        actualM = omgs(angles=np.pi/4,mus=1)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test4x4(self,datatype):
        atol=1e-10
        
        # Expected values
        expctdNorm = 1

        # Instantiation of target class
        ang = 2*np.pi*rand(6,1)
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        unitvec = torch.randn(4,dtype=datatype)
        unitvec /= unitvec.norm()
        actualNorm = omgs(angles=ang,mus=1).mv(unitvec).norm()

        # Evaluation
        message = "normActual=%s differs from 1" % str(actualNorm)
        self.assertTrue(np.isclose(actualNorm,expctdNorm,rtol=0.,atol=atol),message)        

if __name__ == '__main__':
    unittest.main()