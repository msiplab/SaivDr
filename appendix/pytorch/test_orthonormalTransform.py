import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import numpy as np
from orthonormalTransform import OrthonormalTransform

datatype = [ torch.float, torch.double ]
ncols = [ 1, 2, 4 ]

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
        atol=1e-6

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)
        expctdZ = X
        expctdNParams = 1

        # Instantiation of target class
        target = OrthonormalTransform()

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)
        actualNParams = len(target.parameters().__next__())

        # Evaluation
        self.assertTrue(isinstance(target,nn.Module))        
        self.assertTrue(torch.isclose(actualZ,expctdZ,rtol=0.,atol=atol).all())    
        self.assertEqual(actualNParams,expctdNParams)

    @parameterized.expand(
        list(itertools.product(datatype,ncols))
    )
    def testCallWithAngles(self,datatype,ncols):
        atol=1e-6

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)        
        R = torch.tensor([
            [ np.cos(np.pi/4), -np.sin(np.pi/4) ],
            [ np.sin(np.pi/4),  np.cos(np.pi/4) ] ],
            dtype=datatype)
        expctdZ = R @ X

        # Instantiation of target class
        target = OrthonormalTransform()
        target.angles.data = torch.tensor([np.pi/4])

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.isclose(actualZ,expctdZ,rtol=0.,atol=atol).all())        

    @parameterized.expand(
        list(itertools.product(datatype,ncols))
    )
    def testCallWithAnglesAndMus(self,datatype,ncols):
        atol=1e-6

        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)        
        R = torch.tensor([
            [ np.cos(np.pi/4), -np.sin(np.pi/4) ],
            [ -np.sin(np.pi/4), -np.cos(np.pi/4) ] ],
            dtype=datatype)
        expctdZ = R @ X

        # Instantiation of target class
        target = OrthonormalTransform()
        target.angles.data = torch.tensor([np.pi/4])
        target.mus = torch.tensor([1, -1])        

        # Actual values
        with torch.no_grad():
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.isclose(actualZ,expctdZ,rtol=0.,atol=atol).all())        

    @parameterized.expand(
        list(itertools.product(datatype,ncols))
    )
    def testSetAngles(self,datatype,ncols):
        atol=1e-6
    
        # Expected values
        X = torch.randn(2,ncols,dtype=datatype)  
        R = torch.eye(2,dtype=datatype)
        expctdZ = R @ X

        # Instantiation of target class
        target = OrthonormalTransform()

        # Actual values
        with torch.no_grad():        
            actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.isclose(actualZ,expctdZ,rtol=0.,atol=atol).all())        

        # Expcted values
        R = torch.tensor([
            [ np.cos(np.pi/4), -np.sin(np.pi/4) ],
            [ np.sin(np.pi/4), np.cos(np.pi/4) ] ],
            dtype=datatype)
        expctdZ = R @ X

        # Actual values
        target.angles.data = torch.tensor([np.pi/4])
        actualZ = target.forward(X)

        # Evaluation
        self.assertTrue(torch.isclose(actualZ,expctdZ,rtol=0.,atol=atol).all())        

    """
    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test4x4(self,datatype):
        atol=1e-6
        
        # Expected values
        expctdNorm = 1.

        # Instantiation of target class
        ang = 2*np.pi*rand(6)
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        unitvec = torch.randn(4,dtype=datatype)
        unitvec /= unitvec.norm()
        actualNorm = omgs(angles=ang,mus=1).mv(unitvec).norm().numpy()

        # Evaluation
        message = "actualNorm=%s differs from %s" % ( str(actualNorm), str(expctdNorm) )
        self.assertTrue(np.isclose(actualNorm,expctdNorm,rtol=0.,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test8x8(self,datatype):
        atol=1e-6
        
        # Expected values
        expctdNorm = 1.

        # Instantiation of target class
        ang = 2*np.pi*rand(28)
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        unitvec = torch.randn(8,dtype=datatype)
        unitvec /= unitvec.norm()
        actualNorm = omgs(angles=ang,mus=1).mv(unitvec).norm().numpy()

        # Evaluation
        message = "actualNorm=%s differs from %s" % ( str(actualNorm), str(expctdNorm) )
        self.assertTrue(np.isclose(actualNorm,expctdNorm,rtol=0.,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test4x4red(self,datatype):
        atol=1e-6

        # Expected values
        expctdLeftTop = 1.

        # Instantiation of target class
        ang = 2*np.pi*rand(6)
        nSize = 4
        ang[:nSize-1] = np.zeros(nSize-1)
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        matrix = omgs(angles=ang,mus=1)
        actualLeftTop = matrix[0,0]

        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )        
        self.assertTrue(np.isclose(actualLeftTop,expctdLeftTop,rtol=0.,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test8x8red(self,datatype):
        atol=1e-6

        # Expected values
        expctdLeftTop = 1.

        # Instantiation of target class
        ang = 2*np.pi*rand(28)
        nSize = 8
        ang[:nSize-1] = np.zeros(nSize-1)
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        matrix = omgs(angles=ang,mus=1)
        actualLeftTop = matrix[0,0]

        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )        
        self.assertTrue(np.isclose(actualLeftTop,expctdLeftTop,rtol=0.,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference(self,datatype):
        atol=1e-6

        # Expected values
        expctdM = torch.tensor([
            [ 0., -1.],
            [ 1., 0.] ],
            dtype=datatype)

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=0,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())                

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifferenceAngs(self,datatype):
        atol=1e-6

        # Expected values
        expctdM = torch.tensor([
            [ np.cos(np.pi/4+np.pi/2), -np.sin(np.pi/4+np.pi/2)],
            [ np.sin(np.pi/4+np.pi/2),  np.cos(np.pi/4+np.pi/2)] ],
            dtype=datatype)

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=np.pi/4,mus=1,index_pd_angle=0)
            
        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())                

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifferenceAngsAndMus(self,datatype):
        atol=1e-6

        # Expected values
        expctdM = torch.tensor([
            [ np.cos(np.pi/4+np.pi/2), -np.sin(np.pi/4+np.pi/2)],
            [ -np.sin(np.pi/4+np.pi/2),  -np.cos(np.pi/4+np.pi/2)] ],
            dtype=datatype)

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=np.pi/4,mus=[1,-1],index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())                

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifferenceSetAngles(self,datatype):
        atol=1e-6

        # Expected values
        expctdM = torch.tensor([
            [ 0., -1.],
            [ 1., 0.] ],
            dtype=datatype)

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=0,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())                

        # Expected values
        expctdM = torch.tensor([
            [ np.cos(np.pi/4+np.pi/2), -np.sin(np.pi/4+np.pi/2)],
            [ np.sin(np.pi/4+np.pi/2),  np.cos(np.pi/4+np.pi/2)] ],
            dtype=datatype)

        # Actual values
        actualM = omgs(angles=np.pi/4,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())                

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test4x4RandAngs(self,datatype):
        atol=1e-6

        # Expcted values
        mus = [ -1, 1, -1, 1 ]
        angs = 2*np.pi*rand(6)

        expctdM = torch.tensor(
            np.diag(mus).dot(np.array(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, np.cos(angs[5]), -np.sin(angs[5]) ],
                 [0, 0, np.sin(angs[5]), np.cos(angs[5]) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, np.cos(angs[4]), 0, -np.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, np.sin(angs[4]), 0, np.cos(angs[4]) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, np.cos(angs[3]), -np.sin(angs[3]), 0 ],
                 [0, np.sin(angs[3]), np.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
                [ [ np.cos(angs[2]), 0, 0, -np.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ np.sin(angs[2]), 0, 0, np.cos(angs[2]) ] ]
            )).dot(np.array(
               [ [np.cos(angs[1]), 0, -np.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [np.sin(angs[1]), 0, np.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
               [ [ np.cos(angs[0]), -np.sin(angs[0]), 0, 0 ],
                 [ np.sin(angs[0]), np.cos(angs[0]), 0, 0 ],
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
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference4x4RandAngPdAng2(self,datatype):
        atol=1e-6

        # Expcted values
        mus = [ -1, 1, -1, 1 ]
        angs = 2*np.pi*rand(6)
        pdAng = 2

        expctdM = torch.tensor(
            np.diag(mus).dot(np.array(
                [ [1, 0, 0, 0. ],
                 [0, 1, 0, 0. ],
                 [0, 0, np.cos(angs[5]), -np.sin(angs[5]) ],
                 [0, 0, np.sin(angs[5]), np.cos(angs[5]) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, np.cos(angs[4]), 0, -np.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, np.sin(angs[4]), 0, np.cos(angs[4]) ] ]
            )).dot(np.array( 
                [ [1, 0, 0, 0 ], 
                 [0, np.cos(angs[3]), -np.sin(angs[3]), 0 ],
                 [0, np.sin(angs[3]), np.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array( # Partial Diff.
                [ [ np.cos(angs[2]+np.pi/2), 0, 0, -np.sin(angs[2]+np.pi/2) ],
                 [0, 0, 0, 0 ],
                 [0, 0, 0, 0 ],
                 [ np.sin(angs[2]+np.pi/2), 0, 0, np.cos(angs[2]+np.pi/2) ] ]
            )).dot(np.array(
               [ [np.cos(angs[1]), 0, -np.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [np.sin(angs[1]), 0, np.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
               [ [ np.cos(angs[0]), -np.sin(angs[0]), 0, 0 ],
                 [ np.sin(angs[0]), np.cos(angs[0]), 0, 0 ],
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
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference4x4RandAngPdAng5(self,datatype):
        atol=1e-6

        # Expcted values
        mus = [ 1, 1, -1, -1 ]
        angs = 2*np.pi*rand(6)
        pdAng = 5

        expctdM = torch.tensor(
            np.diag(mus).dot(np.array( # Partial Diff.
                [ [0, 0, 0, 0. ],
                 [0, 0, 0., 0. ],
                 [0, 0, np.cos(angs[5]+np.pi/2), -np.sin(angs[5]+np.pi/2) ],
                 [0., 0, np.sin(angs[5]+np.pi/2), np.cos(angs[5]+np.pi/2) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, np.cos(angs[4]), 0, -np.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, np.sin(angs[4]), 0, np.cos(angs[4]) ] ]
            )).dot(np.array( 
                [ [1, 0, 0, 0 ], 
                 [0, np.cos(angs[3]), -np.sin(angs[3]), 0 ],
                 [0, np.sin(angs[3]), np.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array( 
                [ [ np.cos(angs[2]), 0, 0, -np.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ np.sin(angs[2]), 0, 0, np.cos(angs[2]) ] ]
            )).dot(np.array(
               [ [np.cos(angs[1]), 0, -np.sin(angs[1]), 0 ],
                 [0, 1, 0, 0 ],
                 [np.sin(angs[1]), 0, np.cos(angs[1]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
               [ [ np.cos(angs[0]), -np.sin(angs[0]), 0, 0 ],
                 [ np.sin(angs[0]), np.cos(angs[0]), 0, 0 ],
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
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())    

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference4x4RandAngPdAng1(self,datatype):
        atol=1e-6

        # Expcted values
        mus = [ -1, -1, -1, -1 ]
        angs = 2*np.pi*rand(6)
        pdAng = 1
        delta = 1e-10

        expctdM = torch.tensor(
            (1/delta)*np.diag(mus).dot(np.array( 
                [ [1, 0, 0, 0. ],
                 [0, 1, 0., 0. ],
                 [0, 0, np.cos(angs[5]), -np.sin(angs[5]) ],
                 [0., 0, np.sin(angs[5]), np.cos(angs[5]) ] ]
            )).dot(np.array(
                [ [1, 0, 0, 0 ],
                 [0, np.cos(angs[4]), 0, -np.sin(angs[4]) ],
                 [0, 0, 1, 0 ],
                 [0, np.sin(angs[4]), 0, np.cos(angs[4]) ] ]
            )).dot(np.array( 
                [ [1, 0, 0, 0 ], 
                 [0, np.cos(angs[3]), -np.sin(angs[3]), 0 ],
                 [0, np.sin(angs[3]), np.cos(angs[3]), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array( 
                [ [ np.cos(angs[2]), 0, 0, -np.sin(angs[2]) ],
                 [0, 1, 0, 0 ],
                 [0, 0, 1, 0 ],
                 [ np.sin(angs[2]), 0, 0, np.cos(angs[2]) ] ]
            )).dot(np.array( 
               [ [np.cos(angs[1]+delta/2), 0, -np.sin(angs[1]+delta/2), 0 ],
                 [0, 1, 0, 0 ],
                 [np.sin(angs[1]+delta/2), 0, np.cos(angs[1]+delta/2), 0 ],
                 [0, 0, 0, 1 ] ]
            ) - np.array( 
               [ [np.cos(angs[1]-delta/2), 0, -np.sin(angs[1]-delta/2), 0 ],
                 [0, 1, 0, 0 ],
                 [np.sin(angs[1]-delta/2), 0, np.cos(angs[1]-delta/2), 0 ],
                 [0, 0, 0, 1 ] ]
            )).dot(np.array(
               [ [ np.cos(angs[0]), -np.sin(angs[0]), 0, 0 ],
                 [ np.sin(angs[0]), np.cos(angs[0]), 0, 0 ],
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
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())    

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference8x8RandAngPdAng13(self,datatype):
        atol=1e-2

        # Expcted values
        pdAng = 13
        delta = 1e-5
        angs0 = 2*np.pi*rand(28)
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
        self.assertTrue(torch.isclose(actualM,expctdM,rtol=0.,atol=atol).all())    
    """

if __name__ == '__main__':
    unittest.main()