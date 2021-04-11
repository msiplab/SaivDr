import itertools
import unittest
from parameterized import parameterized
import torch
import math
from nsoltUtility import OrthonormalMatrixGenerationSystem

datatype = [ torch.float, torch.double ]
isdevicetest = True

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
        rtol,atol=1e-5,1e-8

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
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testConstructorWithDevice(self,datatype):
        rtol,atol=1e-5,1e-8

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
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))              

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testCallWithAngles(self,datatype):
        rtol,atol=1e-5,1e-8

        # Expected values
        expctdM = torch.tensor([
            [math.cos(math.pi/4), -math.sin(math.pi/4)],
            [math.sin(math.pi/4),  math.cos(math.pi/4)] ],
            dtype=datatype)
            
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )
            
        # Actual values
        actualM = omgs(angles=math.pi/4,mus=1)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testCallWithAnglesAndMus(self,datatype):
        rtol,atol=1e-5,1e-8

        # Expected values
        expctdM = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4)],
            [-math.sin(math.pi/4), -math.cos(math.pi/4)] ],
            dtype=datatype)
            
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )
            
        # Actual values
        actualM = omgs(angles=math.pi/4, mus=[ 1, -1] )

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testSetAngles(self,datatype):
        rtol,atol=1e-5,1e-8

        # Expected values
        expctdM = torch.eye(2,dtype=datatype)

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        actualM = omgs(angles=0,mus=1)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))        

        # Expected values
        expctdM = torch.tensor([
            [math.cos(math.pi/4), -math.sin(math.pi/4)],
            [math.sin(math.pi/4),  math.cos(math.pi/4)] ],
            dtype=datatype)

        actualM = omgs(angles=math.pi/4,mus=1)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test4x4(self,datatype):
        rtol,atol=1e-5,1e-8
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Expected values
        expctdNorm = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        ang = 2*math.pi*torch.rand(6).to(device)
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        unitvec = torch.randn(4,dtype=datatype).to(device)
        unitvec /= unitvec.norm()
        actualNorm = omgs(angles=ang,mus=1).mv(unitvec).norm() #.numpy()

        # Evaluation
        message = "actualNorm=%s differs from %s" % ( str(actualNorm), str(expctdNorm) )
        self.assertTrue(torch.isclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test8x8(self,datatype):
        rtol,atol=1e-5,1e-8
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Expected values
        expctdNorm = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        ang = 2*math.pi*torch.rand(28).to(device)
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        unitvec = torch.randn(8,dtype=datatype).to(device)
        unitvec /= unitvec.norm()
        actualNorm = omgs(angles=ang,mus=1).mv(unitvec).norm() #.numpy()

        # Evaluation
        message = "actualNorm=%s differs from %s" % ( str(actualNorm), str(expctdNorm) )
        self.assertTrue(torch.isclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test4x4red(self,datatype):
        rtol,atol=1e-5,1e-8
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Expected values
        expctdLeftTop = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        ang = 2*math.pi*torch.rand(6).to(device)
        nSize = 4
        ang[:nSize-1] = torch.zeros(nSize-1).to(device)
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        matrix = omgs(angles=ang,mus=1)
        actualLeftTop = matrix[0,0]

        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )        
        self.assertTrue(torch.isclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test8x8red(self,datatype):
        rtol,atol=1e-5,1e-8
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Expected values
        expctdLeftTop = torch.tensor(1.,dtype=datatype).to(device)

        # Instantiation of target class
        ang = 2*math.pi*torch.rand(28).to(device)
        nSize = 8
        ang[:nSize-1] = torch.zeros(nSize-1).to(device)
        omgs = OrthonormalMatrixGenerationSystem(
            dtype=datatype
        )

        # Actual values
        matrix = omgs(angles=ang,mus=1)
        actualLeftTop = matrix[0,0]

        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )        
        self.assertTrue(torch.isclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)        

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference(self,datatype):
        rtol,atol=1e-4,1e-7

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
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))                

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifferenceAngs(self,datatype):
        rtol,atol=1e-4,1e-7

        # Expected values
        expctdM = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2)],
            [ math.sin(math.pi/4+math.pi/2),  math.cos(math.pi/4+math.pi/2)] ],
            dtype=datatype)

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=math.pi/4,mus=1,index_pd_angle=0)
            
        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))                

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifferenceAngsAndMus(self,datatype):
        rtol,atol=1e-4,1e-7

        # Expected values
        expctdM = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2)],
            [ -math.sin(math.pi/4+math.pi/2),  -math.cos(math.pi/4+math.pi/2)] ],
            dtype=datatype)

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=math.pi/4,mus=[1,-1],index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))                

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifferenceSetAngles(self,datatype):
        rtol,atol=1e-4,1e-7

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
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))                

        # Expected values
        expctdM = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2)],
            [ math.sin(math.pi/4+math.pi/2),  math.cos(math.pi/4+math.pi/2)] ],
            dtype=datatype)

        # Actual values
        actualM = omgs(angles=math.pi/4,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))                

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def test4x4RandAngs(self,datatype):
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Expcted values
        angs = 2*math.pi*torch.rand(6).to(device)
        mus = [ -1, 1, -1, 1 ]

        expctdM = torch.as_tensor(
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
            ),dtype=datatype).to(device)
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype
            )

        # Actual values
        actualM = omgs(angles=angs,mus=mus)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference4x4RandAngPdAng2(self,datatype):
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Expcted values
        angs = 2*math.pi*torch.rand(6).to(device)
        mus = [ -1, 1, -1, 1 ]
        pdAng = 2

        expctdM = torch.as_tensor(
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
            ) @ torch.tensor( # Partial Diff.
                [ [ math.cos(angs[2]+math.pi/2), 0, 0, -math.sin(angs[2]+math.pi/2) ],
                 [0, 0, 0, 0 ],
                 [0, 0, 0, 0 ],
                 [ math.sin(angs[2]+math.pi/2), 0, 0, math.cos(angs[2]+math.pi/2) ] ]
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
            ),dtype=datatype).to(device)
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference4x4RandAngPdAng5(self,datatype):
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Expcted values
        angs = 2*math.pi*torch.rand(6).to(device)
        mus = [ 1, 1, -1, -1 ]
        pdAng = 5

        expctdM = torch.as_tensor(
            torch.tensor(mus).view(-1,1) * \
            torch.tensor( # Partial Diff.
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
            ),dtype=datatype).to(device)
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))    

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference4x4RandAngPdAng1(self,datatype):
        rtol,atol=1e-1,1e-3
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Expcted values
        angs = 2*math.pi*torch.rand(6).to(device)
        mus = [ -1, -1, -1, -1 ]
        pdAng = 1
        delta = 1e-3

        expctdM = torch.as_tensor(
            (1/delta)*torch.tensor(mus).view(-1,1) * \
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
               [ [math.cos(angs[1]+delta/2), 0, -math.sin(angs[1]+delta/2), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]+delta/2), 0, math.cos(angs[1]+delta/2), 0 ],
                 [0, 0, 0, 1 ] ] ) - \
                torch.tensor( 
               [ [math.cos(angs[1]-delta/2), 0, -math.sin(angs[1]-delta/2), 0 ],
                 [0, 1, 0, 0 ],
                 [math.sin(angs[1]-delta/2), 0, math.cos(angs[1]-delta/2), 0 ],
                 [0, 0, 0, 1 ] ] )
            ) @ torch.tensor(
               [ [ math.cos(angs[0]), -math.sin(angs[0]), 0, 0 ],
                 [ math.sin(angs[0]), math.cos(angs[0]), 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 0, 0, 1 ] ]
            ),dtype=datatype).to(device)
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=True
            )

        # Actual values
        actualM = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))    

    @parameterized.expand(
        list(itertools.product(datatype))
    )
    def testPartialDifference8x8RandAngPdAng13(self,datatype):
        rtol,atol=1e-1,1e-3
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")   

        # Expcted values
        pdAng = 13
        delta = 1e-3
        angs0 = 2*math.pi*torch.rand(28).to(device)
        angs1 = angs0.clone()
        angs2 = angs0.clone()
        angs1[pdAng] = angs0[pdAng]-delta/2
        angs2[pdAng] = angs0[pdAng]+delta/2

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=False
            )
        expctdM = ( omgs(angles=angs2,mus=1) - omgs(angles=angs1,mus=1) )/delta
        expctdM.to(device)
        
        # Instantiation of target class
        omgs.partial_difference=True
        actualM = omgs(angles=angs0,mus=1,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(actualM,expctdM,rtol=rtol,atol=atol))    

if __name__ == '__main__':
    unittest.main()