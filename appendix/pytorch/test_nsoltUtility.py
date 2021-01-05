import unittest
import torch
from nsoltUtility import OrthonormalMatrixGenerationSystem

class OrthonormalMatrixGenerationSystemTestCase(unittest.TestCase):

    def testConstructor(self):
        target = OrthonormalMatrixGenerationSystem()
        self.assertTrue(isinstance(target, object))

    def testForward(self):
        target = OrthonormalMatrixGenerationSystem()
        x = 0.0
        valueExpctd = 0.0
        valueActual = target(x)
        self.assertEqual(valueExpctd, valueActual)

if __name__ == '__main__':
    unittest.main()