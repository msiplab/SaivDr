import unittest
import torch
import torch.nn as nn
from nsoltInitialRotation2dLayer import NsoltInitialRotation2dLayer

class NsoltAtomExtention2dLayerTestCase(unittest.TestCase):

    def testConstructor(self):
        target = NsoltInitialRotation2dLayer()
        self.assertTrue(isinstance(target, nn.Module))

    def testForward(self):
        target = NsoltInitialRotation2dLayer()
        x = 0.0
        valueExpctd = 0.0
        valueActual = target.forward(x)
        self.assertEqual(valueExpctd, valueActual)

if __name__ == '__main__':
    unittest.main()
