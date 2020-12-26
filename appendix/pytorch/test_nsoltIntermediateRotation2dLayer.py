import unittest
import torch
import torch.nn as nn
from nsoltIntermediateRotation2dLayer import NsoltIntermediateRotation2dLayer

class TestNsoltAtomExtention2dLayer(unittest.TestCase):

    def testConstructor(self):
        target = NsoltIntermediateRotation2dLayer()
        self.assertTrue(isinstance(target, nn.Module))

    def testForward(self):
        target = NsoltIntermediateRotation2dLayer()
        x = 0.0
        valueExpctd = 0.0
        valueActual = target.forward(x)
        self.assertEqual(valueExpctd, valueActual)

if __name__ == '__main__':
    unittest.main()
