import unittest
import torch
import torch.nn as nn
from nsoltSubbandSerialization2dLayer import NsoltSubbandSerialization2dLayer

class NsoltSubbandSerialization2dLayerTestCase(unittest.TestCase):

    def testConstructor(self):
        target = NsoltSubbandSerialization2dLayer()
        self.assertTrue(isinstance(target, nn.Module))

    def testForward(self):
        target = NsoltSubbandSerialization2dLayer()
        x = 0.0
        valueExpctd = 0.0
        valueActual = target.forward(x)
        self.assertEqual(valueExpctd, valueActual)

if __name__ == '__main__':
    unittest.main()