import unittest
import torch
import torch.nn as nn
from nsoltSynthesis2dNetwork import NsoltSynthesis2dNetwork

class NsoltSynthesis2dNetworkTestCase(unittest.TestCase):

    def testConstructor(self):
        target = NsoltSynthesis2dNetwork()
        self.assertTrue(isinstance(target, nn.Module))

    def testDummy(self):
        target = NsoltSynthesis2dNetwork()
        x = 0.0
        valueExpctd = 0.0
        valueActual = target.forward(x)
        self.assertEqual(valueExpctd, valueActual)

if __name__ == '__main__':
    unittest.main()

