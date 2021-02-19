import unittest
import torch
import torch.nn as nn
from nsoltAnalysis2dNetwork import NsoltAnalysis2dNetwork

class NsoltAnalysis2dNetworkTestCase(unittest.TestCase):

    def testConstructor(self):
        target = NsoltAnalysis2dNetwork()
        self.assertTrue(isinstance(target, nn.Module))

    def testDummy(self):
        target = NsoltAnalysis2dNetwork()
        x = 0.0
        valueExpctd = 0.0
        valueActual = target.forward(x)
        self.assertEqual(valueExpctd, valueActual)

if __name__ == '__main__':
    unittest.main()