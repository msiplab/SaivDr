import torch
import torch.nn as nn

class NsoltChannelConcatenation2dLayer(nn.Module):
    """
    NSOLTCHANNELCONCATENATION2DLAYER
    
       ２コンポーネント入力(nComponents=2のみサポート):
          nSamples x nRows x nCols x (nChsTotal-1) 
          nSamples x nRows x nCols 
    
       １コンポーネント出力(nComponents=1のみサポート):
          nSamples x nRows x nCols x nChsTotal
    
     Requirements: Python 3.7.x, PyTorch 1.7.x
    
     Copyright (c) 2020-2021, Shogo MURAMATSU
    
     All rights reserved.
    
     Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/
    """

    def __init__(self,
        name=''):
        super(NsoltChannelConcatenation2dLayer, self).__init__()
        self.name = name
        self.description = "Channel concatenation"
        #self.type = ''
        #self.input_names = [ 'ac', 'dc' ]

    def forward(self,Xac,Xdc):
        """
        Forward input data through the layer at prediction time and
        output the result.
            
            Inputs:
                layer       - Layer to forward propagate through
                X1, X2      - Input data (2 components)
            Outputs:
                Z           - Outputs of layer forward function
        """
            
        # Layer forward function for prediction goes here.
        return torch.cat((Xdc.unsqueeze(dim=3),Xac),dim=3)
