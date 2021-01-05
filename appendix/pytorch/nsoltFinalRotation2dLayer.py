import torch
import torch.nn as nn
from nsoltUtility import Direction
import numpy as np

class NsoltFinalRotation2dLayer(nn.Module):
    """
    NSOLTFINALROTATION2DLAYER 
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nChs
    
       コンポーネント別に出力(nComponents):
          nSamples x nRows x nCols x nDecs
    
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
        number_of_channels=[],
        decimation_factor=[],
        name=''):
        super(NsoltFinalRotation2dLayer, self).__init__()
        self.name = name
        self.number_of_channels = number_of_channels
        self.decimation_factor = decimation_factor
        self.description = "NSOLT final rotation " \
                + "(ps,pa) = (" \
                + str(self.number_of_channels[0]) + "," \
                + str(self.number_of_channels[1]) + "), "  \
                + "(mv,mh) = (" \
                + str(self.decimation_factor[Direction.VERTICAL]) + "," \
                + str(self.decimation_factor[Direction.HORIZONTAL]) + ")"

    def forward(self,X):
        nSamples = X.size(dim=0)
        nrows = X.size(dim=1)
        ncols = X.size(dim=2)
        ps, pa = self.number_of_channels
        stride = self.decimation_factor
        nDecs = np.prod(stride)

        W0T = torch.eye(ps,dtype=X.dtype)
        U0T = torch.eye(pa,dtype=X.dtype)

        Y = X
        Ys = Y[:,:,:,:ps].view(-1,ps).T
        Ya = Y[:,:,:,ps:].view(-1,pa).T 
        Zsa = torch.cat(        
            ( W0T[:np.ceil(nDecs/2.).astype(int),:].mm(Ys),
             U0T[:np.floor(nDecs/2.).astype(int),:].mm(Ya) ),
             dim=0 )
        return Zsa.T.view(nSamples,nrows,ncols,nDecs)
        """
            import saivdr.dcnn.fcn_orthmtxgen
            
            % Layer forward function for prediction goes here.
            if isempty(layer.Mus)
                layer.Mus = ones(ps+pa,1);
            elseif isscalar(layer.Mus)
                layer.Mus = layer.Mus*ones(ps+pa,1);
            end
            if layer.NoDcLeakage
                layer.Mus(1) = 1;
                layer.Angles(1:ps-1) = ...
                    zeros(ps-1,1,'like',layer.Angles);
            end            
            muW = layer.Mus(1:ps);
            muU = layer.Mus(ps+1:end);
            anglesW = layer.Angles(1:length(layer.Angles)/2);
            anglesU = layer.Angles(length(layer.Angles)/2+1:end);
            W0T = transpose(fcn_orthmtxgen(anglesW,muW));
            U0T = transpose(fcn_orthmtxgen(anglesU,muU));

            Y = X; %permute(X,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            Zsa = [ W0T(1:ceil(nDecs/2),:)*Ys; U0T(1:floor(nDecs/2),:)*Ya ];
            %Z = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            Z = reshape(Zsa,nDecs,nrows,ncols,nSamples);
        """
        return X