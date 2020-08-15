classdef nsoltInitialRotation3dLayer < nnet.layer.Layer
    %NSOLTINITIALROTATION3DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nRows x nCols x nLays x nDecs x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nRows x nCols x nLays x nChs x nSamples
    %
    % Requirements: MATLAB R2020a
    %
    % Copyright (c) 2020, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    
    properties
        % (Optional) Layer properties.
        NumberOfChannels
        DecimationFactor
        NoDcLeakage
        Mus
        
        % Layer properties go here.
    end
    
    properties (Learnable)
        Angles
    end
    
    methods
        function layer = nsoltInitialRotation3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            import saivdr.dictionary.utility.Direction
            p = inputParser;
            addParameter(p,'NumberOfChannels',[])
            addParameter(p,'DecimationFactor',[])
            addParameter(p,'Name','')
            addParameter(p,'Mus',[])
            addParameter(p,'Angles',[])
            addParameter(p,'NoDcLeakage',false);
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.Name = p.Results.Name;
            layer.Mus = p.Results.Mus;
            layer.Angles = p.Results.Angles;
            layer.NoDcLeakage = p.Results.NoDcLeakage;
            layer.Description = "NSOLT initial rotation " ...
                + "(ps,pa) = (" ...
                + layer.NumberOfChannels(1) + "," ...
                + layer.NumberOfChannels(2) + "), "  ...
                + "(mv,mh,md) = (" ...
                + layer.DecimationFactor(Direction.VERTICAL) + "," ...
                + layer.DecimationFactor(Direction.HORIZONTAL) + "," ...
                + layer.DecimationFactor(Direction.DEPTH) + ")";
            layer.Type = '';
            
            nChsTotal = sum(layer.NumberOfChannels);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            if isempty(layer.Angles)
                layer.Angles = zeros(nAngles,1);
            end
            if length(layer.Angles)~=nAngles
                error('Invalid # of angles')
            end
            
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data (n: # of components)
            % Outputs:
            %         Z           - Outputs of layer forward function
            %
            import saivdr.dcnn.fcn_orthmtxgen
            
            % Layer forward function for prediction goes here.
            nrows = size(X,1);
            ncols = size(X,2);
            nlays = size(X,3);
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            nSamples = size(X,5);
            stride = layer.DecimationFactor;
            nDecs = prod(stride);
            nChsTotal = ps + pa;
            %
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
            W0 = fcn_orthmtxgen(anglesW,muW);
            U0 = fcn_orthmtxgen(anglesU,muU);
            
            Y = reshape(permute(X,[4 1 2 3 5]),nDecs,nrows*ncols*nlays*nSamples);
            Zs = W0(:,1:nDecs/2)*Y(1:nDecs/2,:);
            Za = U0(:,1:nDecs/2)*Y(nDecs/2+1:end,:);
            Z = ipermute(reshape([Zs;Za],nChsTotal,nrows,ncols,nlays,nSamples),...
                [4 1 2 3 5]);
            
        end
        
        function [dLdX, dLdW] = ...
                backward(layer, X, ~, dLdZ, ~)
            % (Optional) Backward propagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %
            import saivdr.dcnn.fcn_orthmtxgen
            
            nrows = size(dLdZ,1);
            ncols = size(dLdZ,2); 
            nlays = size(dLdZ,3); 
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            nAngles = length(layer.Angles);
            nSamples = size(dLdZ,5);
            stride = layer.DecimationFactor;
            nDecs = prod(stride);
            %
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
            W0T = transpose(fcn_orthmtxgen(anglesW,muW,0));
            U0T = transpose(fcn_orthmtxgen(anglesU,muU,0));
            
            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            Y = permute(dLdZ,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            Zsa = [ W0T(1:nDecs/2,:)*Ys; U0T(1:nDecs/2,:)*Ya ];
            dLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
                [4 1 2 3 5]);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dLdW = zeros(nAngles,1,'like',dLdZ);
            dldz_ = permute(dLdZ,[4 1 2 3 5]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            % (dVdWi)X
            for iAngle = 1:nAngles/2
                dW0 = fcn_orthmtxgen(anglesW,muW,iAngle);
                dU0 = fcn_orthmtxgen(anglesU,muU,iAngle);
                a_ = permute(X,[4 1 2 3 5]);
                c_upp = reshape(a_(1:nDecs/2,:,:,:,:),nDecs/2,nrows*ncols*nlays*nSamples);
                c_low = reshape(a_(nDecs/2+1:nDecs,:,:,:,:),nDecs/2,nrows*ncols*nlays*nSamples);
                d_upp = dW0(:,1:nDecs/2)*c_upp;
                d_low = dU0(:,1:nDecs/2)*c_low;
                dLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                dLdW(nAngles/2+iAngle) = sum(dldz_low.*d_low,'all');
            end
        end
    end
    
end

