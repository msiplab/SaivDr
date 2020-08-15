classdef nsoltFinalRotation3dLayer < nnet.layer.Layer
    %NSOLTFINALROTATION2DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nRows x nCols x nLays x nChs x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nRows x nCols x nLays x nDecs x nSamples
    %
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
        function layer = nsoltFinalRotation3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            import saivdr.dictionary.utility.Direction
            p = inputParser;
            addParameter(p,'NumberOfChannels',[])
            addParameter(p,'DecimationFactor',[])
            addParameter(p,'Mus',[]);
            addParameter(p,'Angles',[]);
            addParameter(p,'Name','')
            addParameter(p,'NoDcLeakage',false);
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.Mus = p.Results.Mus;
            layer.Angles = p.Results.Angles;
            layer.NoDcLeakage = p.Results.NoDcLeakage;
            layer.Name = p.Results.Name;
            layer.Description = "NSOLT final rotation " ...
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
            import saivdr.dcnn.fcn_orthonormalmatrixgenerate
            
            % Layer forward function for prediction goes here.
            nrows = size(X,1);
            ncols = size(X,2);
            nlays = size(X,3);
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            nSamples = size(X,5);
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
            W0T = transpose(fcn_orthonormalmatrixgenerate(anglesW,muW));
            U0T = transpose(fcn_orthonormalmatrixgenerate(anglesU,muU));
            
            Y = permute(X,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            Zsa = [ W0T(1:nDecs/2,:)*Ys; U0T(1:nDecs/2,:)*Ya ];
            Z = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
                [4 1 2 3 5]);
        end
        
        function [Z, memory] = forward(layer, X)
            % (Optional) Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            %         memory      - Memory value for custom backward propagation
            
            % Layer forward function for training goes here.
            Z = layer.predict(X);
            memory = X;
        end
        
        function [dLdX, dLdW] = ...
                backward(layer, ~, ~, dLdZ, memory)
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
            %                             learnable parameter
            import saivdr.dcnn.*
            nrows = size(dLdZ,1);
            ncols = size(dLdZ,2);
            nlays = size(dLdZ,3);
            nSamples = size(dLdZ,5);
            nDecs = prod(layer.DecimationFactor);
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            nAngles = length(layer.Angles);
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
            anglesW = layer.Angles(1:nAngles/2);
            anglesU = layer.Angles(nAngles/2+1:end);
            
            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            W0 = fcn_orthonormalmatrixgenerate(anglesW,muW,0);
            U0 = fcn_orthonormalmatrixgenerate(anglesU,muU,0);
            adldz_ = permute(dLdZ,[4 1 2 3 5]);
            cdLd_ = reshape(adldz_,nDecs,nrows*ncols*nlays*nSamples);
            cdLd_upp = W0(:,1:nDecs/2)*cdLd_(1:nDecs/2,:);
            cdLd_low = U0(:,1:nDecs/2)*cdLd_(nDecs/2+1:nDecs,:);
            adLd_ = reshape([cdLd_upp;cdLd_low],...
                pa+ps,nrows,ncols,nlays,nSamples);
            dLdX = ipermute(adLd_,[4 1 2 3 5]);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dLdW = zeros(nAngles,1,'like',dLdZ);
            dldz_ = permute(dLdZ,[4 1 2 3 5]);
            dldz_upp = reshape(dldz_(1:nDecs/2,:,:,:),nDecs/2,nrows*ncols*nlays*nSamples);
            dldz_low = reshape(dldz_(nDecs/2+1:nDecs,:,:,:),nDecs/2,nrows*ncols*nlays*nSamples);
            for iAngle = 1:nAngles/2
                dW0_T = transpose(fcn_orthonormalmatrixgenerate(anglesW,muW,iAngle));
                dU0_T = transpose(fcn_orthonormalmatrixgenerate(anglesU,muU,iAngle));
                a_ = permute(memory,[4 1 2 3 5]);
                c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols*nlays*nSamples);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nlays*nSamples);
                d_upp = dW0_T(1:nDecs/2,:)*c_upp;
                d_low = dU0_T(1:nDecs/2,:)*c_low;
                dLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                dLdW(nAngles/2+iAngle) = sum(dldz_low.*d_low,'all');
            end
        end
        
        
    end
    
    
end

