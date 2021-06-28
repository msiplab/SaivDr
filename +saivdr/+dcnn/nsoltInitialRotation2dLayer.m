classdef nsoltInitialRotation2dLayer < nnet.layer.Layer %#codegen
    %NSOLTINITIALROTATION2DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nDecs x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x nRows x nCols x nSamples
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2020-2021, Shogo MURAMATSU
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
    end
    
    properties (Dependent)
        NoDcLeakage
    end
    
    properties (Dependent)
        Mus
    end
    
    properties (Learnable,Dependent)
        Angles
    end
    
    properties (Access = private)
        PrivateNoDcLeakage
        PrivateAngles
        PrivateMus
        isUpdateRequested
    end
    
    properties (Hidden)
        W0
        U0
    end
        
    methods
        
        function layer = nsoltInitialRotation2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
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
                + "(mv,mh) = (" ...
                + layer.DecimationFactor(1) + "," ...
                + layer.DecimationFactor(2) + ")";
            layer.Type = '';

            nChsTotal = sum(layer.NumberOfChannels);            
            nAngles = (nChsTotal-2)*nChsTotal/4;
            if length(layer.PrivateAngles)~=nAngles
                error('Invalid # of angles')
            end
            
            layer = layer.updateParameters();
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
            
            % Layer forward function for prediction goes here.
            nrows = size(X,2);
            ncols = size(X,3);            
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            nSamples = size(X,4);
            stride = layer.DecimationFactor;
            nDecs = prod(stride);
            nChsTotal = ps + pa;
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            W0_ = layer.W0;
            U0_ = layer.U0;
            %Y = reshape(permute(X,[3 1 2 4]),nDecs,nrows*ncols*nSamples);
            Y = reshape(X,nDecs,nrows*ncols*nSamples);
            Zs = W0_(:,1:ceil(nDecs/2))*Y(1:ceil(nDecs/2),:);
            Za = U0_(:,1:floor(nDecs/2))*Y(ceil(nDecs/2)+1:end,:);
            %Z = ipermute(reshape([Zs;Za],nChsTotal,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            Z = reshape([Zs;Za],nChsTotal,nrows,ncols,nSamples);
            
        end
        
        function [dLdX, dLdW] = backward(layer, X, ~, dLdZ, ~)
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
            import saivdr.dcnn.get_fcn_orthmtxgen_diff
            
            nrows = size(dLdZ,2);
            ncols = size(dLdZ,3);            
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            nAngles = length(layer.PrivateAngles);
            nSamples = size(dLdZ,4);
            stride = layer.DecimationFactor;
            nDecs = prod(stride);
            %{
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
            %}
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            angles = layer.PrivateAngles;
            mus = cast(layer.Mus,'like',angles);            
            anglesW = angles(1:nAngles/2);
            anglesU = angles(nAngles/2+1:end);
            muW = mus(1:ps);
            muU = mus(ps+1:end);
            %[W0_,dW0Pst,dW0Pre] = fcn_orthmtxgen_diff(anglesW,muW,0,[],[]);
            %[U0_,dU0Pst,dU0Pre] = fcn_orthmtxgen_diff(anglesU,muU,0,[],[]);
            W0_ = layer.W0; %transpose(fcn_orthmtxgen(anglesW,muW,0));
            U0_ = layer.U0; %transpose(fcn_orthmtxgen(anglesU,muU,0));
            W0T = transpose(W0_);
            U0T = transpose(U0_);
            %if isdlarray(W0_)
            %    dW0Pst = dlarray(muW(:).*W0_);
            %    dU0Pst = dlarray(muU(:).*U0_);
            %    dW0Pre = dlarray(eye(ps,W0_.underlyingType));
            %    dU0Pre = dlarray(eye(pa,U0_.underlyingType));
            %else
            dW0Pst = bsxfun(@times,muW(:),W0_);
            dU0Pst = bsxfun(@times,muU(:),U0_);
            dW0Pre = eye(ps,'like',W0_);
            dU0Pre = eye(pa,'like',U0_);
            %end
            
            % Layer backward function goes here.
            % dLdX = dZdX x dLdZ
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            Zsa = [ W0T(1:ceil(nDecs/2),:)*Ys; U0T(1:floor(nDecs/2),:)*Ya ];
            %dLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            dLdX = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            fcn_orthmtxgen_diff = get_fcn_orthmtxgen_diff(angles);                        
            % dLdWi = <dLdZ,(dVdWi)X>
            dLdW = zeros(nAngles,1,'like',dLdZ);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols*nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:ceil(nDecs/2),:,:,:),ceil(nDecs/2),nrows*ncols*nSamples);
            c_low = reshape(a_(ceil(nDecs/2)+1:nDecs,:,:,:),floor(nDecs/2),nrows*ncols*nSamples);
            for iAngle = uint32(1:nAngles/2)
                %dW0 = fcn_orthmtxgen(anglesW,muW,iAngle);
                %dU0 = fcn_orthmtxgen(anglesU,muU,iAngle);
                [dW0,dW0Pst,dW0Pre] = fcn_orthmtxgen_diff(anglesW,muW,iAngle,dW0Pst,dW0Pre);
                [dU0,dU0Pst,dU0Pre] = fcn_orthmtxgen_diff(anglesU,muU,iAngle,dU0Pst,dU0Pre);
                d_upp = dW0(:,1:ceil(nDecs/2))*c_upp;
                d_low = dU0(:,1:floor(nDecs/2))*c_low;
                dLdW(iAngle) = sum(bsxfun(@times,dldz_upp,d_upp),'all');
                dLdW(nAngles/2+iAngle) = sum(bsxfun(@times,dldz_low,d_low),'all');
            end
        end
        
        function nodcleak = get.NoDcLeakage(layer)
            nodcleak = layer.PrivateNoDcLeakage;
        end
        
        function angles = get.Angles(layer)
            angles = layer.PrivateAngles;
        end
        
        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
        end
        
        function layer = set.NoDcLeakage(layer,nodcleak)
            layer.PrivateNoDcLeakage = nodcleak;
            %
            layer.isUpdateRequested = true;
        end
        
        function layer = set.Angles(layer,angles)
            nChsTotal = sum(layer.NumberOfChannels);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            if isempty(angles)
                angles = zeros(nAngles,1);
            elseif isscalar(angles)
                angles = angles*ones(nAngles,1,'like',angles);
            end
            %
            layer.PrivateAngles = angles;
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = set.Mus(layer,mus)
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            %
            if isempty(mus)
                mus = ones(ps+pa,1);
            elseif isscalar(mus)
                mus = mus*ones(ps+pa,1);
            end
            %
            layer.PrivateMus = mus;
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = updateParameters(layer)
            import saivdr.dcnn.get_fcn_orthmtxgen
            ps = layer.NumberOfChannels(1);
            %
            if layer.NoDcLeakage
                layer.PrivateMus(1) = 1;                
                layer.PrivateAngles(1:ps-1) = ...
                    zeros(ps-1,1,'like',layer.PrivateAngles);
            end            
            %
            angles = layer.PrivateAngles;            
            mus = cast(layer.PrivateMus,'like',angles);
            nAngles = length(angles);
            muW = mus(1:ps);
            muU = mus(ps+1:end);
            anglesW = angles(1:nAngles/2);
            anglesU = angles(nAngles/2+1:end);
            fcn_orthmtxgen = get_fcn_orthmtxgen(angles);                        
            layer.W0 = fcn_orthmtxgen(anglesW,muW);
            layer.U0 = fcn_orthmtxgen(anglesU,muU);
            layer.isUpdateRequested = false;
        end
        
    end
    

    
end

