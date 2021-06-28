classdef nsoltIntermediateRotation2dLayer < nnet.layer.Layer %#codegen
    %NSOLTINTERMEDIATEROTATION2DLAYER
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
        Mode
    end
    
    properties (Dependent)
        Mus
    end
    
    properties (Learnable,Dependent)
        Angles
    end
    
    properties (Access = private)
        PrivateAngles
        PrivateMus
        isUpdateRequested
    end
    
    properties (Hidden)
        Un
    end
    
    methods
        function layer = nsoltIntermediateRotation2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'NumberOfChannels',[])
            addParameter(p,'Angles',[])
            addParameter(p,'Mus',[])
            addParameter(p,'Mode','Synthesis')
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.Name = p.Results.Name;
            layer.Mode = p.Results.Mode;
            layer.Angles = p.Results.Angles;
            layer.Mus = p.Results.Mus;
            layer.Description = layer.Mode ...
                + " NSOLT intermediate rotation " ...
                + "(ps,pa) = (" ...
                + layer.NumberOfChannels(1) + "," ...
                + layer.NumberOfChannels(2) + ")";
            layer.Type = '';
            
            nChsTotal = sum(layer.NumberOfChannels);
            nAngles = (nChsTotal-2)*nChsTotal/8;
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
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            Un_ = layer.Un;
            Y = X; %permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            if strcmp(layer.Mode,'Analysis')
                Za = Un_*Ya;
            elseif strcmp(layer.Mode,'Synthesis')
                Za = Un_.'*Ya;
            else
                throw(MException('NsoltLayer:InvalidMode',...
                    '%s : Mode should be either of Synthesis or Analysis',...
                    layer.Mode))
            end
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            Z = Y; %ipermute(Y,[3 1 2 4]);
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
            %                             learnable parameter
            import saivdr.dcnn.get_fcn_orthmtxgen_diff
            
            % Layer backward function goes here.            
            nrows = size(dLdZ,2);
            ncols = size(dLdZ,3);
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);            
            nSamples = size(dLdZ,4);
            %
            if layer.isUpdateRequested
                layer = layer.updateParameters();
            end
            anglesU = layer.PrivateAngles;
            musU = cast(layer.PrivateMus,'like',anglesU);
            
            % dLdX = dZdX x dLdZ
            %Un = fcn_orthmtxgen(anglesU,musU,0);
            %[Un_,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,musU,0,[],[]);
            Un_ = layer.Un;
            dUnPst = bsxfun(@times,musU(:),Un_);
            dUnPre = eye(pa,'like',Un_);
            %
            dLdX = dLdZ; %permute(dLdZ,[3 1 2 4]);
            cdLd_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),...
                pa,nrows*ncols*nSamples);
            if strcmp(layer.Mode,'Analysis')
                cdLd_low = Un_.'*cdLd_low;
            else
                cdLd_low = Un_*cdLd_low;
            end
            dLdX(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,...
                pa,nrows,ncols,nSamples);
            %dLdX = dLdX; %ipermute(adLd_,[3 1 2 4]);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = get_fcn_orthmtxgen_diff(anglesU);
            nAngles = length(anglesU);
            dLdW = zeros(nAngles,1,'like',dLdZ);
            dVdW_X = zeros(size(X),'like',dLdZ);
            if isgpuarray(X)
                x_low = X(ps+1:ps+pa,:,:,:);
                for iAngle = uint32(1:nAngles)
                    [dUn,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,musU,iAngle,dUnPst,dUnPre);
                    if strcmp(layer.Mode,'Analysis')
                        dVdW_X(ps+1:ps+pa,:,:,:) = ...
                            pagefun(@mtimes,dUn,x_low);
                    else
                        dVdW_X(ps+1:ps+pa,:,:,:) = ...
                            pagefun(@mtimes,dUn.',x_low);
                    end
                    dLdW(iAngle) = sum(bsxfun(@times,dLdZ,dVdW_X),'all');
                end
            else
                x_low = reshape(X(ps+1:ps+pa,:,:,:),pa,[]);
                for iAngle = uint32(1:nAngles)
                    [dUn,dUnPst,dUnPre] = fcn_orthmtxgen_diff(anglesU,musU,iAngle,dUnPst,dUnPre);
                    
                    if strcmp(layer.Mode,'Analysis')
                        c_low = dUn*x_low;
                    else
                        c_low = dUn.'*x_low;
                    end
                    dVdW_X(ps+1:ps+pa,:,:,:) = reshape(c_low,pa,nrows,ncols,nSamples);
                    dLdW(iAngle) = sum(bsxfun(@times,dLdZ,dVdW_X),'all');
                end
            end
        end
        
        function angles = get.Angles(layer)
            angles = layer.PrivateAngles;
        end
        
        function mus = get.Mus(layer)
            mus = layer.PrivateMus;
        end
        
        function layer = set.Angles(layer,angles)
            nChsTotal = sum(layer.NumberOfChannels);
            nAngles = (nChsTotal-2)*nChsTotal/8;
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
            pa = layer.NumberOfChannels(2);
            if isempty(mus)
                mus = ones(pa,1);   
            elseif isscalar(mus)
                mus = mus*ones(pa,1,'like',mus);   
            end
            %
            layer.PrivateMus = mus;
            %layer = layer.updateParameters();
            layer.isUpdateRequested = true;
        end
        
        function layer = updateParameters(layer)
            import saivdr.dcnn.get_fcn_orthmtxgen
            anglesU = layer.PrivateAngles;
            musU = cast(layer.PrivateMus,'like',anglesU);
            fcn_orthmtxgen = get_fcn_orthmtxgen(anglesU);
            layer.Un = fcn_orthmtxgen(anglesU,musU);
            layer.isUpdateRequested = false;
        end
        
    end

end

