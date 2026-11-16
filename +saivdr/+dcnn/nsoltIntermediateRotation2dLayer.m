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
        BoundaryOperation % 'Circular' or 'Termination'
        Direction         % 'Horizontal' or 'Vertical' (boundary for 'Termination')
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
            addParameter(p,'BoundaryOperation','Circular')
            addParameter(p,'Direction','')
            parse(p,varargin{:})
            
            % Boundary operation
            % 'Termination' replaces the rotation of the boundary slice
            % (first column for 'Horizontal', first row for 'Vertical')
            % with -I, as in saivdr.dictionary.nsoltx.NsoltAtomExtender2d.
            % See S. Muramatsu et al., IEEE Trans. Image Process.,
            % vol. 21, no. 4, pp. 2314-2318, 2012.
            % Each value is validated by its set method.
            layer.BoundaryOperation = p.Results.BoundaryOperation;
            layer.Direction = p.Results.Direction;
            layer.validateDirection_();

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
            % The boundary slice is rotated by -I for 'Termination'
            if strcmp(layer.BoundaryOperation,'Termination')
                Y = layer.negateBoundarySlice_(Y,X);
            end
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
            %import saivdr.dcnn.get_fcn_orthmtxgen_diff
            
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
            
            % For 'Termination', the boundary slice is rotated by the
            % fixed matrix -I, so it gives -dLdZ and does not contribute
            % to the angle gradients
            if strcmp(layer.BoundaryOperation,'Termination')
                dLdX = layer.negateBoundarySlice_(dLdX,dLdZ);
                dLdZ = layer.zeroBoundarySlice_(dLdZ);
            end

            % dLdWi = <dLdZ,(dVdWi)X>
            fcn_orthmtxgen_diff = saivdr.dcnn.get_fcn_orthmtxgen_diff(anglesU);
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
            %import saivdr.dcnn.get_fcn_orthmtxgen
            anglesU = layer.PrivateAngles;
            musU = cast(layer.PrivateMus,'like',anglesU);
            fcn_orthmtxgen = saivdr.dcnn.get_fcn_orthmtxgen(anglesU);
            layer.Un = fcn_orthmtxgen(anglesU,musU);
            layer.isUpdateRequested = false;
        end
        
    end

    methods

        function layer = set.BoundaryOperation(layer,boundaryOperation)
            if ~any(strcmp(boundaryOperation,{'Circular','Termination'}))
                throw(MException('NsoltLayer:InvalidBoundaryOperation',...
                    '%s : BoundaryOperation should be either of Circular or Termination',...
                    boundaryOperation))
            end
            layer.BoundaryOperation = boundaryOperation;
            layer.isUpdateRequested = true; %#ok<MCSUP>
        end

        function layer = set.Direction(layer,direction)
            % Empty Direction is allowed only for 'Circular'
            if ~isempty(direction) ...
                    && ~any(strcmp(direction,{'Horizontal','Vertical'}))
                throw(MException('NsoltLayer:InvalidDirection',...
                    '%s : Direction should be either of Horizontal or Vertical',...
                    direction))
            end
            layer.Direction = direction;
            layer.isUpdateRequested = true; %#ok<MCSUP>
        end

    end

    methods (Access = private)

        function validateDirection_(layer)
            % 'Termination' requires Direction
            if strcmp(layer.BoundaryOperation,'Termination') ...
                    && ~any(strcmp(layer.Direction,{'Horizontal','Vertical'}))
                throw(MException('NsoltLayer:InvalidDirection',...
                    'Direction should be either of Horizontal or Vertical for Termination'))
            end
        end

        function idx = boundarySliceIndex_(layer)
            % Index of the difference channels on the boundary slice
            layer.validateDirection_();
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            if strcmp(layer.Direction,'Horizontal')
                idx = {ps+1:ps+pa,':',1,':'}; % first column
            else % 'Vertical'
                idx = {ps+1:ps+pa,1,':',':'}; % first row
            end
        end

        function Y = negateBoundarySlice_(layer,Y,X)
            % Replace the boundary slice of Y with that of -X
            idx = layer.boundarySliceIndex_();
            Y(idx{:}) = -X(idx{:});
        end

        function Y = zeroBoundarySlice_(layer,Y)
            % Set the boundary slice of Y to zero
            idx = layer.boundarySliceIndex_();
            Y(idx{:}) = 0;
        end

    end

end

