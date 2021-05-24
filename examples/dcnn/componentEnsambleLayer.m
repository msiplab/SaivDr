classdef componentEnsambleLayer < nnet.layer.Layer
    %COMPONENTENSAMBLELAYER
    %
    % SSCB,SSCB,...,SSCB -> SSCB
    %
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    %
    
    properties
        NumberOfComponents
    end
    
    methods
        function layer = componentEnsambleLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfComponents',1)
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            layer.NumberOfComponents = p.Results.NumberOfComponents;
            
            % Check # of components
            if mod(layer.NumberOfComponents,2) == 0
                error('# of components must be odd.');
            end
            %layer.NumInputs = 2;
            layer.NumInputs = layer.NumberOfComponents;
            
            % Set layer description.
            layer.Description = "Component ensamble " + ...
                "(# of components: " + layer.NumberOfComponents + ")";
        end
        
        function Z = predict(layer, varargin)
            if length(varargin) ~= layer.NumberOfComponents
                error('Invalid # of components')
            end
            xx = zeros(layer.NumberOfComponents,numel(varargin{1}),'like',varargin{1});
            for iCmp = 1:layer.NumberOfComponents
                x = varargin{iCmp};
                xx(iCmp,:) = x(:).';
            end
            if isdlarray(xx)
                z = dlarray(median(extractdata(xx),1));
            else
                z = median(xx,1);
            end
            sz = size(varargin{1});
            Z = reshape(z,sz);
        end
        
        
        function varargout = backward(layer,varargin)
            idxZ = layer.NumInputs + 1;
            idxdLdZ = layer.NumInputs + layer.NumOutputs + 1;
            %
            Z = varargin{idxZ};
            dLdZ = varargin{idxdLdZ};            
            xx = zeros(layer.NumInputs,numel(varargin{1}),'like',varargin{1});
            %
            for iCmp = 1:layer.NumInputs
                x = varargin{iCmp};
                xx(iCmp,:) = x(:).';
            end
            z = Z(:).';
            zx = bsxfun(@eq,xx,z);
            fwdmed = bsxfun(@rdivide,zx,sum(zx,1));
            varargout = cell(1,layer.NumInputs);
            for iCmp = 1:layer.NumInputs
                cmap = fwdmed(iCmp,:);
                wmap = bsxfun(@times,cmap,dLdZ(:).');
                varargout{iCmp} = reshape(wmap,size(Z));
            end
        end
        
    end
end

