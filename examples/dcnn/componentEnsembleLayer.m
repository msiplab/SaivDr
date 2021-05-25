classdef componentEnsembleLayer < nnet.layer.Layer
    %COMPONENTEnsembleLAYER
    %
    % SSCB,SSCB,...,SSCB -> SSCB
    %
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    %
    
    properties
        NumberOfComponents
        Mode
    end
    
    methods
        function layer = componentEnsembleLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfComponents',1)
            addParameter(p,'Mode','median')
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            layer.NumberOfComponents = p.Results.NumberOfComponents;
            layer.Mode = lower(p.Results.Mode);
            
            % Check # of components
            if mod(layer.NumberOfComponents,2) == 0
                error('# of components must be odd.')
            end
            %layer.NumInputs = 2;
            layer.NumInputs = layer.NumberOfComponents;
            
            % Check mode
            if ~strcmpi(layer.Mode,'median') && ...
                    ~strcmpi(layer.Mode,'max') && ...
                    ~strcmpi(layer.Mode,'min') && ...
                    ~strcmpi(layer.Mode,'absmax') 
                error('Mode should be MEDIAN, MAX, MIN or ABSMAX.')
            end
            
            % Set layer description.
            layer.Description = "Component ensemble " + ...
                "(# of components: " + layer.NumberOfComponents + ", " + ...
                "Mode: " + lower(layer.Mode) + ")";
        end
        
        function Z = predict(layer, varargin)
            if length(varargin) ~= layer.NumberOfComponents
                error('Invalid # of components')
            end
            if strcmpi(layer.Mode,'max')
                fcn = @(x) max(x,[],1);
            elseif strcmpi(layer.Mode,'min')
                fcn = @(x) min(x,[],1);
            elseif strcmpi(layer.Mode,'absmax')
                fcn = @(x) layer.absmax_(x);
            else
                fcn = @(x) median(x,1);
            end
            xx = zeros(layer.NumberOfComponents,numel(varargin{1}),'like',varargin{1});
            for iCmp = 1:layer.NumberOfComponents
                x = varargin{iCmp};
                xx(iCmp,:) = x(:).';
            end
            if isdlarray(xx)
                z = dlarray(fcn(extractdata(xx)));
            else
                z = fcn(xx);
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
    
    methods(Access = private)
        function y = absmax_(~,x)
            [xm,im] = max(abs(x),[],1);
            %{
            sx = zeros(size(xm),'like',xm);
            for iCol = 1:length(im)
                iRow = im(iCol);
                sx(iCol) = sign(x(iRow,iCol));
            end
            %}
            idx = im+(0:length(im)-1)*height(x);
            sx = sign(x(idx));
            y = sx.*xm;
        end
    end
end

