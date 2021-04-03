classdef sfthLayer < nnet.layer.Layer 
    % Example custom Soft-thresholding layer.

    properties %(Learnable)
        % Layer learnable parameters
        % Scaling coefficient
        Lambda
    end
    
    methods
        function layer = sfthLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Lambda',0)
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;

            % Initialize scaling coefficient.
            layer.Lambda = p.Results.Lambda;            
            
            % Set layer description.
            layer.Description = "Soft-thresholding with Lamda = " + layer.Lambda;
        end
        
        function Z = predict(layer, X)
            nc = abs(X)-layer.Lambda;
            nc = (nc+abs(nc))/2;
            Z = sign(X).*nc;
        end
    end
end