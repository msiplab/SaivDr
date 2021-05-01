classdef sfthLayer < nnet.layer.Layer 
    % SFTHLAYER
    %
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    % 

    properties
        NumberOfChannels
    end
    
    properties(Learnable)
        % Layer learnable parameters
        % Scaling coefficient
        Lambda
    end
    
    methods
        function layer = sfthLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Lambda',0)
            addParameter(p,'NumberOfChannels',1)
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            
            % Initialize scaling coefficient.
            layer.Lambda = permute(p.Results.Lambda.*ones(layer.NumberOfChannels,1),[2 3 1]);            
            
            % Set layer description.
            layer.Description = "Soft-thresholding for " + layer.NumberOfChannels + " channels";
        end
        
        function Z = predict(layer, X)
            nc = abs(X)-layer.Lambda;
            nc = (nc+abs(nc))/2;
            Z = sign(X).*nc;
        end
    end
end