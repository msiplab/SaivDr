classdef channelMeanLayer < nnet.layer.Layer
    % CHANNELMEANLAYER
    %
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    % 
    
    properties
    end
    
    properties(Learnable)
        % Layer learnable parameters
    end
    
    methods
        function layer = channelMeanLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;

            % Set layer description.
            layer.Description = "Channel-wise mean"; % for " + layer.NumberOfChannels + " channels";
        end
        
        function Z = predict(~, X)
            Z = mean(X,3);
        end
    end
end