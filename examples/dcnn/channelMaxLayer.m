classdef channelMaxLayer < nnet.layer.Layer
    % CHANNELMAXAYER
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
        function layer = channelMaxLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;

            % Set layer description.
            layer.Description = "Channel-wise max"; % for " + layer.NumberOfChannels + " channels";
        end
        
        function Z = predict(~, X)
            Z = max(X,[],3);
        end
        
        function dLdX = backward(~,X,Z,dLdZ,~)
            maxpos = (Z == X);
            fwdmax = bsxfun(@rdivide,maxpos,sum(maxpos,3));
            dLdX = bsxfun(@times,fwdmax,dLdZ);
        end
    end
end