classdef learnableScalingLayer < nnet.layer.Layer
    % LEARNABLESCALINGLAYER
    % 
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    % 
    
    properties(Learnable)
        % Layer learnable parameters
        Scale
    end
    
    methods
        function layer = learnableScalingLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Scale',1)
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            
            % Initialize scaling coefficient.
            layer.Scale = p.Results.Scale;           
            layer.NumInputs = 1;

            % Set layer description.
            layer.Description = "Learnable scaling ( " + layer.Scale + " ).";
        end
        
        function Z = predict(layer, X)            
            Z = layer.Scale * X ;
        end
    end
end