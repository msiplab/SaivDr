classdef negativeLayer < nnet.layer.Layer
    % NEGATIVELAYER
    % 
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    % 
       
    methods
        function layer = negativeLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            
            % Initialize scaling coefficient.        
            layer.NumInputs = 1;

            % Set layer description.
            layer.Description = "Negative";
        end
        
        function Z = predict(~, X)            
            Z = -X ;
        end
    end
end