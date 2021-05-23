classdef scaledActivationLayer < nnet.layer.Layer
    % SUBBANDACTIVATIONLAYER
    %
    % SSCB -> SSCB
    % 
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    % 
    
    properties
        NumberOfChannels
        ActivationFunction
    end
    
    properties(Learnable)
        % Layer learnable parameters
        ScaleIn
        BiasIn
        ScaleOut
        BiasOut
    end
    
    methods
        function layer = scaledActivationLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'ScaleIn',1)
            addParameter(p,'BiasIn',0)
            addParameter(p,'ScaleOut',1)
            addParameter(p,'BiasOut',0)
            addParameter(p,'NumberOfChannels',1)
            addParameter(p,'ActivationFunction','None')
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            
            % Initialize scaling coefficient.
            layer.ScaleIn = permute(p.Results.ScaleIn.*ones(layer.NumberOfChannels,1),[2 3 1]);            
            layer.BiasIn = permute(p.Results.BiasIn.*ones(layer.NumberOfChannels,1),[2 3 1]);   
            layer.ScaleOut = permute(p.Results.ScaleOut.*ones(layer.NumberOfChannels,1),[2 3 1]);            
            layer.BiasOut = permute(p.Results.BiasOut.*ones(layer.NumberOfChannels,1),[2 3 1]);
            layer.ActivationFunction = p.Results.ActivationFunction;
            layer.NumInputs = 1;

            % Set layer description.
            layer.Description = "Scaled activation " + ...
                "(# of Chs.: " + layer.NumberOfChannels + ", " + ...
                "Activation function: " + layer.ActivationFunction + ")";
        end
        
        function Z = predict(layer, X)            
            U = layer.ScaleIn.*(X + layer.BiasIn);
            if strcmpi(layer.ActivationFunction,'tanh')
                V = tanh(U);
            elseif strcmpi(layer.ActivationFunction,'relu')
                V = relu(U);
            elseif strcmpi(layer.ActivationFunction,'clippedrelu')
                W = relu(U)-1;
                V = -relu(-W)+1;
            elseif strcmpi(layer.ActivationFunction,'none')
                V = U;
            else
                error("Invalid activation function: " + layer.ActivationFunction);
            end
            Z = layer.ScaleOut.*V+layer.BiasOut;
        end
    end
end