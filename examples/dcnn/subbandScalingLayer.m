classdef subbandScalingLayer < nnet.layer.Layer
    % 
    
    properties
        NumberOfChannels
    end
    
    properties(Learnable)
        % Layer learnable parameters
        Scale
        Bias
    end
    
    methods
        function layer = subbandScalingLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Scale',1)
            addParameter(p,'Bias',0)
            addParameter(p,'NumberOfChannels',1)
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            
            % Initialize scaling coefficient.
            layer.Scale = permute(p.Results.Scale.*ones(layer.NumberOfChannels,1),[2 3 1]);            
            layer.Bias = permute(p.Results.Bias.*ones(layer.NumberOfChannels,1),[2 3 1]);            
            
            layer.NumInputs = 1;

            % Set layer description.
            layer.Description = "Scaling and bias for " + layer.NumberOfChannels + " channels";
        end
        
        function Z = predict(layer, X)
            Z = layer.Scale.*X + layer.Bias;
        end
    end
end