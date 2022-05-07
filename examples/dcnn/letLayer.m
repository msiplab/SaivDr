classdef letLayer < nnet.layer.Layer
    % LETLAYER
    %
    % Reference: 
    % (21) in F. Luisier, T. Blu and M. Unser, "A New SURE Approach to Image Denoising: Interscale Orthonormal Wavelet Thresholding," in IEEE Transactions on Image Processing, vol. 16, no. 3, pp. 593-606, March 2007, doi: 10.1109/TIP.2007.891064.Example custom Soft-thresholding layer.
    %  
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    %
    
    properties
        NumberOfChannels
        IsInterScale
    end
    
    properties(Learnable)
        % Layer learnable parameters
        Sigma
        a1
        a2
        b1
        b2
    end
    
    methods
        function layer = letLayer(varargin) 
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Sigma',1)
            addParameter(p,'IsInterScale',true)
            addParameter(p,'NumberOfChannels',1)
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            layer.IsInterScale = p.Results.IsInterScale;
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            
            % Initialize scaling coefficient.
            layer.Sigma = permute(p.Results.Sigma.*ones(layer.NumberOfChannels,1),[2 3 1]);            
            %layer.Sigma = p.Results.Sigma;
            
            % Initialize scaling coefficient.
            layer.a1 = permute(ones(layer.NumberOfChannels,1),[2 3 1]);
            layer.a2 = permute(-ones(layer.NumberOfChannels,1),[2 3 1]);
            layer.b1 = permute(ones(layer.NumberOfChannels,1),[2 3 1]);
            layer.b2 = permute(zeros(layer.NumberOfChannels,1),[2 3 1]);
            
            %layer.NumInputs = 2;
            if layer.IsInterScale
                layer.InputNames = { 'child', 'parent' };
            else
                layer.NumInputs = 1;
            end
            
            % Set layer description.
            layer.Description = "LET for " + layer.NumberOfChannels + " channels";
        end
        
        function Z = predict(layer, varargin)
            X = varargin{1};
            V = exp(-X.^2./(12*layer.Sigma.^2));
            % codegen does not support dlarrays
            if layer.NumInputs < 2
                Z = (layer.a1 + layer.a2.*V).*X;
            else
                P = varargin{2};
                Y = dlresize(P,'OutputSize',size(X,1:2),'DataFormat','SSCB'); % Enlarged parent
                U = exp(-Y.^2./(12*layer.Sigma.^2));
                Z = U.*(layer.a1 + layer.a2.*V).*X ...
                    + (1-U).*(layer.b1 + layer.b2.*V).*X;
            end
        end
    end
end