classdef adjointMeasurement2dLayer < nnet.layer.Layer
    % adjointMeasurment2dLayer
    %
    % SSCB -> SSCB
    %
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    %
    properties
        Sigma
        PsfSize
        PadOption
        DecimationFactor
    end
        
    properties (Hidden)
        psf
    end
    
    properties (Access=private, Hidden)
        phase
    end
    
    methods
        function layer = adjointMeasurement2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Sigma',2)
            addParameter(p,'PsfSize',9);
            addParameter(p,'PadOption',0);
            addParameter(p,'DecimationFactor',1);
            parse(p,varargin{:})
            
            % Set layer name.
            layer.Name = p.Results.Name;
            layer.Sigma = p.Results.Sigma;
            layer.PadOption = p.Results.PadOption;
            layer.PsfSize = p.Results.PsfSize;
            layer.DecimationFactor = p.Results.DecimationFactor;
            
            %
            if isscalar(layer.PsfSize)
                layer.PsfSize = [1 1]*layer.PsfSize;
            end
            if isscalar(layer.DecimationFactor)
                layer.DecimationFactor = [1 1]*layer.DecimationFactor;
            end
            
            %
            layer.psf = fspecial('gaussian',...
                layer.PsfSize,layer.Sigma);
            
            % Set layer description.
            layer.Description = "Adjoint of measurement process " + ...
                "PsfSize: [" + layer.PsfSize(1) + " " + layer.PsfSize(2) + "], " + ...
                "PadOption: " + layer.PadOption + ", " + ...
                "Stride: [" + layer.DecimationFactor(1) + " " + layer.DecimationFactor(2) + "])";
            
            %
            layer.phase = mod(layer.PsfSize+1,[2 2]);              
        end
        
        function Z = predict(layer, X)
            padopt = layer.PadOption;
            decfactor = layer.DecimationFactor;
            ph = layer.phase;
            upsample2 = @(x,m) ipermute(upsample(permute(upsample(x,m(1),ph(1)),[2 1 3 4]),m(2),ph(2)),[2 1 3 4]);
            szBatch = size(X,4);
            Y = upsample2(X,decfactor);
            Z = zeros(size(Y),'like',X);
            for idx = 1:szBatch
                y = Y(:,:,:,idx);
                Z(:,:,:,idx) = imfilter(y,layer.psf,'corr',padopt);
            end
        end        
        
        function dLdX = backward(layer,~,~,dLdZ,~)
            padopt = layer.PadOption;
            decfactor = layer.DecimationFactor;
            downsample2 = @(x,m) ipermute(downsample(permute(downsample(x,m(1)),[2 1 3 4]),m(2)),[2 1 3 4]);
            szBatch = size(dLdZ,4);
            dldy = zeros(size(dLdZ),'like',dLdZ);
            for idx = 1:szBatch
                dldz = dLdZ(:,:,:,idx);
                dldy(:,:,:,idx) = imfilter(dldz,layer.psf,'conv',padopt);
            end
            dLdX = downsample2(dldy,decfactor);
        end
  
    end
    
end