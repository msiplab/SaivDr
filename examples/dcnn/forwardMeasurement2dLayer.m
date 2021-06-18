classdef forwardMeasurement2dLayer < nnet.layer.Layer
    % forwardMeasurment2dLayer
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
        function layer = forwardMeasurement2dLayer(varargin)
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
            layer.Description = "Forward measurement process " + ...
                "PsfSize: [" + layer.PsfSize(1) + " " + layer.PsfSize(2) + "], " + ...
                "PadOption: " + layer.PadOption + ", " + ...
                "Stride: [" + layer.DecimationFactor(1) + " " + layer.DecimationFactor(2) + "])";
            
            %
            layer.phase = mod(layer.PsfSize+1,[2 2]);              
        end
        
        function Z = predict(layer, X)
            padopt = layer.PadOption;
            decfactor = layer.DecimationFactor;
            downsample2 = @(x,m) ipermute(downsample(permute(downsample(x,m(1)),[2 1 3 4]),m(2)),[2 1 3 4]);
            szBatch = size(X,4);
            if any(mod([size(X,1) size(X,2)],decfactor))
                warning('The image size is not a multiple of decimation factor.');
            end
            y = zeros(size(X),'like',X);
            for idx = 1:szBatch
                x = X(:,:,:,idx);
                y(:,:,:,idx) = imfilter(x,layer.psf,'conv',padopt);
            end
            Z = downsample2(y,decfactor);
        end
        
        function dLdX = backward(layer,X,~,dLdZ,~)
            padopt = layer.PadOption;
            decfactor = layer.DecimationFactor;
            ph = layer.phase;
            upsample2 = @(x,m) ipermute(upsample(permute(upsample(x,m(1),ph(1)),[2 1 3 4]),m(2),ph(2)),[2 1 3 4]);
            szBatch = size(dLdZ,4);
            dLdY = upsample2(dLdZ,decfactor);
            dLdX = zeros(size(X),'like',dLdZ);
            for idx = 1:szBatch
                dldy = dLdY(:,:,:,idx);
                dLdX(:,:,:,idx) = imfilter(dldy,layer.psf,'corr',padopt);
            end
        end
        
        function adjoint = createAdjointLayer(layer)
            adjoint = adjointMeasurement2dLayer(...
                'Name',[layer.Name '~'],...
                'Sigma',layer.Sigma,...
                'PadOption',layer.PadOption,...
                'PsfSize',layer.PsfSize,...
                'DecimationFactor',layer.DecimationFactor);
        end
    end
    
end