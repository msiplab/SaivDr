classdef minLayer < nnet.layer.Layer
    % minLAYER
    %
    % SSCB -> SSCB
    %
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    %
   
    properties
        NumberOfChannels
        PadOption
        WindowSize
    end
    
    properties (Access=private, Hidden)

    end
    
    methods
        function layer = minLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfChannels',1)
            addParameter(p,'PadOption','Symmetric');
            addParameter(p,'WindowSize',[3 3]);
            parse(p,varargin{:})

            % Set layer name.
            layer.Name = p.Results.Name;
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.PadOption = p.Results.PadOption;
            layer.WindowSize = p.Results.WindowSize;
            
            % Check PadOption
            if ~strcmp(layer.PadOption,'Symmetric')
                layer.PadOption = 'Zeros';
            end
            
            % Set layer description.
            layer.Description = "min filter " +...
                "(# of Chs.: " + layer.NumberOfChannels + ", " + ...
                "WindowSize: [" + layer.WindowSize(1) + " " + layer.WindowSize(2) + "], " + ...
                "PadOption: " + layer.PadOption + ")";
        end
        
        function Z = predict(layer, X)
            if size(X,3) ~= layer.NumberOfChannels
                error('Invalid # of channels')
            end
            szBatch = size(X,4);
            if isgpuarray(X)
                dtype = underlyingType(X);
                x = gather(X);
            else
                dtype = class(X);
                x = X;
            end
            %
            z = zeros(size(x),dtype);
            padopt = layer.PadOption;
            windowSize_ = layer.WindowSize;
            nPads = (windowSize_-1)/2;
            nDims = [size(X,1) size(X,2)];
            for idx = 1:szBatch
                for iChannel = 1:layer.NumberOfChannels
                    if strcmpi(padopt,'Symmetric')
                        % w = medfilt2(x(:,:,iChannel,idx),windowSize_,'symmetric'); 
                        u = padarray(x(:,:,iChannel,idx),nPads,'both','symmetric');
                        v = ordfilt2(u,1,ones(windowSize_,dtype));
                        w = v(nPads(1)+1:nPads(1)+nDims(1),nPads(2)+1:nPads(2)+nDims(2));
                    else
                        w = ordfilt2(x(:,:,iChannel,idx),1,ones(windowSize_,dtype));
                    end
                    z(:,:,iChannel,idx) = w; 
                end
            end
            if isgpuarray(X)
                Z = gpuArray(z);
            else
                Z = z;
            end
        end
        
        function dLdX = backward(layer,X,Z,dLdZ,~)
            windowSize_ = layer.WindowSize;
            nChannels_ = layer.NumberOfChannels;
            nRows = windowSize_(1);
            nCols = windowSize_(2);
            nDims = [size(dLdZ,1) size(dLdZ,2)];
            nPads = (windowSize_-1)/2;
            szBatch = size(dLdZ,4);
            padopt = layer.PadOption;
            dldz = dLdZ;
            dldx = zeros(size(dldz),'like',dldz);
            for idx = 1:szBatch
                for iChannel = 1:nChannels_
                    ax = dldz(:,:,iChannel,idx);
                    xx = im2col(padarray(X(:,:,iChannel,idx),nPads,'both'),...
                        windowSize_,'sliding');
                    z = Z(:,:,iChannel,idx);
                    zx = bsxfun(@eq,xx,z(:).');
                    fwdmed = bsxfun(@rdivide,zx,sum(zx));
                    %
                    bx = zeros(nDims+windowSize_-1,'like',dldz);
                    for iCol = 1:nCols
                        for iRow = 1:nRows
                            bx(iRow:iRow+nDims(1)-1,iCol:iCol+nDims(2)-1) = ...
                                bsxfun(@plus,bx(iRow:iRow+nDims(1)-1,iCol:iCol+nDims(2)-1),...
                                bsxfun(@times,reshape(fwdmed(iRow+(iCol-1)*nRows,:),nDims),ax));
                        end
                    end
                    %
                    if strcmpi(padopt,'symmetric')
                        bx(nPads(1)+1:2*nPads(1),:) = ...
                            bsxfun(@plus,bx(nPads(1)+1:2*nPads(1),:),bx(nPads(1):-1:1,:));
                        bx(end-nPads(1):-1:end-2*nPads(1)+1,:) = ...
                            bsxfun(@plus,bx(end-nPads(1):-1:end-2*nPads(1)+1,:),bx(end-nPads(1)+1:end,:));
                        bx(:,nPads(2)+1:2*nPads(2)) = ...
                            bsxfun(@plus,bx(:,nPads(2)+1:2*nPads(2)),bx(:,nPads(2):-1:1));
                        bx(:,end-nPads(2):-1:end-2*nPads(2)+1) = ...
                            bsxfun(@plus,bx(:,end-nPads(2):-1:end-2*nPads(2)+1),bx(:,end-nPads(2)+1:end));
                    end
                    %
                    dldx(:,:,iChannel,idx) = ...
                        bx(nPads(1)+1:nPads(1)+nDims(1),nPads(2)+1:nPads(2)+nDims(2));
                end
            end
            dLdX = dldx;
        end
    end
end

