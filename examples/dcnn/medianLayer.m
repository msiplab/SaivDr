classdef medianLayer < nnet.layer.Layer
    % MEDIANLAYER
    %
    % Copyright (c) Shogo MURAMATSU, 2021
    % All rights reserved.
    %

    
    properties
        NumberOfChannels
    end
    
    properties (Access=private, Hidden)
        windowSize = [3 3];
    end
    
    methods
        function layer = medianLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfChannels',1)
            parse(p,varargin{:})

            % Set layer name.
            layer.Name = p.Results.Name;
            layer.NumberOfChannels = p.Results.NumberOfChannels;

            % Set layer description.
            layer.Description = "Median filter for " + layer.NumberOfChannels + " channels";
        end
        
        function Z = predict(layer, X)
            if size(X,3) ~= layer.NumberOfChannels
                error('Invalid # of channels')
            end
            szBatch = size(X,4);
            if isdlarray(X)
                x = extractdata(X);
            else
                x = X;
            end
            %
            z = zeros(size(x),'like',x);
            for idx = 1:szBatch
                for iChannel = 1:layer.NumberOfChannels
                    z(:,:,iChannel,idx) = medfilt2(x(:,:,iChannel,idx),layer.windowSize);
                end
            end
            if isdlarray(X)
                Z = dlarray(z);
            else
                Z = z;
            end
        end
        
        function dLdX = backward(layer,X,Z,dLdZ,~)
            windowSize_ = layer.windowSize;
            nChannels_ = layer.NumberOfChannels;
            nRows = windowSize_(1);
            nCols = windowSize_(2);
            nDims = [size(dLdZ,1) size(dLdZ,2)];
            nPads = (windowSize_-1)/2;
            szBatch = size(dLdZ,4);
            %
            if isdlarray(dLdZ)
                dldz = extractdata(dLdZ);
            else
                dldz = dLdZ;
            end
            %
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
                    dldx(:,:,iChannel,idx) = ...
                        bx(nPads(1)+1:nPads(1)+nDims(1),nPads(2)+1:nPads(2)+nDims(2));
                end
            end
            %
            if isdlarray(dLdZ)
                dLdX = dlarray(dldx);
            else
                dLdX = dldx;
            end
        end
    end
end

