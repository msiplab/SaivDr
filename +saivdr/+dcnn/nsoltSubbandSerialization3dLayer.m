classdef nsoltSubbandSerialization3dLayer < nnet.layer.Layer
    %NSOLTSUBBANDSERIALIZATION2DLAYER
    %
    %   複数コンポーネント入力 (SSSCB):（ツリーレベル数）
    %      nRowsLv1 x nColsLv1 x nLaysLv1 x nChsTotal x nSamples
    %      nRowsLv2 x nColsLv2 x nLaysLv2 x (nChsTotal-1) x nSamples
    %       :
    %      nRowsLvN x nColsLvN x nLaysLvN x (nChsTotal-1) x nSamples
    %
    %   １コンポーネント出力(SSSCB):
    %      nElements x 1 x 1 x 1 x nSamples
    %
    % Requirements: MATLAB R2020a
    %
    % Copyright (c) 2020, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    
    properties
        % (Optional) Layer properties.
        OriginalDimension
        NumberOfChannels
        DecimationFactor
        NumberOfLevels
        Scales
        % Layer properties go here.
    end
    
    methods
        function layer = nsoltSubbandSerialization3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsoltx.ChannelGroup
            
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'OriginalDimension',[8 8 8]);
            addParameter(p,'NumberOfChannels',[4 4]);
            addParameter(p,'DecimationFactor',[2 2 2]);
            addParameter(p,'NumberOfLevels',1);
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.OriginalDimension = p.Results.OriginalDimension;
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.NumberOfLevels = p.Results.NumberOfLevels;
            layer.Name = p.Results.Name;

            nLevels = layer.NumberOfLevels;
            height = layer.OriginalDimension(Direction.VERTICAL);
            width = layer.OriginalDimension(Direction.HORIZONTAL);  
            depth = layer.OriginalDimension(Direction.DEPTH);
            layer.Description = "Subband serialization " ...
                + "(h,w,d) = (" ...
                + height + "," + width + "," + depth + "), "  ...                
                + "lv = " ...
                + nLevels + ", " ...
                + "(ps,pa) = (" ...
                + layer.NumberOfChannels(ChannelGroup.UPPER) + "," + layer.NumberOfChannels(ChannelGroup.LOWER) + "), "  ...
                + "(mv,mh,md) = (" ...
                + layer.DecimationFactor(Direction.VERTICAL) + "," + layer.DecimationFactor(Direction.HORIZONTAL) + "," + layer.DecimationFactor(Direction.DEPTH) + ")";
            layer.Type = '';
            inputNames = cell(1,layer.NumberOfLevels);
            for iLv = 1:nLevels
                inputNames{iLv} = [ 'Lv' num2str(iLv) '_SbIn' ];
            end            
            layer.InputNames = inputNames;

            %
            nChsTotal = sum(layer.NumberOfChannels);
            stride = layer.DecimationFactor;
            
            nrows = height*stride(Direction.VERTICAL).^(-nLevels);
            ncols = width*stride(Direction.HORIZONTAL).^(-nLevels);     
            nlays = depth*stride(Direction.DEPTH).^(-nLevels);                 
            layer.Scales = zeros(nLevels,4);            
            layer.Scales(1,:) = [nrows ncols nlays nChsTotal];
            for iRevLv = 2:nLevels
                layer.Scales(iRevLv,:) = ...
                    [nrows*stride(Direction.VERTICAL)^(iRevLv-1) ncols*stride(Direction.HORIZONTAL)^(iRevLv-1) nlays*stride(Direction.DEPTH)^(iRevLv-1) nChsTotal-1];
            end
        end
       
        function Z = predict(layer, varargin)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X           - Input data (1 component)
            % Outputs:
            %         Z1, Z2      - Outputs of layer forward function
            %  
            
            % Layer forward function for prediction goes here.
            nLevels = layer.NumberOfLevels;
            nSamples = size(varargin{1},5);
            nElements = sum(prod(layer.Scales,2));
            scales = layer.Scales;
            Z = zeros(nElements,1,1,1,nSamples,'like',varargin{1});
            for iSample = 1:nSamples
                x = zeros(nElements,1,'like',Z);
                sidx = 0;
                for iRevLv = 1:nLevels
                    nSubElements = prod(scales(iRevLv,:));
                    a = varargin{nLevels-iRevLv+1}(:,:,:,:,iSample);
                    x(sidx+1:sidx+nSubElements) = a(:);
                    sidx = sidx+nSubElements;
                end
                Z(:,1,1,1,iSample) = x;
            end
        end
        
         function varargout = backward(layer,varargin)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X           - Input data (1 component)
            % Outputs:
            %         Z1, Z2      - Outputs of layer forward function
            %
            nLevels = layer.NumberOfLevels;
            dLdZ = varargin{nLevels+2};            
            nChsTotal = sum(layer.NumberOfChannels);
            scales = layer.Scales;
            varargout = cell(1,nLevels);
            sidx = 0;
            for iRevLv = 1:nLevels
                if iRevLv == 1
                    wodc = 0;
                else
                    wodc = 1;
                end
                nSubElements = prod(scales(iRevLv,:));
                subHeight = scales(iRevLv,1);
                subWidth = scales(iRevLv,2);
                subDepth = scales(iRevLv,3);
                varargout{nLevels-iRevLv+1} = ...
                    reshape(dLdZ(sidx+1:sidx+nSubElements,:),...
                    subHeight,subWidth,subDepth,nChsTotal-wodc,[]);
                sidx = sidx + nSubElements;
            end
        end
    end
    
end

