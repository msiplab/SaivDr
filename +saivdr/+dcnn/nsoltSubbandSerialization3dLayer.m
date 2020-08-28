classdef nsoltSubbandSerialization3dLayer < nnet.layer.Layer
    %NSOLTSUBBANDSERIALIZATION2DLAYER
    %
    %   複数コンポーネント入力 (SSSCB):（ツリーレベル数）
    %      (nChsTotal-1) x nRowsLv1 x nColsLv1 x nLaysLv1 x nSamples
    %      (nChsTotal-1) x nRowsLv2 x nColsLv2 x nLaysLv2 x nSamples
    %       :
    %      (nChsTotal-1) x nRowsLvN x nColsLvN x nLaysLvN x nSamples
    %      1 x nRowsLvN x nColsLvN x nSamples
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
            
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'OriginalDimension',[8 8 8]);
            addParameter(p,'NumberOfChannels',[4 4]);
            addParameter(p,'DecimationFactor',[2 2 2]);
            addParameter(p,'NumberOfLevels',1);
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.NumberOfLevels = p.Results.NumberOfLevels;
            layer.Name = p.Results.Name;
            layer = layer.setOriginalDimension(p.Results.OriginalDimension);
            layer.Type = '';            

            nLevels = layer.NumberOfLevels;            
            %inputNames = cell(1,layer.NumberOfLevels);
            inputNames = cell(1,layer.NumberOfLevels+1);
            for iLv = 1:nLevels
                inputNames{iLv} = [ 'Lv' num2str(iLv) '_SbAcIn' ];
            end            
            inputNames{nLevels+1} = [ 'Lv' num2str(iLv) '_SbDcIn' ]; %
            layer.InputNames = inputNames;

        end
       
        function layer = setOriginalDimension(layer,orgdim)
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsoltx.ChannelGroup
            layer.OriginalDimension = orgdim;
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

            %
            nChsTotal = sum(layer.NumberOfChannels);
            stride = layer.DecimationFactor;
            
            nrows = height*stride(Direction.VERTICAL).^(-nLevels);
            ncols = width*stride(Direction.HORIZONTAL).^(-nLevels);     
            nlays = depth*stride(Direction.DEPTH).^(-nLevels);                 
            layer.Scales = zeros(nLevels,4);            
            %layer.Scales(1,:) = [nrows ncols nlays nChsTotal];
            layer.Scales(1,:) = [nrows ncols nlays 1];
            for iRevLv = 1:nLevels %2:nLevels
                %layer.Scales(iRevLv,:) = ...
                layer.Scales(iRevLv+1,:) = ...
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
                nSubElements = prod(scales(1,:));
                a = varargin{nLevels+1}(:,:,:,:,iSample);
                x(1:nSubElements) = a(:);
                sidx = sidx+nSubElements;
                for iRevLv = 1:nLevels
                    %nSubElements = prod(scales(iRevLv,:));
                    nSubElements = prod(scales(iRevLv+1,:));
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
            %dLdZ = varargin{nLevels+2};            
            dLdZ = varargin{nLevels+3};            
            nChsTotal = sum(layer.NumberOfChannels);
            scales = layer.Scales;
            %varargout = cell(1,nLevels);
            varargout = cell(1,nLevels+1);
            sidx = 0;
            nSubElements = prod(scales(1,:));
            subHeight = scales(1,1);
            subWidth = scales(1,2);
            subDepth = scales(1,3);
            varargout{nLevels+1} = ...
                reshape(dLdZ(1:nSubElements,:),...
                subHeight,subWidth,subDepth,1,[]);
            sidx = sidx + nSubElements;
            for iRevLv = 1:nLevels
                %nSubElements = prod(scales(iRevLv,:));
                nSubElements = prod(scales(iRevLv+1,:));
                %subHeight = scales(iRevLv,1);
                %subWidth = scales(iRevLv,2);
                %subDepth = scales(iRevLv,3);                
                subHeight = scales(iRevLv+1,1);
                subWidth = scales(iRevLv+1,2);                
                subDepth = scales(iRevLv+1,3);
                varargout{nLevels-iRevLv+1} = ...
                    reshape(dLdZ(sidx+1:sidx+nSubElements,:),...
                    subHeight,subWidth,subDepth,nChsTotal-1,[]);
                sidx = sidx + nSubElements;
            end
         end
    end
    
end

