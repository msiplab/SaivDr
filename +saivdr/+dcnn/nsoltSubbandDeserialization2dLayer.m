classdef nsoltSubbandDeserialization2dLayer < nnet.layer.Layer %#codegen
    %NSOLTSUBBANDSERIALIZATION2DLAYER
    %
    %   １コンポーネント入力(SSCB):
    %      nElements x 1 x 1 x nSamples
    %
    %   複数コンポーネント出力 (SSCB):（ツリーレベル数）
    %      nRowsLv1 x nColsLv1 x (nChsTotal-1) x nSamples
    %      nRowsLv2 x nColsLv2 x (nChsTotal-1) x nSamples
    %       :
    %      nRowsLvN x nColsLvN x (nChsTotal-1) x nSamples
    %      nRowsLvN x nColsLvN x 1 x nSamples    
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2020-2021, Shogo MURAMATSU
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
        InputSize
        % Layer properties go here.
    end
    
    methods
        function layer = nsoltSubbandDeserialization2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'OriginalDimension',[8 8]);
            addParameter(p,'NumberOfChannels',[2 2]);
            addParameter(p,'DecimationFactor',[2 2]);
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
            %outputNames = cell(1,layer.NumberOfLevels);
            outputNames = cell(1,layer.NumberOfLevels+1);
            for iLv = 1:nLevels
                outputNames{iLv} = [ 'Lv' num2str(iLv) '_SbAcOut' ];
            end
            outputNames{nLevels+1} = [ 'Lv' num2str(nLevels) '_SbDcOut' ]; %            
            layer.OutputNames = outputNames;
        end
        
        function layer = setOriginalDimension(layer,orgdim) 
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsoltx.ChannelGroup
            layer.OriginalDimension = orgdim;     
            nLevels = layer.NumberOfLevels;
            height = layer.OriginalDimension(Direction.VERTICAL);
            width = layer.OriginalDimension(Direction.HORIZONTAL);
            layer.Description = "Subband deserialization " ...
                + "(h,w) = (" ...
                + height + "," + width + "), "  ...
                + "lv = " ...
                + nLevels + ", " ...
                + "(ps,pa) = (" ...
                + layer.NumberOfChannels(ChannelGroup.UPPER) + "," + layer.NumberOfChannels(ChannelGroup.LOWER) + "), "  ...
                + "(mv,mh) = (" ...
                + layer.DecimationFactor(Direction.VERTICAL) + "," + layer.DecimationFactor(Direction.HORIZONTAL) + ")";
            
            %
            nChsTotal = sum(layer.NumberOfChannels);
            stride = layer.DecimationFactor;

            nrows = height*stride(Direction.VERTICAL).^(-nLevels);
            ncols = width*stride(Direction.HORIZONTAL).^(-nLevels);

            layer.Scales = zeros(nLevels,3);
            %layer.Scales(1,:) = [nrows ncols nChsTotal];
            layer.Scales(1,:) = [nrows ncols 1];
            for iRevLv = 1:nLevels % 2:nLevels
                %layer.Scales(iRevLv,:) = ...
                layer.Scales(iRevLv+1,:) = ...
                    [nrows*stride(Direction.VERTICAL)^(iRevLv-1) ncols*stride(Direction.HORIZONTAL)^(iRevLv-1)  nChsTotal-1];
            end
            
             layer.InputSize = [sum(prod(layer.Scales,2)) 1 1 ];
        end
                    
        function varargout = predict(layer,X)
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
            nChsTotal = sum(layer.NumberOfChannels);
            scales = layer.Scales;
            %varargout = cell(1,nLevels);
            varargout = cell(1,nLevels+1);            
            sidx = 0;
            nSubElements = prod(scales(1,:));
            subHeight = scales(1,1);
            subWidth = scales(1,2);
            varargout{nLevels+1} = ...
                reshape(X(1:nSubElements,:),...
                subHeight,subWidth,1,[]);
            sidx = sidx + nSubElements;            
            for iRevLv = 1:nLevels
                %nSubElements = prod(scales(iRevLv,:));
                nSubElements = prod(scales(iRevLv+1,:));
                %subHeight = scales(iRevLv,1);
                %subWidth = scales(iRevLv,2);
                subHeight = scales(iRevLv+1,1);
                subWidth = scales(iRevLv+1,2);                
                varargout{nLevels-iRevLv+1} = ...
                    reshape(X(sidx+1:sidx+nSubElements,:),...
                    subHeight,subWidth,nChsTotal-1,[]);
                sidx = sidx + nSubElements;
            end
        end
        
       function dLdX = backward(layer, varargin)
            % (Optional) Backward propagate the derivative of the loss  
            % function through the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function            
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %  
            
            nLevels = layer.NumberOfLevels;
            nSamples = size(varargin{nLevels+3},4);
            nElements = sum(prod(layer.Scales,2));
            scales = layer.Scales;
            dLdX = zeros(nElements,1,1,nSamples,'like',varargin{nLevels+3});
            for iSample = 1:nSamples
                x = zeros(nElements,1,'like',dLdX);
                sidx = 0;
                nSubElements = prod(scales(1,:));
                a = varargin{1+nLevels+1+nLevels+1}(:,:,:,iSample);
                x(1:nSubElements) = a(:);
                sidx = sidx+nSubElements;
                for iRevLv = 1:nLevels
                    nSubElements = prod(scales(iRevLv+1,:));
                    a = varargin{nLevels+3+nLevels-iRevLv}(:,:,:,iSample);
                    x(sidx+1:sidx+nSubElements) = a(:);
                    sidx = sidx+nSubElements;
                end
                dLdX(:,1,1,iSample) = x;
            end
        end
    end
    
end

