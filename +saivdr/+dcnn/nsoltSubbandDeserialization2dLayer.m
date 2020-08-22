classdef nsoltSubbandDeserialization2dLayer < nnet.layer.Layer
    %NSOLTSUBBANDSERIALIZATION2DLAYER
    %
    %   １コンポーネント入力(SB):
    %      nElements x nSamples
    %
    %   複数コンポーネント出力 (SSCB):（ツリーレベル数）
    %      nRowsLv1 x nColsLv1 x nChsTotal x nSamples
    %      nRowsLv2 x nColsLv2 x (nChsTotal-1) x nSamples
    %       :
    %      nRowsLvN x nColsLvN x (nChsTotal-1) x nSamples
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
        function layer = nsoltSubbandDeserialization2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsoltx.ChannelGroup
            
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'OriginalDimension',[8 8]);
            addParameter(p,'NumberOfChannels',[2 2]);
            addParameter(p,'DecimationFactor',[2 2]);
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
            layer.Description = "Subband deserialization " ...
                + "(h,w) = (" ...
                + height + "," + width + "), "  ...
                + "lv = " ...
                + nLevels + ", " ...
                + "(ps,pa) = (" ...
                + layer.NumberOfChannels(ChannelGroup.UPPER) + "," + layer.NumberOfChannels(ChannelGroup.LOWER) + "), "  ...
                + "(mv,mh) = (" ...
                + layer.DecimationFactor(Direction.VERTICAL) + "," + layer.DecimationFactor(Direction.HORIZONTAL) + ")";
            layer.Type = '';
            outputNames = cell(1,layer.NumberOfLevels);
            for iLv = 1:nLevels
                outputNames{iLv} = [ 'Lv' num2str(iLv) '_SbOut' ];
            end
            layer.OutputNames = outputNames;
            
            %
            nChsTotal = sum(layer.NumberOfChannels);
            stride = layer.DecimationFactor;
            
            nrows = height*stride(Direction.VERTICAL).^(-nLevels);
            ncols = width*stride(Direction.HORIZONTAL).^(-nLevels);
            layer.Scales = zeros(nLevels,3);
            layer.Scales(1,:) = [nrows ncols nChsTotal];
            for iRevLv = 2:nLevels
                layer.Scales(iRevLv,:) = ...
                    [nrows*stride(Direction.VERTICAL)^(iRevLv-1) ncols*stride(Direction.HORIZONTAL)^(iRevLv-1)  nChsTotal-1];
            end
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
                varargout{nLevels-iRevLv+1} = ...
                    reshape(X(sidx+1:sidx+nSubElements,:),...
                    subHeight,subWidth,nChsTotal-wodc,[]);
                sidx = sidx + nSubElements;
            end
        end
        
        %{
function varargout = backward(~, varargin)
            % (Optional) Backward propagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer forward function for prediction goes here.

        end
        %}
    end
    
end

