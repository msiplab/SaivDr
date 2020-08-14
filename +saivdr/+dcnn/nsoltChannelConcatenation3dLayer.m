classdef nsoltChannelConcatenation3dLayer < nnet.layer.Layer
    %NSOLTCHANNELSEPARATION2DLAYER
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nRows x nCols x nLays x nChsTotal x nSamples
    %
    %   ２コンポーネント出力(nComponents=2のみサポート):
    %      nRows x nCols x nLays x 1 x nSamples
    %      nRows x nCols x nLays x (nChsTotal-1) x nSamples    
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
        
        % Layer properties go here.
    end
    
    methods
        function layer = nsoltChannelConcatenation3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Name = p.Results.Name;
            layer.Description =  "Channel concatenation";
            layer.Type = '';
            layer.NumInputs = 2;
            
        end
        
        function Z = predict(layer, X1,X2)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, X2      - Input data (2 components)
            % Outputs:
            %         Z           - Outputs of layer forward function
            %  
            
            % Layer forward function for prediction goes here.
            Z = cat(4,X1,X2);
        end
        
    end
    
end

