classdef nsoltChannelSeparation3dLayer < nnet.layer.Layer
    %NSOLTCHANNELSEPARATION3DLAYER
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nLays x nSamples
    %
    %   ２コンポーネント出力(nComponents=2のみサポート):
    %      nRows x nCols x nLays x (nChsTotal-1) x nSamples
    %      nRows x nCols x nLays x 1 x nSamples
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
        function layer = nsoltChannelSeparation3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Name = p.Results.Name;
            layer.Description =  "Channel separation";
            layer.Type = '';
            %layer.NumOutputs = 2;
            layer.OutputNames = { 'ac', 'dc' };                        
        end
        
        function [Zac,Zdc] = predict(~, X)
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
            %Z1 = X(:,:,:,1,:);
            %Z2 = X(:,:,:,2:end,:);
            Zac = permute(X(2:end,:,:,:,:),[2 3 4 1 5]);            
            Zdc = permute(X(1,:,:,:,:),[2 3 4 1 5]);
            
            if isdlarray(X)
                Zac = dlarray(Zac,'SSSCB');
                Zdc = dlarray(Zdc,'SSSCB');
            end            
            
        end
        
        function dLdX = backward(~, ~,~,~, dLdZac,dLdZdc, ~)
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
            %dLdX = cat(4,dLdZ1,dLdZ2);
            dLdX = ipermute(cat(4,dLdZdc,dLdZac),[2 3 4 1 5]);
        end
        
    end
    
end

