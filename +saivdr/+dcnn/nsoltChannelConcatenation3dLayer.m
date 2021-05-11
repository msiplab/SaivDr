classdef nsoltChannelConcatenation3dLayer < nnet.layer.Layer
    %NSOLTCHANNELSEPARATION2DLAYER
    %
    %   ２コンポーネント入力(nComponents=2のみサポート):
    %      nRows x nCols x nLays x 1 x nSamples
    %      nRows x nCols x nLays x (nChsTotal-1) x nSamples
    %
    %   １コンポーネント出力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nLays x nSamples
    %
    % Requirements: MATLAB R2020a
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
            %layer.NumInputs = 2;
            layer.InputNames = { 'ac', 'dc' };            
            
        end
        
        function Z = predict(~, Xac,Xdc)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, X2      - Input data (2 components)
            % Outputs:
            %         Z           - Outputs of layer forward function
            %
            
            if isdlarray(Xac) 
                Xac = stripdims(Xac);
            end                        
            
            if isdlarray(Xdc) 
                Xdc = stripdims(Xdc);
            end                                    
            
            % Layer forward function for prediction goes here.
            %Z = cat(4,Xac,Xdc);
            Z = permute(cat(4,Xdc,Xac),[4 1 2 3 5]);
        end
        
        function [dLdXac,dLdXdc] = backward(~, ~, ~, ~, dLdZ, ~)
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
            %dLdXdc = dLdZ(:,:,:,1,:);
            %dLdXac = dLdZ(:,:,:,2:end,:);
            dLdXac = ipermute(dLdZ(2:end,:,:,:,:),[4 1 2 3 5]);            
            dLdXdc = ipermute(dLdZ(1,:,:,:,:),[4 1 2 3 5]);            
        end
        
    end
    
end

