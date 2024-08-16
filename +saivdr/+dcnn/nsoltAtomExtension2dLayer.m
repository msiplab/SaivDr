classdef nsoltAtomExtension2dLayer < nnet.layer.Layer %#codegen
    %NSOLTATOMEXTENSION2DLAYER
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nSamples
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
        NumberOfChannels
        Direction
        TargetChannels
        
        % Layer properties go here.
    end
    
    methods
        function layer = nsoltAtomExtension2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfChannels',[])
            addParameter(p,'Direction','')
            addParameter(p,'TargetChannels','')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.Name = p.Results.Name;
            layer.Direction = p.Results.Direction;
            layer.TargetChannels = p.Results.TargetChannels;
            layer.Description =  layer.Direction ...
                + " shift the " ...
                + lower(layer.TargetChannels) ...
                + "-channel Coefs. " ...
                + "(ps,pa) = (" ...
                + layer.NumberOfChannels(1) + "," ...
                + layer.NumberOfChannels(2) + ")";
            
            layer.Type = '';
            
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data (n: # of components)
            % Outputs:
            %         Z           - Outputs of layer forward function
            %
            
            % Layer forward function for prediction goes here.
            dir = layer.Direction;
            %
            if strcmp(dir,'Right')
                shift = [ 0 0 1 0 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 ];
            elseif strcmp(dir,'Down')
                shift = [ 0 1 0 0 ];
            elseif strcmp(dir,'Up')
                shift = [ 0 -1 0 0 ];
            else
                throw(MException('NsoltLayer:InvalidDirection',...
                    '%s : Direction should be either of Right, Left, Down or Up',...
                    layer.Direction))
            end
            %
            Z = layer.atomext_(X,shift);
        end
        
        function dLdX = backward(layer, ~, ~, dLdZ, ~)
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
            dir = layer.Direction;

            %
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 0 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0 1 0 ];  % Reverse
            elseif strcmp(dir,'Down')
                shift = [ 0 -1 0 0 ];  % Reverse
            elseif strcmp(dir,'Up')
                shift = [ 0 1 0 0 ];  % Reverse
            else
                throw(MException('NsoltLayer:InvalidDirection',...
                    '%s : Direction should be either of Right, Left, Down or Up',...
                    layer.Direction))
            end
            %
            dLdX = layer.atomext_(dLdZ,shift);
        end
        
        function Z = atomext_(layer,X,shift)
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            target = layer.TargetChannels;            
            %
            % Block butterfly
            Xs = X(1:ps,:,:,:);
            Xa = X(ps+1:ps+pa,:,:,:);
            Ys =  bsxfun(@plus,Xs,Xa);
            Ya =  bsxfun(@minus,Xs,Xa);
            % Block circular shift
            if strcmp(target,'Difference')
                Ya = circshift(Ya,shift);
            elseif strcmp(target,'Sum')
                Ys = circshift(Ys,shift);
            else
                throw(MException('NsoltLayer:InvalidTargetChannels',...
                    '%s : TaregetChannels should be either of Sum or Difference',...
                    layer.TargetChannels))
            end
            % Block butterfly
            Y =  cat(1,bsxfun(@plus,Ys,Ya),bsxfun(@minus,Ys,Ya));
            % Output
            Z = 0.5*Y; %ipermute(Y,[3 1 2 4])/2.0;
        end
        
    end

end

