classdef nsoltAtomExtension3dLayer < nnet.layer.Layer
    %NSOLTATOMEXTENSION3DLAYER
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nRows x nCols x nLays x nChsTotal x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nRows x nCols x nLays x nChsTotal x nSamples
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
        NumberOfChannels
        Direction
        TargetChannels
        
        % Layer properties go here.
    end
    
    methods
        function layer = nsoltAtomExtension3dLayer(varargin)
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
                + " shift " ...
                + layer.TargetChannels ...
                + " Coefs. " ...
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
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            dir = layer.Direction;
            target = layer.TargetChannels;
            %
            if strcmp(dir,'Right')
                shift = [ 0 0 1 0 0 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 0 ];
            elseif strcmp(dir,'Down')
                shift = [ 0 1 0 0 0 ];
            elseif strcmp(dir,'Up')
                shift = [ 0 -1 0 0 0 ];
            elseif strcmp(dir,'Back')
                shift = [ 0 0 0 1 0 ];
            elseif strcmp(dir,'Front')
                shift = [ 0 0 0 -1 0 ];                
            else
                throw(MException('NsoltLayer:InvalidDirection',...
                    '%s : Direction should be either of Right, Left, Down, Up, Back or Front',...
                    layer.Direction))
            end
            %
            Y = permute(X,[4 1 2 3 5]); % [ch ver hor dep smpl]
            % Block butterfly
            Ys = Y(1:ps,:,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ];
            % Block circular shift
            if strcmp(target,'Lower')
                Y(ps+1:ps+pa,:,:,:,:) = circshift(Y(ps+1:ps+pa,:,:,:,:),shift);
            elseif strcmp(target,'Upper')
                Y(1:ps,:,:,:,:) = circshift(Y(1:ps,:,:,:,:),shift);
            else
                throw(MException('NsoltLayer:InvalidTargetChannels',...
                    '%s : TaregetChannels should be either of Lower or Upper',...
                    layer.TargetChannels))
            end
            % Block butterfly
            Ys = Y(1:ps,:,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ];
            % Output
            Z = ipermute(Y,[4 1 2 3 5])/2.0;
        end
        
    end

end