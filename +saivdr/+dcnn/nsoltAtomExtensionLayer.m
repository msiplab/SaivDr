classdef nsoltAtomExtensionLayer < nnet.layer.Layer
    %NSOLTATOMEXTENSIONLAYER
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nRows x nCols x nChsTotal x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nRows x nCols x nChsTotal x nSamples
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
        function layer = nsoltAtomExtensionLayer(varargin)
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
            Y = permute(X,[3 1 2 4]); % [ch ver hor smpl]
            % Block butterfly
            Ys = Y(1:ps,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ];
            % Block circular shift
            if strcmp(target,'Lower')
                Y(ps+1:ps+pa,:,:,:) = circshift(Y(ps+1:ps+pa,:,:,:),shift);
            elseif strcmp(target,'Upper')
                Y(1:ps,:,:,:) = circshift(Y(1:ps,:,:,:),shift);
            else
                throw(MException('NsoltLayer:InvalidTargetChannels',...
                    '%s : TaregetChannels should be either of Lower or Upper',...
                    layer.TargetChannels))
            end
            % Block butterfly
            % Block butterfly
            Ys = Y(1:ps,:,:,:);
            Ya = Y(ps+1:ps+pa,:,:,:);
            Y =  [ Ys+Ya ; Ys-Ya ];
            % Output
            Z = ipermute(Y,[3 1 2 4])/2.0;
        end
        
    end
    
    methods (Static, Access = private)
        
        function value = permuteIdctCoefs_(x,blockSize)
            coefs = x;
            decY_ = blockSize(1);
            decX_ = blockSize(2);
            nQDecsee = ceil(decY_/2)*ceil(decX_/2);
            nQDecsoo = floor(decY_/2)*floor(decX_/2);
            nQDecsoe = floor(decY_/2)*ceil(decX_/2);
            cee = coefs(         1:  nQDecsee);
            coo = coefs(nQDecsee+1:nQDecsee+nQDecsoo);
            coe = coefs(nQDecsee+nQDecsoo+1:nQDecsee+nQDecsoo+nQDecsoe);
            ceo = coefs(nQDecsee+nQDecsoo+nQDecsoe+1:end);
            value = zeros(decY_,decX_,'like',x);
            value(1:2:decY_,1:2:decX_) = reshape(cee,ceil(decY_/2),ceil(decX_/2));
            value(2:2:decY_,2:2:decX_) = reshape(coo,floor(decY_/2),floor(decX_/2));
            value(2:2:decY_,1:2:decX_) = reshape(coe,floor(decY_/2),ceil(decX_/2));
            value(1:2:decY_,2:2:decX_) = reshape(ceo,ceil(decY_/2),floor(decX_/2));
        end
        
        function matrix = orthmtxgen_(angles,mus,pdAng)
            
            if nargin < 3
                pdAng = 0;
            end
            
            if isempty(angles)
                matrix = diag(mus);
            else
                nDim_ = (1+sqrt(1+8*length(angles)))/2;
                matrix = eye(nDim_);
                iAng = 1;
                for iTop=1:nDim_-1
                    vt = matrix(iTop,:);
                    for iBtm=iTop+1:nDim_
                        angle = angles(iAng);
                        if iAng == pdAng
                            angle = angle + pi/2;
                        end
                        c = cos(angle); %
                        s = sin(angle); %
                        vb = matrix(iBtm,:);
                        %
                        u  = s*(vt + vb);
                        vt = (c + s)*vt;
                        vb = (c - s)*vb;
                        vt = vt - u;
                        if iAng == pdAng
                            matrix = 0*matrix;
                        end
                        matrix(iBtm,:) = vb + u;
                        %
                        iAng = iAng + 1;
                    end
                    matrix(iTop,:) = vt;
                end
                matrix = diag(mus)*matrix;
            end
        end

    end
end

