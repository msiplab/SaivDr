classdef nsoltInitialRotation2dLayer < nnet.layer.Layer
    %NSOLTINITIALROTATION2DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nRows x nCols x nDecs x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nRows x nCols x nChs x nSamples
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
        DecimationFactor
        NoDcLeakage
        Mus
        
        % Layer properties go here.
    end
    
    properties (Learnable)
        Angles
    end
    
    methods
        function layer = nsoltInitialRotation2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'NumberOfChannels',[])
            addParameter(p,'DecimationFactor',[])
            addParameter(p,'Name','')
            addParameter(p,'Mus',[])
            addParameter(p,'Angles',[])
            addParameter(p,'NoDcLeakage',false);
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.Name = p.Results.Name;
            layer.Mus = p.Results.Mus;
            layer.Angles = p.Results.Angles;
            layer.NoDcLeakage = p.Results.NoDcLeakage;
            layer.Description = "NSOLT initial rotation " ...
                + "(ps,pa) = (" ...
                + layer.NumberOfChannels(1) + "," ...
                + layer.NumberOfChannels(2) + "), "  ...
                + "(mv,mh) = (" ...
                + layer.DecimationFactor(1) + "," ...
                + layer.DecimationFactor(2) + ")";
            layer.Type = '';
            
            nChsTotal = sum(layer.NumberOfChannels);
            nAngles = (nChsTotal-2)*nChsTotal/4;
            if isempty(layer.Angles)
                layer.Angles = zeros(nAngles,1);
            end
            if length(layer.Angles)~=nAngles
                error('Invalid # of angles')
            end
            
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
            import saivdr.dcnn.fcn_orthonormalmatrixgenerate
            
            % Layer forward function for prediction goes here.
            nrows = size(X,1);
            ncols = size(X,2);
            ps = layer.NumberOfChannels(1);
            pa = layer.NumberOfChannels(2);
            nSamples = size(X,4);
            stride = layer.DecimationFactor;
            nDecs = prod(stride);
            nChsTotal = ps + pa;
            %
            if isempty(layer.Mus)
                muW = 1;
                muU = 1;
            else
                if layer.NoDcLeakage
                    layer.Mus(1) = 1;
                end
                muW = layer.Mus(1:ps);
                muU = layer.Mus(ps+1:end);
            end
            if isempty(layer.Angles)
                W0 = eye(ps);
                U0 = eye(pa);
            else
                if layer.NoDcLeakage
                    layer.Angles(1:length(layer.Angles)/2-1) = ...
                        zeros(length(layer.Angles)/2-1,1,'like',layer.Angles);
                end
                anglesW = layer.Angles(1:length(layer.Angles)/2);
                anglesU = layer.Angles(length(layer.Angles)/2+1:end);
                W0 = fcn_orthonormalmatrixgenerate(anglesW,muW);
                U0 = fcn_orthonormalmatrixgenerate(anglesU,muU);
            end
            Y = reshape(permute(X,[3 1 2 4]),nDecs,nrows*ncols*nSamples);
            Zs = W0(:,1:nDecs/2)*Y(1:nDecs/2,:);
            Za = U0(:,1:nDecs/2)*Y(nDecs/2+1:end,:);
            Z = ipermute(reshape([Zs;Za],nChsTotal,nrows,ncols,nSamples),...
                [3 1 2 4]);
            
        end
        
    end
    
end

