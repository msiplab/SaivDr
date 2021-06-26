classdef nsoltBlockIdct3dLayer < nnet.layer.Layer
    %NSOLTBLOCKIDCT3DLAYER
    %
    %   コンポーネント別に入力(nComponents=1):
    %       nDecs x nRows x nCols x nLays x nSamples
    %
    %   ベクトル配列をブロック配列にして出力:
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x (Stride(3)xnLays) x nComponents x nSamples
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
        DecimationFactor
        
        % Layer properties go here.
    end
    
    properties (Access = private)
        Cvhd
    end
    
    methods
        function layer = nsoltBlockIdct3dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            import saivdr.dictionary.utility.Direction
            p = inputParser;
            addParameter(p,'DecimationFactor',[])
            addParameter(p,'Name','')
            addParameter(p,'NumberOfComponents',1)
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.Name = p.Results.Name;
            layer.Description = "Block IDCT of size " ...
                + layer.DecimationFactor(Direction.VERTICAL) + "x" ...
                + layer.DecimationFactor(Direction.HORIZONTAL) + "x" ...
                + layer.DecimationFactor(Direction.DEPTH);
            layer.Type = '';
            layer.NumInputs = p.Results.NumberOfComponents;
            layer.NumOutputs = 1;
            
            decV = layer.DecimationFactor(Direction.VERTICAL);
            decH = layer.DecimationFactor(Direction.HORIZONTAL);
            decD = layer.DecimationFactor(Direction.DEPTH);
            %
            Cv_ = dctmtx(decV);
            Ch_ = dctmtx(decH);
            Cd_ = dctmtx(decD);
            Cv_ = [ Cv_(1:2:end,:) ; Cv_(2:2:end,:) ];
            Ch_ = [ Ch_(1:2:end,:) ; Ch_(2:2:end,:) ];
            Cd_ = [ Cd_(1:2:end,:) ; Cd_(2:2:end,:) ];
            %
            Cve = Cv_(1:ceil(decV/2),:);
            Cvo = Cv_(ceil(decV/2)+1:end,:);
            Che = Ch_(1:ceil(decH/2),:);
            Cho = Ch_(ceil(decH/2)+1:end,:);
            Cde = Cd_(1:ceil(decD/2),:);
            Cdo = Cd_(ceil(decD/2)+1:end,:);
            %
            Cee = kron(Che,Cve);
            Coo = kron(Cho,Cvo);
            Coe = kron(Che,Cvo);
            Ceo = kron(Cho,Cve);
            %
            Ceee = kron(Cde,Cee);
            Ceoo = kron(Cdo,Ceo);
            Cooe = kron(Cde,Coo);
            Coeo = kron(Cdo,Coe);
            Ceeo = kron(Cdo,Cee);
            Ceoe = kron(Cde,Ceo);
            Cooo = kron(Cdo,Coo);
            Coee = kron(Cde,Coe);
            layer.Cvhd = ...
                [ Ceee; Ceoo; Cooe; Coeo; Ceeo; Ceoe; Cooo; Coee ]; % Cyxz
            
        end
        
        function Z = predict(layer, varargin)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            import saivdr.dictionary.utility.Direction
            
            % Layer forward function for prediction goes here.
            nComponents = layer.NumInputs;
            decFactor = layer.DecimationFactor;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            decD = decFactor(Direction.DEPTH);
            nDec = decV*decH*decD;
            Cvhd_T = layer.Cvhd.';
            %
            X = varargin{1};
            nRows = size(X,2);
            nCols = size(X,3);
            nLays = size(X,4);
            nSamples = size(X,5);
            %
            height = decV*nRows;
            width = decH*nCols;
            depth = decD*nLays;
            Z = zeros(height,width,depth,nComponents,nSamples,'like',X);
            %
            for iComponent = 1:nComponents
                X = varargin{iComponent};
                if isgpuarray(X)
                    arrayY = pagefun(@mtimes,Cvhd_T,X);
                else
                    arrayY = Cvhd_T*reshape(X,nDec,[]);
                end
                Z(:,:,:,iComponent,:) = reshape(ipermute(reshape(arrayY,...
                    decV,decH,decD,nRows,nCols,nLays,nSamples),...
                    [1 3 5 2 4 6 7]),...
                    height,width,depth,1,nSamples);
            end
            if isdlarray(X)
                Z = dlarray(Z,'SSSCB');
            end
            
        end
        
        function varargout = backward(layer,~,~,dLdZ,~)
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
            import saivdr.dictionary.utility.Direction
            nComponents = layer.NumInputs;
            varargout = cell(1,nComponents);
            
            % Layer forward function for prediction goes here.
            decFactor = layer.DecimationFactor;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            decD = decFactor(Direction.DEPTH);
            nDec = decV*decH*decD;
            %
            Cvhd_ = layer.Cvhd;
            %
            height = size(dLdZ,1);
            width = size(dLdZ,2);
            depth = size(dLdZ,3);
            nRows = height/decV;
            nCols = width/decH;
            nLays = depth/decD;
            nSamples = size(dLdZ,5);
            %
            for iComponent = 1:nComponents
                arrayX = permute(reshape(dLdZ(:,:,:,iComponent,:),...
                    decV,nRows,decH,nCols,decD,nLays,nSamples),...
                    [1 3 5 2 4 6 7]);
                if isgpuarray(dLdZ)
                    varargout{iComponent} = ...
                        pagefun(@mtimes,Cvhd_,...
                        reshape(arrayX,nDec,nRows,nCols,nLays,nSamples));
                else
                    varargout{iComponent} = reshape(...
                        Cvhd_*reshape(arrayX,nDec,[]),...
                        nDec,nRows,nCols,nLays,nSamples);
                end
            end
        end
    end
end
