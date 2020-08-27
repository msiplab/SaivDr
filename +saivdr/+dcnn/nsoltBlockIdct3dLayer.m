classdef nsoltBlockIdct3dLayer < nnet.layer.Layer
    %NSOLTBLOCKIDCT3DLAYER
    %
    %   コンポーネント別に入力(nComponents=1):
    %      nRows x nCols x nLays x nDecs x nSamples
    %
    %   ベクトル配列をブロック配列にして出力:
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x (Stride(3)xnLays) x nComponents x nSamples
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
            nComponents = layer.NumInputs;
            
            % Layer forward function for prediction goes here.
            decFactor = layer.DecimationFactor;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            decD = decFactor(Direction.DEPTH);            
            %
            Cvhd_T = layer.Cvhd.';
            for iComponent = 1:nComponents
                X = varargin{iComponent};
                if iComponent == 1
                    nRows = size(X,1);
                    nCols = size(X,2);
                    nLays = size(X,3);
                    height = decV*nRows;
                    width = decH*nCols;
                    depth = decD*nLays;
                    nSamples = size(X,5);
                    Z = zeros(height,width,depth,nComponents,nSamples,'like',X);
                end
                A = permute(X,[4 1 2 3 5]);
                for iSample = 1:nSamples
                    for iLay = 1:nLays
                        for iCol = 1:nCols
                            for iRow = 1:nRows
                                coefs = A(:,iRow,iCol,iLay,iSample);
                                Z((iRow-1)*decV+1:iRow*decV,...
                                    (iCol-1)*decH+1:iCol*decH,...
                                    (iLay-1)*decD+1:iLay*decD,...
                                    iComponent,iSample) = ...
                                    reshape(Cvhd_T*coefs,decV,decH,decD);
                            end
                        end
                    end
                end
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
            %
            Cvhd_ = layer.Cvhd;
            %
            nRows = size(dLdZ,1)/decV;
            nCols = size(dLdZ,2)/decH;
            nLays = size(dLdZ,3)/decD;
            nDecs = prod(decFactor);
            nSamples = size(dLdZ,5);
            %
            A = zeros(nDecs,nRows,nCols,nLays,nSamples,'like',dLdZ);
            for iComponent = 1:nComponents
                for iSample = 1:nSamples
                    for iLay = 1:nLays
                        for iCol = 1:nCols
                            for iRow = 1:nRows
                                x = dLdZ((iRow-1)*decV+1:iRow*decV,...
                                    (iCol-1)*decH+1:iCol*decH,...
                                    (iLay-1)*decD+1:iLay*decD,...
                                    iComponent,iSample);
                                A(:,iRow,iCol,iLay,iSample) = Cvhd_*x(:);
                            end
                        end
                    end
                end
                varargout{iComponent} = permute(A,[2 3 4 1 5]);
            end
        end
    end
    
end

