classdef nsoltBlockDct3dLayer < nnet.layer.Layer
    %NSOLTBLOCKDCT3DLAYER
    %
    %   ベクトル配列をブロック配列を入力: 
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x (Stride(3)xnLays) x nComponents x nSamples
    %
    %   コンポーネント別に出力(nComponents=1):
    %      nDecs x nRows x nCols x nSamples
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
        
        function layer = nsoltBlockDct3dLayer(varargin)
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
            layer.Description = "Block DCT of size " ...
                + layer.DecimationFactor(1) + "x" ...
                + layer.DecimationFactor(2) + "x" ...
                + layer.DecimationFactor(3);
            layer.Type = '';
            layer.NumOutputs = p.Results.NumberOfComponents;
            layer.NumInputs = 1;
            
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
        
        function varargout = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X           - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            import saivdr.dictionary.utility.Direction
            varargout = cell(1,nargout);
            
            if isdlarray(X)
                X = stripdims(X);
            end
            
            % Layer forward function for prediction goes here.
            decFactor = layer.DecimationFactor;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            decD = decFactor(Direction.DEPTH);
            %
            Cvhd_ = layer.Cvhd;
            %
            height = size(X,1);
            width = size(X,2);
            depth = size(X,3);
            nComponents = size(X,4);
            nSamples = size(X,5);
            nRows = height/decV;
            nCols = width/decH;
            nLays = depth/decD;
            %
            if isgpuarray(X)
                for iComponent = 1:nComponents
                    arrayX = X(:,:,:,iComponent,:);
                    arrayY = reshape(permute(reshape(arrayX,...
                        decV,nRows,decH,nCols,decD,nLays,nSamples),...
                        [1 3 5 2 4 6 7]),...
                        decV*decH*decD,nRows,nCols,nLays,nSamples);
                    varargout{iComponent} = pagefun(@mtimes,Cvhd_,arrayY);
                end
            else
                parfor iComponent = 1:nComponents
                    arrayX = X(:,:,:,iComponent,:);
                    arrayY = reshape(permute(reshape(arrayX,...
                        decV,nRows,decH,nCols,decD,nLays,nSamples),...
                        [1 3 5 2 4 6 7]),...
                        decV*decH*decD,[]);
                    varargout{iComponent} = reshape(Cvhd_*arrayY,...
                        decV*decH*decD,nRows,nCols,nLays,nSamples);
                end
                %{
                inputComponent = zeros(height,width,depth,1,nSamples,'like',X);
                outputSample = zeros(nDecs,nRows,nCols,nLays,'like',X);
                outputLay = zeros(nDecs,nRows,nCols,'like',X);
                for iComponent = 1:nComponents
                    inputComponent(:,:,:,1,:) = X(:,:,:,iComponent,:);
                    for iSample = 1:nSamples
                        inputSample = inputComponent(:,:,:,1,iSample);
                        for iLay = 1:nLays
                            inputLay = inputSample(:,:,...
                                (iLay-1)*decD+1:iLay*decD);
                            for iCol = 1:nCols
                                inputCol = inputLay(:,...
                                    (iCol-1)*decH+1:iCol*decH,:);
                                Y = reshape(permute(reshape(...
                                    inputCol,decV,nRows,decH,decD),...
                                    [1 3 4 2]),decV*decH*decD,nRows);
                                outputLay(:,:,iCol) = Cvhd_*Y;
                            end
                            outputSample(:,:,:,iLay) = outputLay;
                        end
                        outputComponent(:,:,:,:,iSample) = outputSample;
                    end
                    varargout{iComponent} = outputComponent;
                end
                %}
            end
            %{
            A = zeros(nDecs,nRows,nCols,nLays,nSamples,'like',X);
            for iComponent = 1:nComponents
                for iSample = 1:nSamples
                    for iLay = 1:nLays
                        for iCol = 1:nCols
                            for iRow = 1:nRows
                                x = X((iRow-1)*decV+1:iRow*decV,...
                                    (iCol-1)*decH+1:iCol*decH,...
                                    (iLay-1)*decD+1:iLay*decD,...
                                    iComponent,iSample);
                                A(:,iRow,iCol,iLay,iSample) = Cvhd_*x(:);
                            end
                        end
                    end
                end
                %varargout{iComponent} = permute(A,[2 3 4 1 5]);
                varargout{iComponent} = A;
            end
            %}
        end
        
        function dLdX = backward(layer, varargin)
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
            nComponents = layer.NumOutputs;
            
            % Layer forward function for prediction goes here.
            decFactor = layer.DecimationFactor;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            decD = decFactor(Direction.DEPTH);
            Cvhd_T = layer.Cvhd.';
            %
            dLdZ = varargin{layer.NumInputs+layer.NumOutputs+1};            
            nRows = size(dLdZ,2);
            nCols = size(dLdZ,3);
            nLays = size(dLdZ,4);
            height = decV*nRows;
            width = decH*nCols;
            depth = decD*nLays;
            nSamples = size(dLdZ,5);
            dLdX = zeros(height,width,depth,nComponents,nSamples,'like',dLdZ);
            %
            if isgpuarray(dLdZ)
                for iComponent = 1:nComponents
                    dLdZ = varargin{layer.NumInputs+layer.NumOutputs+iComponent};                                
                    arrayX = pagefun(@mtimes,Cvhd_T,dLdZ);
                    dLdX(:,:,:,iComponent,:) = reshape(ipermute(reshape(arrayX,...
                    decV,decH,decD,nRows,nCols,nLays,nSamples),...
                    [1 3 5 2 4 6 7]),...
                    decV*nRows,decH*nCols,decD*nLays,1,nSamples);
                end
            else
                arrayY = cell(1,nComponents);
                for iComponent = 1:nComponents
                    dLdZ =  varargin{layer.NumInputs+layer.NumOutputs+iComponent};
                    arrayY{iComponent} = dLdZ;
                end
                parfor iComponent = 1:nComponents
                    arrayX = Cvhd_T*reshape(arrayY{iComponent},decV*decH*decD,[]);
                    dLdX(:,:,:,iComponent,:) = reshape(ipermute(reshape(arrayX,...
                        decV,decH,decD,nRows,nCols,nLays,nSamples),...
                        [1 3 5 2 4 6 7]),...
                        decV*nRows,decH*nCols,decD*nLays,1,nSamples);
                end                
                %{
                if iComponent == 1
                    outputLay = zeros(height,width,decD,'like',dLdZ);
                    outputSample = zeros(height,width,depth,'like',dLdZ);
                    outputComponent = zeros(height,width,depth,1,nSamples,'like',dLdZ);
                end
                for iSample = 1:nSamples
                    inputSample = dLdZ(:,:,:,:,iSample);
                    for iLay = 1:nLays
                        inputLay = inputSample(:,:,:,iLay);
                        for iCol = 1:nCols
                            X = Cvhd_T*inputLay(:,:,iCol);
                            outputLay(:,(iCol-1)*decH+1:iCol*decH,:) = ...
                                reshape(permute(reshape(X,decV,decH,decD,nRows),...
                                [1 4 2 3]),decV*nRows,decH,decD);
                        end
                        outputSample(:,:,(iLay-1)*decD+1:iLay*decD,:) = ...
                            outputLay;
                    end
                    outputComponent(:,:,:,1,iSample) = outputSample;
                end
                %}
            end
            %{
                %A = permute(dLdZ,[4 1 2 3 5]);
                A = dLdZ;
                for iSample = 1:nSamples
                    for iLay = 1:nLays
                        for iCol = 1:nCols
                            for iRow = 1:nRows
                                coefs = A(:,iRow,iCol,iLay,iSample);
                                dLdX((iRow-1)*decV+1:iRow*decV,...
                                    (iCol-1)*decH+1:iCol*decH,...
                                    (iLay-1)*decD+1:iLay*decD,...
                                    iComponent,iSample) = ...
                                    reshape(Cvhd_T*coefs,decV,decH,decD);
                            end
                        end
                    end
                end
            %}
        end
    end
end

