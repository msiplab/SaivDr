classdef nsoltBlockDct2dLayer < nnet.layer.Layer %#codegen
    %NSOLTBLOCKDCT2DLAYER
    %
    %   ベクトル配列をブロック配列を入力:
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nDecs x nRows x nCols x nSamples
    %    
    % Requirements: MATLAB R2020b
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
        Cvh
    end
    
    methods
        function layer = nsoltBlockDct2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            import saivdr.dictionary.utility.Direction                                                            
            p = inputParser;
            addParameter(p,'DecimationFactor',[])
            addParameter(p,'Name','')
            addParameter(p,'NumberOfComponents',1);
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.Name = p.Results.Name;
            layer.Description = "Block DCT of size " ...
                + layer.DecimationFactor(Direction.VERTICAL) + "x" ...
                + layer.DecimationFactor(Direction.HORIZONTAL);
            layer.Type = '';
            layer.NumOutputs = p.Results.NumberOfComponents;
            layer.NumInputs = 1;
            
            decV = layer.DecimationFactor(Direction.VERTICAL);
            decH = layer.DecimationFactor(Direction.HORIZONTAL);
            %
            Cv_ = dctmtx(decV);
            Ch_ = dctmtx(decH);
            Cv_ = [ Cv_(1:2:end,:) ; Cv_(2:2:end,:) ];
            Ch_ = [ Ch_(1:2:end,:) ; Ch_(2:2:end,:) ];
            %
            Cve = Cv_(1:ceil(decV/2),:);
            Cvo = Cv_(ceil(decV/2)+1:end,:);
            Che = Ch_(1:ceil(decH/2),:);
            Cho = Ch_(ceil(decH/2)+1:end,:);
            Cee = kron(Che,Cve);
            Coo = kron(Cho,Cvo);
            Coe = kron(Che,Cvo);
            Ceo = kron(Cho,Cve);
            layer.Cvh = [Cee; Coo; Coe; Ceo];            
            
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
            nComponents = layer.NumOutputs;            
            varargout = cell(1,nComponents);
                        
            % Layer forward function for prediction goes here.
            decFactor = layer.DecimationFactor;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            %
            Cvh_ = layer.Cvh;
            %
            height = size(X,1);
            width = size(X,2);
            nRows = height/decV;
            nCols = width/decH;
            nDecs = prod(decFactor);
            nSamples = size(X,4);
            %
            inputComponent = zeros(height,width,1,nSamples,'like',X);
            inputSample = zeros(height,width,'like',X);
            inputCol = zeros(height,decH,'like',X);      
            outputComponent = zeros(nDecs,nRows,nCols,nSamples,'like',X);
            outputSample = zeros(nDecs,nRows,nCols,'like',X);
            outputCol = zeros(nDecs,nRows,'like',X);
            for iComponent = 1:nComponents
                inputComponent(:,:,1,:) = X(:,:,iComponent,:);
                for iSample = 1:nSamples
                    inputSample(:,:) = inputComponent(:,:,1,iSample);
                    for iCol = 1:nCols
                        inputCol(:,:) = inputSample(:,...
                            (iCol-1)*decH+1:iCol*decH);
                        for iRow = 1:nRows
                            x = inputCol((iRow-1)*decV+1:iRow*decV,:);      
                            outputCol(:,iRow) = Cvh_*x(:); 
                        end
                        outputSample(:,:,iCol) = outputCol;
                    end
                    outputComponent(:,:,:,iSample) = outputSample;
                end
                varargout{iComponent} = outputComponent;
            end
            %{
            A = zeros(nDecs,nRows,nCols,nSamples,'like',X);
            for iComponent = 1:nComponents
                for iSample = 1:nSamples
                    for iCol = 1:nCols
                        for iRow = 1:nRows
                            x = X((iRow-1)*decV+1:iRow*decV,...
                                (iCol-1)*decH+1:iCol*decH,...
                                iComponent,iSample);
                            A(:,iRow,iCol,iSample) = Cvh_*x(:);
                        end
                    end
                end
                %varargout{iComponent} = permute(A,[2 3 1 4]);                
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
            decFactor = layer.DecimationFactor;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            Cvh_T = layer.Cvh.';
            for iComponent = 1:nComponents
                dLdZ = varargin{layer.NumInputs+layer.NumOutputs+iComponent};
                if iComponent == 1
                    nElements = size(dLdZ,1);
                    nRows = size(dLdZ,2);
                    nCols = size(dLdZ,3);                    
                    height = decV*nRows;
                    width = decH*nCols;
                    nSamples = size(dLdZ,4);
                    dLdX = zeros(height,width,nComponents,nSamples,'like',dLdZ);
                    %
                    inputSample = zeros(nElements,nRows,nCols,'like',dLdZ);
                    inputCol = zeros(nElements,nRows,'like',dLdZ);
                    outputCol = zeros(height,decH,'like',dLdZ);
                    outputSample = zeros(height,width,'like',dLdZ);
                    outputComponent = zeros(height,width,1,nSamples,'like',dLdZ);
                end
                for iSample = 1:nSamples
                    inputSample(:,:,:) = dLdZ(:,:,:,iSample);
                    for iCol = 1:nCols
                        inputCol(:,:) = inputSample(:,:,iCol);
                        for iRow = 1:nRows
                            coefs = inputCol(:,iRow);
                            outputCol((iRow-1)*decV+1:iRow*decV,:) ...
                                = reshape(Cvh_T*coefs,decV,decH);
                        end
                        outputSample(:,(iCol-1)*decH+1:iCol*decH) = ...
                            outputCol;
                    end
                    outputComponent(:,:,1,iSample) = outputSample;
                end
                dLdX(:,:,iComponent,:) = outputComponent;               
                %{
                %A = permute(X,[3 1 2 4]);
                A = X;
                for iSample = 1:nSamples
                    for iCol = 1:nCols
                        for iRow = 1:nRows
                            coefs = A(:,iRow,iCol,iSample);
                            dLdX((iRow-1)*decV+1:iRow*decV,...
                                (iCol-1)*decH+1:iCol*decH,...
                                iComponent,iSample) = ...
                                reshape(Cvh_T*coefs,decV,decH);
                        end
                    end
                end
                %}
            end
        end
    end
    
end

