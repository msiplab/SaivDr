classdef nsoltBlockIdct2dLayer < nnet.layer.Layer %#codegen
    %NSOLTBLOCKIDCT2DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nDecs x nRows x nCols x nSamples
    %
    %   ベクトル配列をブロック配列にして出力:
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
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
        function layer = nsoltBlockIdct2dLayer(varargin)
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
                + layer.DecimationFactor(Direction.HORIZONTAL);
            layer.Type = '';
            layer.NumInputs = p.Results.NumberOfComponents;
            layer.NumOutputs = 1;
            
            decV = layer.DecimationFactor(Direction.VERTICAL);
            decH = layer.DecimationFactor(Direction.HORIZONTAL);
            
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
            nDec = decV*decH;
            Cvh_T = layer.Cvh.';
            %
            X = varargin{1};
            nRows = size(X,2);
            nCols = size(X,3);
            height = decFactor(1)*nRows;
            width = decFactor(2)*nCols;
            nSamples = size(X,4);
            Z = zeros(height,width,nComponents,nSamples,'like',X);
            %
            for iComponent = 1:nComponents
                X = varargin{iComponent};
                if isgpuarray(X)
                    arrayY = pagefun(@mtimes,Cvh_T,X);
                else
                    arrayY = Cvh_T*reshape(X,nDec,[]);
                end
                Z(:,:,iComponent,:) = ...
                    reshape(ipermute(reshape(arrayY,...
                    decV,decH,nRows,nCols,nSamples),[1 3 2 4 5]),...
                    height,width,1,nSamples);
            end
            if isdlarray(X)
                Z = dlarray(Z,'SSCB');
            end
            
        end
        
        function varargout = backward(layer,varargin)
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
            dLdZ = varargin{layer.NumInputs+layer.NumOutputs+1};
            varargout = cell(1,nComponents);
            
            % Layer forward function for prediction goes here.
            decFactor = layer.DecimationFactor;
            decV = decFactor(Direction.VERTICAL);
            decH = decFactor(Direction.HORIZONTAL);
            nDec = decV*decH;
            %
            Cvh_ = layer.Cvh;
            %
            height = size(dLdZ,1);
            width = size(dLdZ,2);
            nRows = height/decV;
            nCols = width/decH;
            nSamples = size(dLdZ,4);
            %
            for iComponent = 1:nComponents
                arrayX = permute(reshape(dLdZ(:,:,iComponent,:),...
                    decV,nRows,decH,nCols,nSamples),[1 3 2 4 5]);
                if isgpuarray(dLdZ)
                    varargout{iComponent} = ...
                        pagefun(@mtimes,Cvh_,...
                        reshape(arrayX,nDec,nRows,nCols,nSamples));
                else
                    varargout{iComponent} = reshape(...
                        Cvh_*reshape(arrayX,nDec,[]),...
                        nDec,nRows,nCols,nSamples);
                end
            end
        end
    end
    
end

