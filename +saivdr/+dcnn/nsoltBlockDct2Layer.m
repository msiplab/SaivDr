classdef nsoltBlockDct2Layer < nnet.layer.Layer
    %NSOLTBLOCKDCT2LAYER
    %
    %   ベクトル配列をブロック配列を入力:
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nRows x nCols x nDecs x nSamples
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
        Cvh
    end
    
    methods
        function layer = nsoltBlockDct2Layer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'DecimationFactor',[])
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.Name = p.Results.Name;
            layer.Description = "Block DCT of size " ...
                + layer.DecimationFactor(1) + "x" + layer.DecimationFactor(2);
            layer.Type = '';
            
            Cv_ = dctmtx(layer.DecimationFactor(1));
            Ch_ = dctmtx(layer.DecimationFactor(2));
            Cv_ = [ Cv_(1:2:end,:) ; Cv_(2:2:end,:) ];
            Ch_ = [ Ch_(1:2:end,:) ; Ch_(2:2:end,:) ];
            %
            decV = layer.DecimationFactor(1);
            decH = layer.DecimationFactor(2);
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
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            decFactor = layer.DecimationFactor;
            decV = decFactor(1);
            decH = decFactor(2);
            %
            Cvh_ = layer.Cvh;
            %
            nRows = size(X,1)/decV;
            nCols = size(X,2)/decH;
            nDecs = prod(decFactor);
            nComponents = size(X,3);
            nSamples = size(X,4);
            %
            A = zeros(nDecs,nRows,nCols,nSamples,'like',X);
            for iSample = 1:nSamples
                for iComponent = 1:nComponents
                    for iCol = 1:nCols
                        for iRow = 1:nRows
                            x = X((iRow-1)*decV+1:iRow*decV,...
                                (iCol-1)*decH+1:iCol*decH,...
                                iComponent,iSample);
                            %coefs = Cv_*x*Ch_T;
                            %cee = coefs(1:ceil(decV/2),    1:ceil(decH/2));
                            %coo = coefs(ceil(decV/2)+1:end,ceil(decH/2)+1:end);
                            %coe = coefs(ceil(decV/2)+1:end,1:ceil(decH/2));
                            %ceo = coefs(1:ceil(decV/2),    ceil(decH/2)+1:end);
                            %z =  [ cee(:); coo(:); coe(:); ceo(:) ];
                            %
                            A(:,iRow,iCol,iSample) = Cvh_*x(:);
                        end
                    end
                end
            end
            Z = permute(A,[2 3 1 4]);
        end        

    end
    
end

