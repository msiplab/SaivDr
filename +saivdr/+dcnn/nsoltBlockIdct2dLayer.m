classdef nsoltBlockIdct2dLayer < nnet.layer.Layer
    %NSOLTBLOCKIDCT2DLAYER
    %
    %   コンポーネント別に入力(nComponents):
    %      nRows x nCols x nDecs x nSamples
    %
    %   ベクトル配列をブロック配列にして出力:
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
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
        function layer = nsoltBlockIdct2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            import saivdr.dictionary.utility.Direction                                                
            p = inputParser;
            addParameter(p,'DecimationFactor',[])
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.Name = p.Results.Name;
            layer.Description = "Block IDCT of size " ...
                + layer.DecimationFactor(Direction.VERTICAL) + "x" ...
                + layer.DecimationFactor(Direction.HORIZONTAL);
            layer.Type = '';

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
            
            % Layer forward function for prediction goes here.
            nComponents = nargin - 1;
            decFactor = layer.DecimationFactor;
            decV = decFactor(1);
            decH = decFactor(2);
            Cvh_T = layer.Cvh.';
            for iComponent = 1:nComponents
                X = varargin{iComponent};
                if iComponent == 1
                    nRows = size(X,1);
                    nCols = size(X,2);
                    height = decFactor(1)*nRows;
                    width = decFactor(2)*nCols;
                    nSamples = size(X,4);
                    Z = zeros(height,width,nComponents,nSamples,'like',X);
                end
                A = permute(X,[3 1 2 4]);
                for iSample = 1:nSamples
                    for iCol = 1:nCols
                        for iRow = 1:nRows
                            coefs = A(:,iRow,iCol,iSample);
                            Z((iRow-1)*decV+1:iRow*decV,...
                                (iCol-1)*decH+1:iCol*decH,...
                                iComponent,iSample) = ...
                                reshape(Cvh_T*coefs,decV,decH);
                        end
                    end
                end
            end
        end
    end
    
end

