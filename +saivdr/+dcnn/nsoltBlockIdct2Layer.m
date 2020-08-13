classdef nsoltBlockIdct2Layer < nnet.layer.Layer
    %NSOLTBLOCKIDCT2LAYER
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
        function layer = nsoltBlockIdct2Layer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'DecimationFactor',[])
            addParameter(p,'Name','')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.DecimationFactor = p.Results.DecimationFactor;
            layer.Name = p.Results.Name;
            layer.Description = "Block IDCT of size " ...
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
            %Cv_T = layer.Cv.';
            %Ch_ = layer.Ch;
            Cvh_T = layer.Cvh.';
            nRows = size(X,1);
            nCols = size(X,2);
            nComponents = 1;
            nSamples = size(X,4);
            %nQDecsee = ceil(decV/2)*ceil(decH/2);
            %nQDecsoo = floor(decV/2)*floor(decH/2);
            %nQDecsoe = floor(decV/2)*ceil(decH/2);
            %
            height = decFactor(1)*nRows;
            width = decFactor(2)*nCols;
            Z = zeros(height,width,nComponents,nSamples,'like',X);
            A = permute(X,[3 1 2 4]);
            for iSample = 1:nSamples
                for iComponent = 1:nComponents
                    for iCol = 1:nCols
                        for iRow = 1:nRows
                            coefs = A(:,iRow,iCol,iSample);
                            %cee = coefs(         1:  nQDecsee);
                            %coo = coefs(nQDecsee+1:nQDecsee+nQDecsoo);
                            %coe = coefs(nQDecsee+nQDecsoo+1:nQDecsee+nQDecsoo+nQDecsoe);
                            %ceo = coefs(nQDecsee+nQDecsoo+nQDecsoe+1:end);
                            %x = [
                            %    reshape(cee,ceil(decV/2),ceil(decH/2)) reshape(ceo,ceil(decV/2),floor(decH/2));
                            %    reshape(coe,floor(decV/2),ceil(decH/2)) reshape(coo,floor(decV/2),floor(decH/2))
                            %    ];
                            %
                            %z = Cv_T*x*Ch_;
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

