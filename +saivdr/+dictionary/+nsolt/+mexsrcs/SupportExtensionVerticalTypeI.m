classdef SupportExtensionVerticalTypeI < matlab.System %#codegen
    %SUPPORTEXTENSIONVERTICALTYPEI Type-I vertical support extension
    %
    % SVN identifier:
    % $Id: SupportExtensionVerticalTypeI.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2015, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    
    properties (Access=protected, Nontunable)
        hLen
    end
    
    methods
        function obj = SupportExtensionVerticalTypeI(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods ( Access = protected )
        
        function validateInputsImpl(~,~,~,~,paramMtx1,paramMtx2,~)
            if any(size(paramMtx1) ~= size(paramMtx2))
                error('ps and pa must be the same as each other.')
            end           
        end        
        
        function setupImpl(obj,~,~,~,paramMtx1,~,~)
            obj.hLen = size(paramMtx1,1);
        end
        
        function arrayCoefs = stepImpl(obj,arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt)
            %
            % Phase 1 for vertical symtif.Direction
            Uy1 = paramMtx1;                                    
            I = eye(size(Uy1));
            arrayCoefs = blockButterfly_(obj,arrayCoefs);
            arrayCoefs = downShiftLowerCoefs_(obj,arrayCoefs,nRows,nCols);
            arrayCoefs = blockButterfly_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            for iCol = 1:nCols
                for iRow = 1:nRows
                    if iRow == 1 && ~isPeriodicExt
                        U = -I;
                    else
                        U = Uy1;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,nRows,iRow,iCol,U);
                end
            end
            
            % Phase 2 for vertical symtif.Direction
            Uy2 = paramMtx2;            
            arrayCoefs = blockButterfly_(obj,arrayCoefs);
            arrayCoefs = upShiftUpperCoefs_(obj,arrayCoefs,nRows,nCols);
            arrayCoefs = blockButterfly_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            for iCol = 1:nCols
                for iRow = 1:nRows
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,nRows,iRow,iCol,Uy2);
                end
            end
      end
      
        function numIn = getNumInputsImpl(~)
            numIn = 6;
        end
        
        function numOut = getNumOutputsImpl(~)
            numOut = 1;
        end      
        
    end
    
    methods( Access = private)
        
        function arrayCoefs = downShiftLowerCoefs_(obj,arrayCoefs,nRows,nCols)
            for iCol = 1:nCols
                indexCol1 = (iCol-1)*nRows+1;
                colData0 = arrayCoefs(:,indexCol1);
                lowerCoefsPre = colData0(obj.hLen+1:end);
                for iRow = 2:nRows
                    indexCol = (iCol-1)*nRows+iRow;
                    colData = arrayCoefs(:,indexCol);
                    lowerCoefsCur = colData(obj.hLen+1:end);
                    colData(obj.hLen+1:end) = lowerCoefsPre;
                    arrayCoefs(:,indexCol) = colData;
                    lowerCoefsPre = lowerCoefsCur;
                end
                colData0(obj.hLen+1:end) = lowerCoefsPre;
                arrayCoefs(:,indexCol1) = colData0;
            end
        end
        
        function arrayCoefs = upShiftUpperCoefs_(obj,arrayCoefs,nRows,nCols)
            for iCol = 1:nCols
                indexCol1 = (iCol-1)*nRows+1;
                colData0 = arrayCoefs(:,indexCol1);
                upperCoefsPost = colData0(1:obj.hLen);
                for iRow = nRows:-1:2
                    indexCol = (iCol-1)*nRows+iRow;
                    colData = arrayCoefs(:,indexCol);
                    upperCoefsCur = colData(1:obj.hLen);
                    colData(1:obj.hLen) = upperCoefsPost;
                    arrayCoefs(:,indexCol) = colData;
                    upperCoefsPost = upperCoefsCur;
                end
                colData0(1:obj.hLen) = upperCoefsPost;
                arrayCoefs(:,indexCol1) = colData0;
            end
        end
        
        function arrayCoefs = blockButterfly_(obj,arrayCoefs)
            upper = arrayCoefs(1:obj.hLen,:);
            lower = arrayCoefs(obj.hLen+1:end,:);
            arrayCoefs = [
                upper + lower;
                upper - lower ];
        end
        
        function arrayCoefs = lowerBlockRot_(obj,arrayCoefs,nRows,iRow,iCol,U)
            indexCol = (iCol-1)*nRows+iRow;
            colData = arrayCoefs(:,indexCol);
            colData(obj.hLen+1:end) = U*colData(obj.hLen+1:end); 
            arrayCoefs(:,indexCol) = colData;
        end      
        
    end
    
end
