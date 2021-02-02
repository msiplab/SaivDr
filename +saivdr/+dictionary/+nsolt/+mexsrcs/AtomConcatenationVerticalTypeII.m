classdef AtomConcatenationVerticalTypeII < matlab.System %#codegen
    %ATOMCONCATENATIONVERTICALTYPEII Type-II vertical bases combination
    %
    % SVN identifier:
    % $Id: AtomConcatenationVerticalTypeII.m 683 2015-05-29 08:22:13Z sho $
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
    properties (Access=protected,Nontunable)
        hLen
    end
    
    methods
        function obj = AtomConcatenationVerticalTypeII(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end    

    methods ( Access = protected )
        
                
        function validateInputsImpl(~,~,~,~,paramMtx1,paramMtx2,~)
            if all(size(paramMtx1) == size(paramMtx2))
                error('ps and pa must be differnt from each other.')
            end
        end           
        
        function setupImpl(obj,~,~,~,paramMtx1,paramMtx2,~)
            obj.hLen = [ size(paramMtx1,1) size(paramMtx2,1) ];            
        end
        
        function arrayCoefs = stepImpl(obj,arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt)
            %obj.hLen = [ size(paramMtx1,1) size(paramMtx2,1) ];            
            % Phase 1 for vertical symtif.Direction
            Wy = paramMtx1.';
            for iCol = 1:nCols
                for iRow = 1:nRows
                    arrayCoefs = upperBlockRot_(obj,arrayCoefs,nRows,iRow,iCol,Wy);
                end
            end
            arrayCoefs = blockButterfly_(obj,arrayCoefs);
            arrayCoefs = downShiftUpperCoefs_(obj,arrayCoefs,nRows,nCols);
            arrayCoefs = blockButterfly_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            
            % Phase 2 for vertical symtif.Direction
            Uy = paramMtx2.';
            I = eye(size(Uy));
            for iCol = 1:nCols
                for iRow = 1:nRows
                    if (iRow == 1 && ~isPeriodicExt)
                        U = -I;
                    else
                        U = Uy;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,nRows,iRow,iCol,U);
                end
            end
            arrayCoefs = blockButterfly_(obj,arrayCoefs);
            arrayCoefs = upShiftLowerCoefs_(obj,arrayCoefs,nRows,nCols);
            arrayCoefs = blockButterfly_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
        end
          
        function numIn = getNumInputsImpl(~)
            numIn = 6;
        end
        
        function numOut = getNumOutputsImpl(~)
            numOut = 1;
        end
        
    end
    
    methods ( Access = private )
        
        function arrayCoefs = downShiftUpperCoefs_(obj,arrayCoefs,nRows,nCols)
            for iCol = 1:nCols
                indexCol1 = (iCol-1)*nRows+1;
                colData0 = arrayCoefs(:,indexCol1);
                upperCoefsPre = colData0(1:obj.hLen(2));
                for iRow = 2:nRows
                    indexCol = (iCol-1)*nRows+iRow;
                    colData = arrayCoefs(:,indexCol);
                    upperCoefsCur = colData(1:obj.hLen(2));
                    colData(1:obj.hLen(2)) = upperCoefsPre;
                    arrayCoefs(:,indexCol) = colData;
                    upperCoefsPre = upperCoefsCur;
                end
                colData0(1:obj.hLen(2)) = upperCoefsPre;
                arrayCoefs(:,indexCol1) = colData0;
            end
        end
        
        function arrayCoefs = upShiftLowerCoefs_(obj,arrayCoefs,nRows,nCols)
            for iCol = 1:nCols
                indexCol1 = (iCol-1)*nRows+1;
                colData0 = arrayCoefs(:,indexCol1);
                lowerCoefsPost = colData0(obj.hLen(1)+1:end);
                for iRow = nRows:-1:2
                    indexCol = (iCol-1)*nRows+iRow;
                    colData = arrayCoefs(:,indexCol);
                    lowerCoefsCur = colData(obj.hLen(1)+1:end);
                    colData(obj.hLen(1)+1:end) = lowerCoefsPost;
                    arrayCoefs(:,indexCol) = colData;
                    lowerCoefsPost = lowerCoefsCur;
                end
                colData0(obj.hLen(1)+1:end) = lowerCoefsPost;
                arrayCoefs(:,indexCol1) = colData0;
            end
        end
        
        function arrayCoefs = blockButterfly_(obj,arrayCoefs)
            nChMx = max(obj.hLen);            
            nChMn = min(obj.hLen);
            upper  = arrayCoefs(1:nChMn,:);
            middle = arrayCoefs(nChMn+1:nChMx,:);
            lower  = arrayCoefs(nChMx+1:end,:);
            arrayCoefs = [
                upper + lower;
                1.414213562373095*middle;
                upper - lower];
        end
        
        function arrayCoefs = lowerBlockRot_(obj,arrayCoefs,nRows,iRow,iCol,U)
            indexCol = (iCol-1)*nRows+iRow;
            colData = arrayCoefs(:,indexCol);
            colData(obj.hLen(1)+1:end) = U*colData(obj.hLen(1)+1:end); 
            arrayCoefs(:,indexCol) = colData;
        end     
        
        function arrayCoefs = upperBlockRot_(obj,arrayCoefs,nRows,iRow,iCol,W)
            indexCol = (iCol-1)*nRows+iRow;
            colData = arrayCoefs(:,indexCol);
            colData(1:obj.hLen(1)) = W*colData(1:obj.hLen(1)); 
            arrayCoefs(:,indexCol) = colData;
        end        
        
    end
    
end

