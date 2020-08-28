classdef OlsOlaProcess3d < saivdr.restoration.AbstOlsOlaProcess
    %OLSOLAPROCESS3D OLS/OLA wrapper for 3-D analysis and synthesis system
    %
    % Reference:
    %   Shogo Muramatsu and Hitoshi Kiya,
    %   ''Parallel Processing Techniques for Multidimensional Sampling
    %   Lattice Alteration Based on Overlap-Add and Overlap-Save Methods,''
    %   IEICE Trans. on Fundamentals, Vol.E78-A, No.8, pp.939-943, Aug. 1995
    %
    % Requirements: MATLAB R2018a
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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
    
    properties (Access = protected, Constant = true)
        DATA_DIMENSION = 3
    end
    
    properties (Nontunable, PositiveInteger, Hidden)
        VerticalSplitFactor = 1
        HorizontalSplitFactor = 1
        DepthSplitFactor = 1
    end
    
    methods
        
        % Constractor
        function obj = OlsOlaProcess3d(varargin)
            import saivdr.dictionary.utility.Direction
            obj = obj@saivdr.restoration.AbstOlsOlaProcess(...
                varargin{:});
            setProperties(obj,nargin,varargin{:})
        end
        
        function [coefs,scales] = getCoefficients(obj)
            nChs = size(obj.refScales,1);
            nRows = obj.VerticalSplitFactor;
            nCols = obj.HorizontalSplitFactor;
            nLays = obj.DepthSplitFactor;
            coefsCell = cell(nRows,nCols,nLays);
            coefs = [];
            for iCh = 1:nChs
                for iLay = 1:nLays
                    for iCol = 1:nCols
                        for iRow = 1:nRows
                            tmp = obj.States{...
                                (iLay-1)*nRows*nCols+(iCol-1)*nRows+iRow};
                            coefsCell{iRow,iCol,iLay} = tmp{iCh};
                        end
                    end
                end
                
                subCoefs = cell2mat(coefsCell);
                coefs = [ coefs subCoefs(:).'];
            end
            scales = obj.refScales;
        end
        
    end
    
    methods(Access = protected)
        
        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'VerticalSplitFactor') || ...
                    strcmp(propertyName,'HorizontalSplitFactor') || ...
                    strcmp(propertyName,'DepthSplitFactor')
                flag = ~isempty(obj.SplitFactor);
            else
                flag = false;
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.restoration.AbstOlsOlaProcess(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.restoration.AbstOlsOlaProcess(obj,s,wasLocked);
        end
        
        function setupImpl(obj,srcImg)
            obj.setupSplitFactor()
            setupImpl@saivdr.restoration.AbstOlsOlaProcess(obj,srcImg);
        end
        
        function setupSplitFactor(obj)
            import saivdr.dictionary.utility.Direction            
            if ~isempty(obj.SplitFactor)
                obj.VerticalSplitFactor = obj.SplitFactor(Direction.VERTICAL);
                obj.HorizontalSplitFactor = obj.SplitFactor(Direction.HORIZONTAL);
                obj.DepthSplitFactor = obj.SplitFactor(Direction.DEPTH);
            else
                obj.SplitFactor(Direction.VERTICAL) = obj.VerticalSplitFactor;
                obj.SplitFactor(Direction.HORIZONTAL) = obj.HorizontalSplitFactor;
                obj.SplitFactor(Direction.DEPTH) = obj.DepthSplitFactor;
            end
        end
        
        function recImg = circular_ola_(obj,subRecImg)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            depthSplitFactor = obj.DepthSplitFactor;
            stepsize = obj.refSubSize;
            overlap = size(subRecImg{1})-stepsize;
            recImg = zeros(obj.refSize+overlap,'like',subRecImg{1});
            % Overlap add
            iSplit = 0;
            tIdxDep = 0;
            for iDepSplit = 1:depthSplitFactor
                sIdxDep = tIdxDep + 1;
                tIdxDep = sIdxDep + stepsize(Direction.DEPTH) - 1;
                eIdxDep = tIdxDep + overlap(Direction.DEPTH);
                tIdxHor = 0;
                for iHorSplit = 1:horizontalSplitFactor
                    sIdxHor = tIdxHor + 1;
                    tIdxHor = sIdxHor + stepsize(Direction.HORIZONTAL) - 1;
                    eIdxHor = tIdxHor + overlap(Direction.HORIZONTAL);
                    tIdxVer = 0;
                    for iVerSplit = 1:verticalSplitFactor
                        iSplit = iSplit + 1;
                        sIdxVer = tIdxVer + 1;
                        tIdxVer = sIdxVer + stepsize(Direction.VERTICAL) - 1;
                        eIdxVer = tIdxVer + overlap(Direction.VERTICAL);
                        recImg(sIdxVer:eIdxVer,sIdxHor:eIdxHor,sIdxDep:eIdxDep) = ...
                            recImg(sIdxVer:eIdxVer,sIdxHor:eIdxHor,sIdxDep:eIdxDep) + ...
                            subRecImg{iSplit};
                    end
                end
            end
            
            % Folding
            recImg(1:overlap(Direction.VERTICAL),:,:) = ...
                recImg(1:overlap(Direction.VERTICAL),:,:) + ...
                recImg(end-overlap(Direction.VERTICAL)+1:end,:,:);
            recImg(:,1:overlap(Direction.HORIZONTAL),:) = ...
                recImg(:,1:overlap(Direction.HORIZONTAL),:) + ...
                recImg(:,end-overlap(Direction.HORIZONTAL)+1:end,:);
            recImg(:,:,1:overlap(Direction.DEPTH)) = ...
                recImg(:,:,1:overlap(Direction.DEPTH)) + ...
                recImg(:,:,end-overlap(Direction.DEPTH)+1:end);
            % Cropping & circular shift
            recImg = circshift(recImg(...
                1:obj.refSize(Direction.VERTICAL),...
                1:obj.refSize(Direction.HORIZONTAL),...
                1:obj.refSize(Direction.DEPTH)),-overlap/2);
        end
        
        function subCoefArrayOut = padding_ola_(obj,subCoefArrayIn)
            import saivdr.dictionary.utility.Direction
            nChs = size(subCoefArrayIn,2);
            subPadSize_ = obj.subPadSize;
            subPadArrays_ = obj.subPadArrays;
            subCoefArrayOut = cell(size(subCoefArrayIn));
            for iCh = 1:nChs
                sRowIdx = subPadSize_(iCh,Direction.VERTICAL)+1;
                eRowIdx = sRowIdx + size(subCoefArrayIn{iCh},Direction.VERTICAL)-1;
                sColIdx = subPadSize_(iCh,Direction.HORIZONTAL)+1;
                eColIdx = sColIdx + size(subCoefArrayIn{iCh},Direction.HORIZONTAL)-1;
                sLayIdx = subPadSize_(iCh,Direction.DEPTH)+1;
                eLayIdx = sLayIdx + size(subCoefArrayIn{iCh},Direction.DEPTH)-1;
                if isa(subCoefArrayIn{iCh},'gpuArray')
                    subCoefArrayOut{iCh} = gpuArray(subPadArrays_{iCh});
                else
                    subCoefArrayOut{iCh} = subPadArrays_{iCh};
                end
                subCoefArrayOut{iCh}(sRowIdx:eRowIdx,sColIdx:eColIdx,sLayIdx:eLayIdx) ...
                    = subCoefArrayIn{iCh};
            end
        end
        
        function [coefsCrop,scalesCrop] = ...
                extract_ols_(obj,coefsSplit,scalesSplit)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            depthSplitFactor = obj.DepthSplitFactor;
            refSubScales = obj.refScales*diag(...
                1./[verticalSplitFactor,horizontalSplitFactor,depthSplitFactor]);
            nChs = size(refSubScales,1);
            %
            coefsCrop = cell(1,nChs);
            eIdx = 0;
            for iCh = 1:nChs
                stepsize = refSubScales(iCh,:);
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scalesSplit(iCh,:)) - 1;
                tmpVec = coefsSplit(sIdx:eIdx);
                tmpArray = reshape(tmpVec,scalesSplit(iCh,:));
                %
                offset = (scalesSplit(iCh,:) - refSubScales(iCh,:))/2;
                sRowIdx = offset(Direction.VERTICAL) + 1;
                eRowIdx = sRowIdx + stepsize(Direction.VERTICAL) - 1;
                sColIdx = offset(Direction.HORIZONTAL) + 1;
                eColIdx = sColIdx + stepsize(Direction.HORIZONTAL) - 1;
                sLayIdx = offset(Direction.DEPTH) + 1;
                eLayIdx = sLayIdx + stepsize(Direction.DEPTH) - 1;
                %
                tmpArrayCrop = tmpArray(sRowIdx:eRowIdx,sColIdx:eColIdx,sLayIdx:eLayIdx);
                coefsCrop{iCh} = tmpArrayCrop;
            end
            if nargout > 1
                scalesCrop = refSubScales;
            end
        end
        
        function subImgs = split_ols_(obj,srcImg)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            depthSplitFactor = obj.DepthSplitFactor;
            nSplit = verticalSplitFactor*...
                horizontalSplitFactor*...
                depthSplitFactor;
            stepsize = obj.refSubSize;
            overlap = 2*obj.PadSize;
            %
            subImgs = cell(nSplit,1);
            idx = 0;
            for iLaySplit = 1:depthSplitFactor
                sLayIdx = (iLaySplit-1)*stepsize(Direction.DEPTH) + 1;
                eLayIdx = iLaySplit*stepsize(Direction.DEPTH) + ...
                    overlap(Direction.DEPTH);
                for iHorSplit = 1:horizontalSplitFactor
                    sColIdx = (iHorSplit-1)*stepsize(Direction.HORIZONTAL) + 1;
                    eColIdx = iHorSplit*stepsize(Direction.HORIZONTAL) + ...
                        overlap(Direction.HORIZONTAL);
                    for iVerSplit = 1:verticalSplitFactor
                        idx = idx + 1;
                        sRowIdx = (iVerSplit-1)*stepsize(Direction.VERTICAL) + 1;
                        eRowIdx = iVerSplit*stepsize(Direction.VERTICAL) + ...
                            overlap(Direction.VERTICAL);
                        subImgs{idx} = srcImg(sRowIdx:eRowIdx,...
                            sColIdx:eColIdx,...
                            sLayIdx:eLayIdx);
                    end
                end
            end
        end
        
    end
    
end