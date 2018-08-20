classdef Process2dOlsOlaWrapper < matlab.System
    %PROCESS2DOLSOLAWRAPPER OLS/OLA wrapper for 2-D analysis and synthesis system
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
    
    properties (Nontunable)
        Analyzer
        Synthesizer
        BoundaryOperation
        PadSize = [0 0]
        SplitFactor = []
    end
    
    properties (Logical)
        UseParallel = false
        IsIntegrityTest = true
    end
    
    properties (Nontunable, PositiveInteger, Hidden)
        VerticalSplitFactor = 1
        HorizontalSplitFactor = 1
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
    end
    
    properties (Access = private, Nontunable)
        synthesizers
        analyzers
        refSize
        refSubSize
        refScales
        subPadSize
        subPadArrays
        nWorkers
    end
    
    methods
        
        % Constractor
        function obj = Process2dOlsOlaWrapper(varargin)
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.olaols.*
            setProperties(obj,nargin,varargin{:})
            if ~isempty(obj.Analyzer)
                obj.BoundaryOperation = obj.Analyzer.BoundaryOperation;
            end
            if ~isempty(obj.SplitFactor)
                obj.VerticalSplitFactor = obj.SplitFactor(Direction.VERTICAL);
                obj.HorizontalSplitFactor = obj.SplitFactor(Direction.HORIZONTAL);
            end
        end
        
    end
    
    methods(Access = protected)
        
        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'VerticalSplitFactor') || ...
                    strcmp(propertyName,'HorizontalSplitFactor')
                flag = ~isempty(obj.SplitFactor);
            else
                flag = false;
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.Synthesizer = matlab.System.saveObject(obj.Synthesizer);
            s.Analyzer = matlab.System.saveObject(obj.Analyzer);
            s.nWorkers = obj.nWorkers;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.nWorkers = s.nWorkers;
            obj.Synthesizer = matlab.System.loadObject(s.Synthesizer);
            obj.Analyzer = matlab.System.loadObject(s.Analyzer);
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(obj,srcImg,nLevels)
            % Preperation
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            nSplit = verticalSplitFactor*horizontalSplitFactor;
            
            % Analyzers
            obj.Analyzer.release();
            refAnalyzer = obj.Analyzer.clone();
            [refCoefs,refScales_] = refAnalyzer.step(srcImg,nLevels);
            
            % Synthesizers
            obj.Synthesizer.release();
            refSynthesizer = obj.Synthesizer.clone();
    
            % Parameters
            obj.refSize = size(srcImg);
            obj.refSubSize = obj.refSize*...
                diag(1./[verticalSplitFactor,horizontalSplitFactor]);            
            scaleRatio = refScales_*diag(1./obj.refSize);    
            obj.subPadSize = scaleRatio*diag(obj.PadSize);   
            obj.refScales = refScales_;
            
            % Clone
            obj.synthesizers = cell(nSplit,1);
            obj.analyzers = cell(nSplit,1);
            if obj.UseParallel
                obj.nWorkers = Inf;
                for iSplit=1:nSplit
                    obj.analyzers{iSplit} = clone(obj.Analyzer);
                    obj.synthesizers{iSplit} = clone(obj.Synthesizer);
                end
            else
                obj.nWorkers = 0;
                for iSplit=1:nSplit
                    obj.analyzers{iSplit} = obj.Analyzer;
                    obj.synthesizers{iSplit} = obj.Synthesizer;
                end
            end
            %
            % Evaluate
            % Check if srcImg is divisible by split factors
            exceptionId = 'SaivDr:IllegalSplitFactorException';
            message = 'Split factor must be a divisor of array size.';
            if sum(mod(obj.refSubSize,1)) ~= 0
                throw(MException(exceptionId,message))
            end
            % Check if scales are divisible by split factors
            if sum(mod(obj.subPadSize,1)) ~= 0
                throw(MException('SaivDr','Illegal Pad Size.'))                
            end

            % Allocate memory for zero padding of arrays
            nChs = size(refScales_,1);
            obj.subPadArrays = cell(nChs,1);
            nCoefs = 0;
            for iCh = 1:nChs
                subScale = refScales_(iCh,:) * ...
                diag(1./[verticalSplitFactor,horizontalSplitFactor]);
                nDim = subScale+2*obj.subPadSize(iCh,:);
                obj.subPadArrays{iCh} = zeros(nDim);
                nCoefs = nCoefs + prod(nDim);
            end            
            % Check integrity
            if obj.IsIntegrityTest
                exceptionId = 'SaivDr:ReconstructionFailureException';
                message = 'Failure occurs in reconstruction. Please check the split and padding size.';
                %
                refCoefsOut = refCoefs;
                imgExpctd = refSynthesizer.step(refCoefsOut,refScales_);
                imgActual = obj.stepImpl(srcImg,nLevels);
                diffImg = imgExpctd - imgActual;
                if norm(diffImg(:))/numel(diffImg) > 1e-6
                    throw(MException(exceptionId,message))
                end
            end
            %
            % Delete reference analyzer and synthesizer
            refAnalyzer.delete()
            refSynthesizer.delete()
        end
        
        function recImg = stepImpl(obj,srcImg,nLevels)
            % TODO: Include all major steps in parfor loop
            nWorkers_ = obj.nWorkers;
            
            % Analysis
            % 1. Circular padding
            srcImg_ = padarray(srcImg,obj.PadSize,'circular');
            % 2. Overlap save (Split)
            subImgs = split_ols_(obj,srcImg_);
            % 3. Analyze
            nSplit = length(subImgs);
            subCoefs_ = cell(nSplit,1);
            subScales_ = cell(nSplit,1);
            %
            analyzers_ = obj.analyzers;
            parfor (iSplit=1:nSplit,nWorkers_)
                [subCoefs_{iSplit}, subScales_{iSplit}] = ...
                    step(analyzers_{iSplit},subImgs{iSplit},nLevels);
            end
            % 4. Extract & Concatinate
            [coefsin,scales] = obj.extract_(subCoefs_,subScales_);
            
            % Process
            coefsout = coefsin;
            
            % Synthesis
            % 1. Split
            subCoefArrays = obj.merge_(coefsout,scales);
            % 2. Zero Padding
            subCoefArrays = padding_(obj,subCoefArrays); 
            [subCoefs,subScales] = convert_(obj,subCoefArrays);
            % 3. Synthesize
            nSplit = length(subCoefs);
            subRecImg = cell(nSplit,1);
            %
            synthesizers_ = obj.synthesizers;
            parfor (iSplit=1:nSplit,nWorkers_)
               subCoefs_ = subCoefs{iSplit};
               subRecImg{iSplit} = step(synthesizers_{iSplit},subCoefs_,subScales);            
            end
            % 4. Overlap add (Circular)
            recImg = circular_ola_(obj,subRecImg);
        end
    end
    
    methods (Access = private)
        
        function recImg = circular_ola_(obj,subRecImg)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            stepsize = obj.refSubSize;
            overlap = size(subRecImg{1})-stepsize;
            recImg = zeros(obj.refSize+overlap);
            % Overlap add
            iSplit = 0;
            tIdxHor = 0;
            for iHorSplit = 1:horizontalSplitFactor
                sIdxHor = tIdxHor + 1;
                tIdxHor= sIdxHor + stepsize(Direction.HORIZONTAL) - 1;
                eIdxHor = tIdxHor + overlap(Direction.HORIZONTAL);
                tIdxVer = 0;
                for iVerSplit = 1:verticalSplitFactor
                    iSplit = iSplit + 1;
                    sIdxVer = tIdxVer + 1;
                    tIdxVer = sIdxVer + stepsize(Direction.VERTICAL) - 1;
                    eIdxVer = tIdxVer + overlap(Direction.VERTICAL);
                    recImg(sIdxVer:eIdxVer,sIdxHor:eIdxHor) = ...
                        recImg(sIdxVer:eIdxVer,sIdxHor:eIdxHor) + ...
                        subRecImg{iSplit};
                end
            end
            % Folding
            recImg(1:overlap(Direction.VERTICAL),:) = ...
                recImg(1:overlap(Direction.VERTICAL),:) + ...
                recImg(end-overlap(Direction.VERTICAL)+1:end,:);
            recImg(:,1:overlap(Direction.HORIZONTAL)) = ...
                recImg(:,1:overlap(Direction.HORIZONTAL)) + ...
                recImg(:,end-overlap(Direction.HORIZONTAL)+1:end);
            % Cropping & circular shift
            recImg = circshift(imcrop(recImg,[1 1 fliplr(obj.refSize-1)]),...
                -overlap/2);
        end
        
        function [subCoefs,subScales] = convert_(~,subCoefArrays)
            nSplit = size(subCoefArrays,1);
            nChs = size(subCoefArrays,2);
            subScales = zeros(nChs,2);
            subCoefs = cell(nSplit,1);
            for iSplit = 1:nSplit
                tmpCoefs_ = cell(1,nChs);
                for iCh = 1:nChs
                    tmpArray = subCoefArrays{iSplit,iCh};
                    if iSplit == 1
                        subScales(iCh,:) = size(tmpArray);
                    end
                    tmpCoefs_{1,iCh} = tmpArray(:).';
                end
                subCoefs{iSplit} = cell2mat(tmpCoefs_);
            end
        end
        
        function subCoefArrays = padding_(obj,subCoefArrays)
            import saivdr.dictionary.utility.Direction
            nSplit = size(subCoefArrays,1);
            nChs = size(subCoefArrays,2);
            subPadSize_ = obj.subPadSize;
            subPadArrays_ = obj.subPadArrays;
            for iCh = 1:nChs
                sRowIdx = subPadSize_(iCh,Direction.VERTICAL)+1;
                eRowIdx = sRowIdx + size(subCoefArrays{1,iCh},Direction.VERTICAL)-1;
                sColIdx = subPadSize_(iCh,Direction.HORIZONTAL)+1;
                eColIdx = sColIdx + size(subCoefArrays{1,iCh},Direction.HORIZONTAL)-1;
                for iSplit = 1:nSplit
                    tmpArray = subPadArrays_{iCh};
                    tmpArray(sRowIdx:eRowIdx,sColIdx:eColIdx) ...
                        = subCoefArrays{iSplit,iCh};
                    subCoefArrays{iSplit,iCh} = tmpArray;
                end
            end
        end
        
        
        function [subCoefs,subScales] = split_(obj,coefs,scales)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            nSplit = verticalSplitFactor*horizontalSplitFactor;
            % # of channels
            nChs = size(scales,1);
            subScales = scales*diag(...
                [1/verticalSplitFactor, 1/horizontalSplitFactor]);
            subCoefs = cell(nSplit,1);
            %
            eIdx = 0;
            for iCh = 1:nChs
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scales(iCh,:)) - 1;
                nRows = scales(iCh,Direction.VERTICAL);
                nCols = scales(iCh,Direction.HORIZONTAL);
                coefArrays = reshape(coefs(sIdx:eIdx),[nRows nCols]);
                %
                nSubRows = subScales(iCh,Direction.VERTICAL);
                nSubCols = subScales(iCh,Direction.HORIZONTAL);
                iSplit = 0;
                for iHorSplit = 1:horizontalSplitFactor
                    sColIdx = (iHorSplit-1)*nSubCols + 1;
                    eColIdx = iHorSplit*nSubCols;
                    for iVerSplit = 1:verticalSplitFactor
                        iSplit = iSplit + 1;
                        sRowIdx = (iVerSplit-1)*nSubRows + 1;
                        eRowIdx = iVerSplit*nSubRows;
                        tmpArray = ...
                            coefArrays(sRowIdx:eRowIdx,sColIdx:eColIdx);
                        %
                        tmpVec = tmpArray(:).';
                        subCoefs{iSplit} = [ subCoefs{iSplit} tmpVec ];
                    end
                end
            end
        end
        
        
        function subCoefArrays = merge_(obj,subCoefs,subScales)
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            nSplit = verticalSplitFactor*horizontalSplitFactor;
            % # of channels
            nChs = size(subScales,1);
            subCoefArrays = cell(nSplit,nChs);
            %
            for iSplit = 1:nSplit
                coefsSplit = subCoefs{iSplit};
                eIdx = 0;
                for iCh = 1:nChs
                    sIdx = eIdx + 1;
                    eIdx = sIdx + prod(subScales(iCh,:)) - 1;
                    tmpSubCoefs = coefsSplit(sIdx:eIdx);
                    subCoefArrays{iSplit,iCh} = ...
                        reshape(tmpSubCoefs,subScales(iCh,:));
                end
            end
        end
        
        
        function [coefsCrop,scalesCrop] = extract_(obj,subCoefs,subScales)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            refSubScales = obj.refScales*...
                diag(1./[verticalSplitFactor,horizontalSplitFactor]);
            scalesSplit = subScales{1}; % Partial scales
            nSplit = length(subCoefs);
            nChs = size(refSubScales,1);
            %
            coefsCrop = cell(nSplit,1);
            for iSplit = 1:nSplit
                coefsCrop{iSplit} = [];
            end
            %
            for iSplit = 1:nSplit
                coefsSplit = subCoefs{iSplit}; % Partial Coefs.
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
                    %
                    tmpArrayCrop = tmpArray(sRowIdx:eRowIdx,sColIdx:eColIdx);
                    tmpVecCrop = tmpArrayCrop(:).';
                    %
                    coefsCrop{iSplit} = [coefsCrop{iSplit} tmpVecCrop];
                end
            end
            if nargout > 1
                scalesCrop = refSubScales;
            end
        end
        
        function coefs = concatenate_(obj,coefsCrop,scalesCrop)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            %
            nSplit = length(coefsCrop);
            nChs = size(scalesCrop,1);
            tmpSubArrays = cell(verticalSplitFactor,horizontalSplitFactor);
            coefVec = cell(1,nChs);
            %
            eIdx = 0;
            for iCh = 1:nChs
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scalesCrop(iCh,:)) - 1;
                for iSplit = 1:nSplit
                    coefsCrop_ = coefsCrop{iSplit};
                    %
                    iRow = mod((iSplit-1),verticalSplitFactor)+1;
                    iCol = floor((iSplit-1)/horizontalSplitFactor)+1;
                    %
                    tmpSubVec = coefsCrop_(sIdx:eIdx);
                    tmpSubArray = reshape(tmpSubVec,scalesCrop(iCh,:));
                    tmpSubArrays{iRow,iCol} = tmpSubArray;
                end
                tmpArray = cell2mat(tmpSubArrays);
                coefVec{iCh} = tmpArray(:).';
            end
            %
            coefs = cell2mat(coefVec);
        end
        
        function subImgs = split_ols_(obj,srcImg)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            nSplit = verticalSplitFactor*horizontalSplitFactor;
            stepsize = obj.refSubSize;
            overlap = 2*obj.PadSize;
            %
            subImgs = cell(nSplit,1);
            idx = 0;
            for iHorSplit = 1:horizontalSplitFactor
                sColIdx = (iHorSplit-1)*stepsize(Direction.HORIZONTAL) + 1;
                eColIdx = iHorSplit*stepsize(Direction.HORIZONTAL) + ...
                    overlap(Direction.HORIZONTAL);
                for iVerSplit = 1:verticalSplitFactor
                    idx = idx + 1;
                    sRowIdx = (iVerSplit-1)*stepsize(Direction.VERTICAL) + 1;
                    eRowIdx = iVerSplit*stepsize(Direction.VERTICAL) + ...
                        overlap(Direction.VERTICAL);
                    subImgs{idx} = srcImg(sRowIdx:eRowIdx,sColIdx:eColIdx);
                end
            end
        end
    end
end

