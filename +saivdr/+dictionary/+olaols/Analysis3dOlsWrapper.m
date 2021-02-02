classdef Analysis3dOlsWrapper < saivdr.dictionary.AbstAnalysisSystem
    %ANALYSIS3DOLSWRAPPER OLS wrapper for 3-D analysis system
    %
    % Reference:
    %   Shogo Muramatsu and Hitoshi Kiya,
    %   ''Parallel Processing Techniques for Multidimensional Sampling
    %   Lattice Alteration Based on Overlap-Add and Overlap-Save Methods,'' 
    %   IEICE Trans. on Fundamentals, Vol.E78-A, No.8, pp.939-943, Aug. 1995
    %
    % Requirements: MATLAB R2018a
    %
    % Copyright (c) 2018-2020, Shogo MURAMATSU
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
        BoundaryOperation
        PadSize = [0 0 0]
        OutputType = 'Vector'
        SplitFactor = []
    end
    
    properties (Logical)
        UseParallel = false
        IsIntegrityTest = true
    end
    
    properties (Nontunable, PositiveInteger, Hidden)
        VerticalSplitFactor = 1
        HorizontalSplitFactor = 1
        DepthSplitFactor = 1
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
        OutputTypeSet = ...
            matlab.system.StringSet({'Vector','Cell'});        
    end
    
    properties (Access = private, Nontunable)
        refScales
        refSubSize
        analyzers
        nWorkers
    end
    
    methods
        
        % Constractor
        function obj = Analysis3dOlsWrapper(varargin)
            import saivdr.dictionary.utility.Direction
            setProperties(obj,nargin,varargin{:})
            if ~isempty(obj.Analyzer)
                obj.BoundaryOperation = obj.Analyzer.BoundaryOperation;
            end
            if ~isempty(obj.SplitFactor)
                obj.VerticalSplitFactor = obj.SplitFactor(Direction.VERTICAL);
                obj.HorizontalSplitFactor = obj.SplitFactor(Direction.HORIZONTAL);                
                obj.DepthSplitFactor = obj.SplitFactor(Direction.DEPTH);                                
            end            
        end
    end
    
    methods (Access=protected)
       
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
            s = saveObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj);
            s.Analyzer = matlab.System.saveObject(obj.Analyzer);
            s.nWorkers = obj.nWorkers;            
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.nWorkers = s.nWorkers;            
            obj.Analyzer = matlab.System.loadObject(s.Analyzer);
            loadObjectImpl@saivdr.dictionary.AbstAnalysisystem(obj,s,wasLocked);
        end
        
        
        function setupImpl(obj,srcImg)
            obj.Analyzer.release();
            refAnalyzer = obj.Analyzer.clone();
            %
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            depthSplitFactor = obj.DepthSplitFactor;            
            nSplit = verticalSplitFactor*horizontalSplitFactor*depthSplitFactor;            
            %
            [coefs,scales] = step(refAnalyzer,srcImg);
            obj.refScales = scales;
            obj.refSubSize = size(srcImg)*...
                diag(1./[verticalSplitFactor,...
                horizontalSplitFactor,...
                depthSplitFactor]);
            %
            obj.analyzers = cell(nSplit,1);
            if obj.UseParallel
                obj.nWorkers = Inf;
                for iSplit=1:nSplit
                    obj.analyzers{iSplit} = clone(obj.Analyzer);
                end
            else
                obj.nWorkers = 0;
                for iSplit=1:nSplit
                    obj.analyzers{iSplit} = obj.Analyzer;
                end
            end
                        
            % Evaluate
            % Check if srcImg is divisible by split factors
            exceptionId = 'SaivDr:IllegalSplitFactorException';            
            message = 'Split factor must be a divisor of array size.';
            if sum(mod(obj.refSubSize,1)) ~= 0
                throw(MException(exceptionId,message))                
            end            
            % Check integrity
            if obj.IsIntegrityTest
                exceptionId = 'SaivDr:ReconstructionFailureException';
                message = 'Failure occurs in reconstruction. Please check the split and padding size.';
                if strcmp(obj.OutputType,'Cell')
                    [coefsCrop,scalesCrop] = stepImpl(obj,srcImg);
                    newcoefs = obj.concatenate_(coefsCrop,scalesCrop);
                else
                    newcoefs = stepImpl(obj,srcImg);
                end
                diffCoefs = coefs - newcoefs;
                if norm(diffCoefs(:))/numel(diffCoefs) > 1e-6
                    throw(MException(exceptionId,message))
                end
            end
            % Delete reference synthesizer
            refAnalyzer.delete()
        end
        
        function [coefs, scales] = stepImpl(obj,srcImg)
            
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
           nWorkers_ = obj.nWorkers;
           parfor (iSplit=1:nSplit,nWorkers_)
               [subCoefs_{iSplit}, subScales_{iSplit}] = ...
                   step(analyzers_{iSplit},subImgs{iSplit});
           end
           % 4. Concatinate
           if strcmp(obj.OutputType,'Cell')
               [coefs,scales] = obj.extract_(subCoefs_,subScales_);
           else
               [coefsCrop,scalesCrop] = obj.extract_(subCoefs_,subScales_);
               coefs = obj.concatenate_(coefsCrop,scalesCrop);
               scales = obj.refScales;
           end
        end
        
    end
    
    methods (Access = private)
        
        function [coefsCrop,scalesCrop] = extract_(obj,subCoefs,subScales)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            depthSplitFactor = obj.DepthSplitFactor;
            refSubScales = obj.refScales* diag(...
                1./[verticalSplitFactor,horizontalSplitFactor,depthSplitFactor]);
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
                    sLayIdx = offset(Direction.DEPTH) + 1;
                    eLayIdx = sLayIdx + stepsize(Direction.DEPTH) - 1;
                    %
                    tmpArrayCrop = tmpArray(sRowIdx:eRowIdx,sColIdx:eColIdx,sLayIdx:eLayIdx);
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
            depthSplitFactor = obj.DepthSplitFactor;
            %
            nSplit = length(coefsCrop);
            nChs = size(scalesCrop,1);
            tmpSubArrays = cell(verticalSplitFactor,horizontalSplitFactor,depthSplitFactor);
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
                    iCol = mod(floor((iSplit-1)/(verticalSplitFactor)),horizontalSplitFactor)+1;
                    iLay = floor((iSplit-1)/(verticalSplitFactor*horizontalSplitFactor))+1;                    
                    %
                    tmpSubVec = coefsCrop_(sIdx:eIdx);
                    tmpSubArray = reshape(tmpSubVec,scalesCrop(iCh,:));
                    tmpSubArrays{iRow,iCol,iLay} = tmpSubArray;
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