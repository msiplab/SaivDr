classdef Analysis2dOlsWrapper < saivdr.dictionary.AbstAnalysisSystem
    %ANALYSIS2DOLSWRAPPER OLS wrapper for 2-D analysis system
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
        BoundaryOperation
        PadSize = [0 0]
        OutputType = 'Vector'
    end
    
    properties (Logical)
        UseParallel = false
    end
    
    properties (Nontunable)
        SplitFactor = []
    end
    
    properties (Nontunable, PositiveInteger, Hidden)
        VerticalSplitFactor = 1
        HorizontalSplitFactor = 1
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
        refAnalyzer
        analyzers
        nWorkers
    end
    
    methods
        
        % Constractor
        function obj = Analysis2dOlsWrapper(varargin)
            import saivdr.dictionary.utility.Direction
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
    
    methods (Access=protected)
       
        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'VerticalSplitFactor') || ...
                    strcmp(propertyName,'HorizontalSplitFactor') 
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
        
        
        function setupImpl(obj,srcImg,nLevels)
            obj.Analyzer.release();
            obj.refAnalyzer = obj.Analyzer.clone();
            [coefs,scales] = step(obj.refAnalyzer,srcImg,nLevels);
            obj.refScales = scales;
            obj.refSubSize = size(srcImg)*...
                diag(1./[obj.VerticalSplitFactor,obj.HorizontalSplitFactor]);
            %
            nSplit = obj.VerticalSplitFactor*obj.HorizontalSplitFactor;
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
            % Check identity
            exceptionId = 'SaivDr:ReconstructionFailureException';
            message = 'Failure occurs in reconstruction. Please check the split and padding size.';
            if strcmp(obj.OutputType,'Cell')
                [subCoefs_,subScales_] = stepImpl(obj,srcImg,nLevels);
                newcoefs = concatinate_(obj,subCoefs_,subScales_);
            else
                newcoefs = stepImpl(obj,srcImg,nLevels);
            end
            diffCoefs = coefs - newcoefs;
            if norm(diffCoefs(:))/numel(diffCoefs) > 1e-6
                throw(MException(exceptionId,message))
            end
            % Delete reference synthesizer
            obj.refAnalyzer.delete()
        end
        
        function [coefs, scales] = stepImpl(obj,srcImg,nLevels)
            
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
           parfor (iSplit=1:nSplit,obj.nWorkers)
               [subCoefs_{iSplit}, subScales_{iSplit}] = ...
                   step(analyzers_{iSplit},subImgs{iSplit},nLevels);               
           end
           % 4. Concatinate
           if strcmp(obj.OutputType,'Cell')
               coefs = subCoefs_;
               scales = subScales_;
           else
               coefs = concatinate_(obj,subCoefs_,subScales_);
               scales = obj.refScales;
           end
        end
        
    end
    
    methods (Access = private)
        
        function coefs = concatinate_(obj,subCoefs,subScales)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            refSubScales = obj.refScales*...
                diag(1./[verticalSplitFactor,horizontalSplitFactor]);            
            %
            scalesSplit = subScales{1}; % Partial scales
            nSplit = length(subCoefs);           
            nChs = size(refSubScales,1);            
            coefCrop = cell(verticalSplitFactor,horizontalSplitFactor);
            coefVec = cell(1,nChs);
            %
            eIdx = 0;                            
            for iCh = 1:nChs         
                stepsize = refSubScales(iCh,:);
                offset = (scalesSplit(iCh,:) - refSubScales(iCh,:))/2;
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scalesSplit(iCh,:)) - 1;
                for iSplit = 1:nSplit
                    coefsSplit = subCoefs{iSplit}; % Partial Coefs. 
                    %
                    tmpVec = coefsSplit(sIdx:eIdx);
                    tmpArray = reshape(tmpVec,scalesSplit(iCh,:));
                    %
                    sRowIdx = offset(Direction.VERTICAL) + 1;
                    eRowIdx = sRowIdx + stepsize(Direction.VERTICAL) - 1;
                    sColIdx = offset(Direction.HORIZONTAL) + 1; 
                    eColIdx = sColIdx + stepsize(Direction.HORIZONTAL) - 1;
                    %
                    iRow = mod((iSplit-1),verticalSplitFactor)+1;
                    iCol = floor((iSplit-1)/horizontalSplitFactor)+1;
                    coefCrop{iRow,iCol} = ...
                        tmpArray(sRowIdx:eRowIdx,sColIdx:eColIdx);
                end
                tmpArray = cell2mat(coefCrop);
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