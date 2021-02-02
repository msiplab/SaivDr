classdef OlsOlaProcess2d < matlab.System
    %OLSOLAPROCESS2D OLS/OLA wrapper for 2-D analysis and synthesis system
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
        Synthesizer
        BoundaryOperation
        PadSize = [0 0]
        SplitFactor = []
        CoefsManipulator = []
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
        coefsmanipulators
        refSize
        refSubSize
        refScales
        subPadSize
        subPadArrays
        nWorkers
    end
    
    methods
        
        % Constractor
        function obj = OlsOlaProcess2d(varargin)
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
            if isempty(obj.CoefsManipulator)
                import saivdr.utility.CoefsManipulator
                obj.CoefsManipulator = CoefsManipulator();
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
            s.refScales = obj.refScales;
            s.subPadSize = obj.subPadSize;
            s.subPadArrays = obj.subPadArrays;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.subPadArrays = s.subPadArrays;            
            obj.subPadSize = s.subPadSize;            
            obj.refScales = s.refScales;
            obj.nWorkers = s.nWorkers;
            obj.Synthesizer = matlab.System.loadObject(s.Synthesizer);
            obj.Analyzer = matlab.System.loadObject(s.Analyzer);
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(obj,srcImg)
            % Preperation
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            nSplit = verticalSplitFactor*horizontalSplitFactor;
            
            % Analyzers
            obj.Analyzer.release();
            refAnalyzer = obj.Analyzer.clone();
            [refCoefs,refScales_] = refAnalyzer.step(srcImg);
            
            % Synthesizers
            obj.Synthesizer.release();
            refSynthesizer = obj.Synthesizer.clone();
            
            % Manipulators
            obj.CoefsManipulator.release();
            refCoefsManipulator = obj.CoefsManipulator.clone();
    
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
            obj.coefsmanipulators = cell(nSplit,1);
            if obj.UseParallel
                obj.nWorkers = Inf;
                for iSplit=1:nSplit
                    obj.analyzers{iSplit} = clone(obj.Analyzer);
                    obj.synthesizers{iSplit} = clone(obj.Synthesizer);
                    obj.coefsmanipulators{iSplit} = clone(obj.CoefsManipulator);
                end
            else
                obj.nWorkers = 0;
                for iSplit=1:nSplit
                    obj.analyzers{iSplit} = obj.Analyzer;
                    obj.synthesizers{iSplit} = obj.Synthesizer;
                    obj.coefsmanipulators{iSplit} = clone(obj.CoefsManipulator);
                end
            end

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
                refCoefsOut = refCoefsManipulator.step(refCoefs);
                imgExpctd = refSynthesizer.step(refCoefsOut,refScales_);
                %
                imgActual = obj.stepImpl(srcImg);
                diffImg = imgExpctd - imgActual;
                if norm(diffImg(:))/numel(diffImg) > 1e-6
                    throw(MException(exceptionId,message))
                end
            end

            % Delete reference analyzer and synthesizer
            refAnalyzer.delete()
            refSynthesizer.delete()
            refCoefsManipulator.delete()
        end
        
        function recImg = stepImpl(obj,srcImg)

            % Parameters
            nWorkers_ = obj.nWorkers;
            analyzers_ = obj.analyzers;
            synthesizers_ = obj.synthesizers;
            coefsmanipulators_ = obj.coefsmanipulators;
            
            % Define support functions 
            extract_ols = @(c,s) obj.extract_ols_(c,s);
            padding_ola = @(c) obj.padding_ola_(c);
            arr2vec = @(a) obj.arr2vec_(a);            
            
            % Circular global padding
            srcImg_ = padarray(srcImg,obj.PadSize,'circular');
            
            % Overlap save split
            subImgs = obj.split_ols_(srcImg_);
            
            % Parallel processing
            nSplit = length(subImgs);
            subRecImg = cell(nSplit,1);            
            %
            parfor (iSplit=1:nSplit,nWorkers_)
                % Analyze
                [subCoefs, subScales] = ...
                    analyzers_{iSplit}.step(subImgs{iSplit});
                
                % Extract significant coefs.
                coefspre = extract_ols(subCoefs,subScales);
                
                % Process for coefficients
                coefspst = ...
                    coefsmanipulators_{iSplit}.step(coefspre);
                
                % Zero padding for convolution
                subCoefArray = padding_ola(coefspst);
                
                % Synthesis
                [subCoefs,subScales] = arr2vec(subCoefArray);
                subRecImg{iSplit} = ...
                    step(synthesizers_{iSplit},subCoefs,subScales);
            end
            % Overlap add (Circular)
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
        
        function subCoefArray = padding_ola_(obj,subCoefArray)
            import saivdr.dictionary.utility.Direction
            nChs = size(subCoefArray,2);
            subPadSize_ = obj.subPadSize;
            subPadArrays_ = obj.subPadArrays;
            for iCh = 1:nChs
                sRowIdx = subPadSize_(iCh,Direction.VERTICAL)+1;
                eRowIdx = sRowIdx + size(subCoefArray{iCh},Direction.VERTICAL)-1;
                sColIdx = subPadSize_(iCh,Direction.HORIZONTAL)+1;
                eColIdx = sColIdx + size(subCoefArray{iCh},Direction.HORIZONTAL)-1;
                tmpArray = subPadArrays_{iCh};
                tmpArray(sRowIdx:eRowIdx,sColIdx:eColIdx) ...
                    = subCoefArray{iCh};
                subCoefArray{iCh} = tmpArray;
            end
        end
        
        
        function [coefsCrop,scalesCrop] = ...
                extract_ols_(obj,coefsSplit,scalesSplit)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            refSubScales = obj.refScales*...
                diag(1./[verticalSplitFactor,horizontalSplitFactor]);
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
                %
                tmpArrayCrop = tmpArray(sRowIdx:eRowIdx,sColIdx:eColIdx);
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
    
    methods (Access = private, Static)

        function [subCoefs,subScales] = arr2vec_(subCoefArray)
            nChs = size(subCoefArray,2);
            subScales = zeros(nChs,2);
            tmpCoefs_ = cell(1,nChs);
            for iCh = 1:nChs
                tmpArray = subCoefArray{iCh};
                subScales(iCh,:) = size(tmpArray);
                tmpCoefs_{iCh} = tmpArray(:).';
            end
            subCoefs = cell2mat(tmpCoefs_);
        end
        
    end
end

