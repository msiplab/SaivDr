classdef IterativeHardThresholding < ...
        saivdr.sparserep.AbstSparseApproximation %#codegen
    %ITERATIVEHARDTHRESHOLDING Iterative hard thresholding
    %
    % References
    %  - Thomas Blumensath and Mike E. Davies, gIterative hard
    %    thresholding for compressed sensing,h Appl. Comput. Harmon.
    %    Anal., vol. 29, pp. 265-274, 2009.
    %
    %  - Thomas Blumensath and Mike E. Davies, gNormalized iterative
    %    hard thresholding: Guaranteed stability and performance,h
    %    IEEE J. Sel. Topics Signal Process., vol. 4, no. 2, pp. 298-309,
    %    Apr. 2010.
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2018, Shogo MURAMATSU
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
        Synthesizer
        AdjOfSynthesizer
    end
    
    properties
        TolRmse  = 1e-7
        Mu = (1-1e-3)
    end
       
    properties (PositiveInteger)
        MaxIter = 1000
        NumberOfSparseCoefficients = 1
    end    
    
    methods
        function obj = IterativeHardThresholding(varargin)
            obj = ...
                obj@saivdr.sparserep.AbstSparseApproximation(varargin{:});        
        end
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.sparserep.AbstSparseApproximation(obj);
            s.Synthesizer = matlab.System.saveObject(obj.Synthesizer);
            s.AdjOfSynthesizer = ...
                matlab.System.saveObject(obj.AdjOfSynthesizer);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.sparserep.AbstSparseApproximation(obj,s,wasLocked);
            obj.Synthesizer = matlab.System.loadObject(s.Synthesizer);
            obj.AdjOfSynthesizer = ...
                matlab.System.loadObject(s.AdjOfSynthesizer);
        end
        
        function validatePropertiesImpl(obj)
            if isempty(obj.Synthesizer)
                me = MException('SaivDr:InstantiationException',...
                    'Synthesizer must be given.');
                throw(me)
            end
            if isempty(obj.AdjOfSynthesizer)
                me = MException('SaivDr:InstantiationException',...
                    'AdjOfSynthesizer must be given.');
                throw(me)
            end
        end
        
        function [ result, coefvec, scales ] = stepImpl(obj, srcImg)
            nCoefs = obj.NumberOfSparseCoefficients;
            source = im2double(srcImg);
            
            % Initalization
            iIter    = 0;
            result = 0*source;            
            [coefvec,scales] = obj.AdjOfSynthesizer.step(result);
            % Iteration
            while true
                iIter = iIter + 1;
                precoefvec = coefvec;
                % Residual
                residual = source - result;
                % g = Phi.'*r
                [gradvec,~] = obj.AdjOfSynthesizer.step(residual);
                coefvec = precoefvec + obj.Mu*gradvec;
                % Hard thresholding
                [~, idxsort ] = sort(abs(coefvec(:)),1,'descend');
                indexSet = idxsort(1:nCoefs);
                mask = 0*coefvec;
                mask(indexSet) = 1;
                coefvec = mask.*coefvec;
                % Reconstruction
                result= obj.Synthesizer.step(coefvec,scales);
                if ~isempty(obj.StepMonitor) && iIter > 1
                    obj.StepMonitor.step(result);
                end            
                % Evaluation of convergence
                rmse = (norm(coefvec(:)-precoefvec(:),2))/numel(coefvec);
                if (rmse < obj.TolRmse || iIter >= obj.MaxIter)
                    break
                end
                
            end
        end
        
    end
    
end
