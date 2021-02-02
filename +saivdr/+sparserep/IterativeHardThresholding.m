classdef IterativeHardThresholding < ...
        saivdr.sparserep.AbstSparseApproximationSystem %#codegen
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
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
    
    %properties (Nontunable)
    %    Synthesizer
    %    AdjOfSynthesizer
    %end
    
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
                obj@saivdr.sparserep.AbstSparseApproximationSystem(varargin{:});        
        end
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.sparserep.AbstSparseApproximationSystem(obj);
            %s.Synthesizer = matlab.System.saveObject(obj.Synthesizer);
            %s.AdjOfSynthesizer = matlab.System.saveObject(obj.AdjOfSynthesizer);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.sparserep.AbstSparseApproximationSystem(obj,s,wasLocked);
            %obj.Synthesizer = matlab.System.loadObject(s.Synthesizer);
            %obj.AdjOfSynthesizer = matlab.System.loadObject(s.AdjOfSynthesizer);
        end
        
        function [ result, coefvec, scales ] = stepImpl(obj, srcImg)
            fwdDic = obj.Dictionary{obj.FORWARD};
            adjDic = obj.Dictionary{obj.ADJOINT};
            nCoefs = obj.NumberOfSparseCoefficients;
            source = im2double(srcImg);
            
            % Initalization
            iIter    = 0;
            result = 0*source;            
            [coefvec,scales] = adjDic.step(result);
            % Iteration
            while true
                iIter = iIter + 1;
                precoefvec = coefvec;
                % Residual
                residual = source - result;
                % g = Phi.'*r
                [gradvec,~] = adjDic.step(residual);
                coefvec = precoefvec + obj.Mu*gradvec;
                % Hard thresholding
                [~, idxsort ] = sort(abs(coefvec(:)),1,'descend');
                indexSet = idxsort(1:nCoefs);
                mask = 0*coefvec;
                mask(indexSet) = 1;
                coefvec = mask.*coefvec;
                % Reconstruction
                result= fwdDic.step(coefvec,scales);
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
