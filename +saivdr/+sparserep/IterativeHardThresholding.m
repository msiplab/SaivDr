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
    % SVN identifier:
    % $Id: IterativeHardThresholding.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
    %
    
    properties
        TolRes  = 1e-7;
        Mu = (1-1e-3)
    end
       
    properties (PositiveInteger)
        MaxIter = 1000
    end    
    
    methods
        function obj = IterativeHardThresholding(varargin)
            obj = ...
                obj@saivdr.sparserep.AbstSparseApproximation(varargin{:});        
        end
    end
    
    methods (Access=protected)
        
        function [ residual, coefvec, scales ] = ...
                stepImpl(obj, srcImg, nCoefs)
            source = im2double(srcImg);
            
            % Initalization
            iIter    = 0;                
            [coefvec,scales] = step(obj.AdjOfSynthesizer,...
                    source,obj.NumberOfTreeLevels);
            if ~isempty(obj.StepMonitor)
                reset(obj.StepMonitor)
            end
             
            % Iteration
            while true
                iIter = iIter + 1;                
                precoefvec = coefvec;
                % Reconstruction
                reconst = step(obj.Synthesizer,precoefvec,scales);
                if ~isempty(obj.StepMonitor) && iIter > 1
                    step(obj.StepMonitor,reconst);
                end                  
                % Residual
                residual = source - reconst;
                % g = Phi.'*r
                [gradvec,~] = step(obj.AdjOfSynthesizer,...
                    residual,obj.NumberOfTreeLevels);
                coefvec = precoefvec + obj.Mu*gradvec;
                % Hard thresholding
                [~, idxsort ] = sort(abs(coefvec(:)),1,'descend');
                indexSet = idxsort(1:nCoefs);
                mask = 0*coefvec;
                mask(indexSet) = 1;
                coefvec = mask.*coefvec;
                % Evaluation of convergence
                diff = (norm(coefvec(:)-precoefvec(:))/norm(coefvec(:)))^2;
                if (diff < obj.TolRes || iIter >= obj.MaxIter)
                    break
                end
            end
            % Reconstruction
            reconst = step(obj.Synthesizer,coefvec,scales); 
            if ~isempty(obj.StepMonitor) 
                step(obj.StepMonitor,reconst);
            end
            % Residual
            residual = source - reconst;
        end
        
    end
    
end
