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
    
    properties
        TolRes  = 1e-7
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
            residual = source;
            coefvec = 0;
            if ~isempty(obj.StepMonitor)
                reset(obj.StepMonitor)
            end
            iIter = 0;
            diff = Inf;
            
            % Iteration
            while (diff > obj.TolRes && iIter < obj.MaxIter)
                preresidual = residual;
                % g = Phi'*r
                [gradvec,scales] = step(obj.AdjOfSynthesizer,...
                    residual,obj.NumberOfTreeLevels);
                coefvec = coefvec + obj.Mu*gradvec;
                % Hard thresholding
                [~, idxsort ] = sort(abs(coefvec(:)),1,'descend');
                indexSet = idxsort(1:nCoefs);
                mask = 0*coefvec;
                mask(indexSet) = 1;
                coefvec = mask.*coefvec;
                % Reconstruction
                reconst = step(obj.Synthesizer,coefvec,scales);
                % Residual
                residual = source - reconst;
                %
                if ~isempty(obj.StepMonitor)
                    step(obj.StepMonitor,reconst);
                end
                %
                iIter = iIter + 1;
                diff = norm(residual(:)-preresidual(:))^2/numel(residual);
            end
        end
        
    end
    
end
