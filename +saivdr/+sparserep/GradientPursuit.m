classdef GradientPursuit < ...
        saivdr.sparserep.AbstSparseApproximationSystem %#codegen
    %GRADIENTPURSUIT Gradient pursuit
    %
    % Reference
    %  - Thomas Blumensath and Mike E. Davies, "Gradient pursuits,"
    %    IEEE Trans. Signal Process., vol. 56, no. 6, pp. 2370-2382,
    %    July 2008.
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
    
     properties (PositiveInteger)
        NumberOfSparseCoefficients = 1
     end
    
    methods
        function obj = GradientPursuit(varargin)
            obj = ...
                obj@saivdr.sparserep.AbstSparseApproximationSystem(varargin{:});        
        end
    end
    
    methods (Access = protected)
        
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
            residual = source;

            % Initialization
            indexSet = [];
            coefvec  = 0;
            if ~isempty(obj.StepMonitor)
                reset(obj.StepMonitor)
            end
            
            % Iteration
            for iCoef = 1:nCoefs
                % g = Phi.'*r
                [gradvec,scales] = adjDic.step(residual);
                if iCoef == 1
                    dirvec = zeros(size(gradvec));
                end
                % i = argmax|gi|
                [~, idxgmax ] = max(gradvec);
                % 
                indexSet = union(indexSet,idxgmax);
                % Calculate update direction
                dirvec(indexSet) = gradvec(indexSet);
                %
                c = fwdDic.step(dirvec,scales);
                %
                a = (residual(:).'*c(:))/(norm(c(:))^2);
                %
                coefvec  = coefvec  + a*dirvec;
                %
                residual = residual - a*c;
                %
                result = source - residual;
                if ~isempty(obj.StepMonitor)
                    step(obj.StepMonitor,result);
                end
                %
            end
        end

    end
    
end
