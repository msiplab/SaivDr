classdef IterativeSparseApproximater < ...
        saivdr.sparserep.AbstSparseApproximationSystem %#codegen
    %ITERATIVESPARSEAPPROXIMATER Iterative sparse approximation
    %
    % Requirements: MATLAB R2015b
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
        Algorithm
        TolErr = 1e-7
        %
    end
    
    properties (Nontunable, PositiveInteger)
        MaxIter = 1000
    end    
    
    methods
        function obj = IterativeSparseApproximater(varargin)
            obj = ...
                obj@saivdr.sparserep.AbstSparseApproximationSystem(varargin{:});          
        end
    end
    
    methods (Access=protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.sparserep.AbstSparseApproximationSystem(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@saivdr.sparserep.AbstSparseApproximationSystem(obj,s,wasLocked);
        end        
                
        function [ result, coefs, scales ] = stepImpl(obj, srcImg)
            fwdDic = obj.Dictionary{obj.FORWARD};
            adjDic = obj.Dictionary{obj.ADJOINT};            
            obj.Algorithm.Observation = srcImg;
            obj.Algorithm.Dictionary = { fwdDic, adjDic };
            
            % Initialization
            iIter = 0;
            err = Inf; % RMSE between coefficients of successive iterations
            
            % Iteration
            while err >= obj.TolErr && iIter < obj.MaxIter
                iIter = iIter + 1;
                [result,err] = obj.Algorithm.step();
                if ~isempty(obj.StepMonitor) 
                    obj.StepMonitor.step(result);
                end                            
            end
            
            % Output
            if nargout > 1
                [coefs, scales] = obj.Algorithm.getCoefficients();
            end
        end

    end
end
