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
    end
    
    properties (Logical)
        UseParallel = false
    end
    
    properties (Nontunable, PositiveInteger)
        VerticalSplitFactor = 1
        HorizontalSplitFactor = 1
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
    end
    
    properties (Access = private, Nontunable)
        refSize
        refSubSize
        refAnalyzer
    end
    
    methods
        
        % Constractor
        function obj = Analysis2dOlsWrapper(varargin)
            setProperties(obj,nargin,varargin{:})
            if ~isempty(obj.Analyzer)
                obj.BoundaryOperation = obj.Analyzer.BoundaryOperation;
            end
        end
    end
    
    methods (Access=protected)
       
        function [coefs, scales] = stepImpl(obj,srcImg,nLevels)
           [coefs, scales] = step(obj.Analyzer,srcImg,nLevels); 
        end
        
    end
    
end