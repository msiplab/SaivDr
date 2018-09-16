classdef AbstSparseApproximation < matlab.System %#codegen
    %ABSTSPARSEAPPROXIMATION Abstract class of sparse approximation
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
    
    properties
        StepMonitor
    end
    
    methods
        function obj = AbstSparseApproximation(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected,Sealed)
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
        function N = getNumOutputsImpl(~)
            N = 3; 
        end
    end
    
end
