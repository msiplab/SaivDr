classdef AbstSparseApproximationSystem < matlab.System %#codegen
    %ABSTSPARSEAPPROXIMATIONSYSTEM Abstract class of sparse approximation
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
    
    properties (Constant)
        FORWARD = 1
        ADJOINT = 2
    end
    
    properties
        StepMonitor
        Dictionary = cell(2,1)
    end
    
    methods
        function obj = AbstSparseApproximationSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.Dictionary{1} = matlab.System.saveObject(obj.Dictionary{1});
            s.Dictionary{2} = matlab.System.saveObject(obj.Dictionary{2});
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            obj.Dictionary{1} = matlab.System.loadObject(s.Dictionary{1});
            obj.Dictionary{2} = matlab.System.loadObject(s.Dictionary{2});
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
