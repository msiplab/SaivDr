classdef AbstSparseApproximationSystem < matlab.System %#codegen
    %ABSTSPARSEAPPROXIMATIONSYSTEM Abstract class of sparse approximation
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
            s.Dictionary{obj.FORWARD} = ...
                matlab.System.saveObject(obj.Dictionary{obj.FORWARD});
            s.Dictionary{obj.ADJOINT} = ...
                matlab.System.saveObject(obj.Dictionary{obj.ADJOINT});
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            obj.Dictionary{obj.FORWARD} = ...
                matlab.System.loadObject(s.Dictionary{obj.FORWARD});
            obj.Dictionary{obj.ADJOINT} = ...
                matlab.System.loadObject(s.Dictionary{obj.ADJOINT});
        end         
        
        function validatePropertiesImpl(obj)
            if isempty(obj.Dictionary{obj.FORWARD})
                me = MException('SaivDr:InstantiationException',...
                    'Synthesizer(Dictionary{1}) must be given.');
                throw(me)
            end
            if isempty(obj.Dictionary{obj.ADJOINT})
                me = MException('SaivDr:InstantiationException',...
                    'AdjOfSynthesizer(Dictionary{2}) must be given.');
                throw(me)
            end
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
