classdef AbstSparseApproximation < matlab.System %#codegen
    %ABSTSPARSEAPPROXIMATION Abstract class of sparse approximation
    %
    %
    % SVN identifier:
    % $Id: AbstSparseApproximation.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties (Nontunable)
        Synthesizer
        AdjOfSynthesizer
        NumberOfTreeLevels = 1
    end
    
    properties
        StepMonitor
    end
    
    methods
        function obj = AbstSparseApproximation(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected,Sealed)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.Synthesizer = matlab.System.saveObject(obj.Synthesizer);
            s.AdjOfSynthesizer = ...
                matlab.System.saveObject(obj.AdjOfSynthesizer);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@matlab.System(obj,s,wasLocked);
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
        
        function N = getNumInputsImpl(~)
            N = 2;
        end
        
        function N = getNumOutputsImpl(~)
            N = 3; 
        end
    end
    
end
