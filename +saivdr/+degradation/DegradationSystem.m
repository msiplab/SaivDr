classdef DegradationSystem < matlab.System %#codegen
    %DEGRADATIONSYSTEM Degradation system
    %   
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2017, Shogo MURAMATSU
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
    properties(Nontunable)
        LinearProcess
        NoiseProcess
    end

    methods
        % Constractor
        function obj = DegradationSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end

    methods (Access = protected)
        
        function output = stepImpl(obj,input)
            output = step(obj.NoiseProcess,...
                step(obj.LinearProcess,input));
        end
        
        function validatePropertiesImpl(obj)
            % Check linear process
            if ~isa(obj.LinearProcess,'saivdr.degradation.linearprocess.AbstLinearSystem') 
                error('SaivDr: Invalid linear process');
            elseif strcmp(get(obj.LinearProcess,'ProcessingMode'),'Adjoint')
                error('SaivDr: Processing mode should be Normal.');
            end
            
            % Check noise process
            if ~isa(obj.NoiseProcess,'saivdr.degradation.noiseprocess.AbstNoiseSystem') 
                error('SaivDr: Invalid noise process');
            end            
        end
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
        function N = getNumOutputsImpl(~)
            N = 1;
        end
    end
end
