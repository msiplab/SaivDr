classdef Synthesis2dOlaWrapper < saivdr.dictionary.AbstSynthesisSystem
    %SYNTHESIS2DOLAWRAPPER OLA wrapper for 2-D synthesis system
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
        Synthesizer
        BoundaryOperation = 'Circular'
    end
    
    properties (Nontunable, PositiveInteger)
        VerticalSplitFactor = 1
        HorizontalSplitFactor = 1
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
    end
    
    methods
        
        % Constractor
        function obj = Synthesis2dOlaWrapper(varargin)
            setProperties(obj,nargin,varargin{:})
        end
        %{
        function setFrameBound(obj,frameBound)
            obj.FrameBound = frameBound;
        end
        %}
    end
    
    methods (Access = protected)
        
        %{
        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'UseGpu')
                flag = strcmp(obj.FilterDomain,'Frequeny');
            else
                flag = false;
            end
        end        
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.Synthesizer = obj.Synthesizer;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.Synthesizer = s.Synthesizer;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        %}
        function setupImpl(obj,coefs,scales)
            % TODO: 
            % 0. Extract filter support
        end
        
        function recImg = stepImpl(obj,coefs,scales)
            % TODO: 
            % 1. Pad (Circular)
            % 2. Split
            % 3. Synthesize
            % 4. Overlap add
            recImg = stepImpl(obj.Synthesizer,coefs,scales);
        end
        
    end
end