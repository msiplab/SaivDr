classdef AbstSynthesisSystem < matlab.System %#~codegen
    %ABSTSYNTHESISSYSTEM Abstract class of synthesis system
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
        
    properties (Logical)
        UseGpu = false
    end
    
    properties (SetAccess = protected, GetAccess = public)
        FrameBound
    end
        
    methods
        
        % Constractor
        function obj = AbstSynthesisSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            %if ~license('checkout','distrib_computing_toolbox')
            %    obj.UseGpu = false;
            %elseif gpuDeviceCount() < 1
            %    obj.UseGpu = false;
            %end
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.FrameBound = obj.FrameBound;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.FrameBound = s.FrameBound;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
    end
    
    methods (Access = protected, Sealed = true)
        
        function N = getNumInputsImpl(~)
            N = 2; 
        end
        
        function N = getNumOutputsImpl(~)
            N = 1; 
        end
        
    end

end
