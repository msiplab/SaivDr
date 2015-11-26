classdef AbstSynthesisSystem < matlab.System %#~codegen
    %ABSTSYNTHESISSYSTEM Abstract class of synthesis system
    %
    % SVN identifier:
    % $Id: AbstSynthesisSystem.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
    %
    
    properties (SetAccess = protected, GetAccess = public)
        FrameBound
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
