classdef IdentitySynthesisSystem < saivdr.dictionary.AbstSynthesisSystem
    %IDENTITYSYNTHESISSYSTEM Identity synthesis system
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

    
    properties (Nontunable, Access = private)
        dim
    end    
    
    methods
        
        function obj = IdentitySynthesisSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
            %            
            obj.FrameBound = 1;
        end
        
    end
    
    methods (Access=protected)
        
        %function validateInputsImpl(~, ~, scales)
        %end
   
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.dim = obj.dim;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.dim = s.dim;
            loadObjectImpl@matlab.System(obj,s,wasLocked); 
        end
               
        function setupImpl(obj, ~, scales)
            obj.dim = scales(1,:);
        end
        
        function resetImpl(~)
        end
        
        function y = stepImpl(obj, u, ~)
           y = reshape(u,obj.dim(1,:));           
        end

    end
        
end