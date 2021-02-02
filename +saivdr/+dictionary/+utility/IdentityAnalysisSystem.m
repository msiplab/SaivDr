classdef IdentityAnalysisSystem < saivdr.dictionary.AbstAnalysisSystem
    %IDENTITYANALYSISSYSTEM Identity analysis system
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
    properties (Nontunable)
        IsVectorize = true
    end
    
    
    methods
        function obj = IdentityAnalysisSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
            %
        end
    end
    
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(~,~,~)
        end
        
       function [ coefs, scales ] = stepImpl(obj, u, ~)
            scales = size(u);
            if obj.IsVectorize
                coefs = u(:); 
            else
                coefs = u;
            end
        end
    end
    
end
