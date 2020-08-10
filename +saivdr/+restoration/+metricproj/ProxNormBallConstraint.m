classdef ProxNormBallConstraint < matlab.System
    %PROXNORMBALLCONSTRAINT Prox of norm ball constraint
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
        Eps = Inf
    end
    
    properties
        Center = 0
    end
    
    methods
        function obj = ProxNormBallConstraint(varargin)
            setProperties(obj,nargin,varargin{:})
        end
    end
    
    methods (Access = protected)
        function y = stepImpl(obj,x)
            center = obj.Center;
            eps    = obj.Eps;
            d = norm(x(:)-center(:),2);
            if d <= eps
                y = x;
            else
                y = center + (eps/d)*(x-center);
            end            
        end
    end
    
end

