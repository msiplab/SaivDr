classdef ProxBoxConstraint < matlab.System
    %PROXBOXCONSTRAINT Prox of box constraint
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
    
    properties
       Vmin = -Inf
       Vmax = Inf
    end
    

    methods
        function obj = ProxBoxConstraint(varargin)
            setProperties(obj,nargin,varargin{:})
        end
    end
    
    methods (Access = protected)
        function y = stepImpl(obj,x)
            y = x;
            y(y<=obj.Vmin) = obj.Vmin;
            y(y>=obj.Vmax) = obj.Vmax;
        end
    end
end

