classdef CoefsManipulator < matlab.System
    %COEFSMANIPULATOR Coefficient manipulator for OLS/OLA wrapper classes
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (Nontunable)
    end
    
    methods
        
        % Constractor
        function obj = CoefsManipulator(varargin)
            setProperties(obj,nargin,varargin{:})
        end
        
    end
    
    methods(Access = protected)
        
        function varargout = stepImpl(~,varargin)
            nSteps = nargin - 2;
            x = varargin{end};
            if nSteps == 0 && nargout == 1
                varargout{1} = x;
            else
                varargout = cell(nargout,1);
                for iStep = 1:nSteps
                    step = varargin{nSteps-iStep+1};
                    x = step(x);
                    if nSteps-iStep+1 <= nargout 
                        varargout{nSteps-iStep+1} = x;
                    end
                end
            end
        end

    end

end

