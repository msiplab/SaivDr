classdef CoefsManipulator < matlab.System
    %COEFSMANIPULATOR Coefficient manipulator for OLS/OLA wrapper classes
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (Nontunable)
    end
    
    properties
        Steps
        IntermediateData
    end
    
    methods
        
        % Constractor
        function obj = CoefsManipulator(varargin)
            setProperties(obj,nargin,varargin{:})
        end
        
    end
    
    methods(Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(obj,~)
            if ~isempty(obj.Steps) && length(obj.Steps) > 1
                nSteps = length(obj.Steps);
                obj.IntermediateData = cell(nSteps-1,1);
            end
        end
        
        function coefspst = stepImpl(obj,coefspre)
            if isempty(obj.Steps)
                coefspst = coefspre;
            else
                nSteps = length(obj.Steps);
                x = coefspre;
                for iStep = 1:nSteps
                    step = obj.Steps{iStep};
                    x = step(x);
                    if iStep < nSteps
                        obj.IntermediateData{iStep} = x;
                    else
                        coefspst = x;
                    end
                end
            end
        end
    end
end

