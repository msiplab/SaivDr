classdef CoefsManipulator < matlab.System
    %COEFSMANIPULATOR Coefficient manipulator for OLS/OLA wrapper classes
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (Nontunable)
    end
    
    properties
        Steps
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
        

        function coefspst = stepImpl(obj,coefspre)
            if isempty(obj.Steps)
                coefspst = coefspre;
            else
                if iscell(coefspre)
                    nChs = length(coefspre);
                    coefspst = cell(1,nChs);
                    for iCh = 1:nChs
                        coefspst{iCh} = obj.steps_(coefspre{iCh});
                    end
                else
                    coefspst = obj.steps_(coefspre);
                end
            end
        end
    end
    
    methods(Access = private)
        function coefspst = steps_(obj,coefspre)
            nSteps = length(obj.Steps);
            x = coefspre;
            for iStep = 1:nSteps
                step = obj.Steps{iStep};
                x = step(x);

            end
                    coefspst = x;            
        end
    end
end

