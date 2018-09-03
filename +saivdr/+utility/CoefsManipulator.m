classdef CoefsManipulator < matlab.System
    %COEFSMANIPULATOR Coefficient manipulator for OLS/OLA wrapper classes
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (Nontunable)
        Manipulation
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
        
        function [coefspst,statepst] = stepImpl(obj,coefspre,statepre)
            manipulation_ = obj.Manipulation;
            
            if isempty(manipulation_)
                coefspst = coefspre;
                statepst = [];
            elseif iscell(coefspre)
                nChs = length(coefspre);
                coefspst = cell(1,nChs);
                if iscell(statepre)
                    statepst = cell(1,nChs);
                    for iCh = 1:nChs
                        [coefspst{iCh},statepst{iCh}] = ...
                            manipulation_(coefspre{iCh},statepre{iCh});
                    end
                elseif isscalar(statepre)
                    statepst = cell(1,nChs);
                    for iCh = 1:nChs
                        [coefspst{iCh},statepst{iCh}] = ...
                            manipulation_(coefspre{iCh},statepre);
                    end
                else
                    id = 'SaivDr:IllegalStateInitialization';
                    message = 'State must be cell or scalar.';
                    throw(MException(id,message))
                end
            else
                [coefspst,statepst] = manipulation_(coefspre,statepre);
            end
        end
        
    end
end