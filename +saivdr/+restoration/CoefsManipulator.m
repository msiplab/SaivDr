classdef CoefsManipulator < matlab.System
    %COEFSMANIPULATOR Coefficient manipulator for OLS/OLA wrapper classes
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
        
        function coefspst = stepImpl(obj,coefstmp,coefspre)
            manipulation_ = obj.Manipulation;
            
            if isempty(manipulation_)
                coefspst = coefstmp;
            elseif iscell(coefstmp)
                if iscell(coefspre)
                    coefspst = cellfun(...
                        @(x,y) manipulation_(x,y),coefstmp,coefspre,...
                        'UniformOutput',false);
                elseif isscalar(coefspre)
                    coefspst = cellfun(...
                        @(x) manipulation_(x,coefspre),coefstmp,...
                        'UniformOutput',false);
                else
                    id = 'SaivDr:IllegalStateInitialization';
                    message = ['State must be cell or scalar. ' class(coefspre)];
                    throw(MException(id,message))
                end
            else
                coefspst = manipulation_(coefstmp,coefspre);
            end
        end
        
    end
end